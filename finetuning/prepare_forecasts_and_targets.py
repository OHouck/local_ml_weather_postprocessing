#!/usr/bin/env python3
"""
Author: Ozma Houck
Filename: finetuning/prepare_forecasts_and_targets.py

Purpose: Dynamically load forecast and target data on-the-fly for finetuning.
Checks if data exists locally, and if not, downloads it from weatherbench2.
Supports atmospheric variables at specific pressure levels.
Organizes data by region: data_dir/model/model_region_year.zarr
"""

import os
import re
import time
import warnings
from datetime import datetime
from pathlib import Path

import dask
import numpy as np
import pandas as pd
import psutil
import xarray as xr
from dask.diagnostics import ProgressBar
from dask.distributed import Client

warnings.filterwarnings('ignore')


def print_time_and_memory(step_name, start_time):
    """Print elapsed time and current memory usage"""
    elapsed = time.time() - start_time
    memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024  # GB
    print(f"  {step_name}: {elapsed:.2f}s | Memory: {memory:.2f} GB")
    return time.time()


def parse_atmospheric_variable(var_name):
    """
    Parse atmospheric variable names to extract variable and pressure level.

    Examples:
        'temperature_1000hPa' -> ('temperature', 1000)
        '2m_temperature' -> ('2m_temperature', None)
        'geopotential_500hPa' -> ('geopotential', 500)

    Parameters:
    -----------
    var_name : str
        Variable name, possibly with pressure level suffix

    Returns:
    --------
    tuple : (base_var_name, pressure_level)
        base_var_name : str - Variable name without pressure suffix
        pressure_level : int or None - Pressure level in hPa, or None if surface variable
    """
    # Pattern to match variables like "temperature_1000hPa"
    pattern = r'^(.+)_(\d+)hPa$'
    match = re.match(pattern, var_name)

    if match:
        base_var = match.group(1)
        pressure = int(match.group(2))
        return base_var, pressure
    else:
        return var_name, None


def get_data_path(data_dir, data_source, region, year):
    """
    Get the file path for a specific data source, region, and year.

    Parameters:
    -----------
    data_dir : str
        Base data directory
    data_source : str
        Data source name (e.g., 'pangu', 'era5')
    region : str
        Region name (e.g., 'odisha', 'usa_south')
    year : int
        Year

    Returns:
    --------
    str : Path to data file
    """
    data_dir = os.path.expanduser(data_dir)
    model_dir = os.path.join(data_dir, data_source)
    os.makedirs(model_dir, exist_ok=True)

    filename = f"{data_source}_{region}_{year}.zarr"
    return os.path.join(model_dir, filename)


def check_variables_in_dataset(file_path, required_vars):
    """
    Check which required variables are present in a dataset.

    Parameters:
    -----------
    file_path : str
        Path to zarr dataset
    required_vars : list
        List of required variable names (may include atmospheric vars like 'temperature_500hPa')

    Returns:
    --------
    tuple : (present_vars, missing_vars)
        present_vars : list - Variables that exist in the dataset
        missing_vars : list - Variables that are missing
    """
    try:
        ds = xr.open_zarr(file_path)

        present_vars = []
        missing_vars = []

        for var in required_vars:
            # Skip computed variables
            if var == "10m_wind_speed":
                present_vars.append(var)
                continue

            base_var, pressure_level = parse_atmospheric_variable(var)

            if base_var not in ds.data_vars:
                missing_vars.append(var)
            elif pressure_level is not None:
                # Check if pressure level exists
                if 'level' in ds[base_var].dims:
                    available_levels = ds[base_var].level.values
                    if pressure_level not in available_levels:
                        missing_vars.append(var)
                    else:
                        present_vars.append(var)
                else:
                    missing_vars.append(var)
            else:
                present_vars.append(var)

        ds.close()
        return present_vars, missing_vars

    except Exception as e:
        print(f"  Error checking variables in {file_path}: {e}")
        return [], required_vars


def check_data_exists(data_dir, data_source, region, years, variables):
    """
    Check if data files exist for the given parameters and contain required variables.

    Parameters:
    -----------
    data_dir : str
        Directory where data is stored
    data_source : str
        Name of data source (e.g., 'pangu', 'era5', 'hres_t0')
    region : str
        Region name
    years : list
        List of years to check
    variables : list
        List of variables to check (including atmospheric vars like 'temperature_500hPa')

    Returns:
    --------
    dict : {year: {'exists': bool, 'missing_vars': list}}
    """
    data_dir = os.path.expanduser(data_dir)
    status = {}

    for year in years:
        file_path = get_data_path(data_dir, data_source, region, year)

        if not os.path.exists(file_path):
            status[year] = {'exists': False, 'missing_vars': variables}
        else:
            present_vars, missing_vars = check_variables_in_dataset(file_path, variables)
            status[year] = {
                'exists': True,
                'missing_vars': missing_vars,
                'present_vars': present_vars
            }

    return status


def convert_init_to_valid_time(ds):
    """
    Convert a dataset from init_time dimension to valid_time dimension.

    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset with dimensions (init_time, prediction_timedelta, latitude, longitude)

    Returns:
    --------
    xarray.Dataset
        Dataset with dimensions (valid_time, prediction_timedelta, latitude, longitude)
    """
    lead_times = np.unique(ds.prediction_timedelta.values)

    converted_lead_times = []
    for lt in lead_times:
        ds_lt = ds.sel(prediction_timedelta=lt)
        ds_lt = ds_lt.copy()
        ds_lt = ds_lt.rename({'init_time': 'valid_time'})
        ds_lt['valid_time'] = ds_lt.valid_time + lt
        converted_lead_times.append(ds_lt)

    # Combine all lead times into a single dataset
    combined_ds = xr.concat(converted_lead_times, dim='prediction_timedelta')
    combined_ds = combined_ds.sortby('valid_time')

    # Determine dimensions based on what's present
    dims = ['valid_time', 'prediction_timedelta']
    if 'level' in combined_ds.dims:
        dims.append('level')
    dims.extend(['latitude', 'longitude'])

    combined_ds = combined_ds.transpose(*dims)

    return combined_ds


def merge_variables_into_dataset(existing_path, new_ds, variables_to_merge):
    """
    Merge new variables into an existing dataset.

    Parameters:
    -----------
    existing_path : str
        Path to existing zarr dataset
    new_ds : xr.Dataset
        Dataset containing new variables
    variables_to_merge : list
        List of variable names to merge

    Returns:
    --------
    xr.Dataset : Merged dataset
    """
    print(f"    Merging {len(variables_to_merge)} variables into existing dataset...")

    # Load existing dataset
    existing_ds = xr.open_zarr(existing_path)

    # Extract only the variables we want to add from new_ds
    vars_to_add = {}
    for var in variables_to_merge:
        base_var, pressure_level = parse_atmospheric_variable(var)

        if base_var in new_ds.data_vars:
            if pressure_level is not None and 'level' in new_ds[base_var].dims:
                # Select specific pressure level
                vars_to_add[var] = new_ds[base_var].sel(level=pressure_level)
            else:
                vars_to_add[base_var] = new_ds[base_var]

    # Create new dataset with added variables
    new_vars_ds = xr.Dataset(vars_to_add)

    # Merge with existing
    merged_ds = xr.merge([existing_ds, new_vars_ds])

    existing_ds.close()

    return merged_ds


def download_forecast_data(data_dir, model_name, region, years, variables, lead_time_hours,
                           region_lat=None, region_lon=None, use_dask_client=True):
    """
    Download forecast data from weatherbench2 for the given parameters.
    Supports atmospheric variables at specific pressure levels.

    Parameters:
    -----------
    data_dir : str
        Directory to save data
    model_name : str
        Name of forecast model (e.g., 'pangu', 'ifs', 'aifs')
    region : str
        Region name for file organization
    years : list
        List of years to download
    variables : list
        List of variables to download (may include 'temperature_500hPa' etc.)
    lead_time_hours : list
        List of lead times in hours
    region_lat : np.ndarray, optional
        Latitude values for regional subset
    region_lon : np.ndarray, optional
        Longitude values for regional subset
    use_dask_client : bool
        Whether to use dask client for parallel processing

    Returns:
    --------
    list : Paths to downloaded files
    """
    data_dir = os.path.expanduser(data_dir)

    print(f"\nDownloading {model_name} forecast data for region '{region}'...")
    print(f"  Years: {years}")
    print(f"  Variables: {variables}")
    print(f"  Lead times: {lead_time_hours} hours")

    start_time = time.time()

    # Parse variables to separate surface and atmospheric
    surface_vars = []
    atmospheric_vars = {}  # {base_var: set of pressure levels}

    for var in variables:
        if var == "10m_wind_speed":
            continue  # Computed variable

        base_var, pressure_level = parse_atmospheric_variable(var)

        if pressure_level is not None:
            if base_var not in atmospheric_vars:
                atmospheric_vars[base_var] = set()
            atmospheric_vars[base_var].add(pressure_level)
        else:
            surface_vars.append(base_var)

    print(f"  Surface variables: {surface_vars}")
    if atmospheric_vars:
        print(f"  Atmospheric variables: {atmospheric_vars}")

    # Set up Dask client if requested
    client = None
    if use_dask_client:
        print("\nSetting up Dask client...")
        client = Client(
            n_workers=2,
            threads_per_worker=4,
            processes=False,
            memory_limit='8GB',
            silence_logs=30
        )
        print(f"  Dask dashboard: {client.dashboard_link}")
        start_time = print_time_and_memory("Dask setup", start_time)

    try:
        # Open the remote dataset
        print("\nOpening remote dataset...")
        if model_name == 'pangu':
            ds = xr.open_zarr(
                "gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr",
                consolidated=True
            )
        elif model_name == 'ifs':
            ds = xr.open_zarr(
                "gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
                consolidated=True
            )
        elif model_name == 'aifs':
            ds = xr.open_zarr(
                "gs://weatherbench2/datasets/aifs/2020-full-0p25.zarr",
                consolidated=True
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Rename time to init_time if needed
        if 'time' in ds.dims:
            ds = ds.rename({'time': 'init_time'})

        print(f"  Dataset opened successfully")
        print(f"  Available variables: {list(ds.data_vars)[:10]}...")
        if 'level' in ds.dims:
            print(f"  Available pressure levels: {ds.level.values}")
        start_time = print_time_and_memory("Dataset opened", start_time)

        # Check which variables are available
        all_vars_to_download = list(set(surface_vars + list(atmospheric_vars.keys())))
        available_vars = [v for v in all_vars_to_download if v in ds.data_vars]

        if len(available_vars) < len(all_vars_to_download):
            missing = set(all_vars_to_download) - set(available_vars)
            print(f"  Warning: Variables not found in remote dataset: {missing}")

        # Select lead times
        lead_times_td = [np.timedelta64(h, 'h') for h in lead_time_hours]

        downloaded_files = []

        # Download year by year
        for year in years:
            output_path = get_data_path(data_dir, model_name, region, year)

            # Check if we need to download or update this file
            status = check_data_exists(data_dir, model_name, region, [year], variables)
            year_status = status[year]

            if year_status['exists'] and not year_status['missing_vars']:
                print(f"\n  Skipping {year}: all variables already exist")
                downloaded_files.append(output_path)
                continue

            vars_to_download = year_status['missing_vars'] if year_status['exists'] else variables
            print(f"\n  Processing year {year}...")
            print(f"    Variables to download: {vars_to_download}")

            time_range = [f'{year}-01-01', f'{year}-12-31']

            # Select subset
            subset = ds[available_vars].sel(
                prediction_timedelta=lead_times_td
            )

            # Filter for midnight forecasts only
            subset = subset.sel(init_time=subset.init_time.dt.hour.isin([0]))

            # Select time range
            subset = subset.sel(init_time=slice(time_range[0], time_range[1]))

            # Apply regional subset if provided
            if region_lat is not None and region_lon is not None:
                subset = subset.sel(latitude=region_lat, longitude=region_lon)

            # Select specific pressure levels for atmospheric variables
            if atmospheric_vars and 'level' in subset.dims:
                all_levels_needed = set()
                for base_var, levels in atmospheric_vars.items():
                    all_levels_needed.update(levels)

                # Only select the pressure levels we need
                subset = subset.sel(level=sorted(list(all_levels_needed)))

            # Convert to valid_time
            subset = convert_init_to_valid_time(subset)
            subset = subset.rename({'valid_time': 'time'})

            # Rechunk for efficient storage
            n_times = len(subset.time)
            time_chunk = min(240, max(n_times // 10, 1))

            chunk_dict = {
                'time': time_chunk,
                'prediction_timedelta': len(subset.prediction_timedelta),
                'latitude': len(subset.latitude),
                'longitude': len(subset.longitude)
            }

            if 'level' in subset.dims:
                chunk_dict['level'] = len(subset.level)

            subset_rechunked = subset.chunk(chunk_dict)

            # Save or merge
            if year_status['exists'] and year_status['missing_vars']:
                # Merge with existing dataset
                merged_ds = merge_variables_into_dataset(
                    output_path,
                    subset_rechunked,
                    vars_to_download
                )

                print(f"    Saving merged dataset to {output_path}...")
                with ProgressBar():
                    merged_ds.to_zarr(output_path, mode='w', consolidated=True, zarr_version=2)
            else:
                # Save new dataset
                print(f"    Saving to {output_path}...")
                with ProgressBar():
                    subset_rechunked.to_zarr(
                        output_path,
                        mode='w',
                        consolidated=True,
                        zarr_version=2
                    )

            downloaded_files.append(output_path)
            print(f"    Saved successfully")

        print(f"\nForecast data download complete!")

    finally:
        if client is not None:
            client.close()
            print("\nDask client closed")

    return downloaded_files


def download_target_data(data_dir, model_name, ground_truth_source, region, years, variables,
                        region_lat=None, region_lon=None, use_dask_client=True):
    """
    Download target/observation data for the given parameters.
    Supports atmospheric variables at specific pressure levels.

    Parameters:
    -----------
    data_dir : str
        Directory to save data
    model_name : str
        Name of forecast model (used to determine default ground truth)
    ground_truth_source : str
        Name of ground truth source (e.g., 'era5', 'hres_t0', or empty string for default)
    region : str
        Region name for file organization
    years : list
        List of years to download
    variables : list
        List of variables to download (may include 'temperature_500hPa' etc.)
    region_lat : np.ndarray, optional
        Latitude values for regional subset
    region_lon : np.ndarray, optional
        Longitude values for regional subset
    use_dask_client : bool
        Whether to use dask client for parallel processing

    Returns:
    --------
    list : Paths to downloaded files
    """
    data_dir = os.path.expanduser(data_dir)

    # Determine target dataset
    if ground_truth_source == "":
        if model_name == "pangu":
            target = "era5"
        elif model_name == "ifs":
            target = "hres_t0"
        elif model_name == "aifs":
            target = "era5"
        else:
            raise ValueError(f"Unknown model_name '{model_name}' and no ground_truth_source provided")
    else:
        target = ground_truth_source

    print(f"\nDownloading {target} target data for region '{region}'...")
    print(f"  Years: {years}")
    print(f"  Variables: {variables}")

    start_time = time.time()

    # Parse variables to separate surface and atmospheric
    surface_vars = []
    atmospheric_vars = {}  # {base_var: set of pressure levels}

    for var in variables:
        if var == "10m_wind_speed":
            continue  # Computed variable

        base_var, pressure_level = parse_atmospheric_variable(var)

        if pressure_level is not None:
            if base_var not in atmospheric_vars:
                atmospheric_vars[base_var] = set()
            atmospheric_vars[base_var].add(pressure_level)
        else:
            surface_vars.append(base_var)

    print(f"  Surface variables: {surface_vars}")
    if atmospheric_vars:
        print(f"  Atmospheric variables: {atmospheric_vars}")

    # Set up Dask client if requested
    client = None
    if use_dask_client:
        print("\nSetting up Dask client...")
        client = Client(
            n_workers=2,
            threads_per_worker=4,
            processes=False,
            memory_limit='8GB',
            silence_logs=30
        )
        print(f"  Dask dashboard: {client.dashboard_link}")
        start_time = print_time_and_memory("Dask setup", start_time)

    try:
        # Open the remote dataset
        print("\nOpening remote dataset...")
        if target == 'era5':
            ds = xr.open_zarr(
                'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3',
                consolidated=True,
                storage_options={"token": "anon"}
            )
            # Rename precipitation variable if present
            if 'total_precipitation' in ds.data_vars:
                ds = ds.rename({'total_precipitation': 'total_precipitation_6hr'})
        elif target == 'hres_t0':
            ds = xr.open_zarr(
                "gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr",
                consolidated=True,
                storage_options={"token": "anon"}
            )
        else:
            raise ValueError(f"Unknown target: {target}")

        print(f"  Dataset opened successfully")
        print(f"  Available variables: {list(ds.data_vars)[:10]}...")
        if 'level' in ds.dims:
            print(f"  Available pressure levels: {ds.level.values}")
        start_time = print_time_and_memory("Dataset opened", start_time)

        # Check which variables are available
        all_vars_to_download = list(set(surface_vars + list(atmospheric_vars.keys())))
        available_vars = [v for v in all_vars_to_download if v in ds.data_vars]

        if len(available_vars) < len(all_vars_to_download):
            missing = set(all_vars_to_download) - set(available_vars)
            print(f"  Warning: Variables not found in remote dataset: {missing}")

        downloaded_files = []

        # Download year by year
        for year in years:
            output_path = get_data_path(data_dir, target, region, year)

            # Check if we need to download or update this file
            status = check_data_exists(data_dir, target, region, [year], variables)
            year_status = status[year]

            if year_status['exists'] and not year_status['missing_vars']:
                print(f"\n  Skipping {year}: all variables already exist")
                downloaded_files.append(output_path)
                continue

            vars_to_download = year_status['missing_vars'] if year_status['exists'] else variables
            print(f"\n  Processing year {year}...")
            print(f"    Variables to download: {vars_to_download}")

            time_range = [f'{year}-01-01', f'{year}-12-31']

            # Select subset
            subset = ds[available_vars].sel(
                time=slice(time_range[0], time_range[1])
            )

            # Apply regional subset if provided
            if region_lat is not None and region_lon is not None:
                subset = subset.sel(latitude=region_lat, longitude=region_lon)

            # Select specific pressure levels for atmospheric variables
            if atmospheric_vars and 'level' in subset.dims:
                all_levels_needed = set()
                for base_var, levels in atmospheric_vars.items():
                    all_levels_needed.update(levels)

                subset = subset.sel(level=sorted(list(all_levels_needed)))

            # Handle precipitation conversion if needed
            if 'total_precipitation_6hr' in available_vars:
                six_hour_precip = subset['total_precipitation_6hr']

                # Shift time coordinate forward by 6 hours
                six_hour_precip = six_hour_precip.assign_coords(
                    time=six_hour_precip.time + pd.Timedelta(hours=6)
                )

                # Create daily precipitation sums
                daily_precip = six_hour_precip.resample(time='1D').sum()
                daily_precip = daily_precip.rename('total_precipitation')

                # Broadcast back to 6-hourly resolution
                total_precipitation = daily_precip.resample(time='6H').ffill()

                # Update subset
                subset = subset.drop_vars('total_precipitation_6hr')
                subset["total_precipitation"] = total_precipitation

            # Filter for specific hours (0, 6, 12 for flexibility)
            subset = subset.sel(time=subset.time.dt.hour.isin([0, 6, 12]))

            # Rechunk for efficient storage
            n_times = len(subset.time)
            time_chunk = min(240, max(n_times // 10, 1))

            chunk_dict = {
                'time': time_chunk,
                'latitude': len(subset.latitude),
                'longitude': len(subset.longitude)
            }

            if 'level' in subset.dims:
                chunk_dict['level'] = len(subset.level)

            subset_rechunked = subset.chunk(chunk_dict)

            # Save or merge
            if year_status['exists'] and year_status['missing_vars']:
                # Merge with existing dataset
                merged_ds = merge_variables_into_dataset(
                    output_path,
                    subset_rechunked,
                    vars_to_download
                )

                print(f"    Saving merged dataset to {output_path}...")
                with ProgressBar():
                    merged_ds.to_zarr(output_path, mode='w', consolidated=True, zarr_version=2)
            else:
                # Save new dataset
                print(f"    Saving to {output_path}...")
                with ProgressBar():
                    subset_rechunked.to_zarr(
                        output_path,
                        mode='w',
                        consolidated=True,
                        zarr_version=2
                    )

            downloaded_files.append(output_path)
            print(f"    Saved successfully")

        print(f"\nTarget data download complete!")

    finally:
        if client is not None:
            client.close()
            print("\nDask client closed")

    return downloaded_files


def prepare_data_for_finetuning(data_dir, model_name, ground_truth_source, region,
                                training_vars, output_vars, train_start, train_end,
                                test_start, test_end, lead_time_hours,
                                region_lat=None, region_lon=None):
    """
    Main function to prepare forecast and target data for finetuning.
    Checks if data exists, and downloads if necessary.
    Supports atmospheric variables at specific pressure levels.

    Parameters:
    -----------
    data_dir : str
        Directory for data storage
    model_name : str
        Name of forecast model
    ground_truth_source : str
        Name of ground truth source (empty string for default)
    region : str
        Region name for file organization
    training_vars : list
        List of training variables (may include 'temperature_500hPa' etc.)
    output_vars : list
        List of output variables
    train_start : str
        Training start date (YYYY-MM-DD)
    train_end : str
        Training end date (YYYY-MM-DD)
    test_start : str
        Test start date (YYYY-MM-DD)
    test_end : str
        Test end date (YYYY-MM-DD)
    lead_time_hours : list
        List of lead times in hours
    region_lat : np.ndarray, optional
        Latitude values for regional subset
    region_lon : np.ndarray, optional
        Longitude values for regional subset

    Returns:
    --------
    dict : Dictionary with status and paths
    """
    data_dir = os.path.expanduser(data_dir)

    print("="*70)
    print("PREPARING DATA FOR FINETUNING")
    print("="*70)

    # Determine years needed
    train_years = list(range(int(train_start[:4]), int(train_end[:4]) + 1))
    test_years = list(range(int(test_start[:4]), int(test_end[:4]) + 1))
    all_years = sorted(set(train_years + test_years))

    print(f"\nRegion: {region}")
    print(f"Years needed: {all_years}")
    print(f"  Training: {train_years}")
    print(f"  Testing: {test_years}")

    # Determine target dataset
    if ground_truth_source == "":
        if model_name == "pangu":
            target = "era5"
        elif model_name == "ifs":
            target = "hres_t0"
        elif model_name == "aifs":
            target = "era5"
        else:
            raise ValueError(f"Unknown model_name '{model_name}' and no ground_truth_source provided")
    else:
        target = ground_truth_source

    # Combine training and output vars (may overlap)
    all_forecast_vars = list(set(training_vars + output_vars))
    # Remove wind_speed as it's computed, not downloaded
    forecast_vars = [v for v in all_forecast_vars if v != "10m_wind_speed"]

    # Check forecast data
    print(f"\nChecking {model_name} forecast data...")
    fc_status = check_data_exists(data_dir, model_name, region, all_years, forecast_vars)

    years_to_download = []
    for year, status in fc_status.items():
        if not status['exists'] or status['missing_vars']:
            years_to_download.append(year)
            if status['exists']:
                print(f"  Year {year}: missing variables {status['missing_vars']}")
            else:
                print(f"  Year {year}: file does not exist")

    if not years_to_download:
        print(f"  ✓ All forecast data exists with required variables")
    else:
        print(f"  Downloading/updating forecast data for years: {years_to_download}")
        download_forecast_data(
            data_dir, model_name, region, years_to_download, forecast_vars,
            lead_time_hours, region_lat, region_lon, use_dask_client=True
        )

    # Check target data
    print(f"\nChecking {target} target data...")
    target_vars = [v for v in output_vars if v != "10m_wind_speed"]
    tgt_status = check_data_exists(data_dir, target, region, all_years, target_vars)

    years_to_download = []
    for year, status in tgt_status.items():
        if not status['exists'] or status['missing_vars']:
            years_to_download.append(year)
            if status['exists']:
                print(f"  Year {year}: missing variables {status['missing_vars']}")
            else:
                print(f"  Year {year}: file does not exist")

    if not years_to_download:
        print(f"  ✓ All target data exists with required variables")
    else:
        print(f"  Downloading/updating target data for years: {years_to_download}")
        download_target_data(
            data_dir, model_name, ground_truth_source, region, years_to_download,
            target_vars, region_lat, region_lon, use_dask_client=True
        )

    print("\n" + "="*70)
    print("DATA PREPARATION COMPLETE")
    print("="*70)

    return {
        'status': 'success',
        'data_dir': data_dir,
        'region': region,
        'forecast_source': model_name,
        'target_source': target,
        'years': all_years
    }
