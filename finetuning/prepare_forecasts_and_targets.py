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


def flatten_atmospheric_variables(ds, atmospheric_vars):
    """
    Flatten atmospheric variables with level dimension into separate variables.

    For example, if ds has variable 'temperature' with levels [1000, 850, 500],
    this will create variables 'temperature_1000hPa', 'temperature_850hPa', 'temperature_500hPa'.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset with atmospheric variables that have a 'level' dimension
    atmospheric_vars : dict
        Dictionary mapping base variable names to sets of pressure levels
        e.g., {'temperature': {1000, 850}, 'geopotential': {500}}

    Returns:
    --------
    xr.Dataset
        Dataset with flattened atmospheric variables (no level dimension)
    """
    if not atmospheric_vars or 'level' not in ds.dims:
        return ds

    # Start with surface variables (those without level dimension)
    surface_vars = [v for v in ds.data_vars if 'level' not in ds[v].dims]
    result_ds = ds[surface_vars] if surface_vars else xr.Dataset()

    # Flatten atmospheric variables
    for base_var, levels in atmospheric_vars.items():
        if base_var not in ds.data_vars:
            continue

        for level in sorted(levels):
            try:
                # Select this pressure level
                var_data = ds[base_var].sel(level=level)
                # Create new variable name
                new_var_name = f"{base_var}_{level}hPa"
                # Add to result dataset
                result_ds[new_var_name] = var_data
            except (KeyError, ValueError) as e:
                print(f"    Warning: Could not extract {base_var} at level {level}: {e}")

    return result_ds


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
        ds = xr.open_zarr(file_path, chunks='auto', consolidated=True)

        present_vars = []
        missing_vars = []

        for var in required_vars:
            # Skip computed variables
            if var == "10m_wind_speed":
                present_vars.append(var)
                continue

            # After flattening, atmospheric variables are stored with their full name
            # (e.g., 'temperature_1000hPa' not 'temperature' with level dimension)
            if var in ds.data_vars:
                present_vars.append(var)
            else:
                missing_vars.append(var)

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

    Handles flattened atmospheric variables (e.g., 'temperature_1000hPa').

    Parameters:
    -----------
    existing_path : str
        Path to existing zarr dataset
    new_ds : xr.Dataset
        Dataset containing new variables (already flattened)
    variables_to_merge : list
        List of variable names to merge (e.g., ['temperature_1000hPa', 'geopotential_500hPa'])

    Returns:
    --------
    xr.Dataset : Merged dataset
    """
    print(f"    Merging {len(variables_to_merge)} variables into existing dataset...")

    # Load existing dataset
    existing_ds = xr.open_zarr(existing_path)

    # Extract only the variables we want to add from new_ds
    # Note: new_ds already has flattened atmospheric variables
    vars_to_add = {}
    for var in variables_to_merge:
        if var == "10m_wind_speed":
            continue  # Computed variable

        # The variable should already be in new_ds with its full name (flattened)
        if var in new_ds.data_vars:
            vars_to_add[var] = new_ds[var]
        else:
            print(f"      Warning: Variable '{var}' not found in new dataset, skipping")

    if not vars_to_add:
        print(f"      Warning: No variables to merge!")
        existing_ds.close()
        return existing_ds

    # Create new dataset with added variables
    new_vars_ds = xr.Dataset(vars_to_add)

    # Merge with existing
    merged_ds = xr.merge([existing_ds, new_vars_ds])

    existing_ds.close()

    print(f"      Successfully merged {len(vars_to_add)} variables")
    return merged_ds


def download_forecast_data(data_dir, model_name, region, years, variables, lead_time_hours,
                           region_lat=None, region_lon=None, use_dask_client=True, skip_save=False,
                           return_datasets=False):
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
    skip_save : bool
        If True, pull data from weatherbench but skip saving to disk
    return_datasets : bool
        If True, return list of xarray Datasets instead of file paths

    Returns:
    --------
    list : Paths to downloaded files (if return_datasets=False) or xarray Datasets (if return_datasets=True)
    """
    data_dir = os.path.expanduser(data_dir)

    print(f"\nDownloading {model_name} forecast data for region '{region}'...")
    print(f"  Years: {years}")
    print(f"  Variables: {variables}")
    print(f"  Lead times: {lead_time_hours} hours")

    # Early check: see if all requested years already exist with all variables
    # This avoids starting dask client and opening remote dataset unnecessarily
    all_exist = True
    for year in years:
        status = check_data_exists(data_dir, model_name, region, [year], variables)
        year_status = status[year]
        if not (year_status['exists'] and not year_status['missing_vars']):
            all_exist = False
            break

    if all_exist:
        print(f"  All years already exist with required variables. Skipping download.")
        # Return file paths for existing data
        if return_datasets:
            # If datasets are requested, load them and return
            datasets = []
            for year in years:
                output_path = get_data_path(data_dir, model_name, region, year)
                ds = xr.open_zarr(output_path)
                datasets.append(ds)
            return datasets
        else:
            return [get_data_path(data_dir, model_name, region, year) for year in years]

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
        print_time_and_memory("Dataset opened", start_time)

        # Check which variables are available
        all_vars_to_download = list(set(surface_vars + list(atmospheric_vars.keys())))
        available_vars = [v for v in all_vars_to_download if v in ds.data_vars]

        if len(available_vars) < len(all_vars_to_download):
            missing = set(all_vars_to_download) - set(available_vars)
            print(f"  Warning: Variables not found in remote dataset: {missing}")

        # Select lead times
        lead_times_td = [np.timedelta64(h, 'h') for h in lead_time_hours]

        downloaded_files = []
        downloaded_datasets = []

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

            # Determine which base variables we need for this year
            year_surface_vars = []
            year_atmospheric_vars = {}  # {base_var: set of pressure levels}

            for var in vars_to_download:
                if var == "10m_wind_speed":
                    continue

                base_var, pressure_level = parse_atmospheric_variable(var)

                if pressure_level is not None:
                    if base_var not in year_atmospheric_vars:
                        year_atmospheric_vars[base_var] = set()
                    year_atmospheric_vars[base_var].add(pressure_level)
                else:
                    year_surface_vars.append(base_var)

            # Select only the base variables needed for this year
            year_base_vars = list(set(year_surface_vars + list(year_atmospheric_vars.keys())))
            year_available_vars = [v for v in year_base_vars if v in ds.data_vars]

            if not year_available_vars:
                print(f"    Warning: No variables available in remote dataset for {vars_to_download}")
                continue

            print(f"    Downloading base variables: {year_available_vars}")

            time_range = [f'{year}-01-01', f'{year}-12-31']

            # Select subset - only variables needed for THIS year
            subset = ds[year_available_vars].sel(
                prediction_timedelta=lead_times_td
            )

            # Filter for midnight forecasts only
            subset = subset.sel(init_time=subset.init_time.dt.hour.isin([0]))

            # Select time range
            subset = subset.sel(init_time=slice(time_range[0], time_range[1]))

            # Apply regional subset if provided
            if region_lat is not None and region_lon is not None:
                subset = subset.sel(latitude=region_lat, longitude=region_lon)

            # Select specific pressure levels for atmospheric variables (for this year only)
            if year_atmospheric_vars and 'level' in subset.dims:
                all_levels_needed = set()
                for base_var, levels in year_atmospheric_vars.items():
                    all_levels_needed.update(levels)

                # Only select the pressure levels we need for this year
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

            # Flatten atmospheric variables before saving (only for this year)
            print(f"    Flattening atmospheric variables...")
            subset_flattened = flatten_atmospheric_variables(subset_rechunked, year_atmospheric_vars)

            # Rechunk the flattened dataset
            flat_chunk_dict = {
                'time': time_chunk,
                'prediction_timedelta': len(subset_flattened.prediction_timedelta),
                'latitude': len(subset_flattened.latitude),
                'longitude': len(subset_flattened.longitude)
            }
            subset_flattened = subset_flattened.chunk(flat_chunk_dict)

            # Save or merge
            if year_status['exists'] and year_status['missing_vars']:
                # Merge with existing dataset
                merged_ds = merge_variables_into_dataset(
                    output_path,
                    subset_flattened,
                    vars_to_download
                )
                final_ds = merged_ds

                if not skip_save:
                    print(f"    Saving merged dataset to {output_path}...")
                    with ProgressBar():
                        merged_ds.to_zarr(output_path, mode='w', consolidated=True, zarr_version=2)
                else:
                    print(f"    [SKIP SAVE] Data pulled but not saved to disk")
            else:
                # Save new dataset
                final_ds = subset_flattened

                if not skip_save:
                    print(f"    Saving to {output_path}...")
                    with ProgressBar():
                        subset_flattened.to_zarr(
                            output_path,
                            mode='w',
                            consolidated=True,
                            zarr_version=2
                        )
                else:
                    print(f"    [SKIP SAVE] Data pulled but not saved to disk")

            # Collect datasets if requested
            if return_datasets:
                downloaded_datasets.append(final_ds)

            if not skip_save:
                downloaded_files.append(output_path)
                print(f"    Saved successfully")

        print(f"\nForecast data download complete!")

    finally:
        if client is not None:
            client.close()
            print("\nDask client closed")

    if return_datasets:
        return downloaded_datasets
    else:
        return downloaded_files


def download_target_data(data_dir, model_name, ground_truth_source, region, years, variables,
                        region_lat=None, region_lon=None, use_dask_client=True, skip_save=False,
                        return_datasets=False):
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
    skip_save : bool
        If True, pull data from weatherbench but skip saving to disk
    return_datasets : bool
        If True, return list of xarray Datasets instead of file paths

    Returns:
    --------
    list : Paths to downloaded files (if return_datasets=False) or xarray Datasets (if return_datasets=True)
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

    # Early check: see if all requested years already exist with all variables
    # This avoids starting dask client and opening remote dataset unnecessarily
    all_exist = True
    for year in years:
        status = check_data_exists(data_dir, target, region, [year], variables)
        year_status = status[year]
        if not (year_status['exists'] and not year_status['missing_vars']):
            all_exist = False
            break

    if all_exist:
        print(f"  All years already exist with required variables. Skipping download.")
        # Return file paths for existing data
        if return_datasets:
            # If datasets are requested, load them and return
            datasets = []
            for year in years:
                output_path = get_data_path(data_dir, target, region, year)
                ds = xr.open_zarr(output_path)
                datasets.append(ds)
            return datasets
        else:
            return [get_data_path(data_dir, target, region, year) for year in years]

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
        print_time_and_memory("Dataset opened", start_time)

        # Check which variables are available
        all_vars_to_download = list(set(surface_vars + list(atmospheric_vars.keys())))
        available_vars = [v for v in all_vars_to_download if v in ds.data_vars]

        if len(available_vars) < len(all_vars_to_download):
            missing = set(all_vars_to_download) - set(available_vars)
            print(f"  Warning: Variables not found in remote dataset: {missing}")

        downloaded_files = []
        downloaded_datasets = []

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

            # Determine which base variables we need for this year
            year_surface_vars = []
            year_atmospheric_vars = {}  # {base_var: set of pressure levels}

            for var in vars_to_download:
                if var == "10m_wind_speed":
                    continue

                base_var, pressure_level = parse_atmospheric_variable(var)

                if pressure_level is not None:
                    if base_var not in year_atmospheric_vars:
                        year_atmospheric_vars[base_var] = set()
                    year_atmospheric_vars[base_var].add(pressure_level)
                else:
                    year_surface_vars.append(base_var)

            # Select only the base variables needed for this year
            year_base_vars = list(set(year_surface_vars + list(year_atmospheric_vars.keys())))
            year_available_vars = [v for v in year_base_vars if v in ds.data_vars]

            if not year_available_vars:
                print(f"    Warning: No variables available in remote dataset for {vars_to_download}")
                continue

            print(f"    Downloading base variables: {year_available_vars}")

            time_range = [f'{year}-01-01', f'{year}-12-31']

            # Select subset - only variables needed for THIS year
            subset = ds[year_available_vars].sel(
                time=slice(time_range[0], time_range[1])
            )

            # Apply regional subset if provided
            if region_lat is not None and region_lon is not None:
                subset = subset.sel(latitude=region_lat, longitude=region_lon)

            # Select specific pressure levels for atmospheric variables (for this year only)
            if year_atmospheric_vars and 'level' in subset.dims:
                all_levels_needed = set()
                for base_var, levels in year_atmospheric_vars.items():
                    all_levels_needed.update(levels)

                subset = subset.sel(level=sorted(list(all_levels_needed)))

            # Handle precipitation conversion if needed
            if 'total_precipitation_6hr' in year_available_vars:
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

            # Flatten atmospheric variables before saving (only for this year)
            print(f"    Flattening atmospheric variables...")
            subset_flattened = flatten_atmospheric_variables(subset_rechunked, year_atmospheric_vars)

            # Rechunk the flattened dataset
            flat_chunk_dict = {
                'time': time_chunk,
                'latitude': len(subset_flattened.latitude),
                'longitude': len(subset_flattened.longitude)
            }
            subset_flattened = subset_flattened.chunk(flat_chunk_dict)

            # Save or merge
            if year_status['exists'] and year_status['missing_vars']:
                # Merge with existing dataset
                merged_ds = merge_variables_into_dataset(
                    output_path,
                    subset_flattened,
                    vars_to_download
                )
                final_ds = merged_ds

                if not skip_save:
                    print(f"    Saving merged dataset to {output_path}...")
                    with ProgressBar():
                        merged_ds.to_zarr(output_path, mode='w', consolidated=True, zarr_version=2)
                else:
                    print(f"    [SKIP SAVE] Data pulled but not saved to disk")
            else:
                # Save new dataset
                final_ds = subset_flattened

                if not skip_save:
                    print(f"    Saving to {output_path}...")
                    with ProgressBar():
                        subset_flattened.to_zarr(
                            output_path,
                            mode='w',
                            consolidated=True,
                            zarr_version=2
                        )
                else:
                    print(f"    [SKIP SAVE] Data pulled but not saved to disk")

            # Collect datasets if requested
            if return_datasets:
                downloaded_datasets.append(final_ds)

            if not skip_save:
                downloaded_files.append(output_path)
                print(f"    Saved successfully")

        print(f"\nTarget data download complete!")

    finally:
        if client is not None:
            client.close()
            print("\nDask client closed")

    if return_datasets:
        return downloaded_datasets
    else:
        return downloaded_files


# ============================================================================
# COMPREHENSIVE DATA LOADING/PULLING FUNCTION
# ============================================================================

def load_or_pull_forecast_data(data_dir, model_name, region, years, variables, lead_time_hours,
                                region_lat=None, region_lon=None, skip_save=False):
    """
    Load forecast data from local files if they exist, or pull from weatherbench if missing.
    Optionally save pulled data to disk if skip_save=False.

    Returns the combined xarray Dataset for all years.
    """
    data_dir = os.path.expanduser(data_dir)

    print(f"\nLoading/pulling {model_name} forecast data for region '{region}'...")
    print(f"  Years: {years}")
    print(f"  Variables: {variables}")
    print(f"  Lead times: {lead_time_hours} hours")

    # Check which years exist locally and which need to be pulled
    years_to_load = []
    years_to_pull = []

    for year in years:
        status = check_data_exists(data_dir, model_name, region, [year], variables)
        year_status = status[year]

        if year_status['exists'] and not year_status['missing_vars']:
            years_to_load.append(year)
        else:
            years_to_pull.append(year)

    if years_to_load:
        print(f"  Loading from local files: {years_to_load}")
    if years_to_pull:
        if skip_save:
            print(f"  Pulling from weatherbench (not saving): {years_to_pull}")
        else:
            print(f"  Pulling from weatherbench and saving: {years_to_pull}")

    datasets = []

    # Load from local files
    for year in years_to_load:
        output_path = get_data_path(data_dir, model_name, region, year)
        print(f"    Loading {year} from {output_path}")
        ds = xr.open_zarr(output_path, chunks='auto', consolidated=True)
        # Select required lead times
        lead_times_td = [np.timedelta64(h, 'h') for h in lead_time_hours]
        ds = ds.sel(prediction_timedelta=lead_times_td)
        # select only region of interest if lat/lon provided
        if region_lat is not None and region_lon is not None:
            ds = ds.sel(latitude=region_lat, longitude=region_lon).compute()
        datasets.append(ds)

    # Pull from weatherbench for missing years
    if years_to_pull:
        pulled_datasets = download_forecast_data(
            data_dir, model_name, region, years_to_pull, variables,
            lead_time_hours, region_lat, region_lon, use_dask_client=True,
            skip_save=skip_save, return_datasets=True
        )
        datasets.extend(pulled_datasets)

    # Combine all datasets
    if not datasets:
        raise ValueError(f"No data available for {model_name} in {region} for years {years}")

    combined_ds = xr.concat(datasets, dim='time', combine_attrs='override')
    combined_ds = combined_ds.sortby('time')

    # Remove duplicates
    _, unique_indices = np.unique(combined_ds.time.values, return_index=True)
    combined_ds = combined_ds.isel(time=sorted(unique_indices))

    return combined_ds


def load_or_pull_target_data(data_dir, model_name, ground_truth_source, region, years, variables,
                             region_lat=None, region_lon=None, skip_save=False):
    """
    Load target data from local files if they exist, or pull from weatherbench if missing.
    Optionally save pulled data to disk if skip_save=False.

    Returns the combined xarray Dataset for all years.
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

    print(f"\nLoading/pulling {target} target data for region '{region}'...")
    print(f"  Years: {years}")
    print(f"  Variables: {variables}")

    # Check which years exist locally and which need to be pulled
    years_to_load = []
    years_to_pull = []

    for year in years:
        status = check_data_exists(data_dir, target, region, [year], variables)
        year_status = status[year]

        if year_status['exists'] and not year_status['missing_vars']:
            years_to_load.append(year)
        else:
            years_to_pull.append(year)

    if years_to_load:
        print(f"  Loading from local files: {years_to_load}")
    if years_to_pull:
        if skip_save:
            print(f"  Pulling from weatherbench (not saving): {years_to_pull}")
        else:
            print(f"  Pulling from weatherbench and saving: {years_to_pull}")

    datasets = []

    # Load from local files
    for year in years_to_load:
        output_path = get_data_path(data_dir, target, region, year)
        print(f"    Loading {year} from {output_path}")
        ds = xr.open_zarr(output_path)
        datasets.append(ds)

    # Pull from weatherbench for missing years
    if years_to_pull:
        pulled_datasets = download_target_data(
            data_dir, model_name, ground_truth_source, region, years_to_pull,
            variables, region_lat, region_lon, use_dask_client=True,
            skip_save=skip_save, return_datasets=True
        )
        datasets.extend(pulled_datasets)

    # Combine all datasets
    if not datasets:
        raise ValueError(f"No data available for {target} in {region} for years {years}")

    combined_ds = xr.concat(datasets, dim='time', combine_attrs='override')
    combined_ds = combined_ds.sortby('time')
    print("created combined_ds")

    # Remove duplicates
    _, unique_indices = np.unique(combined_ds.time.values, return_index=True)
    combined_ds = combined_ds.isel(time=sorted(unique_indices))

    return combined_ds


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

# ============================================================================
# LEGACY DATA LOADING - REMOVE THIS FUNCTION WHEN NO LONGER NEEDED
# ============================================================================
def load_combined_dataset_legacy_global(lat_values, lon_values, time_values, root_dir, data_source):
    """
    LEGACY: Loads global yearly data files and subsets to region of interest.

    File format: root_dir/data_source/data_source_year.zarr
    Example: ~/data/pangu/pangu_2018.zarr, ~/data/era5/era5_2019.zarr

    This function loads global data files and subsets them spatially.
    TO REMOVE: Delete this entire function when legacy data is no longer needed.
    """
    min_year = min(time_values).astype('datetime64[Y]').astype(int) + 1970
    max_year = max(time_values).astype('datetime64[Y]').astype(int) + 1970

    file_paths = []
    for year in range(min_year, max_year + 1):
        # Legacy format: data_source/data_source_year.zarr (no region in filename)
        file_pattern = f"{data_source}/{data_source}_{year}.zarr"
        file_paths.append(os.path.join(root_dir, file_pattern))

    if len(file_paths) == 0:
        raise ValueError(f"No files found matching pattern: {file_pattern}")

    # Load datasets individually to handle overlapping time coordinates
    datasets = []

    for file_path in file_paths:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required data file not found: {file_path}")

        # Load global zarr file
        ds = xr.open_zarr(file_path)

        # Select spatial subset (this is the key difference from new loading)
        ds = ds.sel(latitude=lat_values, longitude=lon_values).sortby('latitude')
        datasets.append(ds)

    # Concatenate along time dimension, allowing overlaps
    forecast_ds = xr.concat(datasets, dim='time', combine_attrs='override')

    # Sort by time to ensure monotonic order
    forecast_ds = forecast_ds.sortby('time')

    # Remove any duplicate time steps (keeping first occurrence)
    _, unique_indices = np.unique(forecast_ds.time.values, return_index=True)
    forecast_ds = forecast_ds.isel(time=sorted(unique_indices))

    return forecast_ds
# ============================================================================
# END LEGACY DATA LOADING
# ============================================================================


def load_combined_dataset(lat_values, lon_values, time_values, root_dir, data_source, region):
    """
    Finds all files in the subfolders of root_dir matching file_pattern and combines them.
    Uses new file organization: root_dir/data_source/data_source_region_year.zarr

    Updated to handle overlapping time coordinates across years by loading individually
    and concatenating with proper sorting.
    """

    min_year = min(time_values).astype('datetime64[Y]').astype(int) + 1970
    max_year = max(time_values).astype('datetime64[Y]').astype(int) + 1970

    file_paths = []
    for year in range(min_year, max_year + 1):

        if data_source == "imd":
            file_pattern = f"IMD_0p25deg/data_{year}.nc"
        else:
            # New file organization: data_source/data_source_region_year.zarr
            file_pattern = f"{data_source}/{data_source}_{region}_{year}.zarr"

        file_paths.append(os.path.join(root_dir, file_pattern))

    if len(file_paths) == 0:
        raise ValueError(f"No files found matching pattern: {file_pattern}")

    # Load datasets individually to handle overlapping time coordinates
    datasets = []

    for file_path in file_paths:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required data file not found: {file_path}")

        # Load individual file
        if data_source == "imd":
            ds = xr.open_dataset(file_path)
            ds = ds.rename({
                'TIME': 'time',
                'LONGITUDE': 'longitude',
                'LATITUDE': 'latitude',
                'RAINFALL': 'total_precipitation'
            })
        else:
            ds = xr.open_zarr(file_path)

        # Select spatial subset
        ds = ds.sel(latitude=lat_values, longitude=lon_values).sortby('latitude')
        datasets.append(ds)

    # Concatenate along time dimension, allowing overlaps
    # This will handle the case where forecast valid_times extend across year boundaries
    forecast_ds = xr.concat(datasets, dim='time', combine_attrs='override')

    # Sort by time to ensure monotonic order
    forecast_ds = forecast_ds.sortby('time')

    # Remove any duplicate time steps (keeping first occurrence)
    _, unique_indices = np.unique(forecast_ds.time.values, return_index=True)
    forecast_ds = forecast_ds.isel(time=sorted(unique_indices))

    return forecast_ds


def load_forecasts(data_dir, args, lat_values, lon_values, train=True, patch_num=None,
                   use_legacy_global_data=False, skip_download=False):
    """
    Vectorized version that processes all data at once without loops.
    More memory intensive but faster for reasonable data sizes.

    Parameters:
    -----------
    use_legacy_global_data : bool
        If True, load from global yearly files (e.g., pangu_2018.zarr)
        If False, load from region-specific files (e.g., pangu_india_2018.zarr)
    skip_download : bool
        If True, pull from weatherbench but don't save. If False, save pulled data.
    """
    if train:
        ver_str = "train"
    else:
        ver_str = "test"

    time_start = getattr(args, f"{ver_str}_start")
    time_end = getattr(args, f"{ver_str}_end")

    # Create time range
    time_values = pd.date_range(start=time_start, end=time_end, freq='12h')

    # only keep growing season dates: 3-15 to 10-31
    if args.growing_season_only:
        time_values = time_values[
            (((time_values.month > 3) | ((time_values.month == 3) & (time_values.day >= 15))) &
            (time_values.month <= 10))
        ]

    time_values_np = time_values.to_numpy()

    # Define target dataset name
    if args.ground_truth_source == "":
        if args.model_name == "pangu":
            target = "era5"
        elif args.model_name == "ifs":
            target = "hres_t0"
        elif args.model_name == "aifs":
            target = "era5"
        else:
            raise ValueError(f"Unknown model_name '{args.model_name}' and no ground_truth_source provided")
    else:
        # For now only IMD is supported as alternative ground truth
        target = args.ground_truth_source

    # Determine years needed
    min_year = min(time_values_np).astype('datetime64[Y]').astype(int) + 1970
    max_year = max(time_values_np).astype('datetime64[Y]').astype(int) + 1970
    years_needed = list(range(min_year, max_year + 1))

    # ========================================================================
    # LEGACY: Load datasets using appropriate method based on flag
    # TO REMOVE: Remove this entire if/else block when legacy data not needed
    # ========================================================================
    if use_legacy_global_data:
        # Legacy: Load global files and subset spatially
        print(f"  [LEGACY MODE] Loading global yearly data files...")
        forecast_ds = load_combined_dataset_legacy_global(lat_values, lon_values, time_values_np, data_dir, args.model_name)
        obs_ds = load_combined_dataset_legacy_global(lat_values, lon_values, time_values_np, data_dir, target)
    else:
        # New: Load from local files or pull from weatherbench
        forecast_vars = [v for v in args.training_vars if v != "10m_wind_speed"]
        if "10m_wind_speed" in args.training_vars:
            forecast_vars.extend(["10m_u_component_of_wind", "10m_v_component_of_wind"])

        print("  Loading forecast data...")
        forecast_ds = load_or_pull_forecast_data(
            data_dir, args.model_name, args.region, years_needed, forecast_vars,
            args.lead_time_hours, lat_values, lon_values, skip_save=skip_download
        )

        print("  Loading observation data...")
        target_vars = [v for v in args.output_vars if v != "10m_wind_speed"]
        obs_ds = load_or_pull_target_data(
            data_dir, args.model_name, args.ground_truth_source, args.region,
            years_needed, target_vars, lat_values, lon_values, skip_save=skip_download
        )
        # If data was pulled (not loaded from disk), it's a dask array - load into memory for faster processing
        # Check if data is dask-backed by looking at the first variable
        first_forecast_var = list(forecast_ds.data_vars)[0]
        if hasattr(forecast_ds[first_forecast_var].data, 'compute'):
            print("  Data is dask-backed (pulled from weatherbench). Loading into memory for faster processing...")
            import dask
            # Rechunk for optimal memory layout before computing
            optimal_chunks = {
                'time': len(forecast_ds.time) // 4,  # Split time into 4 chunks for parallel loading
                'latitude': len(forecast_ds.latitude),
                'longitude': len(forecast_ds.longitude)
            }
            if 'prediction_timedelta' in forecast_ds.dims:
                optimal_chunks['prediction_timedelta'] = len(forecast_ds.prediction_timedelta)

            forecast_ds = forecast_ds.chunk(optimal_chunks)
            obs_ds = obs_ds.chunk({
                'time': len(obs_ds.time) // 4,
                'latitude': len(obs_ds.latitude),
                'longitude': len(obs_ds.longitude)
            })

            # Compute both datasets in parallel
            forecast_ds, obs_ds = dask.compute(forecast_ds, obs_ds)
            print("  Data loaded into memory successfully")
    # ========================================================================
    # END LEGACY
    # ========================================================================

    # Create wind speed if needed
    if "10m_wind_speed" in args.training_vars:
        forecast_ds["10m_wind_speed"] = np.sqrt(
            forecast_ds["10m_u_component_of_wind"]**2 +
            forecast_ds["10m_v_component_of_wind"]**2
        )

    if "10m_wind_speed" in args.output_vars:
        obs_ds["10m_wind_speed"] = np.sqrt(
            obs_ds["10m_u_component_of_wind"]**2 +
            obs_ds["10m_v_component_of_wind"]**2
        )

    # Convert lead times to timedelta and select
    lead_times_td = [np.timedelta64(h, 'h') for h in args.lead_time_hours]
    forecast_ds = forecast_ds.sel(prediction_timedelta=lead_times_td)

    # Select lead times and common time range, this prevents off by one errors
    # from lining up forecasts for days on the edges of the time range
    common_times = np.intersect1d(forecast_ds.time.values, obs_ds.time.values)
    common_times = np.intersect1d(common_times, time_values_np)
    forecast_ds = forecast_ds.sel(time=common_times)
    obs_ds = obs_ds.sel(time=common_times)

    # Get dimensions
    n_time = len(common_times)
    n_lead_times = len(lead_times_td)
    n_lat = len(forecast_ds.latitude)
    n_lon = len(forecast_ds.longitude)
    n_training_vars = len(args.training_vars)
    n_output_vars = len(args.output_vars)

    # Stack all dimensions except variables
    forecast_stacked = forecast_ds[args.training_vars].stack(
        sample=['time', 'prediction_timedelta']
    ).to_array()


    forecast_output_stacked = forecast_ds[args.output_vars].stack(
        sample=['time', 'prediction_timedelta']
    ).to_array()


    obs_repeated = obs_ds[args.output_vars].expand_dims(
        prediction_timedelta=lead_times_td
    ).stack(
        sample=['time', 'prediction_timedelta']
    ).to_array()

    # Transpose and reshape to (n_samples, n_features)
    # Check if any of the arrays are dask-backed and compute together if so
    if hasattr(forecast_stacked.data, 'compute'):
        print("  Computing dask arrays (this may take a moment)...")
        import dask
        # Compute all three arrays together for efficiency
        fc_vals, fc_out_vals, obs_vals = dask.compute(
            forecast_stacked.values.T,
            forecast_output_stacked.values.T,
            obs_repeated.values.T
        )
        fc_combined = fc_vals.reshape(-1, n_training_vars * n_lat * n_lon)
        fc_output_combined = fc_out_vals.reshape(-1, n_output_vars * n_lat * n_lon)
        obs_combined = obs_vals.reshape(-1, n_output_vars * n_lat * n_lon)
        print("  Arrays computed and reshaped successfully")
    else:
        # Data already in memory, just reshape
        fc_combined = forecast_stacked.values.T.reshape(-1, n_training_vars * n_lat * n_lon)
        fc_output_combined = forecast_output_stacked.values.T.reshape(-1, n_output_vars * n_lat * n_lon)
        obs_combined = obs_repeated.values.T.reshape(-1, n_output_vars * n_lat * n_lon)

    # Create lead time indices
    lead_time_indices = np.tile(
        np.arange(n_lead_times), n_time
    )

    # Create time array
    all_times = np.repeat(common_times, n_lead_times)
    # Create day-of-year sin/cos features
    day_of_year = pd.DatetimeIndex(common_times).dayofyear.to_numpy() # XX maybe should use all_times? I don't think so but flagging
    # Convert to radians: 2*pi*d/365
    day_of_year_rad = 2 * np.pi * day_of_year / 365.0
    day_of_year_sin = np.sin(day_of_year_rad)
    day_of_year_cos = np.cos(day_of_year_rad)
    # Stack to create [n_time, 2] array, then repeat for each lead time
    day_of_year_features = np.stack([day_of_year_sin, day_of_year_cos], axis=1)
    day_of_year_features = np.repeat(day_of_year_features, n_lead_times, axis=0)

    # Remove any samples with NaN
    valid_mask = ~(np.isnan(fc_combined).any(axis=1) | np.isnan(obs_combined).any(axis=1))
    fc_combined = fc_combined[valid_mask]
    fc_output_combined = fc_output_combined[valid_mask]
    obs_combined = obs_combined[valid_mask]
    lead_time_indices_combined = lead_time_indices[valid_mask]
    day_of_year_features_combined = day_of_year_features[valid_mask]
    all_times = all_times[valid_mask]

    # Calculate mean forecast error
    training_mean_forecast_error = {}

    for lt_idx, lead_time_hours in enumerate(args.lead_time_hours):
        mask = lead_time_indices_combined == lt_idx
        if not np.any(mask):
            continue

        fc_output_lt = fc_output_combined[mask].reshape(-1, n_output_vars, n_lat, n_lon)
        obs_lt = obs_combined[mask].reshape(-1, n_output_vars, n_lat, n_lon)

        # print("force compute if dask arrays")
        # # FORCE COMPUTATION if these are dask arrays
        # if hasattr(fc_output_lt, 'compute'):
        #     fc_output_lt = fc_output_lt.compute()
        # if hasattr(obs_lt, 'compute'):
        #     obs_lt = obs_lt.compute()

        mean_error = fc_output_lt.mean(axis=0) - obs_lt.mean(axis=0)

        for var_idx, var_name in enumerate(args.output_vars):
            key = f"{var_name}_lt{lead_time_hours}h"
            training_mean_forecast_error[key] = mean_error[var_idx]

    # each sample represents one forecast for one specific lead time and time combination
    return (fc_combined, fc_output_combined, obs_combined, lead_time_indices_combined,
            day_of_year_features_combined, all_times, forecast_ds.latitude.values,
            forecast_ds.longitude.values, n_lat, n_lon,
            n_training_vars, n_output_vars, training_mean_forecast_error)


# ============================================================================
# DATA PREPARATION AND LOADING ORCHESTRATION
# ============================================================================

def prepare_data_for_finetuning(data_dir, model_name, ground_truth_source, region,
                                training_vars, output_vars, train_start, train_end,
                                test_start, test_end, lead_time_hours,
                                region_lat=None, region_lon=None, skip_download=False,
                                growing_season_only=False, use_legacy_global_data=False,
                                load_data=True, spatial_subsets=None):
    """
    Main function to prepare forecast and target data for finetuning.
    Checks if data exists, downloads if necessary (unless skip_download=True),
    and optionally loads the data for training and testing.
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
    skip_download : bool, optional
        If True, check for data but skip downloading if missing (default: False)
    growing_season_only : bool, optional
        If True, only keep growing season dates (3-15 to 10-31) (default: False)
    use_legacy_global_data : bool, optional
        If True, load from global yearly files instead of region-specific (default: False)
    load_data : bool, optional
        If True, load and return data. If False, only download/check data (default: True)
    spatial_subsets : list of dict, optional
        List of spatial subsets for patches/bootstrap. Each dict should contain:
        - 'lat_vals': np.ndarray of latitude values
        - 'lon_vals': np.ndarray of longitude values
        - 'id': identifier for the subset (e.g., patch number or bootstrap sample number)
        If None, uses region_lat and region_lon for a single region (default: None)

    Returns:
    --------
    dict or list of dict :
        If spatial_subsets is None: Dictionary containing train and test data with keys:
            'train': tuple of training data (or None if load_data=False)
            'test': tuple of test data (or None if load_data=False)
            'metadata': dict with data_dir, region, forecast_source, target_source, years
        If spatial_subsets is provided: List of dictionaries (one per subset) with same structure
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

    # Note: Data checking, loading, and downloading is now handled by load_forecasts()
    # which will check for local files and pull from weatherbench as needed

    # ========================================================================
    # LOAD DATA FOR TRAINING AND TESTING (optional)
    # ========================================================================
    if load_data:
        # Create args-like object for load_forecasts
        class Args:
            pass

        args = Args()
        args.model_name = model_name
        args.ground_truth_source = ground_truth_source
        args.region = region
        args.training_vars = training_vars
        args.output_vars = output_vars
        args.train_start = train_start
        args.train_end = train_end
        args.test_start = test_start
        args.test_end = test_end
        args.lead_time_hours = lead_time_hours
        args.growing_season_only = growing_season_only

        if spatial_subsets is None:
            # Single region case
            print("\nLoading data for finetuning...")

            # Load training data
            print("\n  Loading training data...")
            train_data = load_forecasts(
                data_dir, args, region_lat, region_lon, train=True,
                use_legacy_global_data=use_legacy_global_data,
                skip_download=skip_download
            )

            # Load test data
            print("\n  Loading test data...")
            test_data = load_forecasts(
                data_dir, args, region_lat, region_lon, train=False,
                use_legacy_global_data=use_legacy_global_data,
                skip_download=skip_download
            )

            print("\n" + "="*70)
            print("DATA LOADING COMPLETE")
            print("="*70)

            return {
                'train': train_data,
                'test': test_data,
                'metadata': {
                    'data_dir': data_dir,
                    'region': region,
                    'forecast_source': model_name,
                    'target_source': target,
                    'years': all_years
                }
            }
        else:
            # Multiple spatial subsets case (patches/bootstrap)
            print(f"\nLoading data for {len(spatial_subsets)} spatial subsets...")
            data_bundles = []

            for i, subset in enumerate(spatial_subsets):
                subset_id = subset['id']
                lat_vals = subset['lat_vals']
                lon_vals = subset['lon_vals']

                print(f"\n  [{i+1}/{len(spatial_subsets)}] Loading data for subset {subset_id}...")
                print(f"    Lat range: [{lat_vals.min():.2f}, {lat_vals.max():.2f}]")
                print(f"    Lon range: [{lon_vals.min():.2f}, {lon_vals.max():.2f}]")

                # Load training data for this subset
                print(f"    Loading training data...")
                train_data = load_forecasts(
                    data_dir, args, lat_vals, lon_vals, train=True,
                    patch_num=subset_id,
                    use_legacy_global_data=use_legacy_global_data,
                    skip_download=skip_download
                )

                # Load test data for this subset
                print(f"    Loading test data...")
                test_data = load_forecasts(
                    data_dir, args, lat_vals, lon_vals, train=False,
                    patch_num=subset_id,
                    use_legacy_global_data=use_legacy_global_data,
                    skip_download=skip_download
                )

                data_bundles.append({
                    'train': train_data,
                    'test': test_data,
                    'metadata': {
                        'data_dir': data_dir,
                        'region': region,
                        'forecast_source': model_name,
                        'target_source': target,
                        'years': all_years,
                        'subset_id': subset_id
                    }
                })

            print("\n" + "="*70)
            print(f"DATA LOADING COMPLETE - {len(data_bundles)} subsets loaded")
            print("="*70)

            return data_bundles
    else:
        print("\n[SKIP DATA LOADING] Data preparation complete. Data will be loaded later.")

        if spatial_subsets is None:
            return {
                'train': None,
                'test': None,
                'metadata': {
                    'data_dir': data_dir,
                    'region': region,
                    'forecast_source': model_name,
                    'target_source': target,
                    'years': all_years
                }
            }
        else:
            # Return list of bundles with None data
            return [{
                'train': None,
                'test': None,
                'metadata': {
                    'data_dir': data_dir,
                    'region': region,
                    'forecast_source': model_name,
                    'target_source': target,
                    'years': all_years,
                    'subset_id': subset['id']
                }
            } for subset in spatial_subsets]
