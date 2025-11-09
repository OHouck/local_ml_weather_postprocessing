#!/usr/bin/env python3
"""
Author: Ozma Houck
Filename: finetuning/prepare_forecasts_and_targets.py

Purpose: Dynamically load forecast and target data on-the-fly for finetuning.
Checks if data exists locally, and if not, downloads it from weatherbench2.
"""

import os
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
        where valid_time = init_time + prediction_timedelta
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
    combined_ds = combined_ds.transpose("valid_time", "prediction_timedelta", "latitude", "longitude")

    return combined_ds


def check_data_exists(data_dir, data_source, years, variables):
    """
    Check if data files exist for the given parameters.

    Parameters:
    -----------
    data_dir : str
        Directory where data is stored
    data_source : str
        Name of data source (e.g., 'pangu', 'era5', 'hres_t0')
    years : list
        List of years to check
    variables : list
        List of variables to check

    Returns:
    --------
    tuple : (all_exist, missing_years)
        all_exist : bool - True if all data exists
        missing_years : list - List of years with missing data
    """
    data_dir = os.path.expanduser(data_dir)
    missing_years = []

    for year in years:
        file_path = os.path.join(data_dir, f"{data_source}_{year}.zarr")

        if not os.path.exists(file_path):
            missing_years.append(year)
            continue

        # Check if the file has the required variables
        try:
            ds = xr.open_zarr(file_path)
            missing_vars = [v for v in variables if v not in ds.data_vars and v != "10m_wind_speed"]
            if missing_vars:
                print(f"  Warning: {file_path} missing variables: {missing_vars}")
                missing_years.append(year)
            ds.close()
        except Exception as e:
            print(f"  Error checking {file_path}: {e}")
            missing_years.append(year)

    all_exist = len(missing_years) == 0
    return all_exist, missing_years


def download_forecast_data(data_dir, model_name, years, variables, lead_time_hours,
                           region_lat=None, region_lon=None, use_dask_client=True):
    """
    Download forecast data from weatherbench2 for the given parameters.

    Parameters:
    -----------
    data_dir : str
        Directory to save data
    model_name : str
        Name of forecast model (e.g., 'pangu', 'ifs', 'aifs')
    years : list
        List of years to download
    variables : list
        List of variables to download
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
    os.makedirs(data_dir, exist_ok=True)

    print(f"\nDownloading {model_name} forecast data...")
    print(f"  Years: {years}")
    print(f"  Variables: {variables}")
    print(f"  Lead times: {lead_time_hours} hours")

    start_time = time.time()

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
        start_time = print_time_and_memory("Dataset opened", start_time)

        # Select variables
        available_vars = [v for v in variables if v in ds.data_vars]
        if len(available_vars) < len(variables):
            missing = set(variables) - set(available_vars)
            print(f"  Warning: Variables not found: {missing}")

        # Select lead times
        lead_times_td = [np.timedelta64(h, 'h') for h in lead_time_hours]

        downloaded_files = []

        # Download year by year
        for year in years:
            output_path = os.path.join(data_dir, f"{model_name}_{year}.zarr")

            if os.path.exists(output_path):
                print(f"\n  Skipping {year}: file already exists")
                downloaded_files.append(output_path)
                continue

            print(f"\n  Processing year {year}...")
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

            subset_rechunked = subset.chunk(chunk_dict)

            # Save to zarr
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


def download_target_data(data_dir, model_name, ground_truth_source, years, variables,
                        region_lat=None, region_lon=None, use_dask_client=True):
    """
    Download target/observation data for the given parameters.

    Parameters:
    -----------
    data_dir : str
        Directory to save data
    model_name : str
        Name of forecast model (used to determine default ground truth)
    ground_truth_source : str
        Name of ground truth source (e.g., 'era5', 'hres_t0', or empty string for default)
    years : list
        List of years to download
    variables : list
        List of variables to download
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
    os.makedirs(data_dir, exist_ok=True)

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

    print(f"\nDownloading {target} target data...")
    print(f"  Years: {years}")
    print(f"  Variables: {variables}")

    start_time = time.time()

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
        start_time = print_time_and_memory("Dataset opened", start_time)

        # Select variables
        available_vars = [v for v in variables if v in ds.data_vars]
        if len(available_vars) < len(variables):
            missing = set(variables) - set(available_vars)
            print(f"  Warning: Variables not found: {missing}")

        downloaded_files = []

        # Download year by year
        for year in years:
            output_path = os.path.join(data_dir, f"{target}_{year}.zarr")

            if os.path.exists(output_path):
                print(f"\n  Skipping {year}: file already exists")
                downloaded_files.append(output_path)
                continue

            print(f"\n  Processing year {year}...")
            time_range = [f'{year}-01-01', f'{year}-12-31']

            # Select subset
            subset = ds[available_vars].sel(
                time=slice(time_range[0], time_range[1])
            )

            # Apply regional subset if provided
            if region_lat is not None and region_lon is not None:
                subset = subset.sel(latitude=region_lat, longitude=region_lon)

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
                available_vars = [v if v != 'total_precipitation_6hr' else 'total_precipitation'
                                for v in available_vars]

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

            subset_rechunked = subset.chunk(chunk_dict)

            # Save to zarr
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


def prepare_data_for_finetuning(data_dir, model_name, ground_truth_source,
                                training_vars, output_vars, train_start, train_end,
                                test_start, test_end, lead_time_hours,
                                region_lat=None, region_lon=None):
    """
    Main function to prepare forecast and target data for finetuning.
    Checks if data exists, and downloads if necessary.

    Parameters:
    -----------
    data_dir : str
        Directory for data storage
    model_name : str
        Name of forecast model
    ground_truth_source : str
        Name of ground truth source (empty string for default)
    training_vars : list
        List of training variables
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

    print(f"\nYears needed: {all_years}")
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
    fc_exists, fc_missing = check_data_exists(data_dir, model_name, all_years, forecast_vars)

    if fc_exists:
        print(f"  ✓ All forecast data exists")
    else:
        print(f"  ✗ Missing forecast data for years: {fc_missing}")
        print(f"  Downloading missing data...")
        download_forecast_data(
            data_dir, model_name, fc_missing, forecast_vars,
            lead_time_hours, region_lat, region_lon, use_dask_client=True
        )

    # Check target data
    print(f"\nChecking {target} target data...")
    target_vars = [v for v in output_vars if v != "10m_wind_speed"]
    tgt_exists, tgt_missing = check_data_exists(data_dir, target, all_years, target_vars)

    if tgt_exists:
        print(f"  ✓ All target data exists")
    else:
        print(f"  ✗ Missing target data for years: {tgt_missing}")
        print(f"  Downloading missing data...")
        download_target_data(
            data_dir, model_name, ground_truth_source, tgt_missing,
            target_vars, region_lat, region_lon, use_dask_client=True
        )

    print("\n" + "="*70)
    print("DATA PREPARATION COMPLETE")
    print("="*70)

    return {
        'status': 'success',
        'data_dir': data_dir,
        'forecast_source': model_name,
        'target_source': target,
        'years': all_years
    }
