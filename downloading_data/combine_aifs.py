#combine_aifs.py

# Create yearly files from daily AIFS data downloaded from downlaod_aifs.sh and aifs_cleaning.py

import sys
import os
import socket
import glob 
import xarray as xr
import pandas as pd
import numpy as np


def setup_directories():
    """Set up directory structure based on environment."""
    nodename = socket.gethostname()
    if nodename == "oMac.local":
        root = os.path.expanduser("~/OneDrive - The University of Chicago/ai_weather_ag/data")
    else:
        raise Exception(f"Unknown environment, Please specify the root directory. "
                        f"Nodename found: {nodename}")

    dirs = {
        'root': root,
        'raw': os.path.join(root, "raw"),
        'globus': os.path.expanduser(f"/Users/ohouck/globus/forecast_data"),
        'processed': os.path.join(root, "processed"),
        'fig': os.path.join(root, "../figures/finetuning"),
        'input': os.path.join(root, "fine_tuning_output")
    }

    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs

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
    common_valid_times = set()
    for lt in lead_times:
        ds_lt = ds.sel(prediction_timedelta=lt)

        ds_lt = ds_lt.rename({'init_time': 'valid_time'})
        ds_lt['valid_time'] = ds_lt.valid_time + lt
        converted_lead_times.append(ds_lt)
        if not common_valid_times:
            # Initialize with the first lead time's valid_time
            common_valid_times = set(ds_lt.valid_time.values)
        else:
            common_valid_times.intersection_update(ds_lt.valid_time.values)

    # Combine all lead times into a single dataset and filter to common valid times
    combined_ds = xr.concat(converted_lead_times, dim='prediction_timedelta')
    combined_ds = combined_ds.sel(valid_time=list(common_valid_times))
    combined_ds = combined_ds.sortby('valid_time')

    return combined_ds

def validate_conversion(original_ds: xr.Dataset, converted_ds: xr.Dataset) -> bool:
    """
    Validate the conversion from init_time to valid_time.
    Returns True if basic conversion appears correct, False otherwise.
    """
    try:
        # Basic dimension check
        expected_dims = {'valid_time', 'prediction_timedelta', 'latitude', 'longitude'}
        if set(converted_ds.dims.keys()) != expected_dims:
            return False
        
        # Basic data variable check
        if set(original_ds.data_vars) != set(converted_ds.data_vars):
            return False
        
        # Non-empty result
        if len(converted_ds.valid_time) == 0:
            return False
        
        # Quick math check on one sample
        first_pred_delta = converted_ds.prediction_timedelta.values[0]
        first_valid_time = converted_ds.valid_time.values[0]
        expected_init_time = pd.Timestamp(first_valid_time) - pd.Timedelta(first_pred_delta)
        original_init_times_pd = pd.to_datetime(original_ds.init_time.values)
        
        if expected_init_time not in original_init_times_pd:
            return False
        
        return True
    
    except Exception:
        return False


def main():

    dirs = setup_directories()
    aifs_dir = os.path.join(dirs['raw'], "aifs")
    
    aifs_files = sorted(glob.glob(os.path.join(aifs_dir, "processed_init*.zarr")))
    ds = None
    for file in aifs_files:
        
        try:
            ds_temp = xr.open_zarr(file, consolidated=True, decode_timedelta = True)
        except Exception as e:
            print(f"Error opening {file}: {e}")
            continue

    for file in aifs_files:
        
        ds_temp = xr.open_zarr(file, consolidated=True, decode_timedelta = True)

        ds_temp = ds_temp.rename({"time": "init_time"})
        ds= xr.concat([ds, ds_temp], dim="init_time") if ds is not None else ds_temp

    ds_valid_time = convert_init_to_valid_time(ds)
    validate_conversion(ds, ds_valid_time)

    print(ds_valid_time)

    # years we have aifs data
    years = [2021, 2022, 2023, 2024]
    for year in years:
        ds_year = ds_valid_time.sel(valid_time=slice(f"{year}-01-01", f"{year}-12-31"))
        # save yearly files to globus forecast data directory 
        out_path = os.path.join(dirs["globus"], f"aifs_{year}.zarr")
        ds_year.to_zarr(out_path, mode="w", consolidated=True)

if __name__ == "__main__":
    main()