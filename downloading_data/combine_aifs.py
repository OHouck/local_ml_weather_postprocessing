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
    min_sup = None # keep track of most recent earliest valid time and earliest latest valid time
    max_inf = None

    for lt in lead_times:
        ds_lt = ds.sel(prediction_timedelta=lt)

        ds_lt = ds_lt.rename({'init_time': 'valid_time'})
        ds_lt['valid_time'] = ds_lt.valid_time + lt
        converted_lead_times.append(ds_lt)
        if len(common_valid_times) == 0: 
            # Initialize with the first lead time's valid_time
            common_valid_times = set(ds_lt.valid_time.values)
            min_sup = ds_lt.valid_time.values.max()
            max_inf = ds_lt.valid_time.values.min()
        else:
            common_valid_times = common_valid_times.union(set(ds_lt.valid_time.values))
            if ds_lt.valid_time.values.min() > max_inf:
                max_inf = ds_lt.valid_time.values.min()
            if ds_lt.valid_time.values.max() < min_sup:
                min_sup = ds_lt.valid_time.values.max()
        

    # Combine all lead times into a single dataset and filter to common valid times
    combined_ds = xr.concat(converted_lead_times, dim='prediction_timedelta')
    common_valid_times = {vt for vt in common_valid_times if (vt >= max_inf and vt <= min_sup)}
    combined_ds = combined_ds.sel(valid_time=list(common_valid_times))
    combined_ds = combined_ds.sortby('valid_time')
    

    return combined_ds

def validate_conversion(original_ds: xr.Dataset, converted_ds: xr.Dataset) -> bool:
    """
    Validate the conversion from init_time to valid_time.
    Checks both structure and actual data values to ensure conversion is correct.
    """
    try:
        # Basic dimension check
        expected_dims = {'valid_time', 'prediction_timedelta', 'latitude', 'longitude'}
        if set(converted_ds.dims.keys()) != expected_dims:
            print(f"Dimension mismatch. Expected: {expected_dims}, Got: {set(converted_ds.dims.keys())}")
            return False
        
        # Basic data variable check
        if set(original_ds.data_vars) != set(converted_ds.data_vars):
            print(f"Data variable mismatch")
            return False
        
        # Check that we have data
        if len(converted_ds.valid_time) == 0:
            print(f"No valid times in converted dataset")
            return False
        
        # Choose a variable to test with (prefer total_precipitation if available)
        test_var = None
        if 'total_precipitation' in converted_ds.data_vars:
            test_var = 'total_precipitation'
        elif len(converted_ds.data_vars) > 0:
            test_var = list(converted_ds.data_vars)[0]
        else:
            print("No data variables to test")
            return False
        
        print(f"Using '{test_var}' for data validation...")
        
        # check a few random samples
        n_samples = 5
        for sample_idx in range(n_samples):

            rand_idx = {dim: np.random.randint(0, converted_ds.sizes[dim]) for dim in converted_ds.dims}
            rand_converted_obs = converted_ds[test_var].isel(**rand_idx)
            var = rand_converted_obs.values

            # if sampled value is nan because valid time and lead time combination doesn't exist, resample
            if np.isnan(var):
                while np.isnan(var):
                    rand_idx = {dim: np.random.randint(0, converted_ds.sizes[dim]) for dim in converted_ds.dims}
                    rand_converted_obs = converted_ds[test_var].isel(**rand_idx)
                    var = rand_converted_obs.values


            valid_time = rand_converted_obs.valid_time.values
            # check if valid time is for noon or midnight forecast
            pred_delta = rand_converted_obs.prediction_timedelta.values
            lat = rand_converted_obs.latitude.values
            lon = rand_converted_obs.longitude.values

            pred_delta_hours = np.timedelta64(pred_delta, 'h')

            expected_init_time = valid_time - pred_delta
            print(f"Sample {sample_idx+1}: valid_time={valid_time}, pred_delta={pred_delta_hours}, "
                  f"expected_init_time={expected_init_time}, lat={lat}, lon={lon}")

            original_obs = original_ds[test_var].sel(
                init_time=expected_init_time,
                prediction_timedelta=pred_delta,
                latitude=lat,
                longitude=lon
            )
            original_var = original_obs.values  

            # Check if values are equal (handling potential NaN values)
            if np.isnan(var) and np.isnan(original_var):
                continue  # Both NaN is okay
            elif not np.isclose(var, original_var, rtol=1e-6):
                print(f"Data mismatch at valid_time={valid_time}, "
                        f"pred_delta={pred_delta}, lat_idx={lat}, lon_idx={lon}")
                print(f"  Converted value: {var}")
                print(f"  Original value: {original_var}")
                print(f"  Expected init_time: {expected_init_time}")
                return False
        
        
        print("Validation passed! Data values match between original and converted datasets.")
        return True
    
    except Exception as e:
        print(f"Validation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():

    dirs = setup_directories()
    aifs_dir = os.path.join(dirs['globus'], "aifs")
    
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

        ds= xr.concat([ds, ds_temp], dim="init_time") if ds is not None else ds_temp

    ds_valid_time = convert_init_to_valid_time(ds)

    # if not validate_conversion(ds, ds_valid_time):
    #     print("Conversion validation failed!")

    # years we have aifs data
    years = [2021, 2022, 2023, 2024]
    for year in years:
        ds_year = ds_valid_time.sel(valid_time=slice(f"{year}-01-01", f"{year}-12-31"))

        ds_year = ds_year.chunk({
            'prediction_timedelta': -1,  # Keep all prediction times in one chunk
            'valid_time': 100,           # Reasonable chunk size for time
            'latitude': 181,             # Keep spatial chunks as they are
            'longitude': 360
        })
        # save yearly files to globus forecast data directory 
        out_path = os.path.join(dirs["globus"], f"aifs_{year}.zarr")
        ds_year.to_zarr(out_path, mode="w", consolidated=True)

if __name__ == "__main__":
    main()