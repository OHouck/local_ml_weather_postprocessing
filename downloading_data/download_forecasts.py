import xarray as xr
import numpy as np  
import dask
from dask.distributed import Client, as_completed
from dask.diagnostics import ProgressBar
import pandas as pd
import time
from datetime import datetime
import psutil
import os
import warnings

from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

warnings.filterwarnings('ignore')

import numpy as np
import xarray as xr
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

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

        ds_lt = ds_lt.copy()
        ds_lt = ds_lt.rename({'init_time': 'valid_time'})
        ds_lt['valid_time'] = ds_lt.valid_time + lt
        converted_lead_times.append(ds_lt)

    # Combine all lead times into a single dataset and filter to common valid times
    combined_ds = xr.concat(converted_lead_times, dim='prediction_timedelta')
    combined_ds = combined_ds.sortby('valid_time')
    combined_ds = combined_ds.transpose("valid_time", "prediction_timedelta", "latitude", "longitude")

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
        
        # Choose a variable to test with (prefer 2m_temperature if available)
        test_var = None
        if '2m_temperature' in converted_ds.data_vars:
            test_var = '2m_temperature'
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


def print_time_and_memory(step_name, start_time):
    """Print elapsed time and current memory usage"""
    elapsed = time.time() - start_time
    memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024  # GB
    print(f"\n✓ {step_name}: {elapsed:.2f} seconds | Memory: {memory:.2f} GB")
    return time.time()

def download_data_by_year(model, year, subset, available_vars, start_time, client):

    # Output path
    output_path = os.path.expanduser(f"/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/data/raw/{model}_{year}.zarr")

    time_range = [f'{year}-01-01', f'{year}-12-31'] 
    print(f"  Time range: {time_range[0]} to {time_range[1]}")

    # Select time range
    subset = subset.sel(
        valid_time=slice(time_range[0], time_range[1])
    )

    # Print subset info
    print(f"\nSubset shape:")
    for var in available_vars:
        var_shape = subset[var].shape
        var_size_gb = subset[var].nbytes / 1024**3
        print(f"  {var}: {var_shape} (~{var_size_gb:.2f} GB)")
    
    total_size_gb = sum(subset[var].nbytes for var in available_vars) / 1024**3
    print(f"\nTotal size: ~{total_size_gb:.2f} GB")
    
    # Show chunk structure
    print(f"\nOriginal chunk structure for {available_vars[0]}:")
    print(f"  {subset[available_vars[0]].chunks}")


    start_time = print_time_and_memory("Subset selected", start_time)
    
    # 5. Rechunk for optimal performance
    print("\n5. Rechunking data for optimal download...")
    
    # Adaptive chunking based on data size
    n_times = len(subset.valid_time)
    time_chunk = min(240, n_times // 10)  # ~10 chunks or 240 timesteps
    
    chunk_dict = {
        'valid_time': time_chunk,
        'prediction_timedelta': len(subset.prediction_timedelta),
        'latitude': len(subset.latitude),
        'longitude': len(subset.longitude)
    }
    
    subset_rechunked = subset.chunk(chunk_dict)
    print(f"  New chunks: {subset_rechunked[available_vars[0]].chunks}")
    
    start_time = print_time_and_memory("Rechunking complete", start_time)
    
    # 6. Save to Zarr with progress tracking
    print("\n6. Saving to Zarr format...")
    print(f"  Output path: {output_path}")
    
    # Try different saving methods
    save_successful = False
    
    # Method 1: Save without explicit encoding
    try:
        print("\n  Trying save without encoding...")
        with ProgressBar():
            save_start = time.time()
            subset_rechunked.to_zarr(
                output_path,
                mode='w',
                consolidated=True,
                zarr_version=2  # Use zarr v2 format
            )
            save_time = time.time() - save_start
            print(f"\n  Save completed in {save_time:.2f} seconds")
            print(f"  Average speed: {total_size_gb / save_time:.2f} GB/s")
            save_successful = True
            
    except Exception as e:
        print(f"  Method 1 failed: {e}")
        
        # Method 2: Save to NetCDF first, then convert
        try:
            print("\n  Trying NetCDF intermediate save...")
            temp_nc = 'temp_era5.nc'
            
            # Compute and save to NetCDF
            with ProgressBar():
                subset_rechunked.compute().to_netcdf(temp_nc)
            
            # Load and save to Zarr
            ds_temp = xr.open_dataset(temp_nc)
            ds_temp.to_zarr(output_path, mode='w', consolidated=True)
            ds_temp.close()
            
            # Clean up
            os.remove(temp_nc)
            save_successful = True
            print("  NetCDF conversion method succeeded!")
            
        except Exception as e2:
            print(f"  Method 2 failed: {e2}")
            
            # Method 3: Save variable by variable
            try:
                print("\n  Trying variable-by-variable save...")
                import zarr
                
                # Create zarr store
                store = zarr.DirectoryStore(output_path)
                root = zarr.open(store, mode='w')
                
                # Save each variable
                for var in available_vars:
                    print(f"    Saving {var}...")
                    data = subset[var].compute()
                    
                    # Create dataset with single variable
                    ds_single = xr.Dataset({var: data})
                    
                    # Save to temporary zarr
                    temp_path = f'temp_{var}.zarr'
                    ds_single.to_zarr(temp_path, mode='w')
                    
                    # Copy to main store
                    temp_store = zarr.open(temp_path)
                    zarr.copy(temp_store[var], root, name=var)
                    
                    # Clean up
                    import shutil
                    shutil.rmtree(temp_path)
                
                # Save coordinates
                subset.coords.to_zarr(output_path, mode='a')
                
                save_successful = True
                print("  Variable-by-variable method succeeded!")
                
            except Exception as e3:
                print(f"  Method 3 failed: {e3}")
                
                # Method 4: Use legacy zarr format
                try:
                    print("\n  Trying legacy zarr format...")
                    
                    # First compute the data
                    computed_data = subset_rechunked.compute()
                    
                    # Save with zarr_version specified
                    computed_data.to_zarr(
                        output_path,
                        mode='w',
                        consolidated=True,
                        zarr_version=2  # Use zarr v2 format
                    )
                    save_successful = True
                    print("  Legacy format method succeeded!")
                    
                except Exception as e4:
                    print(f"  Method 4 failed: {e4}")
                    raise Exception("All save methods failed!")
    
    if not save_successful:
        client.close()
        raise Exception("Failed to save data")
    
    start_time = print_time_and_memory("Data saved", start_time)
    
    # Check file size
    if os.path.exists(output_path):
        import glob
        total_size = sum(
            os.path.getsize(f) 
            for f in glob.glob(f"{output_path}/**/*", recursive=True)
            if os.path.isfile(f)
        ) / 1024**3
        print(f"  Total file size on disk: {total_size:.2f} GB")
    
    
    # Close Dask client
    client.close()
    print("\nDask client closed.")
    
    return output_path

def main():
    # Check package versions
    import zarr
    import numcodecs
    print(f"Package versions:")
    print(f"  xarray: {xr.__version__}")
    print(f"  zarr: {zarr.__version__}")
    print(f"  numcodecs: {numcodecs.__version__}")
    print(f"  dask: {dask.__version__}")

    years = [2021, 2022]
    model = 'pangu'
    # model = "ifs"
    # Start timing
    script_start = time.time()
    start_time = time.time()
    
    print("=== Download Script Started ===")
    print(f"Start time: {datetime.now()}")
    
    # 1. Set up Dask client with 16GB RAM
    print("\n1. Setting up Dask client...")
    
    # For macOS, use threads instead of processes to avoid multiprocessing issues
    client = Client(
        n_workers=2,                    # Fewer workers on macOS
        threads_per_worker=4,           # More threads per worker
        processes=False,                # Use threads instead of processes
        memory_limit='8GB',             # 8GB per worker = 16GB total
        silence_logs=30
    )
    
    print(f"Dask client: {client}")
    print(f"Dashboard: {client.dashboard_link}")
    start_time = print_time_and_memory("Dask setup", start_time)
    
    # 2. Define variables and parameters
    print("\n2. Defining download parameters...")
    
    # Variable mapping for Pangu and ifs
    if model == 'pangu':
        variables_to_try = [
            '2m_temperature',
            '10m_u_component_of_wind',
            '10m_v_component_of_wind'
        ]
        lead_times_hours = [24, 24 + 12, 120, 120 + 12, 216, 216 + 12] # midday and midnight forecasts for 1, 5, 9 days 
    elif model == 'ifs':
        variables_to_try = [
            '2m_temperature',
            '10m_u_component_of_wind',
            '10m_v_component_of_wind',
            'total_precipitation_6hr'
        ]
        base_days = [24,120,216] # 1, 5, 9 days
        lead_times_hours = []
        for b in base_days:
            lead_times_hours += [b + offset for offset in (0, 6, 12, 18, 24)]

    
    start_time = print_time_and_memory("Parameter setup", start_time)
    
    print("\n3. Opening dataset...")
    try:
        if model == 'pangu':
            ds = xr.open_zarr(
                "gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr",
                consolidated=True
            )
        elif model == 'ifs':
            ds = xr.open_zarr(
                "gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
                consolidated=True
            )

        else:
            raise ValueError("Model not recognized.")

        # rename time variable to init_time
        ds = ds.rename({'time': 'init_time'})
        print(f"Dataset opened successfully")
        print(f"Dataset dimensions: {ds.dims}")
        
        # Check which requested variables are available
        available_vars = []
        for var_name in variables_to_try:
            if var_name in ds.data_vars:
                available_vars.append(var_name)
                print(f"  ✓ {var_name} found")
            else:
                # Check for similar variable names
                similar = [v for v in ds.data_vars if var_name.split('_')[-1] in v]
                if similar:
                    print(f"  ✗ {var_name} not found. Similar: {similar[:3]}")
                else:
                    print(f"  ✗ {var_name} not found")
        
        # Show all humidity-related variables
        humidity_vars = [v for v in ds.data_vars if any(h in v.lower() for h in ['humidity', 'dewpoint', 'moisture'])]
        if humidity_vars:
            print(f"\nAvailable humidity-related variables: {humidity_vars}")
        
    except Exception as e:
        print(f"Error opening dataset: {e}")
        client.close()
        raise
    
    start_time = print_time_and_memory("Dataset opened", start_time)
    
    # 4. Select subset
    print("\n4. Selecting data subset...")
    print(f"  Variables: {available_vars}")

    print(f" Lead time options: {lead_times_hours} hours")

    # print unique lead times in dataset
    dataset_lead_times = np.unique(ds.prediction_timedelta.values).astype('timedelta64[h]').astype(int)
    print(f"  Dataset lead times (hours): {dataset_lead_times}")

    subset = ds[available_vars].sel(
        prediction_timedelta=[np.timedelta64(hours, 'h') for hours in lead_times_hours]
    )

    # only look at forecasts initialized at midnight 
    subset = subset.sel(init_time=subset.init_time.dt.hour.isin([0]))

    if 'total_precipitation_6hr' in available_vars:
        print("  Converting 6-hourly total precipitation to daily totals...")
        precip_6hr = subset['total_precipitation_6hr']

        # manually define the lead time needed to calculate daily totals
        daily_precip_steps = {
            "day1": [30, 36, 42, 48],   # 24-48h
            "day5": [126, 132, 138, 144], # 120-144h
            "day9": [222, 228, 234, 240]  # 216-240h
        }
        
        # Create the dataset with only the lead times you need
        # Midnight: 24h (day 1), 120h (day 5), 216h (day 9)
        # Midday: 36h (day 1), 132h (day 5), 228h (day 9)
        
        total_precip_list = []
        keep_lead_times =[]

        for day, hours in daily_precip_steps.items():

            # sum the 6h totals across 24h window
            day_total = precip_6hr.sel(prediction_timedelta=[np.timedelta64(h, 'h') for h in hours]).sum(dim='prediction_timedelta')
        
            for td in [np.timedelta64(hours[0]-6, 'h'), np.timedelta64(hours[1], 'h')]:
                expanded = day_total.expand_dims(prediction_timedelta=[td])
                total_precip_list.append(expanded)
                keep_lead_times.append(td)
        
        # Concatenate
        total_precipitation = xr.concat(total_precip_list, dim='prediction_timedelta')

        # replace subset lead times with only the ones we want
        subset = subset.sel(prediction_timedelta=keep_lead_times)
        subset['total_precipitation'] = total_precipitation


    # convert init_time to valid_time
    subset_valid_time = convert_init_to_valid_time(subset)
    # sort by valid_time
    if not validate_conversion(subset, subset_valid_time):
        print("Validation failed! The conversion does not appear correct.")
        client.close()
        raise Exception("Conversion validation failed")
    subset = subset_valid_time
    
    # save subset of large data one year at a time
    for year in years:
        output = download_data_by_year(model, year, subset, available_vars, start_time, client)
        print(f"\nSuccess! Data saved to: {output}")

    start_time = print_time_and_memory("Verification complete", start_time)
    
    print("\n=== Download Summary ===")
    total_time = time.time() - script_start
    print(f"Total script time: {total_time:.2f} seconds")
    print(f"End time: {datetime.now()}")

if __name__ == '__main__':
    main()


       