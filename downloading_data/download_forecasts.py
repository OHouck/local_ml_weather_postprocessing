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

    years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
    # model = 'pangu'
    model = "ifs"
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
        lead_times_hours = [24, 24 + 12, 120, 120 + 12, 240, 240 + 12] # midday and midnight forecasts for 1, 5, 10 days 
    elif model == 'ifs':
        variables_to_try = [
            '2m_temperature',
            '10m_u_component_of_wind',
            '10m_v_component_of_wind',
            'total_precipitation_6hr'
        ]
        lead_times_hours = [24, 24 + 6, 24 + 12, 24 + 18,
                            120, 120 + 12, 120 + 18, 120 + 12, 
                            216, 216 + 6, 216 + 12, 216 + 18]  # for IFS get all 6 hour intervals to get daily total precip

    
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


    subset = ds[available_vars].sel(
        prediction_timedelta=[np.timedelta64(hours, 'h') for hours in lead_times_hours]
    )

    # only look at forecasts initialized at midnight 
    subset = subset.sel(init_time=subset.init_time.dt.hour.isin([0]))

    if 'total_precipitation_6hr' in available_vars:
        # Convert 6-hourly total precipitation to daily total precipitation
        # currently just for ifs
        print("  Converting 6-hourly total precipitation to daily totals...")
        precip_6hr = subset['total_precipitation_6hr']

        # Convert timedelta to hours and assign day group as coordinate
        hours = precip_6hr.prediction_timedelta / np.timedelta64(1, 'h')
        day_groups = (hours // 24).astype(int)

        # Attach day_groups as a coordinate
        precip_6hr = precip_6hr.assign_coords(day_group=('prediction_timedelta', day_groups.data))

        # Compute daily sums (collapsing 6hr steps into days)
        daily_precip = precip_6hr.groupby('day_group').sum(dim='prediction_timedelta')

        # Broadcast daily_precip back to original prediction_timedelta dimension
        total_precipitation = xr.apply_ufunc(
            lambda dg: daily_precip.sel(day_group=dg),
            precip_6hr['day_group'],
            vectorize=True,
        )

        subset['total_precipitation'] = total_precipitation
        print(subset)
        exit()
        # precip_6hr = subset['total_precipitation_6hr']

        # # Convert timedelta to hours for easier grouping
        # hours = precip_6hr.prediction_timedelta / np.timedelta64(1, 'h')
        # day_groups = (hours // 24).astype(int) 

        # daily_precip = precip_6hr.groupby(day_groups).sum(dim='prediction_timedelta')

        # # Create new array with subset's coordinates
        # total_precipitation = xr.zeros_like(subset['total_precipitation_6hr'])

        # for td in subset.prediction_timedelta.values:
        #     hour = td / np.timedelta64(1, 'h')
        #     day_group = int(hour // 24)
        #     print("hour", hour, "day_group", day_group, "td", td)
        #     total_precipitation.loc[dict(prediction_timedelta=td)] = daily_precip.sel(prediction_timedelta=day_group)
        
        # subset['total_precipitation'] = total_precipitation
        # print(subset)

        exit()

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


        