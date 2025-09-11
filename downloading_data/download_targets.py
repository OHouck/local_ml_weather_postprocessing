import time
from datetime import datetime
import psutil
import os
import warnings
import pandas as pd
import xarray as xr
import numpy as np
import dask
from dask.distributed import Client, as_completed
from dask.diagnostics import ProgressBar
warnings.filterwarnings('ignore')


def print_time_and_memory(step_name, start_time):
    """Print elapsed time and current memory usage"""
    elapsed = time.time() - start_time
    memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024  # GB
    print(f"\n✓ {step_name}: {elapsed:.2f} seconds | Memory: {memory:.2f} GB")
    return time.time()

def download_data(data_name, year):
    """Main download function"""
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
    

    if data_name == 'era5':
        variables_to_try = [
            '2m_temperature',
            '10m_u_component_of_wind',
            '10m_v_component_of_wind',
            'total_precipitation_6hr'
        ]
    elif data_name == 'hres_t0':
        variables_to_try = [
            '2m_temperature',
            '10m_u_component_of_wind',
            '10m_v_component_of_wind',
            'total_precipitation_6hr'
        ]
    
    # Define your domain for each year to be downloaded
    time_range = [f'{year}-01-01', f'{year}-12-31']
    
    # Output path
    output_path = os.path.expanduser(f"/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/data/raw/{data_name}_{year}.zarr")
    
    start_time = print_time_and_memory("Parameter setup", start_time)
    
    # 3. Open the dataset
    print("\n3. Opening dataset...")
    try:

        if data_name == 'era5':
            print("  Using ARCO-ERA5 dataset")
            ds = xr.open_zarr(
                'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3',
                consolidated=True
            )
            # to match hres_t0 variable names to be consistent
            ds = ds.rename({'total_precipitation': 'total_precipitation_6hr'})

        elif data_name == 'hres_t0':
            print("  Using Weatherbenchy-HRES-T0 dataset")
            ds = xr.open_zarr(
                "gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr",
                consolidated=True
            )
        else:
            raise ValueError(f"Unknown dataset name: {data_name}")
        
        # print all available variables
        vars = list(ds.data_vars)
        print(f"Available variables in dataset: {vars}")

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
    
        
    except Exception as e:
        print(f"Error opening dataset: {e}")
        client.close()
        raise

    
    
    start_time = print_time_and_memory("Dataset opened", start_time)
    
    # 4. Select subset
    print("\n4. Selecting data subset...")
    print(f"  Time range: {time_range[0]} to {time_range[1]}")
    print(f"  Variables: {available_vars}")

    subset = ds[available_vars].sel(
        time=slice(time_range[0], time_range[1])
    )

    # calculate cumulative precipitation for the entire day
    # group by 6, 12, 18, 24 hours and take the sum of the 6-hourly precipitation
    if 'total_precipitation_6hr' in available_vars:

        six_hour_precip = subset['total_precipitation_6hr']

        # convert from being cumulative precip at the end of the 6-hour period 
        # to being the precip over the coming 6-hour period
        # This means we can sum by the day
        # move each timestep back by 6 hours
        # Shift time coordinate forward by 6 hours
        six_hour_precip = six_hour_precip.assign_coords(
            time=six_hour_precip.time + pd.Timedelta(hours=6)
        )

        # Create daily precipitation sums
        daily_precip = six_hour_precip.resample(time='1D').sum()
        daily_precip = daily_precip.rename('total_precipitation')

        # Broadcast back to 6-hourly resolution so all timesteps within a day have the same value
        total_precipitation = daily_precip.resample(time='6H').ffill()

        # merge back
        subset = subset.drop_vars('total_precipitation_6hr')
        subset["total_precipitation"] = total_precipitation

    # filter for only hours 0 and 12
    subset = subset.sel(time=subset.time.dt.hour.isin([0, 12]))

    print(subset)

    # update available_vars to reflect changes
    available_vars = list(subset.data_vars)
    
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
    n_times = len(subset.time)
    time_chunk = min(240, n_times // 10)  # ~10 chunks or 240 timesteps
    
    chunk_dict = {
        'time': time_chunk,
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
            temp_nc = 'temp.nc'
            
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
    
    # 7. Verify the saved data
    print("\n7. Verifying saved data...")
    ds_saved = xr.open_zarr(output_path)
    print(f"  Saved dataset shape: {ds_saved.dims}")
    print(f"  Variables saved: {list(ds_saved.data_vars)}")
    print(f"  Time range: {ds_saved.time.values[0]} to {ds_saved.time.values[-1]}")
    
    # Check file size
    if os.path.exists(output_path):
        import glob
        total_size = sum(
            os.path.getsize(f) 
            for f in glob.glob(f"{output_path}/**/*", recursive=True)
            if os.path.isfile(f)
        ) / 1024**3
        print(f"  Total file size on disk: {total_size:.2f} GB")
    
    start_time = print_time_and_memory("Verification complete", start_time)
    
    # 8. Final summary
    print("\n=== Download Summary ===")
    total_time = time.time() - script_start
    print(f"Total execution time: {total_time/60:.2f} minutes")
    print(f"Average download speed: {total_size_gb / (total_time/60):.2f} GB/minute")
    print(f"Output saved to: {output_path}")
    print(f"End time: {datetime.now()}")
    
    # Close Dask client
    client.close()
    print("\nDask client closed.")
    
    return output_path

if __name__ == '__main__':

    # Check package versions
    import zarr
    import numcodecs
    print(f"Package versions:")
    print(f"  xarray: {xr.__version__}")
    print(f"  zarr: {zarr.__version__}")
    print(f"  numcodecs: {numcodecs.__version__}")
    print(f"  dask: {dask.__version__}")

    # years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
    years = [2023, 2024]
    data_source = 'era5'  # hres_t0 or era5
    
    # Try the download
    for year in years:
        output = download_data(data_source, year)
        print(f"\nSuccess! Data saved to: {output}")
        