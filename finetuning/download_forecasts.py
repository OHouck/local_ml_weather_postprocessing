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
warnings.filterwarnings('ignore')

def convert_init_to_valid_time_optimized(ds):
    """
    Optimized conversion from init_time to valid_time dimension.
    
    This version uses xarray's advanced indexing and groupby operations
    to avoid explicit loops and maintain lazy evaluation with Dask.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset with dimensions (init_time, prediction_timedelta, latitude, longitude)
        
    Returns:
    --------
    xarray.Dataset
        Dataset with dimensions (valid_time, prediction_timedelta, latitude, longitude)
    """
    # Calculate valid_time as init_time + prediction_timedelta
    # This creates a 2D array (init_time, prediction_timedelta)
    valid_time_2d = ds.init_time + ds.prediction_timedelta
    
    # Stack init_time and prediction_timedelta into a single dimension
    ds_stacked = ds.stack(time_combo=['init_time', 'prediction_timedelta'])
    
    # Calculate valid_time for the stacked dimension
    valid_time_stacked = valid_time_2d.stack(time_combo=['init_time', 'prediction_timedelta'])
    
    # Add valid_time as a new data variable (not just a coordinate)
    # This is necessary because groupby needs a data variable
    ds_stacked['valid_time'] = valid_time_stacked
    
    # Also need to expand prediction_timedelta to match the stacked dimension
    # Create a 2D array where prediction_timedelta is repeated for each init_time
    pred_td_expanded = ds.prediction_timedelta.expand_dims(init_time=ds.init_time)
    pred_td_stacked = pred_td_expanded.stack(time_combo=['init_time', 'prediction_timedelta'])
    ds_stacked['prediction_timedelta_expanded'] = pred_td_stacked
    
    # Now group by both valid_time and prediction_timedelta
    # Use the data variables we just created
    grouped = ds_stacked.groupby(['valid_time', 'prediction_timedelta_expanded'])
    
    # Take the first occurrence (you could also use .mean() if you want to average)
    ds_grouped = grouped.first()
    
    # The groupby operation creates dimensions with the grouped variable names
    # Rename them to our desired dimension names
    ds_grouped = ds_grouped.rename({
        'prediction_timedelta_expanded': 'prediction_timedelta'
    })
    
    # Drop the temporary variables we created for grouping
    if 'valid_time' in ds_grouped.data_vars:
        ds_grouped = ds_grouped.drop_vars('valid_time')
    if 'prediction_timedelta_expanded' in ds_grouped.data_vars:
        ds_grouped = ds_grouped.drop_vars('prediction_timedelta_expanded')
    
    # Ensure dimensions are in the correct order
    expected_dims = ['valid_time', 'prediction_timedelta', 'latitude', 'longitude']
    actual_dims = list(ds_grouped.dims)
    
    # Only transpose if all expected dimensions exist
    if all(dim in actual_dims for dim in expected_dims):
        ds_final = ds_grouped.transpose(*expected_dims)
    else:
        ds_final = ds_grouped
    
    # Drop any fully NaN time steps
    ds_final = ds_final.dropna(dim='valid_time', how='all')
    
    return ds_final

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
    # Create valid_time coordinate
    # This will be a 2D array of shape (init_time, prediction_timedelta)
    valid_time_2d = ds.init_time + ds.prediction_timedelta

    # Stack init_time and prediction_timedelta into a single dimension
    ds_stacked = ds.stack(stacked=['init_time', 'prediction_timedelta'])
    
    # Assign the flattened valid_time as a coordinate
    valid_time_flat = valid_time_2d.stack(stacked=['init_time', 'prediction_timedelta'])
    ds_stacked = ds_stacked.assign_coords(valid_time=valid_time_flat)
    
    # Get unique valid times and lead times
    unique_valid_times = np.unique(valid_time_flat.values)
    lead_times = ds.prediction_timedelta.values

    
    # Create output dataset structure
    output_vars = {}
    
    for var in ds.data_vars:
        # Create empty array for this variable
        output_shape = (len(unique_valid_times), len(lead_times), 
                       len(ds.latitude), len(ds.longitude))
        output_data = np.full(output_shape, np.nan, dtype=np.float32)
        
        # Fill the array
        for i, vt in enumerate(unique_valid_times):
            for j, lt in enumerate(lead_times):
                # Find where valid_time equals vt and prediction_timedelta equals lt
                mask = (valid_time_flat == vt) & (ds_stacked.prediction_timedelta == lt)
                
                if mask.any():
                    # Get the data for this combination
                    data = ds_stacked[var].where(mask, drop=True)
                    if len(data) > 0:
                        output_data[i, j, :, :] = data.isel(stacked=0).values
        
        # Create DataArray
        output_vars[var] = xr.DataArray(
            output_data,
            dims=['valid_time', 'prediction_timedelta', 'latitude', 'longitude'],
            coords={
                'valid_time': unique_valid_times,
                'prediction_timedelta': lead_times,
                'latitude': ds.latitude,
                'longitude': ds.longitude
            }
        )
    
    # Create output dataset
    result = xr.Dataset(output_vars, attrs=ds.attrs)
    
    # Drop any valid_times where all data is NaN
    result = result.dropna(dim='valid_time', how='all')
    
    return result

def print_time_and_memory(step_name, start_time):
    """Print elapsed time and current memory usage"""
    elapsed = time.time() - start_time
    memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024  # GB
    print(f"\n✓ {step_name}: {elapsed:.2f} seconds | Memory: {memory:.2f} GB")
    return time.time()

def download_pangu_data(year):
    """Main download function"""
    # Start timing
    script_start = time.time()
    start_time = time.time()
    
    print("=== PANGU Download Script Started ===")
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
    
    # Variable mapping for Pangu
    variables_to_try = [
        '2m_temperature',
        '10m_u_component_of_wind',
        '10m_v_component_of_wind',
    ]
    
    # Define your domain
    lat_bounds = [27, 17]  
    lon_bounds = [72, 82]  
    time_range = [f'{year}-01-01', f'{year}-12-31']
    
    # Output path
    output_path = os.path.expanduser(f"/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/data/raw/pangu_{year}.zarr")
    
    start_time = print_time_and_memory("Parameter setup", start_time)
    
    print("\n3. Opening Weatherbench Pangu dataset...")
    try:
        ds = xr.open_zarr(
            "gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr",
            consolidated=True
        )

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
    print(f"  Time range: {time_range[0]} to {time_range[1]}")
    print(f"  Variables: {available_vars}")

    lead_times_hours = [24, 120, 240]
    print(f" Lead time options: {lead_times_hours} hours")

    subset = ds[available_vars].sel(
        init_time=slice(time_range[0], time_range[1]),
        prediction_timedelta=[np.timedelta64(hours, 'h') for hours in lead_times_hours]
    )

    # filter for only hours 0 and 12
    subset = subset.sel(init_time=subset.init_time.dt.hour.isin([0, 12]))

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

    # convert init_time to valid_time
    print("\nConverting init_time to valid_time...")
    subset = convert_init_to_valid_time_optimized(subset)
    start_time = print_time_and_memory("Converted init_time to valid_time", start_time)
    
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

    years = [2019, 2020, 2021, 2022]
    
    # Try the download
    for year in years:
        output = download_pangu_data(year)
        print(f"\nSuccess! Data saved to: {output}")
        