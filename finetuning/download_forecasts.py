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


def convert_init_to_valid_time_fastest(ds):
    """
    Ultimate optimized version combining all performance techniques:
    1. Pre-computed index mappings
    2. Bulk memory operations
    3. Parallel processing for variables
    4. Minimal data copying
    
    Expected performance: 10-50x faster than the 3-minute baseline
    """
    # Extract time arrays as integers for faster comparison
    init_times = ds.init_time.values.view('int64')
    pred_tds = ds.prediction_timedelta.values.view('int64')
    
    # Compute all valid times using broadcasting
    valid_times_2d = init_times[:, None] + pred_tds[None, :]
    
    # Get unique valid times using optimized numpy operations
    unique_valid_times = np.unique(valid_times_2d)
    n_valid = len(unique_valid_times)
    n_pred = len(pred_tds)
    
    # Build the mapping table using searchsorted (O(n log n) instead of O(n²))
    # This is the key optimization - we build a reusable index map
    index_map = np.full((n_valid, n_pred), -1, dtype=np.int32)
    
    # Create a reverse lookup using searchsorted
    valid_time_indices = np.searchsorted(unique_valid_times, valid_times_2d.ravel())
    valid_time_indices = valid_time_indices.reshape(len(init_times), n_pred)
    
    # Fill the index map efficiently
    for init_idx in range(len(init_times)):
        for pred_idx in range(n_pred):
            valid_idx = valid_time_indices[init_idx, pred_idx]
            index_map[valid_idx, pred_idx] = init_idx
    
    # Get variable names and prepare for parallel processing
    var_names = list(ds.data_vars)
    n_vars = len(var_names)
    
    # Function to process a single variable
    def process_variable(var_name):
        source = ds[var_name].values
        output = np.full((n_valid, n_pred) + source.shape[2:], np.nan, dtype=source.dtype)
        
        # Optimized copying using numpy's advanced indexing
        # This is much faster than nested loops
        for pred_idx in range(n_pred):
            # Get all valid init indices for this prediction timedelta
            init_indices = index_map[:, pred_idx]
            valid_mask = init_indices >= 0
            
            if valid_mask.any():
                # Copy all valid data at once for this prediction timedelta
                output[valid_mask, pred_idx] = source[init_indices[valid_mask], pred_idx]
        
        return var_name, output
    
    # Process variables in parallel if beneficial
    if n_vars > 1 and n_vars <= mp.cpu_count():
        # Use thread pool for I/O bound operations
        with ThreadPoolExecutor(max_workers=min(n_vars, 4)) as executor:
            results = list(executor.map(process_variable, var_names))
    else:
        # Process sequentially for single variable or many variables
        results = [process_variable(var_name) for var_name in var_names]
    
    # Convert back to datetime for valid_time coordinate
    unique_valid_times_dt = pd.to_datetime(unique_valid_times)
    
    # Build the output dataset
    data_vars = {}
    for var_name, output in results:
        data_vars[var_name] = xr.DataArray(
            output,
            dims=['valid_time', 'prediction_timedelta', 'latitude', 'longitude'],
            coords={
                'valid_time': unique_valid_times_dt,
                'prediction_timedelta': ds.prediction_timedelta.values,
                'latitude': ds.latitude,
                'longitude': ds.longitude
            },
            attrs=ds[var_name].attrs
        )
    
    # Create the final dataset
    result = xr.Dataset(data_vars, attrs=ds.attrs)
    
    # Remove all-NaN valid times efficiently
    # Check if any variable has data for each valid time
    has_data = np.zeros(n_valid, dtype=bool)
    for var_name, output in results:
        has_data |= ~np.isnan(output).all(axis=(1, 2, 3))
    
    if not has_data.all():
        result = result.isel(valid_time=has_data)
    
    return result

def validate_time_conversion(original_ds, converted_ds, sample_size=10):
    """
    Validate that the conversion from init_time to valid_time was done correctly.
    
    Parameters:
    -----------
    original_ds : xarray.Dataset
        Original dataset with init_time dimension
    converted_ds : xarray.Dataset
        Converted dataset with valid_time dimension
    sample_size : int
        Number of random samples to check
        
    Returns:
    --------
    dict
        Dictionary with validation results
    """
    results = {
        'is_valid': True,
        'errors': [],
        'checks_performed': 0,
        'samples_checked': []
    }
    
    # Check 1: Verify dimensions
    expected_dims = {'valid_time', 'prediction_timedelta', 'latitude', 'longitude'}
    actual_dims = set(converted_ds.dims.keys())
    
    if not expected_dims.issubset(actual_dims):
        results['is_valid'] = False
        results['errors'].append(f"Missing dimensions: {expected_dims - actual_dims}")
    
    # Check 2: Verify that all data is preserved (no data loss)
    for var in original_ds.data_vars:
        if var not in converted_ds.data_vars:
            results['is_valid'] = False
            results['errors'].append(f"Variable {var} missing in converted dataset")
            continue
            
        # Count non-NaN values
        original_count = np.isfinite(original_ds[var].values).sum()
        converted_count = np.isfinite(converted_ds[var].values).sum()
        
        if original_count != converted_count:
            results['is_valid'] = False
            results['errors'].append(
                f"Data count mismatch for {var}: original={original_count}, converted={converted_count}"
            )
    
    # Check 3: Sample-based validation
    # Randomly sample some points and verify the mapping
    np.random.seed(42)  # For reproducibility
    
    n_init = len(original_ds.init_time)
    n_pred = len(original_ds.prediction_timedelta)
    
    for _ in range(sample_size):
        # Random indices
        init_idx = np.random.randint(0, n_init)
        pred_idx = np.random.randint(0, n_pred)
        lat_idx = np.random.randint(0, len(original_ds.latitude))
        lon_idx = np.random.randint(0, len(original_ds.longitude))
        
        # Calculate expected valid_time
        init_time = original_ds.init_time.values[init_idx]
        pred_td = original_ds.prediction_timedelta.values[pred_idx]
        expected_valid_time = init_time + pred_td
        
        # Get the data from original dataset
        original_value = {}
        for var in original_ds.data_vars:
            original_value[var] = original_ds[var].isel(
                init_time=init_idx,
                prediction_timedelta=pred_idx,
                latitude=lat_idx,
                longitude=lon_idx
            ).values.item()
        
        # Find the corresponding point in converted dataset
        valid_time_idx = np.where(converted_ds.valid_time.values == expected_valid_time)[0]
        
        if len(valid_time_idx) == 0:
            results['errors'].append(
                f"Valid time {expected_valid_time} not found in converted dataset"
            )
            continue
            
        valid_time_idx = valid_time_idx[0]
        
        # Compare values
        for var in original_ds.data_vars:
            converted_value = converted_ds[var].isel(
                valid_time=valid_time_idx,
                prediction_timedelta=pred_idx,
                latitude=lat_idx,
                longitude=lon_idx
            ).values.item()
            
            if not np.allclose(original_value[var], converted_value, equal_nan=True):
                results['is_valid'] = False
                results['errors'].append(
                    f"Value mismatch for {var} at init_time={init_time}, "
                    f"pred_td={pred_td}: {original_value[var]} != {converted_value}"
                )
        
        results['checks_performed'] += 1
        results['samples_checked'].append({
            'init_time': init_time,
            'prediction_timedelta': pred_td,
            'valid_time': expected_valid_time,
            'status': 'passed' if len(results['errors']) == 0 else 'failed'
        })
    
    # Check 4: Verify time range consistency
    min_valid_time = original_ds.init_time.min() + original_ds.prediction_timedelta.min()
    max_valid_time = original_ds.init_time.max() + original_ds.prediction_timedelta.max()
    
    if converted_ds.valid_time.min() < min_valid_time or converted_ds.valid_time.max() > max_valid_time:
        results['is_valid'] = False
        results['errors'].append(
            f"Valid time range inconsistency: converted range "
            f"[{converted_ds.valid_time.min().values}, {converted_ds.valid_time.max().values}] "
            f"outside expected range [{min_valid_time.values}, {max_valid_time.values}]"
        )
    
    return results


def print_validation_results(results):
    """Pretty print validation results."""
    print("\n=== Validation Results ===")
    print(f"Overall Status: {'✓ PASSED' if results['is_valid'] else '✗ FAILED'}")
    print(f"Checks Performed: {results['checks_performed']}")
    
    if results['errors']:
        print(f"\nErrors Found ({len(results['errors'])}):")
        for i, error in enumerate(results['errors'], 1):
            print(f"  {i}. {error}")
    else:
        print("\nNo errors found!")
    
    print("\nSample Checks:")
    for sample in results['samples_checked'][:5]:  # Show first 5
        print(f"  {sample['init_time']} + {sample['prediction_timedelta']} "
              f"→ {sample['valid_time']} [{sample['status']}]")

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
    # lat_bounds = [19, 17]  
    # lon_bounds = [80, 82]  
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


    # # subset to a smaller area for testing
    # subset = subset.sel(
    #     latitude=slice(lat_bounds[0], lat_bounds[1]),  # Note: xarray uses (lat_min, lat_max)
    #     longitude=slice(lon_bounds[0], lon_bounds[1])  # Note: xarray uses (lon_min, lon_max)
    # )

    # convert init_time to valid_time
    print("\nConverting init_time to valid_time...")
    # subset_valid_time = convert_init_to_valid_time_fastest(subset)
    # subset_valid_time = benchmark_and_select_fastest(subset)
    subset_valid_time = convert_init_to_valid_time_fastest(subset)

    # check validation
    validation_results = validate_time_conversion(subset, subset_valid_time)
    print_validation_results(validation_results)

    start_time = print_time_and_memory("Converted init_time to valid_time", start_time)

    exit()
    
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

    # years = [2018, 2019, 2020, 2021, 2022]
    years = [2018]
    
    # Try the download
    for year in years:
        output = download_pangu_data(year)
        print(f"\nSuccess! Data saved to: {output}")
        