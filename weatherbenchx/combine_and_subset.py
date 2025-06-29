"""
Name: combine_and_subset.py
Author: Ozma Houck
Date: 6/24/25

Purpose: Take output created from weatherbench_download.py and combine and filter
them into region specific datasets that can be exported to laptop

Enhanced with:
- Dask configuration and optional local cluster
- Comprehensive diagnostics
- Better error handling and chunking strategies
"""

import os
import socket
import xarray as xr
import numpy as np
import glob
import logging
from collections import defaultdict
import psutil
import gc
import json
from datetime import datetime

# Dask imports
import dask
import dask.array as da
from dask.distributed import Client, as_completed
from dask.diagnostics import ProgressBar

# Configure dask for optimal performance
dask.config.set({
    'array.chunk-size': '128MB',  # Target chunk size
    'array.slicing.split_large_chunks': True,
    'distributed.worker.memory.target': 0.8,  # Use 80% of memory before spilling
    'distributed.worker.memory.spill': 0.85,
    'distributed.worker.memory.pause': 0.90,
    'distributed.worker.memory.terminate': 0.95
})

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def log_memory_usage(stage):
    """Log current memory usage"""
    process = psutil.Process()
    mem_info = process.memory_info()
    logger.info(f"Memory at {stage}: RSS={mem_info.rss/1e9:.2f}GB, VMS={mem_info.vms/1e9:.2f}GB")

def setup_dask_client(n_workers=None, threads_per_worker=None, memory_limit=None):
    """
    Setup a local dask client for better control over parallel processing
    
    Parameters:
    -----------
    n_workers : int, optional
        Number of worker processes. If None, uses SLURM_CPUS_PER_TASK / 2
    threads_per_worker : int, optional
        Threads per worker. If None, set to 2
    memory_limit : str, optional
        Memory limit per worker. If None, calculated from available memory
    
    Returns:
    --------
    client : dask.distributed.Client or None
    """
    try:
        # Get SLURM resources if available
        slurm_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', psutil.cpu_count()))
        total_memory = psutil.virtual_memory().total
        
        # Default values based on SLURM allocation
        if n_workers is None:
            n_workers = max(1, slurm_cpus // 2)  # Use half CPUs for workers
        if threads_per_worker is None:
            threads_per_worker = 2
        if memory_limit is None:
            # Divide 80% of total memory among workers
            memory_per_worker = int(0.8 * total_memory / n_workers)
            memory_limit = f"{memory_per_worker}B"
        
        logger.info(f"Starting Dask LocalCluster with {n_workers} workers, "
                   f"{threads_per_worker} threads each, {memory_limit} per worker")
        
        client = Client(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            memory_limit=memory_limit,
            silence_logs=logging.WARNING
        )
        
        logger.info(f"Dask client started: {client}")
        logger.info(f"Dashboard available at: {client.dashboard_link}")
        
        return client
        
    except Exception as e:
        logger.warning(f"Could not start dask client: {e}")
        logger.warning("Proceeding with dask's default threaded scheduler")
        return None

def diagnose_data_structure(n_files=3):
    """
    Diagnose the data structure before processing
    
    Parameters:
    -----------
    n_files : int
        Number of files to inspect
    """
    dirs = setup_directories()
    
    print("\n" + "="*80)
    print("DATA STRUCTURE DIAGNOSTICS")
    print("="*80)
    
    # Check both pangu and era5 files
    for data_type in ['predictions', 'targets']:
        files = sorted(glob.glob(os.path.join(dirs["raw"], "pangu_raw_data", f"{data_type}*.nc")))
        
        if not files:
            print(f"\nNo {data_type} files found!")
            continue
            
        print(f"\n{data_type.upper()} FILES:")
        print(f"Total files found: {len(files)}")
        print(f"Date range: {os.path.basename(files[0])} to {os.path.basename(files[-1])}")
        
        # Inspect first few files
        for i, f in enumerate(files[:n_files]):
            print(f"\n  File {i+1}: {os.path.basename(f)}")
            print(f"  Size: {os.path.getsize(f)/1e9:.3f} GB")
            
            try:
                with xr.open_dataset(f, engine='netcdf4', decode_timedelta=False) as ds:
                    print(f"  Dimensions: {dict(ds.sizes)}")  # Use .sizes instead of .dims
                    print(f"  Coordinates: {list(ds.coords)}")
                    print(f"  Data variables: {list(ds.data_vars)[:5]}")  # First 5 vars
                    if len(ds.data_vars) > 5:
                        print(f"    ... and {len(ds.data_vars) - 5} more variables")
                    
                    # Check native chunking
                    chunked_vars = []
                    for var in list(ds.data_vars)[:2]:  # Check first 2 variables
                        var_data = ds[var]
                        if hasattr(var_data, 'encoding') and 'chunks' in var_data.encoding:
                            chunked_vars.append(f"{var}: {var_data.encoding['chunks']}")
                    
                    if chunked_vars:
                        print(f"  Native chunking found:")
                        for cv in chunked_vars:
                            print(f"    {cv}")
                    else:
                        print(f"  No native chunking detected")
                    
                    # Memory estimate for full dataset
                    total_size = sum(ds[var].nbytes for var in ds.data_vars)
                    print(f"  Estimated memory (uncompressed): {total_size/1e9:.2f} GB")
                    
            except Exception as e:
                print(f"  ERROR reading file: {e}")
    
    # Test region subsetting
    print("\n" + "="*80)
    print("REGION SUBSET TEST")
    print("="*80)
    
    test_file = files[0] if files else None
    if test_file:
        try:
            with xr.open_dataset(test_file, decode_timedelta=False) as ds:
                # Check coordinate ranges
                lat_dim = 'latitude' if 'latitude' in ds.dims else 'lat'
                lon_dim = 'longitude' if 'longitude' in ds.dims else 'lon'
                
                print(f"\nCoordinate ranges in data:")
                print(f"  Latitude: {float(ds[lat_dim].min().values):.2f} to {float(ds[lat_dim].max().values):.2f}")
                print(f"  Longitude: {float(ds[lon_dim].min().values):.2f} to {float(ds[lon_dim].max().values):.2f}")
                
                # Check if latitude is sorted
                lat_diff = np.diff(ds[lat_dim].values)
                if np.all(lat_diff > 0):
                    print(f"  Latitude is sorted: ascending")
                elif np.all(lat_diff < 0):
                    print(f"  Latitude is sorted: descending")
                else:
                    print(f"  WARNING: Latitude is not monotonic!")
                
                for region in ["india", "usa_south"]:
                    lat_bounds, lon_bounds = get_region_bounds(region)
                    
                    print(f"\n{region.upper()}:")
                    print(f"  Requested lat bounds: {lat_bounds}")
                    print(f"  Requested lon bounds: {lon_bounds}")
                    
                    # Get actual subset - handle both ascending and descending latitude
                    if ds[lat_dim].values[0] > ds[lat_dim].values[-1]:  # Descending
                        lat_slice = ds[lat_dim].where(
                            (ds[lat_dim] >= lat_bounds[0]) & (ds[lat_dim] <= lat_bounds[1]), 
                            drop=True
                        )
                    else:  # Ascending
                        lat_slice = ds[lat_dim].sel({lat_dim: slice(lat_bounds[0], lat_bounds[1])})
                    
                    lon_slice = ds[lon_dim].sel({lon_dim: slice(lon_bounds[0], lon_bounds[1])})
                    
                    print(f"  Grid points found: {len(lat_slice)} x {len(lon_slice)} = {len(lat_slice) * len(lon_slice)}")
                    
                    if len(lat_slice) > 0 and len(lon_slice) > 0:
                        print(f"  Actual lat range: {float(lat_slice.min().values):.2f} to {float(lat_slice.max().values):.2f}")
                        print(f"  Actual lon range: {float(lon_slice.min().values):.2f} to {float(lon_slice.max().values):.2f}")
                    else:
                        print(f"  WARNING: No data found in region!")
                    
        except Exception as e:
            print(f"Error in region subset test: {e}")

def setup_directories():
    """Set up directory structure based on environment"""
    nodename = socket.gethostname()
    if nodename == "oMac.local":
        root = os.path.expanduser("~/OneDrive - The University of Chicago/ai_weather_ag/data")
    elif "anvil" in nodename.lower():
        root = os.path.expanduser("/anvil/projects/x-atm170020/ohouck/data")
    else:
        raise ValueError(f"Unknown node {nodename}")

    dirs = {
        "root": root,
        "raw": os.path.join(root, "raw"),
        "processed": os.path.join(root, "processed"),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs

def get_region_bounds(region):
    """Get latitude and longitude bounds for a region"""
    regions = {
        "india": ((16.75, 27.25), (71.75, 82.25)),
        "pakistan": ((23.75, 34.25), (59.75, 70.25)),
        "usa_south": ((29.75, 40.25), (-105.25 + 360, -94.75 + 360)),
        "amazon": ((-10.25, 0.25), (-70.25 + 360, -59.75 + 360)),
        "british_columbia": ((47.75, 58.25), (-130.25 + 360, -119.75 + 360))
    }
    
    if region not in regions:
        raise ValueError(f"Unknown region: {region}")
    
    return regions[region]

def preprocess_and_subset(ds, region):
    """Preprocess dataset and subset to region"""
    # Standardize dimension names
    if 'lat' in ds.dims:
        ds = ds.rename({'lat': 'latitude'})
    if 'lon' in ds.dims:
        ds = ds.rename({'lon': 'longitude'})
    if 'valid_time' in ds.dims:
        ds = ds.rename({'valid_time': 'time'})
    
    # Get region bounds
    lat_bounds, lon_bounds = get_region_bounds(region)
    
    # Handle latitude selection based on ordering
    # Check if latitude is ascending or descending
    if ds['latitude'].values[0] > ds['latitude'].values[-1]:  # Descending
        # For descending latitude, we need to reverse the bounds
        ds = ds.sel(
            latitude=slice(lat_bounds[1], lat_bounds[0]),  # Reversed!
            longitude=slice(lon_bounds[0], lon_bounds[1])
        )
    else:  # Ascending
        ds = ds.sel(
            latitude=slice(lat_bounds[0], lat_bounds[1]),
            longitude=slice(lon_bounds[0], lon_bounds[1])
        )
    
    # Ensure latitude is in ascending order for consistency
    if ds['latitude'].values[0] > ds['latitude'].values[-1]:
        ds = ds.reindex(latitude=ds['latitude'][::-1])
    
    return ds

def determine_optimal_chunks(sample_file, region):
    """
    Determine optimal chunk sizes based on data structure and region
    
    Returns:
    --------
    chunks : dict or 'auto'
    """
    try:
        with xr.open_dataset(sample_file, decode_timedelta=False) as ds:
            # Get dimension sizes after preprocessing
            ds_subset = preprocess_and_subset(ds, region)
            
            # Calculate sizes
            sizes = dict(ds_subset.sizes)  # Already using sizes, good!
            
            logger.info(f"Region {region} subset dimensions: {sizes}")
            
            # Since files have no native chunking and are ~0.4GB each,
            # we should be conservative with chunk sizes
            chunks = {}
            
            # Time-like dimensions - keep small for memory efficiency
            if 'time' in sizes:
                chunks['time'] = 1  # Process one time step at a time
            
            if 'init_time' in sizes:
                chunks['init_time'] = min(2, sizes['init_time'])  # Small chunks
            
            if 'lead_time' in sizes:
                chunks['lead_time'] = -1  # Keep lead_time together (only 7 steps)
            
            # Spatial dimensions - based on region size
            # For a 0.25° grid, India is roughly 43x43 points
            for dim in ['latitude', 'longitude']:
                if dim in sizes:
                    if sizes[dim] < 50:
                        chunks[dim] = -1  # Don't chunk small regions
                    else:
                        # Use ~50-100 points per chunk for spatial dims
                        chunks[dim] = min(100, sizes[dim])
            
            logger.info(f"Recommended chunks for {region}: {chunks}")
            return chunks
            
    except Exception as e:
        logger.warning(f"Could not determine optimal chunks: {e}")
        # Fallback to conservative manual chunks
        return {
            'time': 1,
            'init_time': 2,
            'lead_time': -1,
            'latitude': 50,
            'longitude': 50
        }

def validate_netcdf_file(filepath):
    """Check if a NetCDF file is valid and can be opened"""
    try:
        with xr.open_dataset(filepath, engine='netcdf4', decode_timedelta=False) as ds:
            # Try to access basic metadata
            _ = ds.sizes  # Use sizes instead of dims
            _ = ds.attrs
        return True
    except Exception as e:
        logger.warning(f"Invalid file {filepath}: {e}")
        return False

def group_files_by_year(file_paths):
    """Group files by year, filtering out invalid files"""
    files_by_year = defaultdict(list)
    invalid_files = []
    
    logger.info(f"Validating {len(file_paths)} files...")
    
    for i, file_path in enumerate(file_paths):
        if i % 10 == 0:
            logger.info(f"  Validated {i}/{len(file_paths)} files...")
            
        if validate_netcdf_file(file_path):
            filename = os.path.basename(file_path)
            year = filename.split('_')[1][:4]
            files_by_year[year].append(file_path)
        else:
            invalid_files.append(file_path)
    
    if invalid_files:
        logger.warning(f"Found {len(invalid_files)} invalid files:")
        for f in invalid_files[:5]:  # Show first 5
            logger.warning(f"  - {os.path.basename(f)}")
        if len(invalid_files) > 5:
            logger.warning(f"  ... and {len(invalid_files) - 5} more")
    
    return dict(files_by_year)

def process_year_data_safe(file_paths, region, year, data_type, chunks='auto'):
    """
    Process data for a single year with multiple fallback strategies
    """
    logger.info(f"Processing {data_type} for {region}, year {year} ({len(file_paths)} files)")
    log_memory_usage(f"start_{data_type}_{region}_{year}")
    
    def preprocess_func(ds):
        return preprocess_and_subset(ds, region)
    
    # For regions like India (43x43 points), the subset is small enough to load entirely
    # Each file after subsetting is only ~3MB instead of 400MB
    
    # Strategy 1: Try loading all files with minimal chunking (best for small regions)
    strategies = [
        ("minimal chunks for small region", chunks, False),
        ("no chunks (load all)", None, False),  # Try loading without dask
        ("auto chunks", 'auto', False),
        ("conservative chunks", {'time': 1, 'latitude': -1, 'longitude': -1}, False)
    ]
    
    for strategy_name, chunk_spec, use_parallel in strategies:
        logger.info(f"Trying strategy: {strategy_name}")
        try:
            # Add decode_timedelta=False to avoid warnings
            open_kwargs = {
                'preprocess': preprocess_func,
                'concat_dim': 'time',
                'combine': 'nested',
                'engine': 'netcdf4',
                'parallel': use_parallel,
                'lock': False,
                'decode_timedelta': False
            }
            
            # Only add chunks if specified
            if chunk_spec is not None:
                open_kwargs['chunks'] = chunk_spec
                
            ds = xr.open_mfdataset(file_paths, **open_kwargs)
            
            # Log actual chunk sizes if using dask
            if hasattr(ds, 'chunks') and ds.chunks:
                logger.info(f"Successfully loaded with chunks: {ds.chunks}")
            else:
                logger.info(f"Successfully loaded without chunking (in-memory)")
            
            return ds
            
        except Exception as e:
            logger.warning(f"Strategy '{strategy_name}' failed: {e}")
            gc.collect()  # Clean up before next attempt
            
    # If all strategies fail, try batch processing
    logger.info("All strategies failed, trying batch processing...")
    return batch_process_files(file_paths, region, preprocess_func)

def batch_process_files(file_paths, region, preprocess_func, batch_size=5):
    """Process files in batches when other methods fail"""
    datasets = []
    n_batches = (len(file_paths) - 1) // batch_size + 1
    
    for i in range(0, len(file_paths), batch_size):
        batch = file_paths[i:i+batch_size]
        batch_num = i // batch_size + 1
        logger.info(f"Processing batch {batch_num}/{n_batches}")
        
        try:
            batch_ds = xr.open_mfdataset(
                batch,
                preprocess=preprocess_func,
                concat_dim='time',
                combine='nested',
                chunks='auto',
                engine='netcdf4',
                parallel=False,
                lock=False,
                decode_timedelta=False
            )
            datasets.append(batch_ds)
            log_memory_usage(f"batch_{batch_num}")
            
        except Exception as e:
            logger.error(f"Failed to process batch {batch_num}: {e}")
            # Try files individually in this batch
            for f in batch:
                try:
                    with xr.open_dataset(f, chunks='auto', decode_timedelta=False) as single_ds:
                        processed = preprocess_func(single_ds)
                        # Load the processed subset into memory since it's small
                        processed = processed.load()
                        datasets.append(processed)
                except Exception as e2:
                    logger.error(f"Failed to process individual file {os.path.basename(f)}: {e2}")
    
    if not datasets:
        raise RuntimeError("No files could be processed")
    
    # Combine all datasets
    logger.info(f"Combining {len(datasets)} datasets...")
    combined = xr.concat(datasets, dim='time')
    
    # Clean up
    for ds in datasets:
        if hasattr(ds, 'close'):
            ds.close()
    
    return combined

def process_dataset(file_paths, region, output_path, data_type, use_dask_progress=True):
    """Process entire dataset by combining yearly data"""
    logger.info(f"Processing {data_type} for {region}")
    log_memory_usage(f"start_process_{data_type}_{region}")
    
    # Determine optimal chunks from first file
    if file_paths:
        chunks = determine_optimal_chunks(file_paths[0], region)
    else:
        chunks = 'auto'
    
    # Group files by year
    files_by_year = group_files_by_year(file_paths)
    logger.info(f"Found years: {sorted(files_by_year.keys())}")
    
    if not files_by_year:
        logger.error(f"No valid files found for {data_type}")
        return
    
    # Process each year
    yearly_datasets = []
    for year in sorted(files_by_year.keys()):
        try:
            year_ds = process_year_data_safe(
                files_by_year[year], 
                region, 
                year, 
                data_type,
                chunks
            )
            yearly_datasets.append(year_ds)
            log_memory_usage(f"after_year_{year}")
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error processing {data_type} {region} {year}: {e}")
            continue
    
    if not yearly_datasets:
        logger.error(f"No data processed for {data_type} {region}")
        return
    
    # Combine all years
    logger.info(f"Combining {len(yearly_datasets)} years for {data_type} {region}")
    combined_ds = xr.concat(yearly_datasets, dim='time')
    
    # Save with compression
    logger.info(f"Saving {data_type} {region} to {output_path}")
    encoding = {var: {'zlib': True, 'complevel': 4} for var in combined_ds.data_vars}
    
    # Use dask progress bar if enabled
    if use_dask_progress and hasattr(combined_ds, 'chunks'):
        with ProgressBar():
            combined_ds.to_netcdf(output_path, encoding=encoding, compute=True)
    else:
        combined_ds.to_netcdf(output_path, encoding=encoding)
    
    logger.info(f"Successfully saved {data_type} {region}")
    log_memory_usage(f"after_save_{data_type}_{region}")
    
    # Cleanup
    combined_ds.close()
    for ds in yearly_datasets:
        ds.close()
    gc.collect()

def save_run_info(regions_processed, start_time, end_time):
    """Save information about the run for debugging"""
    dirs = setup_directories()
    run_info = {
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'duration_seconds': (end_time - start_time).total_seconds(),
        'regions_processed': regions_processed,
        'hostname': socket.gethostname(),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID', 'local'),
        'python_version': os.sys.version,
        'xarray_version': xr.__version__,
        'dask_version': dask.__version__
    }
    
    info_file = os.path.join(dirs['processed'], f"run_info_{start_time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(info_file, 'w') as f:
        json.dump(run_info, f, indent=2)
    
    logger.info(f"Run info saved to {info_file}")

def main(use_dask_client=False, diagnose_only=False):
    """
    Main processing function
    
    Parameters:
    -----------
    use_dask_client : bool
        Whether to use a local dask client
    diagnose_only : bool
        If True, only run diagnostics without processing
    """
    start_time = datetime.now()
    
    # Run diagnostics first
    if diagnose_only:
        diagnose_data_structure()
        return
    
    # Setup dask client if requested
    client = None
    if use_dask_client:
        client = setup_dask_client()
    
    try:
        dirs = setup_directories()
        regions = ["india"]  # Start with just India for testing
        # regions = ["india", "usa_south", "amazon", "british_columbia"]  # Full list
        
        regions_processed = []
        
        for region in regions:
            logger.info(f"Starting region: {region}")
            
            # Get file paths
            pangu_files = sorted(glob.glob(os.path.join(dirs["raw"], "pangu_raw_data", "predictions*.nc")))
            era5_files = sorted(glob.glob(os.path.join(dirs["raw"], "pangu_raw_data", "targets*.nc")))
            
            if not pangu_files or not era5_files:
                logger.error(f"Missing files for {region}")
                continue
            
            # Process both datasets
            pangu_output = os.path.join(dirs["processed"], f"pangu_{region}.nc")
            era5_output = os.path.join(dirs["processed"], f"era5_{region}.nc")
            
            try:
                process_dataset(pangu_files, region, pangu_output, "pangu")
                process_dataset(era5_files, region, era5_output, "era5")
                logger.info(f"Completed region: {region}")
                regions_processed.append(region)
                
            except Exception as e:
                logger.error(f"Error processing region {region}: {e}", exc_info=True)
    
    finally:
        # Cleanup
        if client:
            client.close()
        
        # Save run information
        end_time = datetime.now()
        save_run_info(regions_processed, start_time, end_time)
        
        logger.info(f"Total processing time: {(end_time - start_time).total_seconds():.1f} seconds")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process WeatherBench data')
    parser.add_argument('--use-dask-client', action='store_true', 
                       help='Use a local dask client for processing')
    parser.add_argument('--diagnose-only', action='store_true',
                       help='Only run diagnostics without processing')
    
    args = parser.parse_args()
    
    # Run diagnostics first if not diagnose-only mode
    if not args.diagnose_only:
        logger.info("Running initial diagnostics...")
        diagnose_data_structure(n_files=2)
        print("\n" + "="*80 + "\n")
        
        # Print recommendations based on diagnostics
        print("PROCESSING RECOMMENDATIONS:")
        print("-" * 40)
        print("1. Data files have NO native chunking")
        print("2. Each predictions file is ~0.4GB, targets ~0.1GB")
        print("3. India region subset is only ~43x43 grid points")
        print("4. After subsetting, data size reduces by ~500x")
        print("5. Consider processing without dask for small regions")
        print("-" * 40)
        print("\nStarting processing...\n")
    
    main(use_dask_client=args.use_dask_client, diagnose_only=args.diagnose_only)