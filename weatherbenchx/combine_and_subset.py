"""
Name: combine_and_subset_incremental.py
Author: Ozma Houck (Enhanced)
Date: 7/3/25

Purpose: Combine and filter weatherbench data incrementally to avoid memory issues
This version processes data year by year and appends to the output file
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
import tempfile

# Configure logging
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
    dim_mapping = {
        'lat': 'latitude',
        'lon': 'longitude',
        'valid_time': 'time'
    }
    
    for old_name, new_name in dim_mapping.items():
        if old_name in ds.dims and new_name not in ds.dims:
            ds = ds.rename({old_name: new_name})
    
    # Get region bounds
    lat_bounds, lon_bounds = get_region_bounds(region)
    
    # Handle latitude selection based on ordering
    if ds['latitude'].values[0] > ds['latitude'].values[-1]:  # Descending
        ds = ds.sel(
            latitude=slice(lat_bounds[1], lat_bounds[0]),
            longitude=slice(lon_bounds[0], lon_bounds[1])
        )
    else:  # Ascending
        ds = ds.sel(
            latitude=slice(lat_bounds[0], lat_bounds[1]),
            longitude=slice(lon_bounds[0], lon_bounds[1])
        )
    
    # Ensure latitude is in ascending order
    if ds['latitude'].values[0] > ds['latitude'].values[-1]:
        ds = ds.reindex(latitude=ds['latitude'][::-1])
    
    return ds

def process_year_files(file_paths, region, year):
    """Process all files for a single year and return combined dataset"""
    logger.info(f"Processing {len(file_paths)} files for year {year}")
    
    datasets = []
    for i, file_path in enumerate(file_paths):
        try:
            with xr.open_dataset(file_path, engine='netcdf4', decode_timedelta=False) as ds:
                ds_subset = preprocess_and_subset(ds, region)
                ds_loaded = ds_subset.load()
                datasets.append(ds_loaded)
            
            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {i + 1}/{len(file_paths)} files")
                gc.collect()
                
        except Exception as e:
            logger.error(f"Failed to process {os.path.basename(file_path)}: {e}")
    
    if not datasets:
        raise ValueError(f"No datasets processed for year {year}")
    
    # Combine all datasets for this year
    logger.info(f"Combining {len(datasets)} datasets for year {year}")
    year_combined = xr.concat(datasets, dim='time')
    year_combined = year_combined.sortby('time')
    
    # Clean up individual datasets
    for ds in datasets:
        ds.close()
    gc.collect()
    
    return year_combined

def process_region_incremental(region, data_type, file_paths, output_dir):
    """Process region data incrementally, year by year"""
    logger.info(f"\nProcessing {data_type} for {region} (incremental mode)")
    log_memory_usage(f"start_{data_type}_{region}")
    
    # Group files by year
    files_by_year = defaultdict(list)
    for file_path in sorted(file_paths):
        filename = os.path.basename(file_path)
        year = filename.split('_')[1][:4]
        files_by_year[year].append(file_path)
    
    years = sorted(files_by_year.keys())
    logger.info(f"Found years: {years}")
    
    if not years:
        logger.error(f"No valid files found for {data_type}")
        return False
    
    output_path = os.path.join(output_dir, f"{data_type}_{region}.nc")
    temp_path = output_path + '.tmp'
    
    # Process first year to create the base file
    first_year = years[0]
    logger.info(f"\nProcessing first year {first_year} to create base file")
    
    try:
        year_data = process_year_files(files_by_year[first_year], region, first_year)
        
        # Save with compression
        encoding = {var: {'zlib': True, 'complevel': 4} for var in year_data.data_vars}
        year_data.to_netcdf(output_path, encoding=encoding, engine='netcdf4', mode='w')
        logger.info(f"Created base file with year {first_year}")
        
        year_data.close()
        log_memory_usage(f"after_year_{first_year}")
        gc.collect()
        
    except Exception as e:
        logger.error(f"Failed to process first year {first_year}: {e}")
        return False
    
    # Process remaining years one at a time
    for year in years[1:]:
        try:
            logger.info(f"\nProcessing year {year}")
            year_data = process_year_files(files_by_year[year], region, year)
            
            # Read existing data
            logger.info(f"Reading existing data and appending year {year}")
            with xr.open_dataset(output_path, engine='netcdf4', decode_timedelta=False) as existing_data:
                # Combine with new year
                combined = xr.concat([existing_data, year_data], dim='time')
                combined = combined.sortby('time')
                
                # Save to temporary file
                encoding = {var: {'zlib': True, 'complevel': 4} for var in combined.data_vars}
                combined.to_netcdf(temp_path, encoding=encoding, engine='netcdf4', mode='w')
            
            # Replace original with temporary
            os.replace(temp_path, output_path)
            logger.info(f"Successfully appended year {year}")
            
            year_data.close()
            log_memory_usage(f"after_year_{year}")
            gc.collect()
            
        except Exception as e:
            logger.error(f"Failed to process year {year}: {e}")
            # Clean up temp file if it exists
            if os.path.exists(temp_path):
                os.remove(temp_path)
            continue
    
    # Verify final output
    try:
        with xr.open_dataset(output_path) as ds:
            logger.info(f"\nFinal dataset for {data_type} {region}:")
            logger.info(f"  Shape: {dict(ds.sizes)}")
            logger.info(f"  Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
            logger.info(f"  Variables: {list(ds.data_vars)}")
        return True
    except Exception as e:
        logger.error(f"Failed to verify output file: {e}")
        return False

def main():
    """Main processing function"""
    start_time = datetime.now()
    
    try:
        dirs = setup_directories()
        regions = ["india"]  # Start with India
        
        regions_processed = []
        
        for region in regions:
            logger.info(f"\n{'='*60}")
            logger.info(f"Starting region: {region}")
            logger.info(f"{'='*60}")
            
            # Get file paths
            pangu_files = sorted(glob.glob(os.path.join(dirs["raw"], "pangu_raw_data", "predictions*.nc")))
            era5_files = sorted(glob.glob(os.path.join(dirs["raw"], "pangu_raw_data", "targets*.nc")))
            
            if not pangu_files:
                logger.error(f"No pangu files found")
                continue
            if not era5_files:
                logger.error(f"No era5 files found")
                continue
            
            # Process both datasets
            success_pangu = process_region_incremental(region, "pangu", pangu_files, dirs["processed"])
            success_era5 = process_region_incremental(region, "era5", era5_files, dirs["processed"])
            
            if success_pangu and success_era5:
                regions_processed.append(region)
                logger.info(f"\nSuccessfully completed region: {region}")
            else:
                logger.error(f"\nFailed to process region: {region}")
            
            gc.collect()
    
    finally:
        # Save run information
        end_time = datetime.now()
        run_info = {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': (end_time - start_time).total_seconds(),
            'regions_processed': regions_processed,
            'hostname': socket.gethostname(),
            'slurm_job_id': os.environ.get('SLURM_JOB_ID', 'local'),
            'mode': 'incremental'
        }
        
        info_file = os.path.join(dirs['processed'], f"run_info_{start_time.strftime('%Y%m%d_%H%M%S')}.json")
        with open(info_file, 'w') as f:
            json.dump(run_info, f, indent=2)
        
        logger.info(f"\nTotal processing time: {(end_time - start_time).total_seconds():.1f} seconds")
        logger.info(f"Run info saved to {info_file}")

if __name__ == "__main__":
    logger.info("Starting WeatherBench data processing (incremental mode)...")
    main()