"""
Name: combine_and_subset.py
Author: Ozma Houck

Date: 6/24/25

Purpose: Take output created from weatherbench_download.py and combine and filter
them into region specific datasets that can be exported to laptop
"""

import os
import socket
import xarray as xr
import numpy as np
import glob
import logging
from collections import defaultdict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

    ds = ds.sortby('latitude')
    
    # Get region bounds and subset
    lat_bounds, lon_bounds = get_region_bounds(region)
    ds = ds.sel(
        latitude=slice(lat_bounds[0], lat_bounds[1]),
        longitude=slice(lon_bounds[0], lon_bounds[1])
    )
    
    return ds

def group_files_by_year(file_paths):
    """Group files by year based on filename patterns"""
    files_by_year = defaultdict(list)
    
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        # Extract year from filename pattern: predictions_2018-01-01_2018-01-07.nc
        year = filename.split('_')[1][:4]  # Get first 4 chars after first underscore
        files_by_year[year].append(file_path)
    
    return dict(files_by_year)

def process_year_data(file_paths, region, year, data_type):
    """Process data for a single year and region"""
    logger.info(f"Processing {data_type} for {region}, year {year} ({len(file_paths)} files)")
    
    def preprocess_func(ds):
        return preprocess_and_subset(ds, region)
    
    # Load with appropriate chunking
    chunks = {'time': 50, 'init_time': 50, 'lead_time': -1, 'latitude': -1, 'longitude': -1}
    
    ds = xr.open_mfdataset(
        file_paths,
        preprocess=preprocess_func,
        concat_dim='time',
        combine='nested',
        chunks=chunks,
        engine='netcdf4',
        parallel=True
    )
    
    logger.info(f"Loaded {data_type} {region} {year}: {ds.sizes}")
    return ds

def process_dataset(file_paths, region, output_path, data_type):
    """Process entire dataset by combining yearly data"""
    logger.info(f"Processing {data_type} for {region}")
    
    # Group files by year
    files_by_year = group_files_by_year(file_paths)
    logger.info(f"Found years: {sorted(files_by_year.keys())}")
    
    # Process each year
    yearly_datasets = []
    for year in sorted(files_by_year.keys()):
        try:
            year_ds = process_year_data(files_by_year[year], region, year, data_type)
            yearly_datasets.append(year_ds)
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
    
    combined_ds.to_netcdf(output_path, encoding=encoding)
    logger.info(f"Successfully saved {data_type} {region}")
    
    # Cleanup
    combined_ds.close()
    for ds in yearly_datasets:
        ds.close()

def main():
    dirs = setup_directories()
    regions = ["india", "usa_south", "amazon", "british_columbia"]
    
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
        except Exception as e:
            logger.error(f"Error processing region {region}: {e}")

if __name__ == "__main__":
    main()