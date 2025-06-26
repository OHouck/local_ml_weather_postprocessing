"""
Name: combine_and_subset.py
Author: Ozma Houck

Date: 6/24/25

Purpose: Take output created from weatherbench_download.py and combine and filter
them into region specific datasets that can be exported to laptop

This is a similar script to 0_download_data.py in finetuning but created for 
the weatherbenchx pipeline
"""

import os
import socket
import xarray as xr
import numpy as np
import glob
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Map the new region strings to Koppen‐Geiger codes:
CLIMATE_ZONE_MAP = {
    'tropical':  1,
    'arid':       2,
    'temperate':  3,
    'cold':       4,
    'polar':      5,
}


def setup_directories():
    """Set up directory structure based on environment"""
    nodename = socket.gethostname()
    if nodename == "oMac.local":
        root = os.path.expanduser("~/OneDrive - The University of Chicago/ai_weather_ag/data")
    elif "anvil" in nodename.lower():
        root = os.path.expanduser("/anvil/projects/x-atm170020/ohouck/data")
    else:
        raise ValueError(f"Unknown node {nodename} Please add path to setup_directories function")

    dirs = {
        "root": root,
        "raw": os.path.join(root, "raw"),
        "processed": os.path.join(root, "processed"),
        "fig": os.path.join(root, "../figures/finetuning"),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs

def preprocess_data(ds, region):
        # region to download for india
    if region == "india":
        full_lat_values = np.arange(16.75, 27.25, 0.25)
        full_lon_values = np.arange(71.75, 82.25, 0.25)
    elif region == "pakistan":
        # full lat values are by 0.25 degrees
        full_lat_values = np.arange(23.75, 34.25, 0.25) # pakistan/afganistan
        full_lon_values = np.arange(59.75, 70.25, 0.25) 
    elif region == "usa_south":
        full_lat_values = np.arange(29.75, 40.25, 0.25)
        full_lon_values = np.arange(-105.25 + 360, -94.75 + 360, 0.25)
    elif region == "amazon":
        full_lat_values = np.arange(-10.25, 0.25, 0.25)
        full_lon_values = np.arange(-70.25 + 360, -59.75 + 360, 0.25)
    elif region == "british_columbia":
        full_lat_values = np.arange(47.75, 58.25, 0.25) # if rerun should start at 47.75
        full_lon_values = np.arange(-130.25 + 360, -119.75 + 360, 0.25)

    dims = ds.dims
    if 'latitude' not in dims and 'lat' in dims:
        ds = ds.rename({'lat': 'latitude'})
    if 'longitude' not in dims and 'lon' in dims:
        ds = ds.rename({'lon': 'longitude'})
    if 'time' not in dims and 'valid_time' in dims:
        ds = ds.rename({'valid_time': 'time'})

    ds = ds.sortby('latitude')
    
    ds = ds.sel(latitude=slice(full_lat_values.min(), full_lat_values.max()),
               longitude=slice(full_lon_values.min(), full_lon_values.max()))

    return ds

def create_preprocess_function(region):
    """Create a preprocessing function with region parameter bound"""
    def preprocess_func(ds):
        return preprocess_data(ds, region)
    return preprocess_func

def main():

    dirs = setup_directories()

    regions = ["india", "usa_south", "amazon", "british_columbia"]
    for region in regions:

        pangu_file_paths = sorted(glob.glob(os.path.join(dirs["raw"], "pangu_raw_data", "predictions*.nc")))
        era5_file_paths = sorted(glob.glob(os.path.join(dirs["raw"], "pangu_raw_data", "targets*.nc")))

        pangu_output_path = os.path.join(dirs["processed"], f"pangu_{region}.nc")
        era5_output_path = os.path.join(dirs["processed"], f"era5_{region}.nc")

        pangu = xr.open_mfdataset(
            pangu_file_paths,
            preprocess=create_preprocess_function(region),
            concat_dim="time",
            combine="nested",
            engine="netcdf4",
            decode_timedelta=False,
            parallel=True
        )
        logger.info(f"opened pangu for {region}")
        logger.info(pangu)
        pangu.to_netcdf(pangu_output_path)
        logger.info(f"Saved pangu for {region}")

        era5 = xr.open_mfdataset(
            era5_file_paths,
            preprocess=create_preprocess_function(region),
            concat_dim="time",
            combine="nested",
            engine="netcdf4",
            decode_timedelta=False,
            parallel=True
        )
        logger.info(f"opened era5 for {region}")
        logger.info(era5)
        era5.to_netcdf(era5_output_path)
        logger.info(f"Saved era5 for {region}")
    
if __name__ == "__main__":
    main()
