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

def prepare_data(ds, region):
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


def main():

    dirs = setup_directories()

    regions = ["india", "usa_south", "amazon", "british_columbia"]
    model_names = ["pangu", "ifs"]
    region = "india"


    pangu_file_paths = sorted(glob.glob(os.path.join(dirs["raw"], "pangu_raw_data", "predictions*.nc")))
    era5_file_paths = sorted(glob.glob(os.path.join(dirs["raw"], "pangu_raw_data", "targets*.nc")))

    test_pangu = pangu_file_paths[0]
    pangu = xr.open_dataset(test_pangu)
    print(pangu)

    test_era5 = era5_file_paths[0]
    era5 = xr.open_dataset(test_era5)
    print(era5)

    exit()

    pangu_region = xr.open_mfdataset(
        pangu_file_paths,
    )
        
