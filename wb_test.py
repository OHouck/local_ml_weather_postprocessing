"""
Author: Ozma Houck
File name: prepare_aurora_data.py
Date created: 10/20/2025

Purpose: create and save auora weather forecasts for specificed dates, leadtimes
"""

from pathlib import Path
import fsspec
import xarray as xr
import cdsapi
import sys
import os

sys.path.insert(0, str(Path(__file__).parent.parent))
from helper_funcs import setup_directories


dirs = setup_directories()
download_path ="/home/ohouck" 

# We will download from Google Cloud.
url = "gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr"
# ds = xr.open_zarr(fsspec.get_mapper(url), chunks=None)
ds = xr.open_zarr(url, chunks=None, storage_options={"token": "anon"})

# Day to download. This will download all times for that day.
day = "2022-05-11"

# Download the surface-level variables. We write the downloaded data to another file to cache.
file = (Path(download_path) / f"{day}-surface-level.nc")
if not file.exists():
    surface_vars = [
        "2m_temperature",
    ]
    ds_surf = ds[surface_vars].sel(time=day).compute()
    ds_surf.to_netcdf(os.path.join(download_path, f"{day}-surface-level.nc"))
print("Surface-level variables downloaded!")

# Download the atmospheric variables. We write the downloaded data to another file to cache.
file = (Path(download_path) / f"{day}-atmospheric.nc")
if not file.exists():
    atmos_vars = [
        "temperature",
    ]
    ds_atmos = ds[atmos_vars].sel(time=day).compute()
    ds_atmos.to_netcdf(os.path.join(download_path, f"{day}-atmospheric.nc"))
print("Atmos-level variables downloaded!")
