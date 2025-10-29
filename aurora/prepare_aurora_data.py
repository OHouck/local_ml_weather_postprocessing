"""
Author: Ozma Houck
File name: prepare_aurora_data.py
Date created: 10/20/2025
Purpose: create and save aurora weather forecasts for specified dates, leadtimes
"""
from pathlib import Path
import fsspec
import xarray as xr
import cdsapi
import sys
import os
sys.path.insert(0, str(Path(__file__).parent.parent))
from helper_funcs import setup_directories

def download_aurora_data(day):
    """Download aurora data for a specific day."""
    dirs = setup_directories()
    download_path = os.path.join(dirs["raw"], "aurora")
    os.makedirs(download_path, exist_ok=True)
    
    # We will download from Google Cloud.
    url = "gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr"
    ds = xr.open_zarr(url, chunks=None, storage_options={"token": "anon"})
    
    # Download the surface-level variables
    file = Path(download_path) / f"{day}-surface-level.nc"
    if not file.exists():
        surface_vars = [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_temperature",
            "mean_sea_level_pressure",
        ]
        ds_surf = ds[surface_vars].sel(time=day).compute()
        ds_surf.to_netcdf(file)
        print(f"Surface-level variables for {day} downloaded!")
    else:
        print(f"Surface-level file for {day} already exists, skipping.")
    
    # Download the atmospheric variables
    file = Path(download_path) / f"{day}-atmospheric.nc"
    if not file.exists():
        atmos_vars = [
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
            "specific_humidity",
            "geopotential",
        ]
        ds_atmos = ds[atmos_vars].sel(time=day).compute()
        ds_atmos.to_netcdf(file)
        print(f"Atmospheric variables for {day} downloaded!")
    else:
        print(f"Atmospheric file for {day} already exists, skipping.")
    
    # Download the static variables from era5 (only once)
    file = Path(download_path) / "static.nc"
    if not file.exists():
        c = cdsapi.Client()
        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": [
                    "geopotential",
                    "land_sea_mask",
                    "soil_type",
                    "standard_deviation_of_orography",
                ],
                "year": "2023",
                "month": "01",
                "day": "01",
                "time": "00:00",
                "format": "netcdf",
            },
            os.path.join(download_path, "era5_static.nc"),
        )
        print("Static variables downloaded!")

if __name__ == "__main__":
    # Can be called with command line argument or default
    if len(sys.argv) > 1:
        day = sys.argv[1]
    else:
        day = "2022-05-11"
    
    download_aurora_data(day)
