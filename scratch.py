from pathlib import Path

import fsspec
import xarray as xr
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Then your existing code...

# Data will be downloaded here.
download_path = Path("~/downloads/hres_0.25")

download_path = download_path.expanduser()
download_path.mkdir(parents=True, exist_ok=True)

# We will download from Google Cloud.
url = "gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr"

# fs = fsspec.filesystem('gs', token='anon')
# mapper = fs.get_mapper(url)
# ds = xr.open_zarr(mapper, chunks=None)

ds = xr.open_zarr(fsspec.get_mapper(url), chunks=None) # original

# Day to download. This will download all times for that day.
day = "2022-05-11"

# Download the surface-level variables. We write the downloaded data to another file to cache.
if not (download_path / f"{day}-surface-level.nc").exists():
    surface_vars = [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_temperature",
        "mean_sea_level_pressure",
    ]
    ds_surf = ds[surface_vars].sel(time=day).compute()
    ds_surf.to_netcdf(str(download_path / f"{day}-surface-level.nc"))
print("Surface-level variables downloaded!")

# Download the atmospheric variables. We write the downloaded data to another file to cache.
if not (download_path / f"{day}-atmospheric.nc").exists():
    atmos_vars = [
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "specific_humidity",
        "geopotential",
    ]
    ds_atmos = ds[atmos_vars].sel(time=day).compute()
    ds_atmos.to_netcdf(str(download_path / f"{day}-atmospheric.nc"))
print("Atmos-level variables downloaded!")