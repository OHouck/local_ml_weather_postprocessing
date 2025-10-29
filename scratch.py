
import xarray as xr
import zarr

path = "/Users/ohouck/globus/forecast_data/raw/era5_2024.zarr"
# path = "/Users/ohouck/globus/forecast_data/raw/era5_2024.zarr.v2_backup"

#print versions of xarray and zarr
print("xarray version:", xr.__version__)
print("zarr version:", zarr.__version__)

ds = xr.open_zarr(path)
print(ds)