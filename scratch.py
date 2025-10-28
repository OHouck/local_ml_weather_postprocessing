
import xarray as xr
import zarr

# path = "/Users/ohouck/globus/forecast_data/raw/aifs_2022.zarr"
path = "/Users/ohouck/globus/forecast_data/raw/hres_t0_2021.zarr"

#print versions of xarray and zarr
print("xarray version:", xr.__version__)
print("zarr version:", zarr.__version__)

ds = xr.open_zarr(path, zarr_format = 2)
print(ds)