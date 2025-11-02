
import xarray as xr
import zarr
path = "/Users/ohouck/globus/forecast_data/raw/pangu_2019.zarr"

#print versions of xarray and zarr
print("xarray version:", xr.__version__)
print("zarr version:", zarr.__version__)

ds = xr.open_dataset(path, chunks=None)

min_date = ds.valid_time.min().values
max_date = ds.valid_time.max().values
print(f"Data covers from {min_date} to {max_date}")
print(ds)