import xarray as xr
import os


# Test reading a small subset
path = "/anvil/projects/x-atm170020/ohouck/data/processed/pangu_india.nc"

ds = xr.open_dataset(os.path.join(path))

print(ds)
