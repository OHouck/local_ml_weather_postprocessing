import xarray as xr

path = "/Users/ohouck/Documents/data_2023.nc"

ds = xr.open_dataset(path)

print(f"latitude and longitude ranges: {ds.LATITUDE.min().values} to {ds.LATITUDE.max().values}, ")
print(f"and {ds.LONGITUDE.min().values} to {ds.LONGITUDE.max().values}")
print(ds)