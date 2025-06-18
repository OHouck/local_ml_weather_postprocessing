import xarray as xr
import gcsfs

# Test anonymous access
fs = gcsfs.GCSFileSystem(token='anon')

# Test reading a small subset
test_path = 'gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr'
ds = xr.open_zarr(test_path, storage_options={'token': 'anon'})

# Check if you can access the data
print(f"Variables available: {list(ds.data_vars)}")
print(f"Time range: {ds.time.values[0]} to {ds.time.values[-1]}")