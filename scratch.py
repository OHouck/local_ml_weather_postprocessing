import xarray as xr
import time
import os
import numpy as np

import xarray as xr

# Path to the GRIB2 file
path = "/Users/ohouck/Downloads/20250825120000-360h-oper-fc.grib2"

# Open the GRIB2 file using xarray and cfgrib
ds = xr.open_dataset(path, engine="cfgrib")

# Print the dataset to inspect its contents
print(ds)

# print step values in hours
print(np.unique(ds.step.values / 3600))


