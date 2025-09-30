import xarray as xr
import time
import os
import numpy as np

import xarray as xr

path = "/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/IMD/IMD_0p25deg/data_1916.nc"

ds = xr.open_dataset(path, engine = 'netcdf4')

print(ds)



