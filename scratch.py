import xarray as xr
import time
import os
import numpy as np

import xarray as xr

path = "/Users/ohouck/globus/forecast_data/raw/IMD_0p25deg/data_2005.nc"

ds = xr.open_dataset(path, engine = 'netcdf4')

print(ds)



