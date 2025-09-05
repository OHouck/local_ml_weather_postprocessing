import xarray as xr
import time
import os
import numpy as np

import xarray as xr

# Path to the GRIB2 file
path = "/Users/ohouck/test.zarr"

ds = xr.open_dataset(path)
lead_times = np.unique(ds['prediction_timedelta'].values)
lead_time_hours = lead_times / np.timedelta64(1, 'h')
print(lead_time_hours)


