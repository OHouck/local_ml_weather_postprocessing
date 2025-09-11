import xarray as xr
import time
import os
import numpy as np

import xarray as xr

# Path to the GRIB2 file
path = "/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/data/raw/ifs_2020.zarr"


ds = xr.open_zarr(path, consolidated=True)
start_date = ds["valid_time"].values.min()
end_date = ds["valid_time"].values.max()
print(ds)
print(start_date, end_date)


