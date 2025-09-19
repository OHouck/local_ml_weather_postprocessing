import xarray as xr
import time
import os
import numpy as np

import xarray as xr

# Path to the GRIB2 file
# path = "/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/data/fine_tuning_output/pangu/usa_south/train_10m_wind_speed_test_10m_wind_speed_dim10x10_leadtime_24_120_216h_train2018-01-01-2021-12-31_test2022-01-01-2022-12-31_mlp.zarr"
path = "/Users/ohouck/Downloads/processed_init_2024091100.zarr"


ds = xr.open_zarr(path, consolidated=True)
print(ds)


