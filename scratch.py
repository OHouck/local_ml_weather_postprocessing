import xarray as xr
import numpy as np

# path = "/Users/ohouck/Documents/data_2023.nc"
# path = "/Users/ohouck/Documents/aifs2_init_2024073100.nc"
# path = "/Users/ohouck/Documents/aifs1_init_2024071100.nc"
path= "/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/data/raw/pangu_2018.zarr"

ds = xr.open_zarr(path, consolidated=True)

print(ds)

# print max and min longtidue, latitude, and valid time
print("Longitude range:", ds.longitude.min().item(), "to", ds.longitude.max().item())
print("Latitude range:", ds.latitude.min().item(), "to", ds.latitude.max().item())

import pandas as pd

# Convert valid_time from nanoseconds to datetime format
valid_time_min = pd.to_datetime(ds.valid_time.min().item(), unit='ns')
valid_time_max = pd.to_datetime(ds.valid_time.max().item(), unit='ns')

# Print the valid time range in datetime format
print("Valid time range:", valid_time_min, "to", valid_time_max)