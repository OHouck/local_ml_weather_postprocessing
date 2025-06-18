import xarray as xr
import os


# Test reading a small subset
path = "/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/data/raw/pangu2018_raw_data"
ds = xr.open_dataset(os.path.join(path, "targets_2018-01-01_2018-01-01.nc"))

print(ds)

# print all level coordinates
print(ds.level)
print(len(ds.level))
