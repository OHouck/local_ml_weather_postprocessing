import xarray as xr
import numpy as np

path1 = "/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/data/fine_tuning_output/pangu/amazon/train_10m_wind_speed_test_10m_wind_speed_dim10x10_leadtime_24_120_240h_train2018-01-01-2021-12-31_test2022-01-01-2022-12-31_mlp.zarr"
path2 = "/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/data/fine_tuning_output/pangu/amazon/train_10m_wind_speed_test_10m_wind_speed_dim10x10_leadtime_240h_train2018-01-01-2021-12-31_test2022-01-01-2022-12-31_mlp.zarr"

ds = xr.open_zarr(path1, consolidated=True)
print(ds)
ds = xr.open_zarr(path2, consolidated=True)
print(ds)
