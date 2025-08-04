import xarray as xr
import numpy as np

# path = "/Users/ohouck/Documents/aifs2_init_2024073100.nc"
# path = "/Users/ohouck/Documents/aifs1_init_2024071100.nc"
path = "/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/data/fine_tuning_output/pangu/amazon/train_2m_temperature_test_2m_temperature_dim8x8_leadtime_168h_train2018-01-01-2021-12-31_test2022-01-01-2022-12-31_mlp.zarr"

ds = xr.open_zarr(path, consolidated=True)

print(ds)
