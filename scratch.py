import xarray as xr

path = "/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/data/fine_tuning_output/pangu/amazon/train_2m_temperature_test_2m_temperature_dim2x2_leadtime_24_120_240h_train2018-01-01-2021-12-31_test2022-01-01-2022-12-31_mlp.zarr"

ds = xr.open_zarr(path)

print(ds)

# print min and max longitude and latitude values
print("Longitude range:", ds.longitude.min().item(), "to", ds.longitude.max().item())
print("Latitude range:", ds.latitude.min().item(), "to", ds.latitude.max().item())
