import xarray as xr
import os


# Test reading a small subset
path = "/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/data/processed/cleaned_weatherbench_downloads/train_british_columbia/2018-06/pangu_train_forecast_data_2018-06.nc"
ds = xr.open_dataset(os.path.join(path))
print(ds)

# pritn min and max latitude
print("Min latitude:", ds.latitude.min().item())
print("Max latitude:", ds.latitude.max().item())
# Print min and max longitude
print("Min longitude:", ds.longitude.min().item())
print("Max longitude:", ds.longitude.max().item())

