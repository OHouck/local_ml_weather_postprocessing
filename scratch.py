import xarray as xr
import os


# Test reading a small subset
path = "/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/data/processed/cleaned_weatherbench_downloads/test_arid/pangu/pangu_test_forecast_data_arid_2x2_patch_26.nc"
ds = xr.open_dataset(os.path.join(path))
print(ds)

print(ds.prediction_timedelta)

