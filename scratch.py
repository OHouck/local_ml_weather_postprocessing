
import xarray as xr
import zarr
import numpy as np
path = "/Users/ohouck/globus/forecast_data/processed/finetuning_output/pangu/usa_south/train_2m_temperature_10m_wind_speed_test_2m_temperature_10m_wind_speed_dim6x6_leadtime_24h_train2018-01-01-2021-12-31_test2022-01-01-2022-12-31_mlp_joint_temp_wind_loss.zarr"

#print versions of xarray and zarr
print("xarray version:", xr.__version__)
print("zarr version:", zarr.__version__)

def calculate_rmse(predictions, ground_truth):
    """Calculate RMSE between predictions and ground truth."""
    return float(np.sqrt(((predictions- ground_truth) ** 2).mean()))

ds = xr.open_zarr(path, chunks=None)
training_time = ds.training_time_minutes
training_time = round(float(training_time), 2)
print(ds)
print("Training time (minutes):", training_time)




