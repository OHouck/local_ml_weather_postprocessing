
import xarray as xr
import zarr
import numpy as np
path = "/Users/ohouck/globus/forecast_data/processed/finetuning_output/pangu/hilly/train_10m_wind_speed_test_10m_wind_speed_dim2x2_leadtime_24_120_216h_train2018-01-01-2021-12-31_test2022-01-01-2022-12-31_mlp_hilly_bs37.zarr"

#print versions of xarray and zarr
print("xarray version:", xr.__version__)
print("zarr version:", zarr.__version__)

def calculate_rmse(predictions, ground_truth):
    """Calculate RMSE between predictions and ground truth."""
    return float(np.sqrt(((predictions- ground_truth) ** 2).mean()))

ds = xr.open_dataset(path, chunks=None)
print(ds)


