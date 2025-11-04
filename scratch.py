
import xarray as xr
import zarr
import numpy as np
path = "/Users/ohouck/globus/forecast_data/processed/finetuning_output/pangu/india/train_2m_temperature_test_2m_temperature_dim6x6_leadtime_120h_train2018-01-01-2021-12-31_test2022-01-01-2022-12-31_mlp.zarr"


#print versions of xarray and zarr
print("xarray version:", xr.__version__)
print("zarr version:", zarr.__version__)

def calculate_rmse(predictions, ground_truth):
    """Calculate RMSE between predictions and ground truth."""
    return float(np.sqrt(((predictions- ground_truth) ** 2).mean()))

ds = xr.open_dataset(path, chunks=None)

preds = ds['2m_temperature_corrected_lt120h'].values
truth = ds['2m_temperature_ground_truth_lt120h'].values

rmse = calculate_rmse(preds, truth)

print("RMSE:", rmse)

rmse_og = calculate_rmse(ds['2m_temperature_original_lt120h'].values, truth)
print("RMSE Original:", rmse_og)

print("RMSE Improvement (%):", (rmse_og - rmse) / rmse_og * 100 )

