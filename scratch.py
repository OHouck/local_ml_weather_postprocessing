
import xarray as xr
import zarr
import numpy as np
path = "/Users/ohouck/globus/forecast_data/raw/pangu/pangu_india_2022.zarr"

#print versions of xarray and zarr
print("xarray version:", xr.__version__)
print("zarr version:", zarr.__version__)

def calculate_rmse(predictions, ground_truth):
    """Calculate RMSE between predictions and ground truth."""
    return float(np.sqrt(((predictions- ground_truth) ** 2).mean()))

ds = xr.open_dataset(path, chunks=None)
print(ds)


