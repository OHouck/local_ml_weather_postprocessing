
import xarray as xr
import zarr
import numpy as np
path = "/Users/ohouck/Downloads/OneDrive_1_1-28-2026/data_2023.nc"

#print versions of xarray and zarr
print("xarray version:", xr.__version__)
print("zarr version:", zarr.__version__)

def calculate_rmse(predictions, ground_truth):
    """Calculate RMSE between predictions and ground truth."""
    return float(np.sqrt(((predictions- ground_truth) ** 2).mean()))

ds = xr.open_dataset(path, chunks=None)
print(ds)


