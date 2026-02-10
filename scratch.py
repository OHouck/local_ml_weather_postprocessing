
import xarray as xr
import zarr
import numpy as np
path = "/Users/ohouck/globus/forecast_data/raw/ndfd_data/wspd/2025/01/YCUZ88_KWBN_202501101047_texas.nc"

#print versions of xarray and zarr
print("xarray version:", xr.__version__)
print("zarr version:", zarr.__version__)

def calculate_rmse(predictions, ground_truth):
    """Calculate RMSE between predictions and ground truth."""
    return float(np.sqrt(((predictions- ground_truth) ** 2).mean()))

ds = xr.open_dataset(path)

# print unique step and time coordinates
# convert set to hours for easier interpretation
print(ds)

print(ds["latitude"].values.min(), ds["latitude"].values.max())
print(ds["longitude"].values.min(), ds["longitude"].values.max())



