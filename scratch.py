import xarray as xr
import time
import os
import numpy as np
import pickle
import torch

import xarray as xr

import matplotlib.pyplot as plt
import torch
from aurora import Batch

path = "/Users/ohouck/globus/forecast_data/raw/era5_static.nc"

ds = xr.open_dataset(path)
print(ds)




# # Add Batch to safe globals
# torch.serialization.add_safe_globals([Batch])
# preds = torch.load(path, weights_only=False)


# temp_2m = preds[0].surf_vars["2t"]

# print("max and min 2t:")
# print(torch.max(temp_2m) - 273, torch.min(temp_2m) - 273)

# surf_vars = list(preds[0].surf_vars.keys())
# for var in surf_vars:
#     shape = preds[0].surf_vars[var].shape
#     print(f"{var}: {shape}")

# atmos_vars = list(preds[0].atmos_vars.keys())
# for var in atmos_vars:
#     shape = preds[0].atmos_vars[var].shape
#     print(f"{var}: {shape}")

# static_vars = list(preds[0].static_vars.keys())
# for var in static_vars:
#     shape = preds[0].static_vars[var].shape
#     print(f"{var}: {shape}")

# exit()

# fig, ax = plt.subplots(2, 2, figsize=(12, 6.5))

# for i in range(ax.shape[0]):
#     pred = preds[i]

#     ax[i, 0].imshow(pred.surf_vars["2t"][0, 0].numpy() - 273.15, vmin=-50, vmax=50)
#     ax[i, 0].set_ylabel(str(pred.metadata.time[0]))
#     if i == 0:
#         ax[i, 0].set_title("Aurora Prediction")
#     ax[i, 0].set_xticks([])
#     ax[i, 0].set_yticks([])

# plt.tight_layout()
# plt.show()




