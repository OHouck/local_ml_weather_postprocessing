import xarray as xr
import time
import os
import numpy as np

import xarray as xr

path = "/Users/ohouck/globus/forecast_data/aifs_2024.zarr"


ds = xr.open_zarr(path, consolidated=True)
print(ds)


