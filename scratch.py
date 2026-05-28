import numpy as np
import xarray as xr

path = "/Users/ohouck/Desktop/heat_wave_training_subset.nc"

ds = xr.open_dataset(path)

print(ds)


