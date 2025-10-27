import zarr
import xarray as xr
import numpy as np

# print zarr and xarray versions
print(f"zarr version: {zarr.__version__}")
print(f"xarray version: {xr.__version__}")

zarr_path = "/Users/ohouck/globus/forecast_data/processed/finetuning_output/pangu/india/train_2m_temperature_test_2m_temperature_dim6x6_leadtime_120h_growing_season_train2018-01-01-2021-12-31_test2022-01-01-2022-12-31_mlp.zarr"

ds = xr.open_zarr(zarr_path, chunks=None)
print(ds)
