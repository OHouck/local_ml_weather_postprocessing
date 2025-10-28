
import xarray as xr
import zarr

path = "/Users/ohouck/globus/forecast_data/processed/finetuning_output/pangu/india/train_10m_wind_speed_test_10m_wind_speed_dim6x6_leadtime_24_120_216h_train2018-01-01-2021-12-31_test2022-01-01-2022-12-31_mlp.zarr"

#print versions of xarray and zarr
print("xarray version:", xr.__version__)
print("zarr version:", zarr.__version__)

ds = xr.open_zarr(path)
print(ds)