#!/usr/bin/env python3
import glob
import os
import xarray as xr

# root folder where all your monthly .nc files live
root_dir = "/Users/ohouck/wb_finetune_data"

path = "/Volumes/wd_external_hd/weatherbench/test_global/2022-03/pangu_test_forecast_data_2022-03.nc"

ds = xr.open_dataset(path)

print(ds["2m_temperature"].min().item)