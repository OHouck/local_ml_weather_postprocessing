#!/usr/bin/env python3
import glob
import os
import xarray as xr

# root folder where all your monthly .nc files live
root_dir = "/Users/ohouck/wb_finetune_data"

path = "/Users/ohouck/wb_finetune_data/test_british_columbia/2022-06/pangu_test_obs_data_2022-06.nc"
path2 = "/Users/ohouck/wb_finetune_data/test_british_columbia/2022-06/pangu_test_forecast_data_2022-06.nc"

# open the dataset
ds = xr.open_dataset(path)
ds2 = xr.open_dataset(path2)

print(ds)

print(ds2)

max_lat = ds.latitude.max().values
min_lat = ds.latitude.min().values

max_lon = ds.longitude.max().values
min_lon = ds.longitude.min().values

# you can tweak the glob to hit both train_* and test_* if you like
pattern = os.path.join(root_dir, "test_british_columbia", "**", "pangu*.nc")

file_list = glob.glob(pattern, recursive=True)

for fn in file_list:
    ds = xr.open_dataset(fn, decode_timedelta=True)

    is_ds_sorted_lat = (ds.latitude.values == sorted(ds.latitude.values)).all()
    is_ds_sorted_lon = (ds.longitude.values == sorted(ds.longitude.values)).all()

    if not is_ds_sorted_lat:
        print(f"ds sorted by latitude: {is_ds_sorted_lat}")
        print(f"file path: {fn}")   
    if not is_ds_sorted_lon:
        print(f"ds sorted by longitude: {is_ds_sorted_lon}")
        print(f"file path: {fn}")   
    
    # check that matches max and min lat and lon
    if ds.latitude.max().values != max_lat:
        print(f"ds max latitude: {ds.latitude.max()} != {max_lat}")
        print(f"file path: {fn}")   
    if ds.latitude.min().values != min_lat:
        print(f"ds min latitude: {ds.latitude.min()} != {min_lat}")
        print(f"file path: {fn}")   
    if ds.longitude.max().values != max_lon:
        print(f"ds max longitude: {ds.longitude.max()} != {max_lon}")
        print(f"file path: {fn}")   
    if ds.longitude.min() != min_lon:
        print(f"ds min longitude: {ds.longitude.min()} != {min_lon}")
        print(f"file path: {fn}")   

    # # sort by latitude (and then longitude)
    # ds_sorted = ds.sortby(['latitude', 'longitude'])
    # # overwrite the original file with the sorted version
    # ds_sorted.to_netcdf(fn,mode = "w")
    # ds.close()
    # print(f"✅ Sorted & saved: {fn}")
