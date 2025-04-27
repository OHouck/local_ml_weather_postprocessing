#!/usr/bin/env python3
"""
download_data.py
Author: Ozzy Houck
Date: 2025-04-27

Self-contained configuration: uncomment desired model, region, and subregion.
Generates monthly forecast and observation files.
"""
# ------------- CONFIGURATION -------------
# Uncomment the MODEL you want:
# MODEL_NAME = 'pangu'
# MODEL_NAME = 'ifs'
MODEL_NAME = 'pangu'

# Uncomment the REGION:
# REGION = 'amazon'
# REGION = 'usa_south'
# REGION = 'british_columbia'
REGION = 'india'

# Paths (modify if needed)
if MODEL_NAME == 'pangu':
    FORECAST_PATH     = "gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr"
    OBS_PATH          = "gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr"
elif MODEL_NAME == 'ifs':
    FORECAST_PATH     = "gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr"
    OBS_PATH          = "gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr"

DATA_DIR          = '~/wb_finetune_data'
FULL_TRAIN_START  = '2018-01-01'
FULL_TRAIN_END    = '2021-12-31'

FULL_TEST_START   = '2022-01-01'
FULL_TEST_END     = '2022-12-31'

import os
import numpy as np
import xarray as xr
from datetime import datetime, timedelta

# region bounds and offsets
REGION_BOUNDS = {
    'india':            (17, 27,   72,  82),
    'usa_south':        (30, 40, 360-105, 360-95),
    'amazon':           (-10, 0, 360-70, 360-60),
    'british_columbia': (48, 58, 360-130,360-120)
}
SUB_OFFSETS = {'2x2': 4, '4x4': 3, '6x6': 2, '8x8': 1, '10x10': 0}
SURF_VARS   = ['2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind']
ATM_VARS    = ['geopotential', 'v_component_of_wind', 'u_component_of_wind',
               'specific_humidity', 'temperature']
LEAD_TIMES  = [24, 72, 168]


def get_month_ranges(start_str, end_str):
    start = datetime.strptime(start_str, '%Y-%m-%d')
    end   = datetime.strptime(end_str, '%Y-%m-%d')
    cur   = start.replace(day=1)
    ranges = []
    while cur <= end:
        nxt  = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)
        last = nxt - timedelta(days=1)
        ranges.append((max(cur, start), min(last, end)))
        cur = nxt
    return ranges


def save_month(path, vars_s, vars_a, lats, lons, times, leads, out_path):
    ds = xr.open_zarr(path) if path.endswith('.zarr') else xr.open_dataset(path)
    ds = ds.sortby('latitude')
    ds = ds.rename({'lat':'latitude','lon':'longitude'})
    ds_s = ds.sel(time=times, latitude=lats, longitude=lons)[vars_s]
    ds_a = ds.sel(time=times, latitude=lats, longitude=lons, level=1000)[vars_a]
    ds_a = ds_a.drop_vars('level').rename({v:f"{v}_1000hPa" for v in vars_a})
    ds_all = xr.merge([ds_s, ds_a])
    if 'prediction_timedelta' in ds_all.coords:
        parts = [ds_all.sel(prediction_timedelta=np.timedelta64(lt,'h')) for lt in leads]
        ds_all = xr.concat(parts, dim='prediction_timedelta')
    ds_all.to_netcdf(out_path, mode='w')


def main():
    data_root = os.path.expanduser(DATA_DIR)
    lat0, lat1, lon0, lon1 = REGION_BOUNDS[REGION]
    off = SUB_OFFSETS[SUBREGION]
    lats = np.arange(lat0+off, lat1-off+0.25, 0.25)
    lons = np.arange(lon0+off, lon1-off+0.25, 0.25)

    # TRAIN months
    for start, end in get_month_ranges(FULL_TRAIN_START, FULL_TRAIN_END):
        ym = start.strftime('%Y-%m')
        out_dir = os.path.join(data_root, f'train_{REGION}', ym)
        os.makedirs(out_dir, exist_ok=True)
        fc_file = os.path.join(out_dir, f"{MODEL_NAME}_train_forecast_{ym}.nc")
        ob_file = os.path.join(out_dir, f"{MODEL_NAME}_train_obs_{ym}.nc")
        if not (os.path.exists(fc_file) and os.path.exists(ob_file)):
            times = np.arange(np.datetime64(start),
                              np.datetime64(end + timedelta(days=1)),
                              np.timedelta64(24,'h'))
            save_month(FORECAST_PATH, SURF_VARS, ATM_VARS, lats, lons,
                       times, LEAD_TIMES, fc_file)
            save_month(OBS_PATH,      SURF_VARS, ATM_VARS, lats, lons,
                       times, LEAD_TIMES, ob_file)

    # TEST months
    for start, end in get_month_ranges(FULL_TEST_START, FULL_TEST_END):
        ym = start.strftime('%Y-%m')
        out_dir = os.path.join(data_root, f'test_{REGION}', ym)
        os.makedirs(out_dir, exist_ok=True)
        fc_file = os.path.join(out_dir, f"{MODEL_NAME}_test_forecast_{ym}.nc")
        ob_file = os.path.join(out_dir, f"{MODEL_NAME}_test_obs_{ym}.nc")
        if not (os.path.exists(fc_file) and os.path.exists(ob_file)):
            times = np.arange(np.datetime64(start),
                              np.datetime64(end + timedelta(days=1)),
                              np.timedelta64(24,'h'))
            save_month(FORECAST_PATH, SURF_VARS, ATM_VARS, lats, lons,
                       times, LEAD_TIMES, fc_file)
            save_month(OBS_PATH,      SURF_VARS, ATM_VARS, lats, lons,
                       times, LEAD_TIMES, ob_file)

if __name__ == '__main__':
    main()