#!/usr/bin/env python3
"""
download_forecasts.py
Author: Ozma Houck
Revised 2025-04-28 (download by **year** rather than month)

Download all raw forecast and observation data for the selected
model×region pairs.  No command-line arguments are required; select
targets by (un)commenting entries in CONFIGS.
"""
import os
from datetime import datetime, timedelta
import xarray as xr
import numpy as np
import time

def save_data_locally(ds, full_surface_var_list, full_atm_var_list, lat_values, lon_values,
              time_values, lead_time_hours, output_path):
    
    # Ensure consistent ordering of latitude
    ds = ds.sortby('latitude')

    # Rename dims if necessary
    for v in full_surface_var_list + full_atm_var_list:
        if v not in ds:
            print(f"Variable '{v}' not found in dataset. Skipping...")
            print(f"Available variables: {list(ds.data_vars)}")
            continue
        dims = ds[v].dims
        if 'latitude' not in dims and 'lat' in dims:
            ds = ds.rename({'lat': 'latitude'})
        if 'longitude' not in dims and 'lon' in dims:
            ds = ds.rename({'lon': 'longitude'})

    # Select region, time, and variables
    ds_surface = ds.sel(time=time_values,
                latitude=slice(lat_values.min(), lat_values.max()),
                longitude=slice(lon_values.min(), lon_values.max()))[full_surface_var_list]
    # select atm vars for 1000 hPa level
    level = 1000
    ds_atm = ds.sel(time=time_values,
                latitude=slice(lat_values.min(), lat_values.max()),
                longitude=slice(lon_values.min(), lon_values.max()),
                level = level)[full_atm_var_list].drop_vars('level')
    # rename all atm vars to include the level with hte label "1khPa"
    ds_atm = ds_atm.rename({v: f"{v}_{level}hPa" for v in full_atm_var_list})

    # combine surface and atm datasets
    ds = xr.merge([ds_surface, ds_atm])

    if 'prediction_timedelta' in ds.coords:
        selected_datasets = []
        for lead_time in lead_time_hours:
            selected_ds = ds.sel(prediction_timedelta=np.timedelta64(lead_time, 'h'))
            selected_datasets.append(selected_ds)
        ds = xr.concat(selected_datasets, dim='prediction_timedelta')
    
    # save to netcdf
    ds.to_netcdf(output_path, mode='w')
from datetime import datetime, timedelta
from typing import List

def get_day_list(start_date_str: str, end_date_str: str) -> List[datetime]:
    """
    Return a list of individual days between start_date_str and end_date_str (inclusive).
    """

    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date   = datetime.strptime(end_date_str, "%Y-%m-%d")

    current = start_date                          # first day boundary
    ranges  : List[Tuple[datetime, datetime]] = []

    while current <= end_date:
        next_day  = current + timedelta(days=1)   # first moment of the next day
        day_end   = next_day - timedelta(seconds=1)  # 23:59:59 of 'current'
        dstart    = max(current, start_date)         # clip first tuple if needed
        dend      = min(day_end, end_date)           # clip last tuple if needed
        ranges.append((dstart, dend))
        current = next_day

    return ranges


def get_month_ranges(start_date_str, end_date_str):
    """
    Splits the period between start_date and end_date into a list of (month_start, month_end) tuples.
    """
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    current = start_date.replace(day=1)
    ranges = []
    while current <= end_date:
        # Compute the first day of the next month and then the last day of the current month
        next_month = (current.replace(day=28) + timedelta(days=4)).replace(day=1)
        month_end = next_month - timedelta(days=1)
        mstart = max(current, start_date)
        mend = min(month_end, end_date)
        ranges.append((mstart, mend))
        current = next_month
    return ranges
def get_year_ranges(start_date_str, end_date_str):
    """
    Splits the period between start_date and end_date into a list of (year_start, year_end) tuples.
    """
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    current = start_date.replace(month=1, day=1)
    ranges = []
    while current <= end_date:
        # Compute the first day of the next year and then the last day of the current year
        next_year = current.replace(year=current.year + 1, month=1, day=1)
        year_end = next_year - timedelta(days=1)
        ystart = max(current, start_date)
        yend = min(year_end, end_date)
        ranges.append((ystart, yend))
        current = next_year
    return ranges
def main():
    # possible regions
    # regions = ["india", "usa_south", "amazon", "british_columbia"]
    region = "india"
    # model_names = ["pangu", "ifs"]
    model_name = "ifs"

    if model_name == "pangu":
        forecast_path = "gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr"
        obs_path = "gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr"
    elif model_name == "ifs":
        forecast_path = "gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr"
        obs_path = "gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr" 
    else:
        raise ValueError(f"Unknown model '{model_name}'. Please specify a valid model.")


    # data_dir = os.path.expanduser("/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/data/processed/cleaned_weatherbench_downloads")
    data_dir = os.path.expanduser("/Users/ohouck/test_wb_finetune_data")

    os.makedirs(data_dir, exist_ok=True)

    # # Prepare region and time slices
    # if region == "full_india":
    #     lat_min, lat_max = 8.75, 27.25
    #     lon_min, lon_max = 70.75, 87.25
    # elif region == "north_india":
    #     lat_min, lat_max = 21.25, 27.25
    #     lon_min, lon_max = 70.75, 87.25
    # elif region == "uttar_pradesh":
    #     lat_min, lat_max = 24.25, 26
    #     lon_min, lon_max = 78, 87.25
    # elif region =="pixel":
    #     lat_min, lat_max = 24.25, 24.5
    #     lon_min, lon_max = 78, 78.25
    # elif region == "pakistan":
    #     lat_min, lat_max = 25, 34
    #     lon_min, lon_max = 60, 70
    # elif region == "south_pakistan":
    #     lat_min, lat_max = 24, 27.25
    #     lon_min, lon_max = 62, 70

    full_surface_var_list = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"] 
    full_atm_var_list = ["geopotential", "v_component_of_wind", "u_component_of_wind", "specific_humidity", "temperature"]
    full_lead_time_hours = [24, 48, 72, 96, 120, 144, 168] # times for 1 day, 3 days, and 7 days ahead
    full_train_start = "2018-01-01"  
    full_train_end = "2021-12-31"  # full range for training data
    full_test_start = "2022-01-01"  # full range for test data
    full_test_end = "2022-12-31"  # full range for test data

    # region to download for india
    if region == "india":
        full_lat_values = np.arange(16.75, 27.25, 0.25)
        full_lon_values = np.arange(71.75, 82.25, 0.25)
    elif region == "pakistan":
        # full lat values are by 0.25 degrees
        full_lat_values = np.arange(23.75, 34.25, 0.25) # pakistan/afganistan
        full_lon_values = np.arange(59.75, 70.25, 0.25) 
    elif region == "usa_south":
        full_lat_values = np.arange(29.75, 40.25, 0.25)
        full_lon_values = np.arange(-105.25 + 360, -94.75 + 360, 0.25)
    elif region == "amazon":
        full_lat_values = np.arange(-10.25, 0.25, 0.25)
        full_lon_values = np.arange(-70.25 + 360, -59.75 + 360, 0.25)
    elif region == "british_columbia":
        full_lat_values = np.arange(47.75, 58.25, 0.25) # if rerun should start at 47.75
        full_lon_values = np.arange(-130.25 + 360, -119.75 + 360, 0.25)

    # =========================================================================
    # 0) Download and save the data locally (if needed)
    # =========================================================================

    # Open datasets (supporting Zarr or NetCDF)
    ds_forecast = (
        xr.open_zarr(forecast_path) if forecast_path.endswith('.zarr')
        else xr.open_dataset(forecast_path)
    )
    # ds_obs = (
    #     xr.open_zarr(obs_path) if obs_path.endswith('.zarr')
    #     else xr.open_dataset(obs_path)
    # )

     # ---- Training data ---- 
    train_months = get_month_ranges(full_train_start, full_train_end)
    train_dir = os.path.join(data_dir, f"train_{region}")
    os.makedirs(train_dir, exist_ok=True)
    
    for start_dt, end_dt in train_months:
        date_str = start_dt.strftime("%Y-%m")
        date_folder = os.path.join(train_dir, date_str)
        os.makedirs(date_folder, exist_ok=True)
        # print(f"Saving training data for region {region} for {date_str}...")
        # Create time values for the date (ensuring we include the last day)
        time_values = np.arange(np.datetime64(start_dt.strftime("%Y-%m-%d")),
                                np.datetime64((end_dt + timedelta(days=1)).strftime("%Y-%m-%d")),
                                np.timedelta64(24, 'h'))
        forecast_output_path = os.path.join(date_folder, f"{model_name}_train_forecast_data_{date_str}.nc")
        obs_output_path = os.path.join(date_folder, f"{model_name}_train_obs_data_{date_str}.nc")

        # check if the files already exist
        if not os.path.exists(forecast_output_path):
            start_time = time.time()
            save_data_locally(ds_forecast, full_surface_var_list, full_atm_var_list,
                          full_lat_values, full_lon_values, time_values,
                          full_lead_time_hours, forecast_output_path)
            print("Training Forecast data saved successfully for:", date_str, "in region:", region)
            end_time = time.time()
            print("Time taken to save forecast data:", (end_time - start_time) / 3600, "hours")
        # if not os.path.exists(obs_output_path):
        #     start_time = time.time()
        #     save_data_locally(ds_obs, full_surface_var_list, full_atm_var_list,
        #                     full_lat_values, full_lon_values, time_values,
        #                     full_lead_time_hours, obs_output_path)
        #     end_time = time.time()
        #     print("Training Obs data saved successfully for:", date_str, "in region:", region)
        #     print("Time taken to save obs data:", (end_time - start_time) / 3600, "hours")
    
    # ---- Test data ----
    test_months = get_month_ranges(full_test_start, full_test_end)
    test_dir = os.path.join(data_dir, f"test_{region}")
    os.makedirs(test_dir, exist_ok=True)
    
    for start_dt, end_dt in test_months:
        date_str = start_dt.strftime("%Y-%m")
        date_folder = os.path.join(test_dir, date_str)
        os.makedirs(date_folder, exist_ok=True)
        # print(f"Saving test data for region {region} for {date_str}...")
        time_values = np.arange(np.datetime64(start_dt.strftime("%Y-%m-%d")),
                                np.datetime64((end_dt + timedelta(days=1)).strftime("%Y-%m-%d")),
                                np.timedelta64(24, 'h'))
        forecast_output_path = os.path.join(date_folder, f"{model_name}_test_forecast_data_{date_str}.nc")
        obs_output_path = os.path.join(date_folder, f"{model_name}_test_obs_data_{date_str}.nc")

        # check if the files already exist
        if not os.path.exists(forecast_output_path):
            time_start = time.time()
            save_data_locally(ds_forecast, full_surface_var_list, full_atm_var_list,
                            full_lat_values, full_lon_values, time_values,
                            full_lead_time_hours, forecast_output_path)
            time_end = time.time()
            print("Testing Forecast data saved successfully for date:", date_str, "in region:", region)
            # print time in hours
            print("Time taken to save forecast data:", (time_end - time_start) / 3600, "hours")
        
        # if not os.path.exists(obs_output_path):
        #     time_start = time.time()
        #     save_data_locally(ds_obs, full_surface_var_list, full_atm_var_list,
        #                     full_lat_values, full_lon_values, time_values,
        #                     full_lead_time_hours, obs_output_path)
        #     time_end = time.time()
        #     print("Testing Obs data saved successfully for date:", date_str, "in region:", region)
        #     print("Time taken to save obs data:", (time_end - time_start) / 3600, "hours")
    
if __name__ == "__main__":
    main()
