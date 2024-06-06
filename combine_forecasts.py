# Author: Ozzy Houck
# Date Created: 6/6/2024

# Purpose: Combine forecasts and weather data pulled from download_forecasts.py
# into a single file that can be used in compare_forecasts.py

import xarray as xr
import numpy as np

# Config 
forecast_path = "/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/forecasts"
fig_path = "/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/figures"
date = "2024-04-01" # eventually should be part of config file

#-------------------------------------------
# Helper functions to load in and process forecasts
#-------------------------------------------

# Load ERA5 data for the same date
def load_ERA5(date, path):
    era5_path = f"{path}/ERA5_{date}.grib"
    era5 = xr.open_dataset(era5_path, engine='cfgrib')

    # time and valid_time are the same so drop valid_time along with other unnecessary variables
    era5 = era5.drop_vars(['number', 'surface', 'valid_time', 'step']) 

    # convert temperature to Celsius
    era5['t2m'] = era5['t2m'] - 273.15

    # set the coordinates
    era5 = era5.set_coords(['latitude', 'longitude', 'time'])

    return era5

def load_ifs_forecast(date, path):
    ifs_path = f"{path}/ifs_forecast_{date}.grib"
    ifs = xr.open_dataset(ifs_path, engine='cfgrib')

    # longitude range is 0-360 so need to shift to -180 to 180
    ifs = ifs.assign_coords(longitude=(ifs.longitude + 180) % 360 - 180)

    # to match number of steps in ERA5
    ifs = ifs.isel(step=slice(0, 40))

    # drop unnecessary variables
    ifs = ifs.drop_vars(['surface', 'heightAboveGround', 'time', 'step'])

    ifs = ifs.swap_dims({'step': 'time'})

    # rename 'valid_time' dimension to 'time'
    ifs = ifs.rename({'valid_time': 'time'})

    # set the coordinates
    ifs = ifs.set_coords(['latitude', 'longitude', 'time'])

    # convert temperature to Celsius
    ifs['t2m'] = ifs['t2m'] - 273.15

    return ifs

# Load pangu forecast
def load_pangu(date, path):
    pangu_path = f"{path}/pangu_{date}.grib"

    # filter by paramId 167 which is 2m temperature
    pangu = xr.open_dataset(pangu_path, engine='cfgrib', backend_kwargs={'filter_by_keys': {'paramId': 167}})


    # Shift the data along the longitude dimension (not totally sure why I need to do this for pangu but not for IFS)
    pangu = pangu.roll(longitude=pangu.longitude.size // 2, roll_coords=True)

    # Adjust the longitude values to the -180 to 180 range
    pangu['longitude'] = (pangu['longitude'] + 180) % 360 - 180

    pangu = pangu.isel(step=slice(0, 40))

    # drop unnecessary variables
    pangu = pangu.drop_vars(['heightAboveGround', 'time', 'step'])

    # rename 'valid_time' dimension to 'time'
    pangu = pangu.rename({'valid_time': 'time'})

    # rename 'step' dimension to 'time' (this feels weird but it gets the correct dimension name for the merge later on)
    pangu = pangu.swap_dims({'step': 'time'})

    # set the coordinates
    pangu = pangu.set_coords(['latitude', 'longitude', 'time'])

    # convert to celsius
    pangu['t2m'] = pangu['t2m'] - 273.15

    return pangu

# Load FourCastNet data for the same date
def load_FourCastNet(date, path):
    fourcastnet_path = f"{path}/fourcastnet_{date}.grib"

    # filter by paramId 167 which is 2m temperature
    fourcastnet = xr.open_dataset(fourcastnet_path, engine='cfgrib', backend_kwargs={'filter_by_keys': {'paramId': 167}})

    # Shift the data along the longitude dimension (not totally sure why I need to do this for pangu but not for IFS)
    fourcastnet= fourcastnet.roll(longitude=fourcastnet.longitude.size // 2, roll_coords=True)

    # Adjust the longitude values to the -180 to 180 range
    fourcastnet['longitude'] = (fourcastnet['longitude'] + 180) % 360 - 180
    
    # fourcastnet forecast is 41 steps of 6 hours so need to drop the last step to match ERA5 pull
    fourcastnet= fourcastnet.isel(step=slice(0, 40))

    # drop unnecessary variables
    fourcastnet = fourcastnet.drop_vars(['heightAboveGround', 'time', 'step'])

    fourcastnet.set_coords(['latitude', 'longitude', 'valid_time'])

    # rename valid_time coordinate to time
    fourcastnet = fourcastnet.rename_vars({'valid_time': 'time'})

    # swap 'step' dimension to 'time' (this feels weird but it gets the correct dimension name for the merge later on)
    fourcastnet = fourcastnet.swap_dims({'step': 'time'})

    # XX there are some weird na values in t2m 

    return fourcastnet

#-------------------------------------------
# Load the data
#-------------------------------------------
era5 = load_ERA5(date, forecast_path)
era5_t2m = era5['t2m']

ifs = load_ifs_forecast(date, forecast_path)
ifs_t2m = ifs['t2m']

fourcastnet = load_FourCastNet(date, forecast_path)
fourcastnet_t2m = fourcastnet['t2m']

pangu = load_pangu(date, forecast_path)
pangu_t2m = pangu['t2m']

# Check if the 'time' coordinates are equal
if not np.array_equal(ifs_t2m.time.values, era5_t2m.time.values):
    print("Time dimensions do not match: IFS and ERA5")
    exit()
if not np.array_equal(pangu_t2m.time.values, era5_t2m.time.values):
    print("Time dimensions do not match: PanguWeather and ERA5")
    exit()
if not np.array_equal(fourcastnet_t2m.time.values, era5_t2m.time.values):
    print("Time dimensions do not match: FourCastNet and ERA5")
    exit()

# rename t2m for each to be able to merge
era5_t2m = era5_t2m.rename('era5_t2m')
ifs_t2m = ifs_t2m.rename('ifs_t2m')
pangu_t2m = pangu_t2m.rename('pangu_t2m')
fourcastnet_t2m = fourcastnet_t2m.rename('fourcastnet_t2m')

#-------------------------------------------
# Merge the data
#-------------------------------------------

# merge on lat, lon, and valid_time
combined = xr.merge([pangu_t2m, era5_t2m], join='inner')
combined = xr.merge([combined, ifs_t2m], join='inner')
combined = xr.merge([combined, fourcastnet_t2m], join='inner')

# save the combined data to a grib file
combined.to_netcdf(f"{forecast_path}/combined_forecasts_{date}.nc")