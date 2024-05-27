# Author: Ozzy Houck
# Date Created 5/27/2024

# Purpose: Compare t2m measurements between ERA5, FourCastNet, and PanguWeather


import xarray as xr
import matplotlib.pyplot as plt

# Load forecasts
output_path = "/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag"
date = "2024-04-01" # eventually should be part of config file
bbox = [14, 44, 24, 48]

pangu_path = f"{output_path}/pangu_{date}.grib"
pangu = xr.open_dataset(pangu_path, engine='cfgrib').sel(longitude=slice(bbox[0], bbox[2]), latitude=slice(bbox[3], bbox[1]))

print(pangu.coords)

# convert step coordinate to datetime from nano seconds
pangu['time2'] = pangu['time2'].astype('datetime64[ns]')
pangu['step2'] = pangu['step2'].astype('timedelta64[s]')
pangu['time'] = pangu['time'] + pangu['step']

# check if the time coordinate is in the dataset
if 'time' not in pangu.coords:
    print('Time coordinate not found in dataset')






exit()


# Find the nearest pressure level to 1000 hPa which is about the surface level
nearest_level = pangu.isobaricInhPa.sel(isobaricInhPa=1000, method='nearest')

# Extract the temperature at the nearest pressure level and convert to Celsius
pangu_temp = pangu['t'].sel(isobaricInhPa=nearest_level) - 273.15

# average temperatrues also across the latitude and longitude
pangu_temp_avg = pangu_temp.mean(dim='latitude').mean(dim='longitude')
pangu_temp_std = pangu_temp.std(dim = 'latitude').std(dim = 'longitude')

# plot average t2m across the bbox over time
plt.plot(pangu_temp_avg, label='PANGU Weather')
plt.xlabel('Time')
plt.ylabel('Temperature (C)')
plt.title('ERA5 Average Temperature')
plt.legend()
plt.show()

