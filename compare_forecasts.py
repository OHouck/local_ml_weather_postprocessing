# Author: Ozzy Houck
# Date Created 5/27/2024

# Purpose: Compare t2m measurements between ERA5, FourCastNet, and PanguWeather

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Config 
forecast_path = "/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/forecasts"
fig_path = "/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/figures"
date = "2024-04-01" # eventually should be part of config file

# whole world
lon_min, lon_max = -180, 180 
lat_min, lat_max = -90, 90 

# Amerminia 
# lon_min, lon_max = 14, 24
# lat_min, lat_max = 44, 48

# Midwest
# lon_min, lon_max = -103, -85
# lat_min, lat_max = 40, 45

bbox = [lon_min, lat_min, lon_max, lat_max]

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_era5_first_timestep(date, bbox, path, fig_path):
    era5_path = f"{path}/ERA5_{date}.grib"
    era5 = xr.open_dataset(era5_path, engine='cfgrib')
    era5 = era5.sel(longitude=slice(bbox[0], bbox[2]), latitude=slice(bbox[3], bbox[1]))

    # Drop unnecessary variables
    era5 = era5.drop_vars(['number', 'surface', 'valid_time', 'step'])

    # Convert temperature to Celsius
    era5['t2m'] = era5['t2m'] - 273.15

    # Set the coordinates
    era5 = era5.set_coords(['latitude', 'longitude', 'time'])

    # Select the first time step
    era5_first_timestep = era5.isel(time=0)

    # Create the plot
    plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    # Plot the data
    plt.contourf(era5_first_timestep['longitude'], era5_first_timestep['latitude'], era5_first_timestep['t2m'], transform=ccrs.PlateCarree())
    plt.colorbar(label='Temperature (C)')
    plt.title('ERA5 Forecast - First Timestep')
    plt.savefig(f"{fig_path}/era5_forecast_first_timestep.png")
    plt.clf()

# plot_era5_first_timestep(date, bbox, forecast_path, fig_path)

#-------------------------------------------
# Helper functions to load in and process forecasts
#-------------------------------------------


# Load ERA5 data for the same date
def load_ERA5(date, bbox, path):
    era5_path = f"{path}/ERA5_{date}.grib"
    era5 = xr.open_dataset(era5_path, engine='cfgrib')
    # era5 = era5.sel(longitude=slice(bbox[0], bbox[2]), latitude=slice(bbox[3], bbox[1]))

    # time and valid_time are the same so drop valid_time along with other unnecessary variables
    era5 = era5.drop_vars(['number', 'surface', 'valid_time', 'step']) 

    # convert temperature to Celsius
    era5['t2m'] = era5['t2m'] - 273.15

    # set the coordinates
    era5 = era5.set_coords(['latitude', 'longitude', 'time'])

    return era5

def load_ifs_forecast(date, bbox, path):
    ifs_path = f"{path}/ifs_forecast_{date}.grib"
    ifs = xr.open_dataset(ifs_path, engine='cfgrib')
    # longitude range is 0-360 so need to shift to -180 to 180
    ifs = ifs.assign_coords(longitude=(ifs.longitude + 180) % 360 - 180)

    # filter by bbox
    ifs = ifs.where((ifs.longitude >= bbox[0]) & (ifs.longitude <= bbox[2]), drop=True)
    ifs = ifs.where((ifs.latitude >= bbox[1]) & (ifs.latitude <= bbox[3]), drop=True)

    ifs = ifs.isel(step=slice(0, 40))

    # drop unnecessary variables
    ifs = ifs.drop_vars(['surface', 'heightAboveGround', 'time', 'step'])

    # rename 'valid_time' dimension to 'time'
    ifs = ifs.rename({'valid_time': 'time'})

    ifs = ifs.rename({'step': 'time'})

    # set the coordinates
    ifs = ifs.set_coords(['latitude', 'longitude', 'time'])

    # convert temperature to Celsius
    ifs['t2m'] = ifs['t2m'] - 273.15

    return ifs

# Load pangu forecast
def load_pangu(date, bbox, path):
    pangu_path = f"{path}/pangu_{date}.grib"

    # filter by paramId 167 which is 2m temperature
    pangu = xr.open_dataset(pangu_path, engine='cfgrib', backend_kwargs={'filter_by_keys': {'paramId': 167}})

    # longitude range is 0-360 so need to shift to -180 to 180
    pangu = pangu.assign_coords(longitude=(pangu.longitude + 180) % 360 - 180) 

    # filter by bbox
    pangu = pangu.where((pangu.longitude >= bbox[0]) & (pangu.longitude <= bbox[2]), drop=True)
    pangu = pangu.where((pangu.latitude >= bbox[1]) & (pangu.latitude <= bbox[3]), drop=True)

    pangu = pangu.isel(step=slice(0, 40))

    # drop unnecessary variables
    pangu = pangu.drop_vars(['heightAboveGround', 'time', 'step'])

    # rename 'valid_time' dimension to 'time'
    pangu = pangu.rename({'valid_time': 'time'})

    # rename 'step' dimension to 'time' (this feels weird but it gets the correct dimension name for the merge later on)
    pangu = pangu.rename({'step': 'time'})

    # set the coordinates
    pangu = pangu.set_coords(['latitude', 'longitude', 'time'])

    # convert to celsius
    pangu['t2m'] = pangu['t2m'] - 273.15

    return pangu

# Load FourCastNet data for the same date
def load_FourCastNet(date, bbox, path):
    fourcastnet_path = f"{path}/fourcastnet_{date}.grib"

    # filter by paramId 167 which is 2m temperature
    fourcastnet = xr.open_dataset(fourcastnet_path, engine='cfgrib', backend_kwargs={'filter_by_keys': {'paramId': 167}})
    fourcastnet = fourcastnet.sel(longitude=slice(bbox[0], bbox[2]), latitude=slice(bbox[3], bbox[1]))

    # fourcastnet forecast is 41 steps of 6 hours so need to drop the last step to match ERA5 pull
    fourcastnet= fourcastnet.isel(step=slice(0, 40))

    # drop unnecessary variables
    fourcastnet = fourcastnet.drop_vars(['heightAboveGround'])

    fourcastnet.set_coords(['latitude', 'longitude', 'valid_time', 'step'])
    # drop time dimension
    fourcastnet = fourcastnet.drop_vars('time')

    # rename valid_time coordinate to time
    fourcastnet = fourcastnet.rename_vars({'valid_time': 'time'})

    # change step variable to be 0- number of steps
    fourcastnet['step'] = range(0, len(fourcastnet['step']))

    # convert to celsius
    fourcastnet['t2m'] = fourcastnet['t2m'] - 273.15

    # print summary stats for t2m
    print(fourcastnet['t2m'].values.mean())
    print(fourcastnet['t2m'].values.std())
    print(fourcastnet['t2m'].values.min())
    print(fourcastnet['t2m'].values.max())

    # These values are way off from what they should be not going to use this data 
    # for now.
    return fourcastnet
#-------------------------------------------
# Load the data
#-------------------------------------------

era5 = load_ERA5(date, bbox, forecast_path)
era5_t2m = era5['t2m']

ifs = load_ifs_forecast(date, bbox, forecast_path)
ifs_t2m = ifs['t2m']

# foucastnet = load_FourCastNet(date, bbox, forecast_path)
# fourcastnet_t2m = foucastnet['t2m']
pangu = load_pangu(date, bbox, forecast_path)
pangu_t2m = pangu['t2m']

# Check if the 'time' coordinates are equal
if not np.array_equal(ifs_t2m.time.values, era5_t2m.time.values):
    print("Time dimensions do not match")
    exit()
if not np.array_equal(pangu_t2m.time.values, era5_t2m.time.values):
    print("Time dimensions do not match")
    exit()
# if not np.array_equal(fourcastnet_t2m.time.values, era5_t2m.time.values):
#     print("Time dimensions do not match")
#     exit()

# rename t2m for each to be able to merge
era5_t2m = era5_t2m.rename('era5_t2m')
ifs_t2m = ifs_t2m.rename('ifs_t2m')
pangu_t2m = pangu_t2m.rename('pangu_t2m')
# fourcastnet_t2m = fourcastnet_t2m.rename('fourcastnet_t2m')

print("ERA5 coordinates and dimensions:")
print(era5_t2m.coords)
print(era5_t2m.dims)

print("IFS coordinates and dimensions:")
print(ifs_t2m.coords)
print(ifs_t2m.dims)

print("PanguWeather coordinates and dimensions:")
print(pangu_t2m.coords)
print(pangu_t2m.dims)

print("Latitude range for ERA5:", era5_t2m.latitude.min().values, era5_t2m.latitude.max().values)
print("Longitude range for ERA5:", era5_t2m.longitude.min().values, era5_t2m.longitude.max().values)

print("Latitude range for IFS:", ifs_t2m.latitude.min().values, ifs_t2m.latitude.max().values)
print("Longitude range for IFS:", ifs_t2m.longitude.min().values, ifs_t2m.longitude.max().values)

print("Latitude range for PanguWeather:", pangu_t2m.latitude.min().values, pangu_t2m.latitude.max().values)
print("Longitude range for PanguWeather:", pangu_t2m.longitude.min().values, pangu_t2m.longitude.max().values)


#-------------------------------------------
# Merge the data
#-------------------------------------------

# merge on lat, lon, and valid_time
combined = xr.merge([pangu_t2m, era5_t2m], join='inner')
combined = xr.merge([combined, ifs_t2m], join='inner')

print("combined")
print(combined)

# merge on lat, lon, and valid_time
# combined = xr.merge([combined, fourcastnet_t2m], join='inner')

#-------------------------------------------
# Create forecast maps
#-------------------------------------------

# Define a function to plot a single time step of a forecast
def plot_forecast_step(data_array, forecast, title, step=0):
    plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    # Select the data for the given time index and forecast variable
    data = data_array[forecast].isel(time=step)

    # Get the corresponding longitude and latitude values
    lons = data_array['longitude']
    lats = data_array['latitude']

    # Plot the data
    plt.contourf(lons, lats, data, transform=ccrs.PlateCarree())
    plt.colorbar(label='Temperature (C)')
    plt.title(title)
    plt.savefig(f"{fig_path}/{forecast}_forecast.png")
    plt.clf()

# Plot the first time step for each forecast
plot_forecast_step(combined, 'pangu_t2m', 'Pangu Forecast')
plot_forecast_step(combined, 'era5_t2m', 'ERA5 Forecast')
plot_forecast_step(combined, 'ifs_t2m', 'IFS Forecast')
exit()



#-------------------------------------------
# Create loss measures
#-------------------------------------------


def unexpected_freeze_loss(y_true, y_pred):

    # Initialize an empty list to store the loss for each time step
    loss_list = []

    print(y_true.shape)
    print(y_pred.shape)

    # Loop over the first dimension of the arrays
    for t in range(y_true.shape[0]):
        loss = 0
        # Flatten the arrays for the current time step
        y_true_t = y_true[t].flatten()
        y_pred_t = y_pred[t].flatten()

        for i in range(len(y_true_t)):
            # If the actual is below 0 and the forecast is above 0 then increase the loss
            # Returning squared loss
            if y_true_t[i] < 0 and y_pred_t[i] > 0:
                loss += ((y_true_t[i] - y_pred_t[i]) * 2) ** 2
            else:
                loss += (y_true_t[i] - y_pred_t[i]) ** 2
        loss = np.sqrt(loss / len(y_true_t))
        loss_list.append(loss)

    # Convert the list of losses to a numpy array
    loss_array = np.array(loss_list)
    return loss_array

# calculate the difference between ai fourcasts and "truth" (ERA5)
combined['pangu_error'] = combined['pangu_t2m'] - combined['era5_t2m']
combined['ifs_error'] = combined['ifs_t2m'] - combined['era5_t2m']
# combined['fourcastnet_diff'] = combined['fourcastnet_t2m'] - combined['era5_t2m']

combined['pangue_error_squared'] = combined['pangu_error']**2
combined['ifs_error_squared'] = combined['ifs_error']**2


#-------------------------------------------
# EDA
#-------------------------------------------

ifs_error_avg = combined['ifs_error'].mean(dim='latitude').mean(dim='longitude')
pangu_error_avg = combined['pangu_error'].mean(dim='latitude').mean(dim='longitude')
# fourcastnet_diff_avg = combined['fourcastnet_diff'].mean(dim='latitude').mean(dim='longitude')

ifs_rmse = np.sqrt(combined['ifs_error_squared'].mean(dim='latitude').mean(dim='longitude'))
pangu_rmse = np.sqrt(combined['pangue_error_squared'].mean(dim='latitude').mean(dim='longitude'))

ifs_freeze_loss = unexpected_freeze_loss(y_true = combined['era5_t2m'].values, 
                                         y_pred = combined['ifs_t2m'].values)
pangu_freeze_loss = unexpected_freeze_loss(y_true = combined['era5_t2m'].values, 
                                           y_pred = combined['pangu_t2m'].values)

# time in hours
time = combined['era5_t2m'].time.values
# time_hours = combined['era5_t2m'].time.values / 3600
# Assuming `time` is your datetime series
time_hours = (time - time[0]).astype('timedelta64[h]')

# plot average error in the bbox across time
plt.plot(time_hours, ifs_error_avg, label='IFS', color = "lightgreen")
plt.plot(time_hours, pangu_error_avg, label='Pangu', color = "darkgreen")
plt.xlabel('Time (Hours)')
plt.ylabel('Temperature Difference (C)')
plt.title('Mean Error By Time')
plt.legend()
plt.savefig(f"{fig_path}/mean_error_by_time.png")
plt.clf()

# plot RMSE in the bbox across time
plt.plot(time_hours, ifs_rmse, label='IFS', color = "lightgreen")
plt.plot(time_hours, pangu_rmse, label='Pangu', color = "darkgreen")
plt.plot(time_hours, ifs_freeze_loss, label='IFS Freeze Loss', color = "lightgreen", linestyle='dashed')
plt.plot(time_hours, pangu_freeze_loss, label='Pangu Freeze Loss', color = "darkgreen", linestyle='dashed')
plt.xlabel('Time (Hours)')
plt.ylabel('RMSE (C)')
plt.title('RMSE By Time')
plt.legend()
plt.savefig(f"{fig_path}/rmse_by_time.png")
plt.clf()

#-------------------------------------------
# difference ways to measure accuracy



