# Author: Ozzy Houck
# Date Created 5/27/2024

# Purpose: Compare t2m measurements between ERA5, FourCastNet, and PanguWeather
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.animation import FuncAnimation # for creating gifs

#-------------------------------------------
# Function to create forecast gifs 
#-------------------------------------------
def make_forecast_gif(data_array, forecast, title, fig_path, file_name):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    # Get global min and max values for the color scale
    vmin = data_array[forecast].min().values
    vmax = data_array[forecast].max().values

    # Initial plot
    data = data_array[forecast].isel(time=0)
    lons = data_array['longitude']
    lats = data_array['latitude']
    contour = ax.contourf(lons, lats, data, transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax)
    colorbar = fig.colorbar(contour, ax=ax, label='Temperature (C)')

    def animate(i):
        ax.clear()
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        
        # Select the data for the given time index and forecast variable
        data = data_array[forecast].isel(time=i)

        # Plot the data
        contour = ax.contourf(lons, lats, data, transform=ccrs.PlateCarree())

        time = pd.to_datetime(data.time.values).round('h')
        ax.set_title(f"{title} at time {time}")

    anim = FuncAnimation(fig, animate, frames=len(data_array.time), repeat=True)

    # Save the animation to a file
    anim.save(f"{fig_path}/{file_name}.gif", writer='pillow', fps=3)

#-------------------------------------------
# Alternative Loss Functions 
#-------------------------------------------
def unexpected_freeze_loss(y_true, y_pred):
    '''
    RMSE loss function that doubles the penalty for unexpected freezes
    '''

    # Initialize an empty list to store the loss for each time step
    loss_list = []

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

#-------------------------------------------
# Load the data and set up config
#-------------------------------------------
forecast_path = "/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/forecasts"
fig_path = "/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/figures"
date = "2024-04-01" # eventually should be part of config file

# supported regions: 'Global', 'Midwest', 'Pakistan'
region = "Midwest"

# Set bounding box for the data
if region == "Global":
    lat_min = -90   
    lat_max = 90 
    lon_min = -180
    lon_max = 180 
elif region == "Midwest":
    lat_min = 35
    lat_max = 50
    lon_min = -100
    lon_max = -80
elif region == "Pakistan":
    lat_min = 24
    lat_max = 37
    lon_min = 60
    lon_max = 78
else:
    raise ValueError("Region not supported")

bbox = [lon_min, lat_min, lon_max, lat_max]

# Load the data
combined = xr.open_dataset(f"{forecast_path}/combined_forecasts_{date}.nc")

# filter by bbox
combined = combined.where((combined.longitude >= bbox[0]) & (combined.longitude <= bbox[2]), drop=True)
combined = combined.where((combined.latitude >= bbox[1]) & (combined.latitude <= bbox[3]), drop=True)

#-------------------------------------------
# EDA
#-------------------------------------------

# calculate the difference between ai fourcasts and "truth" (ERA5)
combined['pangu_error'] = combined['pangu_t2m'] - combined['era5_t2m']
combined['ifs_error'] = combined['ifs_t2m'] - combined['era5_t2m']
combined['fourcastnet_error'] = combined['fourcastnet_t2m'] - combined['era5_t2m']

combined['pangue_error_squared'] = combined['pangu_error']**2
combined['ifs_error_squared'] = combined['ifs_error']**2

# Plot the first time step for each forecast
make_forecast_gif(data_array = combined, forecast = 'era5_t2m', 
                  title = f'{region} ERA5 Forecast', fig_path =fig_path,
                  file_name = f'era5_{region}_{date}')
make_forecast_gif(data_array = combined, forecast='ifs_t2m', 
                  title=f'{region} IFS Forecast', fig_path=fig_path,
                  file_name = f'ifs_forecast_{region}_{date}')
make_forecast_gif(data_array=combined, forecast='pangu_t2m', 
                  title=f'{region} Pangu Forecast', fig_path=fig_path,
                  file_name = f'pangu_forecast_{region}_{date}')
make_forecast_gif(data_array=combined, forecast='fourcastnet_t2m', 
                  title=f'{region} Fourcastnet Forecast', fig_path=fig_path,
                  file_name = f'fourcastnet_forecast_{region}_{date}')

make_forecast_gif(data_array=combined, forecast='ifs_error', 
                  title=f'{region} IFS Error', fig_path=fig_path,
                  file_name = f'ifs_error_{region}_{date}')
make_forecast_gif(data_array=combined, forecast='pangu_error',
                    title=f'{region} Pangu Error', fig_path=fig_path,
                    file_name = f'pangu_error_{region}_{date}')
make_forecast_gif(data_array=combined, forecast='fourcastnet_error',
                    title=f'{region} Fourcastnet Error', fig_path=fig_path,
                    file_name = f'fourcastnet_error_{region}_{date}')

# Estimate loss over time using RMSE and custom loss functions
ifs_error_avg = combined['ifs_error'].mean(dim='latitude').mean(dim='longitude')
pangu_error_avg = combined['pangu_error'].mean(dim='latitude').mean(dim='longitude')

ifs_rmse = np.sqrt(combined['ifs_error_squared'].mean(dim='latitude').mean(dim='longitude'))
pangu_rmse = np.sqrt(combined['pangue_error_squared'].mean(dim='latitude').mean(dim='longitude'))

ifs_freeze_loss = unexpected_freeze_loss(y_true = combined['era5_t2m'].values, 
                                         y_pred = combined['ifs_t2m'].values)
pangu_freeze_loss = unexpected_freeze_loss(y_true = combined['era5_t2m'].values, 
                                           y_pred = combined['pangu_t2m'].values)

# time in hours
time = combined['era5_t2m'].time.values
time_hours = (time - time[0]).astype('timedelta64[h]')

# plot RMSE in the bbox across time
plt.plot(time_hours, ifs_rmse, label='IFS', color = "lightgreen")
plt.plot(time_hours, pangu_rmse, label='Pangu', color = "darkgreen")
plt.plot(time_hours, ifs_freeze_loss, label='IFS Freeze Loss', color = "lightgreen", linestyle='dashed')
plt.plot(time_hours, pangu_freeze_loss, label='Pangu Freeze Loss', color = "darkgreen", linestyle='dashed')
plt.xlabel('Time (Hours)')
plt.ylabel('RMSE (C)')
plt.title('RMSE By Time')
plt.legend()
plt.savefig(f"{fig_path}/rmse_by_time_{region}.png")
plt.clf()



