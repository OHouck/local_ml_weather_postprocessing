# Author: Ozzy Houck
# Date 7/4/23

# Purpose sample code to teach xarray

import xarray as xr
import numpy as np
import pandas as pd

# plotting libraries
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.animation import FuncAnimation

# Load in data saved as NetCDF
file_path = "/Users/ohouck/OneDrive - The University of Chicago/ai_weather_ag/forecasts"
fig_path = "/Users/ohouck/OneDrive - The University of Chicago/ai_weather_ag/figures"
file_name = "combined_forecasts_2024-04-01.nc"
forecast = xr.open_dataset(f"{file_path}/{file_name}")

print(forecast)

# print the coordinates
print(forecast.coords)

# print the dimensions
print(forecast.dims)

print("attrs")
print(forecast.attrs)

forecast.ifs_t2m.mean(dim="time").plot(x = "longitude", y = "latitude")
plt.show()

exit()

# Let's only focus on the ifs forecast this time
forecast = forecast["ifs_t2m"]

# restrict the data to a specific geographic area (bbox)
lat_min = 40
lat_max = 50
lon_min = -100
lon_max = -90

bbox = [lon_min, lat_min, lon_max, lat_max]

# midwest(ish) 
midwest_forecast = forecast.where((forecast.longitude >= bbox[0]) & 
                                  (forecast.latitude >= bbox[1]) & 
                                  (forecast.longitude <= bbox[2]) &
                                  (forecast.latitude <= bbox[3]), drop = True)

print(midwest_forecast)

# print max and min temperatures predicted using the IFS 
max_temp = forecast.max().values
min_temp = forecast.min().values
print(max_temp, min_temp)

max_temp_midwest = midwest_forecast.max().values
min_temp_midwest = midwest_forecast.min().values
print(max_temp_midwest, min_temp_midwest)


# make forecast gif

def make_forecast_gif(forecast, title, fig_path, file_name):
    fig, ax = plt.subplots(figsize  = (10, 5))
    ax = plt.axes(projection = ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle = ':')

    # initialize the plot using the first timestep
    data = forecast.isel(time=0)
    lons = forecast['longitude']
    lats = forecast['latitude']

    # make filled contour plot
    contour = ax.contourf(lons, lats, data, transform= ccrs.PlateCarree())
    colorbar = fig.colorbar(contour, ax=ax, label = 'Temperature (C)')

    def animate(i):
        ax.clear()
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle = ':')

        data = forecast.isel(time=i)
        contour = ax.contourf(lons, lats, data, transform= ccrs.PlateCarree())

        time = pd.to_datetime(data.time.values).round('h')
        ax.set_title(f"{title} at time {time}")
    animation = FuncAnimation(fig, animate, frames = len(forecast.time), 
                              repeat = True)
    # save the animation
    animation.save(f"{fig_path}/{file_name}.gif", writer = "pillow", fps = 5)

    # clear plot space
    plt.clf()
            

# make plot 
make_forecast_gif(forecast=forecast, title = "Global IFS", fig_path = fig_path, file_name = "global_ifs_example.gif")
make_forecast_gif(forecast=midwest_forecast, title = "Midwest IFS", fig_path = fig_path, file_name = "midwest_ifs_example.gif")
