import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cfgrib
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

# Open the GRIB file using xarray
ds = xr.open_dataset('panguweather.grib', engine='cfgrib')

# Find the nearest pressure level to 1000 hPa which is about the surface level
nearest_level = ds.isobaricInhPa.sel(isobaricInhPa=1000, method='nearest')

# Extract the temperature at the nearest pressure level
temperature = ds['t'].sel(isobaricInhPa=nearest_level)

# Subtract 273.15 from the temperature values to convert from Kelvin to Celsius
temperature_celsius = temperature - 273.15

# Shift the longitude coordinates to align with the map
shift = len(temperature_celsius.longitude) // 2
temperature_celsius = temperature_celsius.roll(longitude=shift, roll_coords=True)

# define the bounding box for Armenia
lon_min, lon_max = 14, 24
lat_min, lat_max = 44, 48

# Subset the data based on the bounding box
temperature_celsius= temperature_celsius.sel(longitude=slice(lon_min, lon_max), latitude=slice(lat_max, lat_min))

# Flip the temperature data vertically
temperature_celsius_subset = temperature_celsius.sel(latitude=slice(None, None, -1))


# Create a figure and axis for the plot
fig = plt.figure(figsize=(8, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

# forecast number: 0-41
forecast_index = 0

# Initialize the plot with the first time step
plot = ax.imshow(temperature_celsius_subset[forecast_index], cmap='viridis',
                 extent=[temperature_celsius_subset.longitude.min(), temperature_celsius_subset.longitude.max(),
                         temperature_celsius_subset.latitude.min(), temperature_celsius_subset.latitude.max()],
                 transform=ccrs.PlateCarree())

fig.colorbar(plot, ax=ax, label='Temperature (°C)')
ax.set_title('Temperature Forecast - Armenia')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

# Add continent outlines
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

# Add country outlines
ax.add_feature(cfeature.BORDERS, linewidth=0.3)

# Customize the map gridlines
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False

# Set the map extent to the bounding box of Poland
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

# Update function for the animation
def update(frame):
    plot.set_data(temperature_celsius_subset[frame])
    ax.set_title(f'Temperature Forecast - Armenia (Frame: {frame+1})')
    return plot,

# Create the animation
ani = FuncAnimation(fig, update, frames=temperature_celsius_subset.shape[0], interval=200, blit=True)

# Save the animation as a GIF
ani.save('temperature_forecast_armenia.gif', writer='pillow')

plt.show()