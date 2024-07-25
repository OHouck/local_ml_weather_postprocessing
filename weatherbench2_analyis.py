# Author: Ozzy Houck
# Date Created: 6/27/2024

# Purpose: take output from weatherbench2_eval.py and create
# figures showing the performance of the different models


import xarray as xr
import matplotlib.pyplot as plt

# local
output_dir = "/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/forecasts/weatherbench2"
fig_dir = "/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/figures/weatherbench2"
code_dir = "/Users/ohouck/vc/ai_weather_ag/"

# anvil
# output_dir = "/anvil/projects/x-atm170020/ohouck/output/weatherbench2"
# code_dir = "/anvil/projects/x-atm170020/ohouck/ai_weather_ag"

region = "pakistan"

# Evaluations created by weatherbench2
pangu = xr.open_dataset(f'{output_dir}/pangu_{region}_deterministic.nc')
# pangu['lead_time'] = (pangu['lead_time'] * 2.77778e-13) / 24 # convert lead time to days (first to hours, then to days)
pangu['lead_time'] = pangu['lead_time'].astype(float) * 1.15741e-14 
print(pangu)

ifs = xr.open_dataset(f'{output_dir}/ifs_hres_{region}_deterministic.nc')
ifs['lead_time'] = ifs['lead_time'].astype(float) * 1.15741e-14

neural_gcm = xr.open_dataset(f'{output_dir}/neural_gcm_{region}_deterministic.nc')
neural_gcm['lead_time'] = neural_gcm['lead_time'].astype(float) * 1.15741e-14

pangu.temperature.sel(level=850).sel(metric='mse').plot(label='Pangu', color='lightgreen')
ifs.temperature.sel(level=850).sel(metric='mse').plot(label='IFS', color='blue')
neural_gcm.temperature.sel(level=850).sel(metric='mse').plot(label='Neural GCM', color='darkgreen')
plt.legend()
plt.ylabel('MSE')
plt.xlabel('Lead Time (days)')
plt.title(f'{region} Temperature at 850 hPa')
plt.savefig(f'{fig_dir}/weatherbench2_temperature_850hPa_{region}.png')
plt.clf()

# plot t2m mse for pangu and ifs (neural gcm doesn't have t2m)
pangu['2m_temperature'].sel(metric='mse').plot(label='Pangu', color='lightgreen')
ifs['2m_temperature'].sel(metric='mse').plot(label='IFS', color='blue')
plt.legend()
plt.ylabel('MSE')
plt.xlabel('Lead Time (days)')
plt.title(f'{region} 2m Temperature')
plt.savefig(f'{fig_dir}/weatherbench2_t2m_{region}.png')
plt.clf()
