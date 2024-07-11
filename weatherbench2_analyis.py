# Author: Ozzy Houck
# Date Created: 6/27/2024

# Purpose: take output from weatherbench2_eval.py and create
# figures showing the performance of the different models


import xarray as xr
import matplotlib.pyplot as plt

output_dir = "/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/forecasts/weatherbench2"
code_dir = "/Users/ohouck/vc/ai_weather_ag/"

# output_dir = "/anvil/projects/x-atm170020/ohouck/output/weatherbench2"
# code_dir = "/anvil/projects/x-atm170020/ohouck/ai_weather_ag"

region = "pakistan"

# Evaluations created by weatherbench2
pangu = xr.open_dataset(f'{output_dir}/pangu_{region}_deterministic.nc')

ifs = xr.open_dataset(f'{output_dir}/ifs_hres_{region}_deterministic.nc')

neural_gcm = xr.open_dataset(f'{output_dir}/neural_gcm_{region}_deterministic.nc')

# rename variables to merge 
pangu = pangu.rename_vars({'temperature': 'pangu_temperature', 
                           '2m_temperature': 'pangu_2m_temperature'})
ifs = ifs.rename_vars({'temperature': 'ifs_temperature',
                     '2m_temperature': 'ifs_2m_temperature',
                     'total_precipitation_24hr': 'ifs_total_precipitation_24hr'})
neural_gcm = neural_gcm.rename_vars({'temperature': 'neural_gcm_temperature'})

# merge datasets
combined = xr.merge([pangu, ifs, neural_gcm])

# convert lead time from nano seconds to days
combined['lead_time'] = combined.lead_time * 1.15741e-14

print(combined['neural_gcm_temperature'].sel(level=850).sel(metric='acc'))

# make plot coming temperature at 850 hPa
combined['pangu_temperature'].sel(level=850).sel(metric='mse').plot(label='pangu', color='lightgreen')
combined['ifs_temperature'].sel(level=850).sel(metric='mse').plot(label='ifs', color='blue')
combined['neural_gcm_temperature'].sel(level=850).sel(metric='mse').plot(label='neural_gcm', color='darkgreen')
plt.legend()
plt.ylabel('MSE')
plt.title('Temperature at 850 hPa')
plt.show()

combined['pangu_2m_temperature'].sel(metric='mse').plot(label='pangu', color='lightgreen')
combined['ifs_2m_temperature'].sel(metric='mse').plot(label='ifs', color='forestgreen')
plt.legend()
plt.xlabel('Lead Time (days)')
plt.ylabel('MSE')
plt.title('2m Temperature MSE')
plt.show()

