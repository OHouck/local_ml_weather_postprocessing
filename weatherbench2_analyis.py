# Author: Ozzy Houck
# Date Created: 6/27/2024

# Purpose: take output from weatherbench2_eval.py and create
# figures showing the performance of the different models


import xarray as xr
import matplotlib.pyplot as plt

output_dir = "/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/forecasts/weatherbench2_output"
code_dir = "/Users/ohouck/vc/ai_weather_ag/"

# output_dir = "/anvil/projects/x-atm170020/ohouck/output/weatherbench2"
# code_dir = "/anvil/projects/x-atm170020/ohouck/ai_weather_ag"

# load in test results
test_results = xr.open_dataset(f'{output_dir}/pangu_small_test_deterministic.nc')
print("TEST RESULTS")
print(test_results)

# print MSE for 2m temperature
print("MSE 2m temperature")
print(test_results['2m_temperature'].sel(metric='mae').values)

# plot mse of 2m temperature
# test_results['2m_temperature'].sel(metric='mse').plot()
# plt.show()



# load in pangu results
#pangu_results = xr.open_dataset(f'{output_dir}/pangu_deterministic.nc')
#hres_results = xr.open_dataset(f'{output_dir}/hres_deterministic.nc')

#print("PANGU")
#print(pangu_results)

#print("HRES")
#print(hres_results)

# rename geopotential variables and merge
#pangu_results = pangu_results.rename_vars({'geopotential': 'pangu_geopotential'})
#hres_results = hres_results.rename_vars({'geopotential': 'hres_geopotential'})

#results = xr.merge([pangu_results, hres_results])
#print("MERGED")
#print(results)

# print metricds
#print("METRICS")
#print(results.metric)

# make plot comparing MSE of geoportential at 500 hPa

#print("PLOTS")
#results['pangu_geopotential'].sel(level=500).sel(metric='mse').plot(label='pangu')
#results['hres_geopotential'].sel(level=500).sel(metric='mse').plot(label='hres')
#plt.legend()
#plt.show()
