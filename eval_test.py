import os
import subprocess
import weatherbench2
import ast
import apache_beam
import time

from absl import app
from absl import flags
from weatherbench2 import config
from weatherbench2 import evaluation
from weatherbench2 import flag_utils
from weatherbench2 import metrics
from weatherbench2 import thresholds
from weatherbench2.derived_variables import DERIVED_VARIABLE_DICT
from weatherbench2.regions import CombinedRegion
from weatherbench2.regions import LandRegion
from weatherbench2.regions import SliceRegion
import xarray as xr

# Set your bucket, project, and region
BUCKET = 'my-bucket'
PROJECT = 'my-project'
REGION = 'us-central1'



def eval_forecast(forecast_name, forecast_path, obs_path, climatology_path, 
                  variables, region):
    print(forecast_name)
    cmd = [
        'python3', 'evaluate.py',
        f'--forecast_path={forecast_path}',
        f'--variables={variables}',
        f'--regions={region}',
        f'--output_file_prefix={forecast_name}_{region}_',
        f'--obs_path={obs_path}',
        f'--climatology_path={climatology_path}',
        f'--output_dir={output_dir}',
        '--input_chunks=time=1,lead_time=1',
        '--eval_configs=deterministic',
    ]
    subprocess.run(cmd, check=True)

output_dir = "/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/forecasts/weatherbench2_output"
# list of supported regions in evaluate.py
region = "small_test"

# coarse resolution for testing
era5_64x32 = 'gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr'
era5_64x32_climatology = 'gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_64x32_equiangular_conservative.zarr'
pangu_test = 'gs://weatherbench2/datasets/pangu/2018-2022_0012_64x32_equiangular_conservative.zarr'

# # start timer
# test_start = time.time()
# # test pangu
# eval_forecast(forecast_name = 'pangu',forecast_path = pangu_test, obs_path = era5_64x32, 
#               climatology_path = era5_64x32_climatology, variables = pangu_vars,
#               region = region)
# test_end = time.time()

# # time elapsed in hours and minutes
# test_elapsed = test_end - test_start
# print(f"Test elapsed time: {test_elapsed/3600} hours")

# 0.25 degree resolution
era5_1440x721_path = 'gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr'
era5_240x121_path = 'gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr'

# resolution for neural gcm
era5_1440x721_climatology_path = 'gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr'
era_240x121_climatology_path = 'gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_240x121_equiangular_with_poles_conservative.zarr'

ifs_hres_pata = 'gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr'
ifs_ens_mean = 'gs://weatherbench2/datasets/ifs_ens/2018-2022-1440x721_mean.zarr'
ifs_vars= '2m_temperature, temperature, total_precipitation_24hr'

pangu_path = 'gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr'
pangu_operational_path = 'gs://weatherbench2/datasets/pangu_hres_init/2020_0012_0p25.zarr'
pangu_vars = '2m_temperature, temperature'

graphcast_path = 'gs://weatherbench2/datasets/graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours_derived.zarr'
graphcast_operational_path = 'gs://weatherbench2/datasets/graphcast_hres_init/2020/date_range_2019-11-16_2021-02-01_12_hours_derived.zarr'
graphcast_vars = '2m_temperature, temperature, total_precipitation_24hr'

fuxi_path = 'gs://weatherbench2/datasets/fuxi/2020-1440x721.zarr'
fuxi_vars = '2m_temperature, temperature, total_precipitation_24hr'

neural_gcm_deterministic_path = 'gs://weatherbench2/datasets/neuralgcm_deterministic/2020-240x121_equiangular_with_poles_conservative.zarr'
neural_gcm_deterministic_vars = 'temperature, P_minus_E_cumulative'

pangu_start = time.time()
# test pangu
eval_forecast(forecast_name = 'pangu',forecast_path = pangu_path, obs_path = era5_1440x721_path, 
              climatology_path = era5_1440x721_climatology_path, variables = pangu_vars,
              region = region)
pangu_end = time.time()

# time elapsed in hours and minutes
pangu_elapsed = pangu_end - pangu_start
print(f"Pangu Evaluation Time: {pangu_elapsed/3600} hours")


    
# load in pangu results
pangu_results = xr.open_dataset(f'{output_dir}/pangu_deterministic.nc')
hres_results = xr.open_dataset(f'{output_dir}/hres_deterministic.nc')

print("PANGU")
print(pangu_results)

print("HRES")
print(hres_results)


# rename geopotential variables and merge
pangu_results = pangu_results.rename_vars({'geopotential': 'pangu_geopotential'})
hres_results = hres_results.rename_vars({'geopotential': 'hres_geopotential'})

results = xr.merge([pangu_results, hres_results])
print("MERGED")
print(results)

# print metricds
print("METRICS")
print(results.metric)

# make plot comparing MSE of geoportential at 500 hPa

import matplotlib.pyplot as plt
print("PLOTS")
results['pangu_geopotential'].sel(level=500).sel(metric='mse').plot(label='pangu')
results['hres_geopotential'].sel(level=500).sel(metric='mse').plot(label='hres')
plt.legend()
plt.show()

