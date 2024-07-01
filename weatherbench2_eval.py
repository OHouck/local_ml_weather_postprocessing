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


# set up file paths and variables
output_dir = "/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/forecasts/weatherbench2_output"
code_dir = "/Users/ohouck/vc/ai_weather_ag/"

# output_dir = "/anvil/projects/x-atm170020/ohouck/output/weatherbench2"
# code_dir = "/anvil/projects/x-atm170020/ohouck/ai_weather_ag"

time_start = '2020-01-01'
time_stop= '2020-07-01'

# list of supported regions in evaluate.py
region = "pakistan"

# coarse resolution for testing
era5_64x32 = 'gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr'
era5_64x32_climatology = 'gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_64x32_equiangular_conservative.zarr'
pangu_test_path = 'gs://weatherbench2/datasets/pangu/2018-2022_0012_64x32_equiangular_conservative.zarr'
pangu_test_vars = 'temperature'

# 0.25 degree resolution
era5_1440x721_path = 'gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr'
era5_1440x721_climatology_path = 'gs://weatherbench2/datasets/era5-hourly-climatology/1990-2019_6h_1440x721.zarr'

# resolution for neural gcm
era5_240x121_path = 'gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr'
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

# Define flags
FLAGS = flags.FLAGS
flags.DEFINE_string('input_chunks', 'time=1,lead_time=1', 'Chunk sizes for input data')
flags.DEFINE_integer('direct_num_workers', 32, 'Number of direct workers for Beam')


def main(argv):
    cmd = [
        'python3', 'evaluate.py',
        f'--forecast_path={forecast_path}',
        f'--obs_path={obs_path}',
        f'--climatology_path={climatology_path}',
        f'--output_dir={output_dir}',
        f'--output_file_prefix={forecast_name}_{region}_',
        '--input_chunks=init_time=1,lead_time=1',
        '--eval_configs=deterministic',
        f'--time_start={time_start}',
        f'--time_stop={time_stop}',
        f'--variables={variables}',
        f'--regions={region}',
        '--use_beam=True',
        '--runner=DirectRunner',
        '--',
        '--direct_num_workers=2'
    ]
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    app.run(main)
# start timer
test_start = time.time()
# test pangu
eval_forecast(forecast_name = 'pangu',forecast_path = pangu_test_path, obs_path = era5_64x32, 
             climatology_path = era5_64x32_climatology, variables = pangu_test_vars,
             region = region, time_start=time_start, time_stop=time_stop)
test_end = time.time()
# time elapsed in hours and minutes
test_elapsed = test_end - test_start
print(f"Test elapsed time: {test_elapsed/3600} hours")
exit()


file = open('run_times.txt', 'w')
output = "start tracking run times :}"
file.write(output)
file.close()


# pangu
pangu_start = time.time()
eval_forecast(forecast_name = 'pangu',forecast_path = pangu_path, obs_path = era5_1440x721_path, 
              climatology_path = era5_1440x721_climatology_path, variables = pangu_vars,
              region = region)
pangu_end = time.time()
pangu_time = str((pangu_end - pangu_start) / 3600)

file = open(code_dir + "run_times.txt", "w")
output = "pangu time: " + pangu_time
file.write(output)
file.close()

# ifs 
ifs_start = time.time()
eval_forecast(forecast_name = 'ifs_hres',forecast_path = ifs_hres_path, obs_path = era5_1440x721_path, 
              climatology_path = era5_1440x721_climatology_path, variables = ifs_vars,
              region = region)
ifs_end = time.time()
ifs_time = str((ifs_end - ifs_start) / 3600)

file = open(code_dir + "run_times.txt", "w")
output = "ifs time: " + ifs_time 
file.write(output)
file.close()

# neural_gcm
neural_start = time.time()
eval_forecast(forecast_name = 'neural_gcm_deterministic',forecast_path = neural_gcm_deterministic_path, obs_path =era5_240x121_path, 
              climatology_path =era_240x121_climatology_path, variables = neural_gcm_deterministic_vars,
              region = region)
neural_end= time.time()
neural_time = str((neural_end - neural_start) / 3600)

file = open(code_dir + "run_times.txt", "w")
output = "neural gcm time: " + neural_time 
file.write(output)
file.close()
