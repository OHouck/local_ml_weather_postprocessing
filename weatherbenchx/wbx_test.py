# trying to run code locally from: https://colab.research.google.com/github/google-research/weatherbenchX/blob/main/docs/source/wbx_quickstart.ipynb#scrollTo=0c7ee5de-bac3-43e5-adda-e82f3b4160c6
import apache_beam as beam
import numpy as np
import xarray as xr
import weatherbenchX
from weatherbenchX.data_loaders import xarray_loaders
from weatherbenchX.metrics import deterministic
from weatherbenchX.metrics import base as metrics_base
from weatherbenchX import aggregation
from weatherbenchX import weighting
from weatherbenchX import binning
from weatherbenchX import time_chunks
from weatherbenchX import beam_pipeline
import dask

from datetime import datetime, timedelta
from custom_weatherbench_download import run_download_pipeline

import time 
import socket
import os

dask.config.set({'distributed.comm.timeouts.tcp': '60s'})

def setup_directories():
    # Determine root directory based on environment.
    nodename = socket.gethostname()
    if nodename == "oMac.local":  # local laptop
        root = os.path.expanduser(
            "~/OneDrive - The University of Chicago/ai_weather_ag/data"
        )
    else:
        raise Exception("Unknown environment, please specify the root directory")

    # file_list = sorted(glob.glob("/Volumes/wd_external_hd/weatherbench/train_global/**/*pangu*.nc", recursive=True))

    dirs = {
        "root": root,
        "raw": os.path.join(root, "raw"),
        "processed": os.path.join(root, "processed"),
        "fig": os.path.join(root, "../figures/finetuning"),
        "external": os.path.join("Volumes" ,"wd_external_hd", "weatherbench")
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs

dirs = setup_directories()

start = time.time()

prediction_path = 'gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr'
target_path = 'gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr'

full_surface_var_list = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"] 
full_atm_var_list = ["geopotential", "v_component_of_wind", "u_component_of_wind", "specific_humidity", "temperature"]

variables = full_surface_var_list + full_atm_var_list
target_data_loader = xarray_loaders.TargetsFromXarray(
    path=target_path,
    variables=variables,
)
prediction_data_loader = xarray_loaders.PredictionsFromXarray(
    path=prediction_path,
    variables=variables,
)

start_date = datetime.strptime("2018-01-01", '%Y-%m-%d')
end_date = datetime.strptime("2018-01-2", '%Y-%m-%d')
date_list = np.arange(
    np.datetime64(start_date), 
    np.datetime64(end_date), 
    dtype='datetime64[D]'
) + np.timedelta64(12, 'h') # to be at noon instead of midnight

init_times = np.array(date_list, dtype='datetime64[ns]')
# lead_times = np.array([24], dtype='timedelta64[h]').astype('timedelta64[ns]')   # To silence xr warnings.
lead_times = np.array([24, 48, 72, 96, 120, 144, 168], dtype='timedelta64[h]').astype('timedelta64[ns]')   # To silence xr warnings.

times = time_chunks.TimeChunks(
    init_times,
    lead_times,
    init_time_chunk_size = 1,
    lead_time_chunk_size = 1
)
# prediction_chunk = prediction_data_loader.load_chunk(init_times, lead_times)
# prediction_chunk
# time_end = time.time()
# print(f"Time taken to load prediction chunk: {(time_end - start)/60} minutes")

output_path = os.path.join(dirs['raw'], 'pangu2018_raw_data')
# make sure output path exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Your existing configuration
run_download_pipeline(
    prediction_path=prediction_path,
    target_path=target_path,
    variables=variables,
    init_times=init_times,
    lead_times=lead_times,
    output_path=output_path,
    runner='DirectRunner'
)

