
import os
import apache_beam as beam
import argparse
import xarray as xr
import weatherbenchX 
from weatherbenchX.data_loaders import xarray_loaders
from weatherbenchX.metrics import deterministic
from weatherbenchX.metrics import base as metrics_base
from weatherbenchX import binning
from weatherbenchX import aggregation
from weatherbenchX import time_chunks 
from weatherbenchX import beam_pipeline

import xbatcher as xb
import numpy as np


def main():

    # Define parameters.
    variables = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"]
    prediction_path = "gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr" 
    target_path = "gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr" 
    region_name = "north_india"
    # Use a longer period for training so that random splitting makes sense.
    train_start = '2020-01-01'
    train_end = '2021-01-01'
    test_start = '2020-01-10T12'
    test_end = '2020-01-10T12'
    output_dir = "~/wb_finetune_test"
    output_name = f"wbx_{region_name}_{train_start}_{train_end}.nc"
    output_path = os.path.join(output_dir, output_name)

    # Define region.
    if region_name == "north_india":
        region = {
            "north_india": ((21, 35.5), (70.75, 87.25)),
        }
    elif region_name == "uttar_pradesh":
        region = {
            "uttar_pradesh": ((24.25, 26), (78, 87.25)),
        }
    elif region_name == "full_india":
        region = {
            "full_india": ((8.75, 27.25), (70.95, 87.25)),
        }
    else:
        region = {
            "global": ((-90, 90), (0, 360)),
        }
    # Bin by region.
    bin_by = [binning.Regions(region)]

    # Load target and prediction datasets.
    target_data_loader = xarray_loaders.TargetsFromXarray(
        path=target_path,
        variables=variables,
    )
    prediction_data_loader = xarray_loaders.PredictionsFromXarray(
        path=prediction_path,
        variables=variables,
    )

    init_times = np.arange(train_start, train_end, np.timedelta64(24, 'h'), dtype='datetime64[ns]')
    # init_times = np.arange('2020-01-01T00', '2020-01-03T00', np.timedelta64(12, 'h'), dtype='datetime64[ns]')
    lead_times = np.arange(24, 48, 96, dtype='timedelta64[h]').astype('timedelta64[ns]') 

    times = time_chunks.TimeChunks(
        init_times,
        lead_times,
        init_time_chunk_size=32,
        lead_time_chunk_size=1
    )

    # Compute forecast error metrics.
    metrics_dict = {
        'rmse': deterministic.RMSE(),
    }

    aggregator = aggregation.Aggregator(
        reduce_dims=[],  # create statistics for each variable
        bin_by=bin_by,
    )

    root = beam.Pipeline(runner='DirectRunner')
    beam_pipeline.define_pipeline(
        root = root,
        times = times,
        predictions_loader = prediction_data_loader,
        targets_loader = target_data_loader,
        metrics = metrics_dict,
        aggregator = aggregator,
        out_path = output_path,
    )

    root.run()

    metrics_all = xr.open_dataset(output_path).compute()
    print(metrics_all)

    exit()

if __name__ == "__main__":
    main()