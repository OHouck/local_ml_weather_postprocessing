import argparse
import torch
import torch.nn as nn
import torch.optim as optim 
import weatherbenchX 
from weatherbenchX.data_loaders import xarray_loaders
from weatherbenchX.metrics import deterministic
from weatherbenchX.metrics import base as metrics_base
from weatherbenchX import binning
from weatherbenchX import aggregation
from weatherbenchX import time_chunks 
import numpy as np

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=1, num_hidden_layers=3):
        super(SimpleMLP, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def main():
    # Set up device: prioritize CUDA, then MPS, then CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # args = parse_args()
    
    variables = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"]
    prediction_path = 'gs://weatherbench2/datasets/hres/2016-2022-0012-64x32_equiangular_conservative.zarr'
    target_path = 'gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr'
    region_name = "north_india"
    train_start = '2020-01-01T00'
    train_end = '2020-01-01T12'
    lead_times = [24] # can train on multiple lead times


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

    # OH: note it is possible to bin by multiple regions
    bin_by = [binning.Regions(region)]



    target_data_loader = xarray_loaders.TargetsFromXarray(
        path = target_path,
        variables = variables,
    )

    prediction_data_loader = xarray_loaders.PredictionsFromXarray(
        path = prediction_path,
        variables = variables,
    )

    init_times = np.array([train_start, train_end], dtype='datetime64[ns]')
    lead_times = np.array(lead_times, dtype='timedelta64[h]').astype('timedelta64[ns]') 

    target_chunk = target_data_loader.load_chunk(init_times, lead_times)
    prediction_chunk = prediction_data_loader.load_chunk(init_times, lead_times)

    metrics = {
        'rmse': deterministic.RMSE(),
    }
    statistics = metrics_base.compute_unique_statistics_for_all_metrics(
        metrics, prediction_chunk, target_chunk
    )

    aggregator = aggregation.Aggregator(
        reduce_dims = [], # create statistics for each variable
        bin_by = bin_by,
    )

    aggregation_state = aggregator.aggregate_statistics(statistics)

    # has naming conventions: <metric>.<variable>
    # this gives is rmse by variable and lat/lon
    metrics = aggregation_state.metric_values(metrics)
    
    print(metrics)


    #-----------------------------------
    # Preprocess errors for training
    #-----------------------------------


    #-----------------------------------
    # Train model
    #-----------------------------------



if __name__ == "__main__":
    main()

