import apache_beam as beam
import argparse
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim 
from torch.utils.data import random_split, DataLoader
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

# Define your MLP model.
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

# Compute target from a batch.
def compute_target(batch):
    # Each batch is an xarray.Dataset containing normalized errors.
    xs = []
    for var in batch.data_vars:
        # Convert to numpy array and flatten.
        xs.append(np.array(batch[var].values).flatten())
    x_input = np.stack(xs, axis=-1)  # shape: (batch_size, num_variables)
    # Our target is the negative of the mean normalized error per sample.
    # (That is, the correction needed to bring the error to 0.)
    target = -np.mean(x_input, axis=1, keepdims=True)
    return target

# Custom PyTorch Dataset that wraps an xbatcher BatchGenerator.
class TorchBatchDataset(torch.utils.data.Dataset):
    def __init__(self, x_bgen, target_map_func):
        # Convert the generator to a list so that we can index it.
        self.x_batches = list(x_bgen)
        self.target_map_func = target_map_func

    def __len__(self):
        return len(self.x_batches)

    def __getitem__(self, idx):
        x_batch = self.x_batches[idx]
        y_batch = self.target_map_func(x_batch)
        return x_batch, y_batch

# Training function for one epoch.
def train_epoch(model, optimizer, criterion, data_loader, device):
    model.train()
    epoch_loss = 0.0
    for x_input, y_target in data_loader:
        # Convert input batch from xarray.Dataset to torch tensor.
        xs = []
        for var in x_input.data_vars:
            xs.append(np.array(x_input[var].values).flatten())
        x_input_tensor = np.stack(xs, axis=-1)
        x_input_tensor = torch.tensor(x_input_tensor, dtype=torch.float32, device=device)
        y_target_tensor = torch.tensor(y_target, dtype=torch.float32, device=device)
        
        optimizer.zero_grad()
        output = model(x_input_tensor)
        loss = criterion(output, y_target_tensor)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(data_loader)

# Validation function for one epoch.
def validate_epoch(model, criterion, data_loader, device):
    model.eval()
    epoch_loss = 0.0
    with torch.no_grad():
        for x_input, y_target in data_loader:
            xs = []
            for var in x_input.data_vars:
                xs.append(np.array(x_input[var].values).flatten())
            x_input_tensor = np.stack(xs, axis=-1)
            x_input_tensor = torch.tensor(x_input_tensor, dtype=torch.float32, device=device)
            y_target_tensor = torch.tensor(y_target, dtype=torch.float32, device=device)
            
            output = model(x_input_tensor)
            loss = criterion(output, y_target_tensor)
            epoch_loss += loss.item()
    return epoch_loss / len(data_loader)

def main():
    # Set up device: prioritize CUDA, then MPS, then CPU.
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # Define parameters.
    variables = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"]
    prediction_path = 'gs://weatherbench2/datasets/hres/2016-2022-0012-64x32_equiangular_conservative.zarr'
    target_path = 'gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr'
    region_name = "north_india"
    # Use a longer period for training so that random splitting makes sense.
    train_start = '2020-01-01T00'
    train_end = '2020-01-10T00'
    test_start = '2020-01-10T00'
    test_end = '2020-01-10T12'

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
        init_time_chunk_size=16,
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
        out_path = 'out.nc',
    )

    root.run()

    metrics_all = xr.open_zarr('out.nc').compute()

    exit()


    metrics_all = aggregation_state.metric_values(metrics_dict)
    
    # Preprocess errors: compute normalized RMSE.
    normalization_stats = {}
    for var in variables:
        key = f'rmse.{var}'
        norm_key = f'normalized_rmse.{var}'
        normalization_stats[f'mean_{var}'] = metrics_all[key].mean()
        normalization_stats[f'std_{var}'] = metrics_all[key].std()
        metrics_all[norm_key] = (metrics_all[key] - normalization_stats[f'mean_{var}']) / normalization_stats[f'std_{var}']

    # Keep only the normalized RMSE variables.
    normalized_var_names = [f'normalized_rmse.{var}' for var in variables]
    metrics_norm = metrics_all[normalized_var_names]

    # -----------------------------------
    # Create a PyTorch-compatible dataset from xbatcher.
    # -----------------------------------
    # Define batch size for xbatcher (each sample will be a batch with this many 'init_time' points).
    batch_size = 8 
    X_bgen_train = xb.BatchGenerator(metrics_norm, input_dims={'init_time': batch_size})
    # Instead of using a .map() method (which is not available), wrap the generator in our custom dataset.
    train_val_dataset = TorchBatchDataset(X_bgen_train, compute_target)

    # Randomly split into training and validation sets.
    n_total = len(train_val_dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_dataset, val_dataset = random_split(train_val_dataset, [n_train, n_val])
    # Create PyTorch DataLoaders.
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    # -----------------------------------
    # Train model using PyTorch DataLoaders.
    # -----------------------------------
    num_epochs = 10
    model = SimpleMLP(input_dim=len(normalized_var_names)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, optimizer, criterion, train_loader, device)
        val_loss = validate_epoch(model, criterion, val_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs} -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # -----------------------------------
    # Test set evaluation using PyTorch.
    # -----------------------------------
    # Define a test time range (different from training period).
    test_init_times = np.array([test_start, test_end], dtype='datetime64[ns]')

    test_target_chunk = target_data_loader.load_chunk(test_init_times, lead_times)
    test_prediction_chunk = prediction_data_loader.load_chunk(test_init_times, lead_times)

    test_statistics = metrics_base.compute_unique_statistics_for_all_metrics(
        metrics_dict, test_prediction_chunk, test_target_chunk
    )
    test_aggregation_state = aggregator.aggregate_statistics(test_statistics)
    test_metrics_all = test_aggregation_state.metric_values(metrics_dict)

    # Compute normalized errors on the test set using training normalization stats.
    for var in variables:
        key = f'rmse.{var}'
        norm_key = f'normalized_rmse.{var}'
        test_metrics_all[norm_key] = (test_metrics_all[key] - normalization_stats[f'mean_{var}']) / normalization_stats[f'std_{var}']

    test_metrics_norm = test_metrics_all[normalized_var_names]
    # Create xbatcher generator and then wrap in our custom torch dataset for the test set.
    X_bgen_test = xb.BatchGenerator(test_metrics_norm, input_dims={'init_time': batch_size})
    test_dataset = TorchBatchDataset(X_bgen_test, compute_target)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    test_loss_total = 0.0
    model.eval()
    with torch.no_grad():
        for x_input, y_target in test_loader:
            xs = []
            for var in x_input.data_vars:
                xs.append(np.array(x_input[var].values).flatten())
            x_input_tensor = np.stack(xs, axis=-1)
            x_input_tensor = torch.tensor(x_input_tensor, dtype=torch.float32, device=device)
            y_target_tensor = torch.tensor(y_target, dtype=torch.float32, device=device)
            
            output = model(x_input_tensor)
            loss = criterion(output, y_target_tensor)
            test_loss_total += loss.item()
    avg_test_loss = test_loss_total / len(test_loader) if len(test_loader) > 0 else float('nan')
    print(f"Test Loss (MSE on corrected normalized error): {avg_test_loss:.4f}")

if __name__ == "__main__":
    main()
