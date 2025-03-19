#!/usr/bin/env python3
"""
weatherbench2_finetuning_combined.py
Author: Ozzy Houck (modified version)
Date: 2025-03-14

This script fine-tunes an MLP correction model to a specific region using model
forecasts and corresponding observations from weatherbench2. 

Example usage:
python3 finetuning/finetune.py \
    --forecast_path="gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr" \
    --obs_path="gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr" \
    --output_dir="~/wb_finetune_test" \
    --model_name="pangu" \
    --region="north_india" \
    --train_start="2018-01-01" --train_end="2021-12-30" \
    --test_start="2022-01-01" --test_end="2022-12-30" \
    --lead_time_hours=48 \
    --training_vars 10m_v_component_of_wind 10m_u_component_of_wind \
    --output_vars 10m_v_component_of_wind 10m_u_component_of_wind \
    --epochs=1000 \
    --mlp_hidden_dim=512 \
    --mlp_layers=5 \
    --use_cupy \
    --use_cloud_dataloader
"""

import argparse
import os
import time
import socket
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
import dask

# Optionally import cupy if available.
try:
    import cupy as cp
except ImportError:
    cp = None

# Import xbatcher and its PyTorch loader
import xbatcher as xb
import xbatcher.loaders.torch

# -------------------------------------------------------------------
# Simple MLP model
# -------------------------------------------------------------------
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=1, num_hidden_layers=3):
        super(SimpleMLP, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # allows for input to be (batch, var, lat, lon) 
        if x.dim() >2:
            x = x.flatten(start_dim=1)
        return self.net(x)

class NormalizedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, stats):
        """
        Wraps a dataset and applies on-the-fly normalization.
        :param base_dataset: The underlying dataset yielding (x, y) pairs.
        :param stats: Dictionary with keys 'mean' and 'std' for normalization.
                      They are expected to be xarray DataArrays (or convertible to numpy arrays)
                      with the 'features' dimension matching the flattened input.
        """
        self.base_dataset = base_dataset
        self.stats = stats
        # Convert stats to numpy arrays if necessary
        self.mean = self.stats['mean'].values if hasattr(self.stats['mean'], 'values') else self.stats['mean']
        self.std = self.stats['std'].values if hasattr(self.stats['std'], 'values') else self.stats['std']

    def __len__(self):
        return len(self.base_dataset)

    def normalize_batch(self, batch):
        # Expect batch to be a numpy array or torch tensor of shape (batch_size, features)
        if isinstance(batch, torch.Tensor):
            # Create torch tensors for mean and std on the same device as the batch.
            mean_tensor = torch.tensor(self.mean, dtype=batch.dtype, device=batch.device)
            std_tensor = torch.tensor(self.std, dtype=batch.dtype, device=batch.device)
            return (batch - mean_tensor.unsqueeze(0)) / (std_tensor.unsqueeze(0) + 1e-8)
        else:
            # Assume numpy array
            return (batch - self.mean[None, :]) / (self.std[None, :] + 1e-8)

    def __getitem__(self, index):
        x, y = self.base_dataset[index]
        # Apply normalization to both x and y (if desired, you might choose to normalize only the input).
        x_norm = self.normalize_batch(x)
        y_norm = self.normalize_batch(y)
        return x_norm, y_norm


# -------------------------------------------------------------------
# pre-processing using xarray/dask
# -------------------------------------------------------------------
def load_data(zarr_path, vars, level, lat_slice, lon_slice, times, lead_time_hours, use_consolidated=True, use_cupy=False):
    open_zarr_kwargs = {'decode_timedelta': True}

    # consolidate metadata
    if use_consolidated:
        open_zarr_kwargs['consolidated'] = True

    # Open dataset (if already an xr.Dataset, pass it through)
    ds = xr.open_zarr(zarr_path, **open_zarr_kwargs) if isinstance(zarr_path, str) else zarr_path

    # Optionally use cupy-backed arrays if enabled.
    if use_cupy and cp is not None:
        ds = ds.cupy.as_cupy()

    # Slice time and select training variables
    ds = ds.sel(time=times, latitude=lat_slice, longitude=lon_slice)[vars]

    # In forecast, select the desired lead time and drop the coordinate if present.
    if 'prediction_timedelta' in ds.coords:
        ds = ds.sel(prediction_timedelta=np.timedelta64(lead_time_hours, 'h'))
        ds = ds.drop_vars('prediction_timedelta')

    # Slice level if specified.
    if level is not None:
        ds = ds.sel(level=level)

    # Convert to a single DataArray with a new 'variable' dimension and transpose.
    # shape is (n_time, n_variable, n_lat, n_lon)
    da = ds.to_array("variable").transpose('time', 'variable', 'latitude', 'longitude')

    return da

    # # stack to create da with shape (n_time, n_variable * n_lat * n_lon)
    # stacked = da.stack(features=('variable', 'latitude', 'longitude'))

# -------------------------------------------------------------------
# Helper functions: argument parsing, run_id generation, normalization, etc.
# -------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune MLP for regional post-processing')
    parser.add_argument('--forecast_path', type=str, required=True,
                        help='Path to forecast data (e.g. Zarr or NetCDF)')
    parser.add_argument('--obs_path', type=str, required=True,
                        help='Path to observation data (e.g. ERA5 Zarr or NetCDF)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the fine-tuned model and corrected forecasts')
    parser.add_argument('--model_name', type=str, default="pangu",
                        help='Name of the base model (e.g. pangu, ifs, neural_gcm)')
    parser.add_argument('--region', type=str, default="north_india",
                        help='Region to train over (e.g. full_india, north_india, uttar_pradesh)')
    parser.add_argument('--train_start', type=str, default='2018-01-01',
                        help='Training start date')
    parser.add_argument('--train_end', type=str, default='2019-12-30',
                        help='Training end date')
    parser.add_argument('--test_start', type=str, default='2020-01-01',
                        help='Test start date')
    parser.add_argument('--test_end', type=str, default='2020-12-31',
                        help='Test end date')
    parser.add_argument('--lead_time_hours', type=int, default=48,
                        help='Lead time in hours for forecast')
    parser.add_argument('--training_vars', type=str, nargs='+', default=["2m_temperature"],
                        help='Variables used to fine-tune (e.g. 2m_temperature precipitation)')
    parser.add_argument('--output_vars', type=str, nargs='+', default=["2m_temperature"],
                        help='Variables to fine-tune subset of training_vars (e.g. 2m_temperature)')
    parser.add_argument('--level', type=int, nargs='?', default=None,
                        help='Pressure level if applicable')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs')
    parser.add_argument('--mlp_hidden_dim', type=int, default=512,
                        help='Number of neurons in the hidden layers')
    parser.add_argument('--mlp_layers', type=int, default=5,
                        help='Number of hidden layers in the MLP')
    parser.add_argument('--use_cupy', action='store_true',
                        help='Enable GPU acceleration for xarray operations using cupy-xarray')
    parser.add_argument('--use_cloud_dataloader', action='store_true',
                        help='Enable cloud native dataloader using Dask and advanced PyTorch DataLoader options')
    return parser.parse_args()

def generate_run_id(args):
    region_str = f"{args.region}"
    dates_str = f"train{args.train_start}-{args.train_end}_test{args.test_start}-{args.test_end}"
    training_vars_str = "_".join(args.training_vars)
    output_vars_str = "_".join(args.output_vars)
    mlp_str = f"mlp{args.mlp_hidden_dim}x{args.mlp_layers}"
    run_id = f"{args.model_name}_{region_str}_{dates_str}_{args.lead_time_hours}h_train_{training_vars_str}_output{output_vars_str}_{mlp_str}"
    return run_id

def train_one_epoch(model, dataloader, optimizer, criterion, device, selected_indices):
    model.train()
    running_loss = 0.0
    for x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        predictions = model(x_batch)
        y_batch_subset = y_batch[:, selected_indices]
        loss = criterion(predictions, y_batch_subset)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x_batch.size(0)
    return running_loss / len(dataloader.dataset)

def validate_one_epoch(model, dataloader, criterion, device, selected_indices):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            predictions = model(x_batch)
            y_batch_subset = y_batch[:, selected_indices]
            loss = criterion(predictions, y_batch_subset)
            running_loss += loss.item() * x_batch.size(0)
    return running_loss / len(dataloader.dataset)

def train_model(model, train_loader, valid_loader, epochs, lr, device, selected_indices):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        epoch_start_time = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, selected_indices)
        valid_loss = validate_one_epoch(model, valid_loader, criterion, device, selected_indices)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")
        print(f"Time taken for epoch {epoch + 1}: {time.time() - epoch_start_time:.2f} seconds")
    return model

def apply_correction(model, forecast_data, device, output_vars, lat_vals, lon_vals):
    """
    Apply the trained model to correct forecast data.
    The input forecast_data is an xarray DataArray with dims ("time", "features")
    where "features" represents a flattened grid over (variable, latitude, longitude)
    for all training variables.
    
    The model outputs a numpy array of shape (n_time, len(output_vars) * n_lat * n_lon).
    This function builds a new DataArray for the corrected forecast with dims:
      ("time", "features")
    and with a new MultiIndex coordinate for "features" corresponding to the output variables.
    """
    model.eval()
    with torch.no_grad():
        # Convert the DataArray values (which are normalized) to a tensor.
        x_tensor = torch.from_numpy(forecast_data.values).float().to(device)
        corrected = model(x_tensor).cpu().numpy()

    # Determine grid dimensions.
    n_time = forecast_data.sizes["time"]
    n_lat = len(lat_vals)
    n_lon = len(lon_vals)
    grid_size = n_lat * n_lon
    n_out = len(output_vars)
    expected_output_dim = n_out * grid_size
    if corrected.shape[1] != expected_output_dim:
        raise ValueError(f"Model output shape {corrected.shape[1]} does not match expected {expected_output_dim}")

    # Create a new MultiIndex for the 'features' dimension.
    # Build a meshgrid for latitude and longitude.
    lon_mesh, lat_mesh = np.meshgrid(lon_vals, lat_vals)
    lat_flat = lat_mesh.flatten()
    lon_flat = lon_mesh.flatten()

    # For each output variable, repeat the grid points.
    variable_list = []
    for var in output_vars:
        variable_list.extend([var] * grid_size)

    import pandas as pd
    features_index = pd.MultiIndex.from_arrays(
        [variable_list, np.tile(lat_flat, n_out), np.tile(lon_flat, n_out)],
        names=["variable", "latitude", "longitude"]
    )

    # Create a new DataArray for the corrected forecast.
    corrected_da = xr.DataArray(
        corrected,
        dims=["time", "features"],
        coords={"time": forecast_data.time, "features": features_index}
    )
    return corrected_da


# def apply_correction(model, forecast_data, device):
#     """Apply the trained model to correct forecast data. forecast_data is an xarray data array
#     returns a corrected numpy array."""

#     model.eval()
#     with torch.no_grad():
#         x_tensor = torch.from_numpy(np.array(forecast_data)).float().to(device)
#         corrected = model(x_tensor).cpu().numpy()

#         corrected_da = xr.DataArray(corrected, dims=forecast_data.dims, coords=forecast_data.coords)
#     return corrected_da

def save_output(output_dir, run_id, training_vars, output_vars, lon_vals, lat_vals,
                time_vals, original_fc, corrected_fc, ground_truth_data=None):
    """
    Unstacks the stacked forecast DataArrays into an xarray.Dataset with human‐readable dimensions.
    It assumes that:
      - original_fc (and ground_truth_data, if provided) were stacked over a MultiIndex with levels
        ('variable', 'latitude', 'longitude').
      - corrected_fc has shape (time, len(output_vars) * (n_lat * n_lon)) but does not yet carry the multi-index.
    
    Parameters:
      output_dir (str): Directory to save the Zarr file.
      run_id (str): Unique identifier used in naming the output.
      training_vars (list of str): List of all variables in the original stacked data.
      output_vars (list of str): Subset of training_vars for which outputs are saved.
      lon_vals (array-like): Array of longitude coordinates.
      lat_vals (array-like): Array of latitude coordinates.
      time_vals (array-like): Array of time coordinates.
      original_fc (xarray.DataArray): Stacked original forecast with dims ('time', 'features').
      corrected_fc (xarray.DataArray): Stacked corrected forecast for output_vars.
      ground_truth_data (xarray.DataArray, optional): Stacked ground truth data.
    
    The function creates an xarray.Dataset with separate DataArrays for each output variable:
      - {var}_original
      - {var}_corrected
      - {var}_groundtruth (if provided)
    and then saves the dataset as a Zarr file.
    """
    import numpy as np
    import pandas as pd
    import xarray as xr
    import os

    # Unstack the original forecasts. We assume original_fc was created with:
    #   da.stack(features=('variable', 'latitude', 'longitude'))
    orig_unstacked = original_fc.unstack("features")
    # Subset to only the output variables (assuming the 'variable' coordinate holds the training_vars names)
    orig_subset = orig_unstacked.sel(variable=output_vars)

    if ground_truth_data is not None:
        gt_unstacked = ground_truth_data.unstack("features")
        gt_subset = gt_unstacked.sel(variable=output_vars)

    # Process the corrected forecasts.
    # Determine the grid dimensions from the coordinate arrays.
    n_lat = len(lat_vals)
    n_lon = len(lon_vals)
    grid_size = n_lat * n_lon
    expected_size = len(output_vars) * grid_size
    if corrected_fc.sizes["features"] != expected_size:
        print("Corrected forecast shape:", corrected_fc.sizes)
        print("Expected size:", expected_size)
        raise ValueError("The size of the 'features' dimension in corrected_fc does not match the expected grid size for output variables.")

    # Create a multi-index for the features of the corrected forecast.
    # Build meshgrids for latitude and longitude.
    lon_mesh, lat_mesh = np.meshgrid(lon_vals, lat_vals)
    lat_flat = lat_mesh.flatten()
    lon_flat = lon_mesh.flatten()

    # For each output variable, assign a block of grid points.
    variable_list = []
    # For each output variable, repeat it grid_size times.
    for var in output_vars:
        variable_list.extend([var] * grid_size)

    # For latitude and longitude, tile the grid for each variable.
    lat_index = np.tile(lat_flat, len(output_vars))
    lon_index = np.tile(lon_flat, len(output_vars))
    # Create a MultiIndex with levels: variable, latitude, and longitude.
    features_index = pd.MultiIndex.from_arrays(
        [variable_list, lat_index, lon_index],
        names=["variable", "latitude", "longitude"]
    )

    # Assign the new multi-index to corrected_fc.
    corrected_fc = corrected_fc.assign_coords(features=("features", features_index))
    corrected_unstacked = corrected_fc.unstack("features")

    # Build the output dataset by extracting, for each output variable, its original, corrected,
    # and (if available) ground truth fields.
    ds_dict = {}
    for var in output_vars:
        ds_dict[f"{var}_original"] = orig_subset.sel(variable=var)
        ds_dict[f"{var}_corrected"] = corrected_unstacked.sel(variable=var)
        if ground_truth_data is not None:
            ds_dict[f"{var}_groundtruth"] = gt_subset.sel(variable=var)

    ds_out = xr.Dataset(ds_dict)
    ds_out = ds_out.assign_coords(time=time_vals, latitude=lat_vals, longitude=lon_vals)
    ds_out.attrs["description"] = f"Original and corrected forecasts for run: {run_id}"

    output_filename = f"{run_id}.zarr"
    output_path = os.path.join(output_dir, output_filename)
    ds_out.to_zarr(output_path, mode="w")
    print(f"Forecasts saved to {output_path} (Zarr format)")



# -------------------------------------------------------------------
# Main function: using load_data() and xbatcher for efficient data loading.
# -------------------------------------------------------------------
def main():
    # Set device (GPU/MPS/CPU)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print("Device:", device)

    args = parse_args()
    run_id = generate_run_id(args)
    print("run id:", run_id)

    # Create output directory.
    if socket.gethostname() == "oMac.local":
        output_dir = os.path.expanduser(args.output_dir)
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Set region slices.
    if args.region == "full_india":
        # lat goes from 90 - -90 so list max first
        lat_slice = slice(27.25, 8.75)
        lon_slice = slice(70.75, 87.25)
    elif args.region == "north_india":
        lat_slice = slice(27.25, 21)
        lon_slice = slice(70.75, 87.25)
    elif args.region == "uttar_pradesh":
        lat_slice = slice(26, 24.25)
        lon_slice = slice(78, 87.25)
    else:
        print("Invalid region specified.")
        exit()

    # Configure Dask if using cloud dataloader.
    # OH: put in notebook and look at dashboard
    if args.use_cloud_dataloader:
        dask.config.set(scheduler="threads", num_workers=8)

    # ----------------------------------------------------------------
    # Load and pre-process training (and observation) data for train/val.
    # ----------------------------------------------------------------
    train_time_values = np.arange(np.datetime64(args.train_start), np.datetime64(args.train_end), np.timedelta64(24, 'h'))
    t0 = time.time()

    fc_train = load_data(
         args.forecast_path,
         vars=args.training_vars,
         level=args.level,
         lat_slice=lat_slice,
         lon_slice=lon_slice,
         times=train_time_values,
         lead_time_hours=args.lead_time_hours,
         use_consolidated=True,
         use_cupy=args.use_cupy
    )

    obs_train = load_data(
         args.obs_path,
         vars=args.output_vars, 
         level=args.level,
         lat_slice=lat_slice,
         lon_slice=lon_slice,
         times=train_time_values,
         lead_time_hours=args.lead_time_hours,
         use_consolidated=True,
         use_cupy=args.use_cupy
    )

    print("Time to load training data:", time.time() - t0)
    # ----------------------------------------------------------------
    # Compute normalization statistics from a small subset.
    # using 10 samples but should probably use more as I scale up
    # ----------------------------------------------------------------

    # If using cupy, bring 1000 random observations to CPU.
    # XX change 10 back to 1000
    if args.use_cupy:
        import cupy as cp
        random_indices = cp.random.choice(fc_train.shape[0], 10, replace=False)
        small_train_obs = cp.asnumpy(obs_train[random_indices])
    else:
        random_indices = np.random.choice(fc_train.shape[0], 10, replace=False)
        small_train_obs = obs_train[random_indices].compute() if hasattr(obs_train, 'compute') else obs_train[random_indices]
    # Compute mean and std along the 'time' dimension for each stacked feature
    print(small_train_obs)
    group_dims = ['variable', 'latitude', 'longitude']  # order of variables stacked in load data
    mean_da = small_train_obs.mean(dim='time')
    std_da = small_train_obs.std(dim='time')

    stats = {
        'mean': mean_da,
        'std': std_da,
        'group_dims': group_dims
    }

    # ----------------------------------------------------------------
    # Select input and output dimensions.
    # ----------------------------------------------------------------
    lon_vals = np.unique(fc_train.longitude.data)
    lat_vals = np.unique(fc_train.latitude.data)
    time_vals = np.unique(fc_train.time.data)

    # input dim: (n_time, n_training_vars * n_lat * n_lon)
    input_dim = len(args.training_vars) * len(lat_vals) * len(lon_vals)

    # output dim: (n_time, n_output_vars * n_lat * n_lon)
    output_dim = len(args.output_vars) * len(lat_vals) * len(lon_vals)

    print("input dim:", input_dim, "output dim:", output_dim)
    exit()

    #----------------------------------------------------------------
    # Create batch generator for training and validation data. and set up 
    # DataLoader for training and validation. Note that the batch size is set along the time dimension.
    #----------------------------------------------------------------

    # Create xbatcher batch generators for training.
    # Set the batch size along the time dimension.
    batch_size = 16
    X_bgen_train = xb.BatchGenerator(fc_train, input_dims={'time': batch_size}, preload_batch=args.use_cloud_dataloader)
    y_bgen_train = xb.BatchGenerator(obs_train, input_dims={'time': batch_size}, preload_batch=args.use_cloud_dataloader)
    # X_bgen_train = xb.BatchGenerator(fc_train, input_dims={'time': batch_size})
    # y_bgen_train = xb.BatchGenerator(obs_train, input_dims={'time': batch_size})
    # Map xbatcher generators to a PyTorch-compatible dataset.
    train_val_dataset = xb.loaders.torch.MapDataset(X_bgen_train, y_bgen_train)

    # Split train_val_dataset into training and validation subsets.
    total_samples = len(train_val_dataset)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [train_size, val_size])

    # Wrap the split datasets so that normalization is applied on-the-fly.
    train_dataset = NormalizedDataset(train_dataset, stats)
    val_dataset = NormalizedDataset(val_dataset, stats)
    
    # Configure DataLoader parameters.
    if args.use_cloud_dataloader:
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=None,  # batching is handled by xbatcher
            shuffle=True,
            num_workers=8,
            prefetch_factor=1,
            persistent_workers=True,
            multiprocessing_context="forkserver"
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=8,
            prefetch_factor=1,
            persistent_workers=True,
            multiprocessing_context="forkserver"
        )
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=None, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=None, shuffle=False)
    print("Training batches:", len(train_dataset), "Validation batches:", len(val_dataset))




    #----------------------------------------------------------------
    # Initialize the model and train
    #----------------------------------------------------------------

    # Initialize and train the model.
    model = SimpleMLP(input_dim=input_dim,
                      hidden_dim=args.mlp_hidden_dim,
                      output_dim=output_dim,
                      num_hidden_layers=args.mlp_layers)
    model.to(device)

    model = train_model(model, train_loader, val_loader, epochs=args.epochs,
                        lr=1e-5, device=device, selected_indices=selected_indices)

    model_path = os.path.join(output_dir, f"{args.model_name}_mlp_correction.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model weights saved to {model_path}")

    # load the model weights
    model.load_state_dict(torch.load(model_path, map_location=device))


    # ----------------------------------------------------------------
    # Load and pre-process test data and apply correction.
    # ----------------------------------------------------------------
    test_time_values = np.arange(np.datetime64(args.test_start), np.datetime64(args.test_end), np.timedelta64(24, 'h'))
    fc_test = load_data(
         args.forecast_path,
         training_vars=args.training_vars,
         level=args.level,
         lat_slice=lat_slice,
         lon_slice=lon_slice,
         times=test_time_values,
         lead_time_hours=args.lead_time_hours,
         use_consolidated=True,
         use_cupy=args.use_cupy
    ).persist()

    obs_test = load_data(
         args.obs_path,
         training_vars=args.training_vars,
         level=args.level,
         lat_slice=lat_slice,
         lon_slice=lon_slice,
         times= test_time_values,
         lead_time_hours=args.lead_time_hours,
         use_consolidated=True,
         use_cupy=args.use_cupy
    ).persist()

    # For testing, if using cupy, work directly on GPU arrays; otherwise, compute lazy arrays.
    if args.use_cupy:
        test_fc = fc_test
        test_obs = obs_test
    else:
        test_fc = fc_test.compute() if hasattr(fc_test, 'compute') else fc_test
        test_obs = obs_test.compute() if hasattr(obs_test, 'compute') else obs_test

    # Retrieve coordinate information from the forecast dataset.
    lon_vals = np.unique(fc_test.longitude.data)
    lat_vals = np.unique(obs_test.latitude.data)
    test_time_vals = np.unique(obs_test.time.data)

    print(test_fc)
    corrected_test_fc_norm = apply_correction(model, test_fc, device, args.output_vars, lat_vals, lon_vals)
    print(corrected_test_fc_norm)
    # use stats from before to denormalize
    # corrected_test_fc = corrected_test_fc_norm * stats['std'].values + stats['mean'].values
    stats_mean_unstacked = stats['mean'].unstack("features")
    stats_std_unstacked = stats['std'].unstack("features")
    stats_mean_subset = stats_mean_unstacked.sel(variable=args.output_vars).stack(features=("variable", "latitude", "longitude"))
    stats_std_subset = stats_std_unstacked.sel(variable=args.output_vars).stack(features=("variable", "latitude", "longitude"))

    corrected_test_fc = corrected_test_fc_norm * stats_std_subset + stats_mean_subset

    save_output(
        output_dir=output_dir,
        run_id=run_id,
        training_vars=args.training_vars,
        output_vars=args.output_vars,
        lon_vals=lon_vals,
        lat_vals=lat_vals,
        time_vals=test_time_vals,
        original_fc=(cp.asnumpy(test_fc) if args.use_cupy else test_fc),
        corrected_fc=corrected_test_fc,
        ground_truth_data=(cp.asnumpy(test_obs) if args.use_cupy else test_obs)
    )

if __name__ == "__main__":
    main()
