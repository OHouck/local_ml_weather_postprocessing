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
def load_data(zarr_path, training_vars, level, lat_slice, lon_slice, time_slice, lead_time_hours, use_consolidated=True, use_cupy=False):
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
    ds = ds.sel(time=time_slice, latitude=lat_slice, longitude=lon_slice)[training_vars]

    # In forecast, select the desired lead time and drop the coordinate if present.
    if 'prediction_timedelta' in ds.coords:
        ds = ds.sel(prediction_timedelta=np.timedelta64(lead_time_hours, 'h'))
        ds = ds.drop_vars('prediction_timedelta')

    # Slice level if specified.
    if level is not None:
        ds = ds.sel(level=level)

    # Convert to a single DataArray with a new 'variable' dimension and transpose.
    da = ds.to_array("variable").transpose('time', 'variable', 'latitude', 'longitude')
    # da = da.compute() # Uncomment if you want to compute the lazy array immediately.
    stacked = da.stack(features=('variable', 'latitude', 'longitude'))

    # Stack spatial and variable dimensions into one 'features' dimension.
    original_shape = da.shape  # e.g. (time, variable, latitude, longitude)
    return stacked
    # return stacked, original_shape, ds

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

def compute_mean_sd(da, group_dims=['variable', 'latitude', 'longitude']):
    """Compute mean and standard deviation for normalization for each variable
    across time."""
        # Check if data is already stacked: if 'features' exists and the group_dims are in its coordinates.
    if 'features' in da.dims and all(dim in da.coords for dim in group_dims):
        da_stacked = da  # Already stacked.
    else:
        da_stacked = da.stack(features=group_dims)
    
    # Compute mean and std along the 'time' dimension for each stacked feature
    mean_da = da_stacked.mean(dim='time')
    std_da = da_stacked.std(dim='time')
    
    stats = {
        'mean': mean_da,
        'std': std_da,
        'group_dims': group_dims
    }
    return stats

def unnormalize_data(da_norm, stats):
    # Stack dimensions to match stats
    da_stacked = da_norm.stack(features=stats['group_dims'])
    mean_da = stats['mean']
    std_da = stats['std']
    
    # Unnormalize
    da_unnorm = da_stacked * (std_da + 1e-8) + mean_da
    return da_unnorm.unstack('features')

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

def apply_correction(model, forecast_data, device):
    model.eval()
    with torch.no_grad():
        x_tensor = torch.from_numpy(np.array(forecast_data)).float().to(device)
        corrected = model(x_tensor).cpu().numpy()
    return corrected

def apply_correction(model, forecast_data, device):
    model.eval()
    with torch.no_grad():
        # If forecast_data is a CuPy array, convert via DLpack to avoid copying to CPU.
        if cp is not None and isinstance(forecast_data, cp.ndarray):
            x_tensor = torch.utils.dlpack.from_dlpack(forecast_data.toDlpack()).to(device)
            x_tensor = x_tensor.float()  # Ensure tensor is float32.
        else:
            # Use torch.as_tensor to avoid unnecessary copies if data is already a NumPy array.
            x_tensor = torch.as_tensor(forecast_data, dtype=torch.float32, device=device)
        corrected = model(x_tensor).cpu().numpy()
    return corrected


def save_output(output_dir, run_id, output_vars, lon_vals, lat_vals,
                time_vals, original_fc, corrected_fc, original_shape,
                ground_truth_data=None):
    if len(original_shape) == 4:
        n_time, n_vars, n_lat, n_lon = original_shape
    else:
        n_time, n_lon, n_lat = original_shape
        n_vars = 1
        original_fc = original_fc.reshape(n_time, n_vars * n_lon * n_lat)
        corrected_fc = corrected_fc.reshape(n_time, n_vars * n_lon * n_lat)
    ds_dict = {}
    for i in range(n_vars):
        start = i * (n_lat * n_lon)
        end = (i + 1) * (n_lat * n_lon)
        orig_slice = original_fc[:, start:end].reshape(n_time, n_lat, n_lon)
        corr_slice = corrected_fc[:, start:end].reshape(n_time, n_lat, n_lon)
        da_orig = xr.DataArray(
            data=orig_slice,
            coords=[time_vals, lat_vals, lon_vals],
            dims=['time', 'latitude', 'longitude'],
            name=f"{output_vars[i]}_original"
        )
        da_corr = xr.DataArray(
            data=corr_slice,
            coords=[time_vals, lat_vals, lon_vals],
            dims=['time', 'latitude', 'longitude'],
            name=f"{output_vars[i]}_corrected"
        )
        ds_dict[f"{output_vars[i]}_original"] = da_orig
        ds_dict[f"{output_vars[i]}_corrected"] = da_corr
        if ground_truth_data is not None:
            gt_slice = ground_truth_data[:, start:end].reshape(n_time, n_lat, n_lon)
            da_gt = xr.DataArray(
                data=gt_slice,
                coords=[time_vals, lat_vals, lon_vals],
                dims=['time', 'latitude', 'longitude'],
                name=f"{output_vars[i]}_groundtruth"
            )
            ds_dict[f"{output_vars[i]}_groundtruth"] = da_gt
    ds_out = xr.Dataset(ds_dict)
    ds_out.attrs['description'] = f'Original and corrected forecasts for run: {run_id}'
    output_filename = f"{run_id}.zarr"
    output_path = os.path.join(output_dir, output_filename)
    ds_out.to_zarr(output_path, mode='w')
    print(f"Forecasts saved to {output_path} (Zarr format)")

def save_output(output_dir, run_id, output_vars, lon_vals, lat_vals,
                time_vals, original_fc, corrected_fc, original_shape,
                ground_truth_data=None):
    # Determine the proper shape.
    if len(original_shape) == 4:
        n_time, n_vars, n_lat, n_lon = original_shape
        original_fc = original_fc.reshape(n_time, n_vars, n_lat, n_lon)
        corrected_fc = corrected_fc.reshape(n_time, n_vars, n_lat, n_lon)
        if ground_truth_data is not None:
            ground_truth_data = ground_truth_data.reshape(n_time, n_vars, n_lat, n_lon)
    else:
        # Assume shape is (time, n_lon, n_lat) and there is only one variable.
        n_time, n_lon, n_lat = original_shape
        n_vars = 1
        original_fc = original_fc.reshape(n_time, 1, n_lat, n_lon)
        corrected_fc = corrected_fc.reshape(n_time, 1, n_lat, n_lon)
        if ground_truth_data is not None:
            ground_truth_data = ground_truth_data.reshape(n_time, 1, n_lat, n_lon)

    ds_dict = {}
    # Loop over variables (usually a small number) and construct DataArrays.
    for i in range(n_vars):
        # Use output_vars if available, otherwise fall back to a default naming.
        orig_name = f"{output_vars[i]}_original" if i < len(output_vars) else f"var{i}_original"
        corr_name = f"{output_vars[i]}_corrected" if i < len(output_vars) else f"var{i}_corrected"
        da_orig = xr.DataArray(
            data=original_fc[:, i, :, :],
            coords=[time_vals, lat_vals, lon_vals],
            dims=['time', 'latitude', 'longitude'],
            name=orig_name
        )
        da_corr = xr.DataArray(
            data=corrected_fc[:, i, :, :],
            coords=[time_vals, lat_vals, lon_vals],
            dims=['time', 'latitude', 'longitude'],
            name=corr_name
        )
        ds_dict[orig_name] = da_orig
        ds_dict[corr_name] = da_corr

        if ground_truth_data is not None:
            gt_name = f"{output_vars[i]}_groundtruth" if i < len(output_vars) else f"var{i}_groundtruth"
            da_gt = xr.DataArray(
                data=ground_truth_data[:, i, :, :],
                coords=[time_vals, lat_vals, lon_vals],
                dims=['time', 'latitude', 'longitude'],
                name=gt_name
            )
            ds_dict[gt_name] = da_gt

    ds_out = xr.Dataset(ds_dict)
    ds_out.attrs['description'] = f'Original and corrected forecasts for run: {run_id}'
    output_filename = f"{run_id}.zarr"
    output_path = os.path.join(output_dir, output_filename)
    ds_out.to_zarr(output_path, mode='w')
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
        lat_slice = slice(35.5, 21)
        lon_slice = slice(70.75, 87.25)
    elif args.region == "uttar_pradesh":
        lat_slice = slice(26, 24.25)
        lon_slice = slice(78, 87.25)
    else:
        print("Invalid region specified.")
        exit()
    n_vars = len(args.training_vars)

    # Configure Dask if using cloud dataloader.
    if args.use_cloud_dataloader:
        dask.config.set(scheduler="threads", num_workers=16)

    # ----------------------------------------------------------------
    # Load and pre-process training (and observation) data for train/val.
    # ----------------------------------------------------------------
    train_time_slice = slice(args.train_start, args.train_end)
    t0 = time.time()

    # fc_train, orig_shape_train, ds_forecast_train = load_data(
    #      args.forecast_path,
    #      training_vars=args.training_vars,
    #      level=args.level,
    #      lat_slice=lat_slice,
    #      lon_slice=lon_slice,
    #      time_slice=train_time_slice,
    #      lead_time_hours=args.lead_time_hours,
    #      use_consolidated=True,
    #      use_cupy=args.use_cupy
    # )

    fc_train = load_data(
         args.forecast_path,
         training_vars=args.training_vars,
         level=args.level,
         lat_slice=lat_slice,
         lon_slice=lon_slice,
         time_slice=train_time_slice,
         lead_time_hours=args.lead_time_hours,
         use_consolidated=True,
         use_cupy=args.use_cupy
    )

    obs_train = load_data(
         args.obs_path,
         training_vars=args.training_vars,
         level=args.level,
         lat_slice=lat_slice,
         lon_slice=lon_slice,
         time_slice=train_time_slice,
         lead_time_hours=args.lead_time_hours,
         use_consolidated=True,
         use_cupy=args.use_cupy
    )
    print("Time to load training data:", time.time() - t0)
    # ----------------------------------------------------------------
    # Compute normalization statistics from a small subset.
    # using 10 samples but should probably use more as I scale up
    # ----------------------------------------------------------------
    # If using cupy, bring small subset to CPU.
    if args.use_cupy:
        small_train_obs = cp.asnumpy(obs_train[:10])
    else:
        small_train_obs = obs_train[:10].compute() if hasattr(obs_train, 'compute') else obs_train[:10]

    # Compute mean and std along the 'time' dimension for each stacked feature
    group_dims = ['variable', 'latitude', 'longitude']  # order of variables stacked in load data
    mean_da = small_train_obs.mean(dim='time')
    std_da = small_train_obs.std(dim='time')

    stats = {
        'mean': mean_da,
        'std': std_da,
        'group_dims': group_dims
    }

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


    # ----------------------------------------------------------------
    # Set up training targets. Note that we might not care to predict all of the 
    # training_vars, only the subset in output_vars
    # ----------------------------------------------------------------
    spatial_dim = small_train_obs.shape[1] // len(args.training_vars)
    selected_indices = []
    for i, var in enumerate(args.training_vars):
        if var in args.output_vars:
            selected_indices.extend(list(range(i * spatial_dim, (i + 1) * spatial_dim)))
    input_dim = small_train_obs.shape[1]
    output_dim = len(selected_indices)

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


    # ----------------------------------------------------------------
    # Load and pre-process test data and apply correction.
    # ----------------------------------------------------------------
    test_time_slice = slice(args.test_start, args.test_end)
    fc_test = load_data(
         args.forecast_path,
         training_vars=args.training_vars,
         level=args.level,
         lat_slice=lat_slice,
         lon_slice=lon_slice,
         time_slice=test_time_slice,
         lead_time_hours=args.lead_time_hours,
         use_consolidated=True,
         use_cupy=args.use_cupy
    )

    obs_test = load_data(
         args.obs_path,
         training_vars=args.training_vars,
         level=args.level,
         lat_slice=lat_slice,
         lon_slice=lon_slice,
         time_slice=test_time_slice,
         lead_time_hours=args.lead_time_hours,
         use_consolidated=True,
         use_cupy=args.use_cupy
    )

    orig_shape_test = fc_test.shape  # e.g. (time, variable, latitude, longitude)

    # For testing, if using cupy, work directly on GPU arrays; otherwise, compute lazy arrays.
    if args.use_cupy:
        test_fc = fc_test
        test_obs = obs_test
    else:
        test_fc = fc_test.compute() if hasattr(fc_test, 'compute') else fc_test
        test_obs = obs_test.compute() if hasattr(obs_test, 'compute') else obs_test

    # Retrieve coordinate information from the forecast dataset.
    lon_vals = fc_test.longitude.data 
    lat_vals = obs_test.latitude.data 
    test_time_vals = obs_test.time.data
    test_original_shape = orig_shape_test

    corrected_test_fc_norm = apply_correction(model, test_fc, device)
    corrected_test_fc = unnormalize_data(corrected_test_fc_norm, stats)

    if args.use_cupy:
        mse_orig = np.mean((cp.asnumpy(test_fc) - cp.asnumpy(test_obs)) ** 2)
    else:
        mse_orig = np.mean((test_fc - test_obs) ** 2)
    mse_corr = np.mean((corrected_test_fc - (cp.asnumpy(test_obs) if args.use_cupy else test_obs)) ** 2)
    print(f"MSE (original forecast, test set): {mse_orig:.6f}")
    print(f"MSE (corrected forecast, test set): {mse_corr:.6f}")

    save_output(
        output_dir=output_dir,
        run_id=run_id,
        output_vars=args.output_vars,
        lon_vals=lon_vals,
        lat_vals=lat_vals,
        time_vals=test_time_vals,
        original_fc=(cp.asnumpy(test_fc) if args.use_cupy else test_fc),
        corrected_fc=corrected_test_fc,
        original_shape=test_original_shape,
        ground_truth_data=(cp.asnumpy(test_obs) if args.use_cupy else test_obs)
    )

if __name__ == "__main__":
    main()
