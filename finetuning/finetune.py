#!/usr/bin/env python3
"""
weatherbench2_finetuning.py
Author: Ozzy Houck 
Date: 12/20/2024 (modified 2025-03-11)

This script fine-tunes an MLP correction model to a specific region using model
forecasts and corresponding observations from weatherbench2.
----------------------------

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
    --mlp_layers=5
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
from torch.utils.data import DataLoader, Dataset
import pandas as pd  

# -------------------------------------------------------------------
# Simple MLP model
# -------------------------------------------------------------------

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=1, num_hidden_layers=3):
        super(SimpleMLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# -------------------------------------------------------------------
# ForecastDataset: Loads forecast and observation data into a stacked array.
# -------------------------------------------------------------------
class ForecastDataset(Dataset):
    def __init__(self, forecast_path, obs_path, training_vars, level, lat_slice, lon_slice, time_slice, lead_time_hours, use_consolidated=True):
        # Use consolidated metadata for faster metadata loading on cloud storage.
        open_zarr_kwargs = {'decode_timedelta': True}
        if use_consolidated:
            open_zarr_kwargs['consolidated'] = True

        if isinstance(forecast_path, str):
            self.ds_forecast = xr.open_zarr(forecast_path, **open_zarr_kwargs)
        else:
            self.ds_forecast = forecast_path

        if isinstance(obs_path, str):
            self.ds_obs = xr.open_zarr(obs_path, **open_zarr_kwargs)
        else:
            self.ds_obs = obs_path

        self.training_vars = training_vars
        self.level = level
        self.lat_slice = lat_slice
        self.lon_slice = lon_slice
        self.time_slice = time_slice
        self.lead_time_hours = lead_time_hours

        # slice by time
        self.ds_forecast = self.ds_forecast.sel(time=time_slice)
        self.ds_obs = self.ds_obs.sel(time=time_slice)

        # slice to only include variables in training_vars
        self.ds_forecast = self.ds_forecast[self.training_vars]
        self.ds_obs = self.ds_obs[self.training_vars]

        # keep only lead time of interest in forecast
        self.ds_forecast = self.ds_forecast.sel(prediction_timedelta=np.timedelta64(lead_time_hours, 'h'))
        if 'prediction_timedelta' in self.ds_forecast.coords:
            self.ds_forecast= self.ds_forecast.drop_vars('prediction_timedelta')

        # if level is not None, slice by level
        if level is not None:
            self.ds_forecast = self.ds_forecast.sel(level=level)
            self.ds_obs = self.ds_obs.sel(level=level)

        # Check if latitude coordinate is sorted; if not, sort it.
        # This takes a long time for large datasets.
        # lat_forecast = self.ds_forecast['latitude'].values
        # lon_forecast = self.ds_forecast['longitude'].values
        # if not (np.all(np.diff(lat_forecast) >= 0) and np.all(np.diff(lon_forecast) >= 0)):
        #     print("Forecast Latitude or Longitude coordinate is not sorted. Sorting...")
        #     self.ds_forecast = self.ds_forecast.sortby(['latitude', 'longitude'])

        # lat_obs = self.ds_obs['latitude'].values
        # lon_obs = self.ds_obs['longitude'].values
        # if not (np.all(np.diff(lat_obs) >= 0) and np.all(np.diff(lon_obs) >= 0)):
        #     print("Observation Latitude or Longitude coordinate is not sorted. Sorting...")
        #     self.ds_obs = self.ds_obs.sortby(['latitude', 'longitude'])

        # # Select forecast and observation variables.
        # fc_vars = []
        # obs_vars = []
        # for v in self.training_vars:
        #     if v not in self.ds_forecast:
        #         print(f"Variable '{v}' not found in forecast dataset. Skipping...")
        #         continue
        #     if v not in self.ds_obs:
        #         print(f"Variable '{v}' not found in obs dataset. Skipping...")
        #         continue
        #     if 'level' in self.ds_forecast[v].dims and level is not None:
        #         fc_var = self.ds_forecast[v].sel(time=time_slice,
        #                                          latitude=lat_slice,
        #                                          longitude=lon_slice,
        #                                          level=level)
        #     else:
        #         fc_var = self.ds_forecast[v].sel(time=time_slice,
        #                                          latitude=lat_slice,
        #                                          longitude=lon_slice)

        #         print(v)
        #         print(fc_var)
        #         exit()
        #     if 'level' in self.ds_obs[v].dims and level is not None:
        #         obs_var = self.ds_obs[v].sel(time=time_slice,
        #                                      latitude=lat_slice,
        #                                      longitude=lon_slice,
        #                                      level=level)
        #     else:
        #         obs_var = self.ds_obs[v].sel(time=time_slice,
        #                                      latitude=lat_slice,
        #                                      longitude=lon_slice)
        #     # Select the desired lead time and drop the coordinate to simplify the data.
        #     fc_var = fc_var.sel(prediction_timedelta=np.timedelta64(lead_time_hours, 'h'))

        #     exit()

        #     if 'prediction_timedelta' in fc_var.coords:
        #         fc_var = fc_var.drop_vars('prediction_timedelta')
        #     fc_vars.append(fc_var)
        #     obs_vars.append(obs_var)

        # t0 = time.time()
        # # Create new Datasets from the selected DataArrays.
        # fc_ds = xr.Dataset({var: fc for var, fc in zip(self.training_vars, fc_vars)})
        # obs_ds = xr.Dataset({var: obs for var, obs in zip(self.training_vars, obs_vars)})

        # Convert to a single DataArray with a new 'variable' dimension.
        fc_concat = self.ds_forecast.to_array("variable").transpose('time', 'variable', 'latitude', 'longitude')
        obs_concat = self.ds_obs.to_array("variable").transpose('time', 'variable', 'latitude', 'longitude')
        
        # If coordinates already match, skip alignment to reduce graph complexity.
        coords_match = (
            np.array_equal(fc_concat.time.values, obs_concat.time.values) and 
            np.array_equal(fc_concat.latitude.values, obs_concat.latitude.values) and 
            np.array_equal(fc_concat.longitude.values, obs_concat.longitude.values)
        )
        if not coords_match:
            print("Aligning forecast and observation datasets...")
            fc_concat, obs_concat = xr.align(fc_concat, obs_concat, join='inner')

        # Store the original shape for later unstacking.
        self.original_shape = fc_concat.shape

        # Stack the non-time dimensions into a single "features" dimension.
        self.fc_data = fc_concat.stack(features=('variable', 'latitude', 'longitude')).data
        self.obs_data = obs_concat.stack(features=('variable', 'latitude', 'longitude')).data

        # Coordinates (they are typically small and can be computed eagerly).
        self.lon_vals = fc_concat.longitude.data
        self.lat_vals = fc_concat.latitude.data
        self.time_vals = fc_concat.time.data

    def __len__(self):
        # Return the number of time steps.
        return self.fc_data.shape[0]

    def __getitem__(self, index):
        # Return one sample (forecast, observation) pair.
        fc_sample = self.fc_data[index].compute() if hasattr(self.fc_data, 'compute') else self.fc_data[index]
        obs_sample = self.obs_data[index].compute() if hasattr(self.obs_data, 'compute') else self.obs_data[index]
        return fc_sample, obs_sample

# -------------------------------------------------------------------
# Other helper functions 
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
    return parser.parse_args()

def generate_run_id(args):
    region_str = f"{args.region}"
    dates_str = f"train{args.train_start}-{args.train_end}_test{args.test_start}-{args.test_end}"
    training_vars_str = "_".join(args.training_vars)
    output_vars_str = "_".join(args.output_vars)
    mlp_str = f"mlp{args.mlp_hidden_dim}x{args.mlp_layers}"
    run_id = f"{args.model_name}_{region_str}_{dates_str}_{args.lead_time_hours}h_train_{training_vars_str}_output{output_vars_str}_{mlp_str}"
    return run_id

def check_dataset_alignment(ds1, ds2, coord_vars=['time', 'latitude', 'longitude']):
    """Helper function to check alignment of coordinates between two datasets."""
    misaligned = []
    for coord in coord_vars:
        if coord in ds1.coords and coord in ds2.coords:
            if not np.array_equal(ds1[coord].values, ds2[coord].values):
                misaligned.append(coord)
        else:
            misaligned.append(coord)
    if misaligned:
        print(f"Warning: The following coordinates are not aligned between forecast and observation datasets: {misaligned}")
    else:
        print("Datasets are aligned on coordinates:", coord_vars)

def create_dataloader(forecast_data, obs_data, batch_size):
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(forecast_data).float(),
        torch.from_numpy(obs_data).float()
    )
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True)
    return dataloader

def normalize_data(train_fc, val_fc, train_obs, val_obs, n_vars):
    space_dim = train_fc.shape[1] // n_vars
    n_time = train_fc.shape[0]

    # Forecast data stats.
    train_fc_reshaped = train_fc.reshape(n_time, n_vars, space_dim)
    val_fc_reshaped = val_fc.reshape(val_fc.shape[0], n_vars, space_dim)
    mean_fc = train_fc_reshaped.mean(axis=(0,2), keepdims=True)
    std_fc = train_fc_reshaped.std(axis=(0,2), keepdims=True)
    train_fc_norm = ((train_fc_reshaped - mean_fc) / (std_fc + 1e-8)).reshape(train_fc.shape)
    val_fc_norm = ((val_fc_reshaped - mean_fc) / (std_fc + 1e-8)).reshape(val_fc.shape)

    # Observation data stats.
    train_obs_reshaped = train_obs.reshape(n_time, n_vars, space_dim)
    val_obs_reshaped = val_obs.reshape(val_obs.shape[0], n_vars, space_dim)
    mean_obs = train_obs_reshaped.mean(axis=(0,2), keepdims=True)
    std_obs = train_obs_reshaped.std(axis=(0,2), keepdims=True)
    train_obs_norm = ((train_obs_reshaped - mean_obs) / (std_obs + 1e-8)).reshape(train_obs.shape)
    val_obs_norm = ((val_obs_reshaped - mean_obs) / (std_obs + 1e-8)).reshape(val_obs.shape)

    stats = {
        'mean_fc': np.atleast_1d(mean_fc.squeeze()),
        'std_fc': np.atleast_1d(std_fc.squeeze()),
        'mean_obs': np.atleast_1d(mean_obs.squeeze()),
        'std_obs': np.atleast_1d(std_obs.squeeze()),
        'n_vars': n_vars,
        'space_dim': space_dim
    }
    return train_fc_norm, val_fc_norm, train_obs_norm, val_obs_norm, stats

def unnormalize_data(corrected_norm, stats, is_obs=True):
    n_time = corrected_norm.shape[0]
    n_vars = stats['n_vars']
    space_dim = stats['space_dim']
    corrected_norm_reshaped = corrected_norm.reshape(n_time, n_vars, space_dim)
    if is_obs:
        unnorm = corrected_norm_reshaped * (stats['std_obs'][None, :, None] + 1e-8) + stats['mean_obs'][None, :, None]
    else:
        unnorm = corrected_norm_reshaped * (stats['std_fc'][None, :, None] + 1e-8) + stats['mean_fc'][None, :, None]
    return unnorm.reshape(corrected_norm.shape)

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
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, selected_indices)
        valid_loss = validate_one_epoch(model, valid_loader, criterion, device, selected_indices)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")
    return model

def apply_correction(model, forecast_data, device):
    model.eval()
    with torch.no_grad():
        x_tensor = torch.from_numpy(forecast_data).float().to(device)
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

# -------------------------------------------------------------------
# Main function: using ForecastDataset for train/val/test splits.
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
        lat_slice = slice(8.68, 27.2)
        lon_slice = slice(70.75, 87.35)
    elif args.region == "north_india":
        lat_slice = slice(21, 35.5)
        lon_slice = slice(70.75, 87.35)
    elif args.region == "uttar_pradesh":
        lat_slice = slice(24.2, 26)
        lon_slice = slice(78, 87.35)
    else:
        print("Invalid region specified.")
        exit()
    n_vars = len(args.training_vars)

    # ----------------------------------------------------------------
    # Create training/validation dataset using ForecastDataset.
    # ----------------------------------------------------------------
    train_time_slice = slice(args.train_start, args.train_end)

    t0 = time.time()
    train_val_dataset = ForecastDataset(
         forecast_path=args.forecast_path,
         obs_path=args.obs_path,
         training_vars=args.training_vars,
         level=args.level,
         lat_slice=lat_slice,
         lon_slice=lon_slice,
         time_slice=train_time_slice,
         lead_time_hours=args.lead_time_hours
    )
    print("Time to load dataset:", time.time() - t0)

    # Split dataset (e.g., 80/20 train/val split).
    train_size = int(0.8 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    print("Using ForecastDataset for training. Total training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    # ----------------------------------------------------------------
    # Compute normalization statistics from a small subset of training data.
    # Here we use the first 10 samples.
    # ----------------------------------------------------------------
    small_train_fc = train_val_dataset.fc_data[:10].compute() if hasattr(train_val_dataset.fc_data, 'compute') else train_val_dataset.fc_data[:10]
    small_train_obs = train_val_dataset.obs_data[:10].compute() if hasattr(train_val_dataset.obs_data, 'compute') else train_val_dataset.obs_data[:10]
    _, _, _, _, stats = normalize_data(small_train_fc, small_train_fc, small_train_obs, small_train_obs, n_vars)

    # ----------------------------------------------------------------
    # Set up training targets.
    # ----------------------------------------------------------------
    spatial_dim = small_train_fc.shape[1] // len(args.training_vars)
    selected_indices = []
    for i, var in enumerate(args.training_vars):
        if var in args.output_vars:
            selected_indices.extend(list(range(i * spatial_dim, (i + 1) * spatial_dim)))
    input_dim = small_train_fc.shape[1]
    output_dim = len(selected_indices)

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
    # Create test dataset and apply correction.
    # ----------------------------------------------------------------
    test_time_slice = slice(args.test_start, args.test_end)
    test_dataset = ForecastDataset(
         forecast_path=args.forecast_path,
         obs_path=args.obs_path,
         training_vars=args.training_vars,
         level=args.level,
         lat_slice=lat_slice,
         lon_slice=lon_slice,
         time_slice=test_time_slice,
         lead_time_hours=args.lead_time_hours
    )
    test_fc = test_dataset.fc_data.compute() if hasattr(test_dataset.fc_data, 'compute') else test_dataset.fc_data
    test_obs = test_dataset.obs_data.compute() if hasattr(test_dataset.obs_data, 'compute') else test_dataset.obs_data
    lon_vals = test_dataset.lon_vals
    lat_vals = test_dataset.lat_vals
    test_time_vals = test_dataset.time_vals
    test_original_shape = test_dataset.original_shape

    corrected_test_fc_norm = apply_correction(model, test_fc, device)
    corrected_test_fc = unnormalize_data(corrected_test_fc_norm, stats, is_obs=True)
    print(f"MSE (original forecast, test set): {np.mean((test_fc - test_obs) ** 2):.6f}")
    print(f"MSE (corrected forecast, test set): {np.mean((corrected_test_fc - test_obs) ** 2):.6f}")

    save_output(
        output_dir=output_dir,
        run_id=run_id,
        output_vars=args.output_vars,
        lon_vals=lon_vals,
        lat_vals=lat_vals,
        time_vals=test_time_vals,
        original_fc=test_fc,
        corrected_fc=corrected_test_fc,
        original_shape=test_original_shape,
        ground_truth_data=test_obs
    )

if __name__ == "__main__":
    main()
