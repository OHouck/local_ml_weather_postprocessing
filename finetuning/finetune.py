#!/usr/bin/env python3
"""
Author: Ozma Houck 

# Purpose: use a simple MLP to post-process weather forecasts trained on
specific regions and variables. Call this script from command line or with 
1_run_experiments.sh script. 

# Modified to support multiple lead times training

# example call
python3 finetuning/1_finetune.py \
    --data_dir="/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/data/raw/ \
    --output_dir="/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/data/fine_tuning_output" \
    --training_vars 2m_temperature \
    --output_vars 2m_temperature \
    --train_start="2018-01-01" --train_end="2021-12-31" \
    --test_start="2022-01-01" --test_end="2022-12-31" \
    --model_name="pangu" \
    --region="india" \
    --subregion="2x2" \
    --lead_time_hours 144 168
"""
import argparse
import os
import socket
import random
import glob
import math
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
from xarray.coding.times import CFDatetimeCoder
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import time

# Map the new region strings to Koppen‐Geiger codes:
CLIMATE_ZONE_MAP = {
    'tropical':  1,
    'arid':       2,
    'temperate':  3,
    'cold':       4,
    'polar':      5,
}

# Purpose: save patches of of climate zones to be used for bootstrapping
def setup_directories():
    # Determine root directory based on environment.
    nodename = socket.gethostname()
    if nodename == "oMac.local":  # local laptop
        root = os.path.expanduser(
            "~/OneDrive - The University of Chicago/ai_weather_ag/data"
        )
    else:
        raise Exception("Unknown environment, please specify the root directory")

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
# ------------------------------
# Simple MLP definition with lead time and month encoding
# ------------------------------
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, output_dim=1, num_hidden_layers=2, 
                n_lead_times=1, lead_time_embedding_dim=8, month_embedding_dim=16,
                dropout_rate=0.0):
        super(SimpleMLP, self).__init__()
        
        # Lead time embedding
        self.n_lead_times = n_lead_times
        self.lead_time_embedding = None
        
        # Month embedding (12 months)
        self.month_embedding = nn.Embedding(12, month_embedding_dim)
        
        # Calculate actual input dimension
        actual_input_dim = input_dim + month_embedding_dim
        
        if n_lead_times > 1:
            self.lead_time_embedding = nn.Embedding(n_lead_times, lead_time_embedding_dim)
            actual_input_dim += lead_time_embedding_dim
            
        layers = [nn.Linear(actual_input_dim, hidden_dim), nn.ReLU()]

        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))

        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]

            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x, lead_time_idx=None, month_idx=None):
        # Always add month embedding
        if month_idx is not None:
            month_emb = self.month_embedding(month_idx)
            x = torch.cat([x, month_emb], dim=-1)
            
        # Add lead time embedding if available
        if self.lead_time_embedding is not None and lead_time_idx is not None:
            lead_time_emb = self.lead_time_embedding(lead_time_idx)
            x = torch.cat([x, lead_time_emb], dim=-1)
            
        return self.net(x)

# ------------------------------
# U-Net definition with lead time and month encoding
# ------------------------------
class UNet(nn.Module):
    """
    U-Net architecture for weather forecast bias correction with lead time and month support.
    """
    
    def __init__(self, input_dim, hidden_dim=128, output_dim=1,
                 n_lat=None, n_lon=None, n_input_vars=None, n_output_vars=None,
                 n_lead_times=1, lead_time_embedding_dim=16, month_embedding_dim=16):
        """
        Initialize U-Net with spatial dimension information, lead time and month support.
        """
        super(UNet, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_lead_times = n_lead_times
        
        # Lead time embedding
        self.lead_time_embedding = None
        if n_lead_times > 1:
            self.lead_time_embedding = nn.Embedding(n_lead_times, lead_time_embedding_dim)
            
        # Month embedding (12 months)
        self.month_embedding = nn.Embedding(12, month_embedding_dim)
        
        # Calculate number of additional channels
        # We'll add 1 channel for month and 1 for lead time (if multi-lead time)
        additional_channels = 1  # month channel
        if n_lead_times > 1:
            additional_channels += 1  # lead time channel
            
        actual_in_channels = n_input_vars + additional_channels

        # Use provided spatial dimensions if available
        if n_lat is not None and n_lon is not None and n_input_vars is not None:
            self.height = n_lat
            self.width = n_lon
            self.in_channels = actual_in_channels
            self.out_channels = n_output_vars if n_output_vars is not None else 1
            
            # Verify dimensions match (approximately, since we might have embeddings)
            expected_input_base = n_input_vars * self.height * self.width
            if abs(expected_input_base - input_dim) > max(lead_time_embedding_dim, month_embedding_dim):
                print(f"Warning: Dimension mismatch. Base input: {expected_input_base}, provided: {input_dim}")
                
            expected_output = self.out_channels * self.height * self.width
            if expected_output != output_dim:
                print(f"Warning: Expected output dim {expected_output} but got {output_dim}")
        else:
            raise ValueError("Dimensions not provided")
        
        # Calculate maximum number of levels based on spatial dimensions
        min_spatial_dim = min(self.height, self.width)
        max_pools = 0
        current_dim = min_spatial_dim
        while current_dim >= 4:  # Need at least 4x4 to pool down to 2x2
            max_pools += 1
            current_dim = current_dim // 2
        self.num_levels = max_pools + 1

        # Build encoder (downsampling path)
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        # Track channel sizes for each encoder level
        self.encoder_channels = []
        
        in_ch = self.in_channels
        out_ch = hidden_dim
        
        for i in range(self.num_levels):
            self.encoders.append(self._make_conv_block(in_ch, out_ch))
            self.encoder_channels.append(out_ch)
            if i < self.num_levels - 1:
                self.pools.append(nn.MaxPool2d(2))
            in_ch = out_ch
            out_ch = out_ch * 2
        
        # Build decoder (upsampling path)
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()

        for i in range(self.num_levels - 1):
            decoder_level = self.num_levels - 1 - i
            skip_level = self.num_levels - 2 - i
            
            in_ch = self.encoder_channels[decoder_level]
            skip_ch = self.encoder_channels[skip_level]
            out_ch = skip_ch
            
            self.upconvs.append(self._make_upconv(in_ch, out_ch))
            combined_ch = out_ch + skip_ch
            self.decoders.append(self._make_conv_block(combined_ch, out_ch))

        # Final output layer
        if self.decoders:
            final_in_ch = self.encoder_channels[0]
        else:
            final_in_ch = self.encoder_channels[-1]
            
        self.final_conv = nn.Conv2d(final_in_ch, self.out_channels, kernel_size=1)
        self.original_n_input_vars = n_input_vars
    
    def _make_conv_block(self, in_channels, out_channels):
        """Create a convolutional block with two conv layers."""
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        block.out_channels = out_channels
        return block
    
    def _make_upconv(self, in_channels, out_channels):
        """Create upsampling layer using transposed convolution."""
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x, lead_time_idx=None, month_idx=None):
        """
        Forward pass through U-Net with lead time and month support.
        """
        batch_size = x.shape[0]
        
        # Reshape flat input to spatial format
        x = x.view(batch_size, self.original_n_input_vars, self.height, self.width)
        
        channels_to_concat = []
        
        # Add month channel
        if month_idx is not None:
            month_emb = self.month_embedding(month_idx)
            # Create spatial month channel by taking mean of embedding dims
            month_channel = month_emb.view(batch_size, -1, 1, 1)
            month_channel = month_channel.mean(dim=1, keepdim=True)
            month_channel = month_channel.expand(batch_size, 1, self.height, self.width)
            channels_to_concat.append(month_channel)
        
        # Add lead time channel if multi-lead time
        if self.lead_time_embedding is not None and lead_time_idx is not None:
            lead_time_emb = self.lead_time_embedding(lead_time_idx)
            # Create spatial lead time channel
            lead_time_channel = lead_time_emb.view(batch_size, -1, 1, 1)
            lead_time_channel = lead_time_channel.mean(dim=1, keepdim=True)
            lead_time_channel = lead_time_channel.expand(batch_size, 1, self.height, self.width)
            channels_to_concat.append(lead_time_channel)
        
        # Concatenate all channels
        if channels_to_concat:
            x = torch.cat([x] + channels_to_concat, dim=1)
        
        # Encoder path with skip connections
        encoder_outputs = []
        
        for i in range(len(self.encoders)):
            x = self.encoders[i](x)
            if i < len(self.encoders) - 1:
                encoder_outputs.append(x)
            if i < len(self.pools):
                x = self.pools[i](x)
        
        # Decoder path with skip connections
        for i, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoders)):
            x = upconv(x)
            skip_connection = encoder_outputs[-(i+1)]
            
            if x.shape[2:] != skip_connection.shape[2:]:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=False)
            
            x = torch.cat([x, skip_connection], dim=1)
            x = decoder(x)
        
        # Final convolution
        x = self.final_conv(x)
        
        # Reshape back to flat output
        x = x.view(batch_size, -1)
        
        return x


# ------------------------------
# Argument parsing
# ------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune MLP for regional post-processing')
    parser.add_argument('--data_dir',     type=str, default="~/weatherbench2_data")
    parser.add_argument('--output_dir',   type=str, required=True)
    parser.add_argument('--climate_zones_file', type=str, default=None)
    parser.add_argument('--model_name',   type=str, required=True)
    parser.add_argument('--region',       type=str, default="india")
    parser.add_argument('--subregion',    type=str, default="2x2")
    parser.add_argument('--lead_time_hours', type=int, nargs='+', default=[24],
                        help='List of lead times in hours (e.g., 24 48 72)')
    parser.add_argument('--training_vars', type=str, nargs='+', default=["2m_temperature"])
    parser.add_argument('--output_vars',   type=str, nargs='+', default=["2m_temperature"])
    parser.add_argument('--train_start',   type=str, default='2018-01-01')
    parser.add_argument('--train_end',     type=str, default='2019-12-31')
    parser.add_argument('--test_start',    type=str, default='2020-01-01')
    parser.add_argument('--test_end',      type=str, default='2020-12-31')
    parser.add_argument('--model_type',   type=str, default='MLP', choices=['MLP', 'UNet'])
    parser.add_argument('--bootstrap',      type=int, default=None,
                        help='If set, run N bootstrap samples of subregions')
    return parser.parse_args()

# ------------------------------
# Region grid and patch helpers
# ------------------------------
def get_region_grid(args):
    """
    Return full region latitude and longitude arrays (unmasked bounding box).
    """
    # region bounds mapping
    if args.region == "india":
        lat0, lat1 = 17, 27
        lon0, lon1 = 72, 82
    elif args.region == "usa_south":
        lat0, lat1 = 30, 40
        lon0, lon1 = -105 + 360, -95 + 360
    elif args.region == "amazon":
        lat0, lat1 = -10, 0
        lon0, lon1 = -70 + 360, -60 + 360
    elif args.region == "british_columbia":
        lat0, lat1 = 48.25, 58 # needs to be 48.25 until data is redownloaded
        lon0, lon1 = -130 + 360, -120 + 360
    elif args.region == "pakistan":
        lat0, lat1 = 25, 34
        lon0, lon1 = 60, 70
    elif args.region == "ethiopia":
        lat0, lat1 = 4, 14
        lon0, lon1 = 34, 44
    elif args.region == "global" or args.region in CLIMATE_ZONE_MAP:
        lat0, lat1 = -90, 90
        lon0, lon1 = 0, 360
    else:
        raise ValueError(f"Unknown region '{args.region}'")
    lat_values = np.arange(lat0, lat1, 0.25)
    lon_values = np.arange(lon0, lon1, 0.25)

    return lat_values, lon_values

def get_patch_shape(args):
    """
    Given args.subregion like '2x2', return number of gridpoints in lat and lon
    """
    deg_lat, deg_lon = map(int, args.subregion.split('x'))
    nlat = int(deg_lat / 0.25)
    nlon = int(deg_lon / 0.25)
    return nlat, nlon

def generate_output_path(args):
    region_str = f"{args.region}"
    subregion_str = f"{args.subregion}"
    dates_str = f"train{args.train_start}-{args.train_end}_test{args.test_start}-{args.test_end}"
    training_vars_str = "_".join(args.training_vars)
    output_vars_str = "_".join(args.output_vars)

    if args.model_type == "UNet":
        model_str = "unet"
    else: 
        model_str = "mlp"
    
    # Format lead times
    lead_times_str = "leadtime_" + "_".join([str(lt) for lt in args.lead_time_hours]) + "h"

    output_path = f"{args.output_dir}/{args.model_name}/{region_str}/train_{training_vars_str}_test_{output_vars_str}_dim{subregion_str}_{lead_times_str}_{dates_str}_{model_str}.zarr"
    return output_path 

def sample_climate_zone_patches(
    cz_da: xr.DataArray,
    zone: int,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    nlat: int,
    nlon: int,
    N: int,
    threshold: float = 0.75
):
    """
    Return a list of N (lat_slice, lon_slice) each of shape (nlat,nlon),
    drawn at random (with replacement) from cz_da restricted to
    lat_vals×lon_vals, such that ≥threshold fraction = zone.
    """
    # restrict to your region grid
    cz = cz_da.sel(latitude=lat_vals, longitude=lon_vals)
    lats = cz.latitude.values
    lons = cz.longitude.values
    H, W = len(lats), len(lons)

    patches = []
    for _ in range(N):
        while True:
            i = random.randint(0, H - nlat)
            j = random.randint(0, W - nlon)
            block = cz.isel(latitude=slice(i, i+nlat),
                            longitude=slice(j, j+nlon))
            frac = (block.values == zone).sum() / float(nlat * nlon)
            if frac >= threshold:
                patches.append((lats[i:i+nlat], lons[j:j+nlon]))
                break
    return patches

def sort_lat_lon(ds):
    # ensure that both lat and lon are sorted ascendingly
    return ds.sortby(['latitude', 'longitude'])

def load_combined_dataset(lat_values, lon_values, time_values, root_dir, data_source):
    """
    Finds all files in the subfolders of root_dir matching file_pattern and combines them.
    """

    min_year = min(time_values).astype('datetime64[Y]').astype(int) + 1970
    max_year = max(time_values).astype('datetime64[Y]').astype(int) + 1970

    file_paths = []
    for year in range(min_year, max_year + 1):

        file_pattern = f"{data_source}_{year}.zarr" 
        file_paths.append(os.path.join(root_dir, file_pattern))
    
    if len(file_paths) == 0:
        raise ValueError(f"No files found matching pattern: {file_pattern}")
    
    return xr.open_mfdataset(
        file_paths,
        combine="by_coords",
        preprocess=lambda ds: ds.sel(latitude = lat_values, longitude = lon_values).sortby('latitude'),
        decode_timedelta=True
    )

def load_forecasts(data_dir, args, lat_values, lon_values, train=True, patch_num=None):
    """
    Vectorized version that processes all data at once without loops.
    More memory intensive but faster for reasonable data sizes.
    """
    if train:
        ver_str = "train"
    else:
        ver_str = "test"

    time_start = getattr(args, f"{ver_str}_start")
    time_end = getattr(args, f"{ver_str}_end")
    
    # Create time range
    time_values = pd.date_range(start=time_start, end=time_end, freq='12h')
    time_values_np = time_values.to_numpy()

    # Define target dataset name
    if args.model_name == "pangu":
        target = "era5"
    if args.model_name == "ifs":
        target = "hres_t0"
    
    # Load datasets
    forecast_ds = load_combined_dataset(lat_values, lon_values, time_values_np, data_dir, args.model_name)
    forecast_ds = forecast_ds.rename({'valid_time': 'time'})

    obs_ds = load_combined_dataset(lat_values, lon_values, time_values_np, data_dir, target)
    
    # Create wind speed if needed
    if "10m_wind_speed" in args.training_vars:
        forecast_ds["10m_wind_speed"] = np.sqrt(
            forecast_ds["10m_u_component_of_wind"]**2 + 
            forecast_ds["10m_v_component_of_wind"]**2
        )
    
    if "10m_wind_speed" in args.output_vars:
        obs_ds["10m_wind_speed"] = np.sqrt(
            obs_ds["10m_u_component_of_wind"]**2 + 
            obs_ds["10m_v_component_of_wind"]**2
        )
    
    # Convert lead times to timedelta and select
    lead_times_td = [np.timedelta64(h, 'h') for h in args.lead_time_hours]
    forecast_ds = forecast_ds.sel(prediction_timedelta=lead_times_td)
    
    # Select lead times and common time range
    common_times = np.intersect1d(forecast_ds.time.values, obs_ds.time.values)
    common_times = np.intersect1d(common_times, time_values_np)
    forecast_ds = forecast_ds.sel(time=common_times)
    obs_ds = obs_ds.sel(time=common_times)
    
    # Get dimensions
    n_time = len(common_times)
    n_lead_times = len(lead_times_td)
    n_lat = len(forecast_ds.latitude)
    n_lon = len(forecast_ds.longitude)
    n_training_vars = len(args.training_vars)
    n_output_vars = len(args.output_vars)
    
    # Stack all dimensions except variables
    forecast_stacked = forecast_ds[args.training_vars].stack(
        sample=['time', 'prediction_timedelta']
    ).to_array()
    
    forecast_output_stacked = forecast_ds[args.output_vars].stack(
        sample=['time', 'prediction_timedelta']
    ).to_array()
    
    # For observations, we need to repeat for each lead time
    obs_repeated = obs_ds[args.output_vars].expand_dims(
        prediction_timedelta=lead_times_td
    ).stack(
        sample=['time', 'prediction_timedelta']
    ).to_array()
    
    # Transpose and reshape to (n_samples, n_features)
    fc_combined = forecast_stacked.values.T.reshape(-1, n_training_vars * n_lat * n_lon)
    fc_output_combined = forecast_output_stacked.values.T.reshape(-1, n_output_vars * n_lat * n_lon)
    obs_combined = obs_repeated.values.T.reshape(-1, n_output_vars * n_lat * n_lon)
    
    # Create lead time indices
    lead_time_indices = np.tile(
        np.arange(n_lead_times), n_time
    )
    
    # Create month indices
    month_values = pd.DatetimeIndex(common_times).month.to_numpy() - 1  
    month_indices = np.repeat(month_values, n_lead_times)
    
    # Create time array
    all_times = np.repeat(common_times, n_lead_times)
    
    # Remove any samples with NaN (XX with updated version of databuild that converts from init to valid time this might not be needed)
    valid_mask = ~(np.isnan(fc_combined).any(axis=1) | np.isnan(obs_combined).any(axis=1))
    fc_combined = fc_combined[valid_mask]
    fc_output_combined = fc_output_combined[valid_mask]
    obs_combined = obs_combined[valid_mask]
    lead_time_indices_combined = lead_time_indices[valid_mask]
    month_indices_combined = month_indices[valid_mask]
    all_times = all_times[valid_mask]

    # Calculate mean forecast error
    training_mean_forecast_error = {}
    
    for lt_idx, lead_time_hours in enumerate(args.lead_time_hours):
        mask = lead_time_indices_combined == lt_idx
        if not np.any(mask):
            continue
            
        fc_output_lt = fc_output_combined[mask].reshape(-1, n_output_vars, n_lat, n_lon)
        obs_lt = obs_combined[mask].reshape(-1, n_output_vars, n_lat, n_lon)
        
        mean_error = fc_output_lt.mean(axis=0) - obs_lt.mean(axis=0)
        
        for var_idx, var_name in enumerate(args.output_vars):
            key = f"{var_name}_lt{lead_time_hours}h"
            training_mean_forecast_error[key] = mean_error[var_idx]
    # each sample represents one forecast for one specific lead time and time combination 
    return (fc_combined, fc_output_combined, obs_combined, lead_time_indices_combined, 
            month_indices_combined, all_times, forecast_ds.latitude.values, 
            forecast_ds.longitude.values, n_lat, n_lon, 
            n_training_vars, n_output_vars, training_mean_forecast_error)


def create_dataloader(forecast_data, obs_data, lead_time_indices, month_indices, batch_size):
    """
    Create a PyTorch DataLoader from forecast, observation data, lead time indices, and month indices.
    """
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(forecast_data).float(),
        torch.from_numpy(obs_data).float(),
        torch.from_numpy(lead_time_indices).long(),
        torch.from_numpy(month_indices).long()
    )
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True)
    return dataloader


def train_model(model, train_loader, valid_loader, epochs, lr, device, weight_decay=0, patience=50, min_delta=9.8e-05):
    """
    Train the model over multiple epochs with early stopping.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), 
                           lr=lr,
                           weight_decay =weight_decay) 

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    # Track training time
    train_start_time = time.time()

    for epoch in range(1, epochs + 1):
        # --- training step ---
        model.train()
        train_loss = 0.0
        for x_batch, y_batch, lead_time_batch, month_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            lead_time_batch, month_batch = lead_time_batch.to(device), month_batch.to(device)
            
            optimizer.zero_grad()
            
            # Pass lead time and month indices to model
            preds = model(x_batch, lead_time_batch, month_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)
        train_loss /= len(train_loader.dataset)

        # --- validation step ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch, lead_time_batch, month_batch in valid_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                lead_time_batch, month_batch = lead_time_batch.to(device), month_batch.to(device)
                
                preds = model(x_batch, lead_time_batch, month_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item() * x_batch.size(0)
        val_loss /= len(valid_loader.dataset)

        # --- early stopping check ---
        if val_loss + min_delta < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"→ Early stopping at epoch {epoch}. "
                      f"No improvement in {patience} epochs.")
                break

    # Calculate training time in minutes
    train_end_time = time.time()
    training_time_minutes = (train_end_time - train_start_time) / 60.0
    
    # Load best weights
    model.load_state_dict(best_model_wts)
    return model, training_time_minutes


def apply_correction(model, forecast_data, lead_time_indices, month_indices, device):
    """
    Apply the MLP-based correction to forecast data with lead times and months.
    """
    model.eval()
    corrected_all = []
    
    # Process in batches to handle memory efficiently
    batch_size = 128
    n_samples = forecast_data.shape[0]
    
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            x_batch = torch.from_numpy(forecast_data[i:end_idx]).float().to(device)
            lt_batch = torch.from_numpy(lead_time_indices[i:end_idx]).long().to(device)
            month_batch = torch.from_numpy(month_indices[i:end_idx]).long().to(device)
            corrected_batch = model(x_batch, lt_batch, month_batch).cpu().numpy()
            corrected_all.append(corrected_batch)
    
    return np.concatenate(corrected_all, axis=0)


def save_output(output_path, model_name, output_vars, lon_values, lat_values,
                time_values, lead_times, original_fc, corrected_fc, lead_time_indices,
                ground_truth_data=None, training_mean_forecast_error=None, training_time_minutes=None):
    """
    Save original and corrected forecasts organized by lead time.
    """
    # Create separate datasets for each lead time
    data_vars = {}
    
    for lt_idx, lead_time in enumerate(lead_times):
        # Get mask for this lead time
        mask = lead_time_indices == lt_idx
        n_time_lt = mask.sum()
        
        if n_time_lt == 0:
            continue
            
        # Extract data for this lead time
        times_lt = [t for i, t in enumerate(time_values) if mask[i]]
        original_lt = original_fc[mask]
        corrected_lt = corrected_fc[mask]
        
        # Reshape data
        n_vars = len(output_vars)
        n_lat = len(lat_values)
        n_lon = len(lon_values)
        
        original_lt = original_lt.reshape(n_time_lt, n_vars, n_lat, n_lon).transpose(1, 0, 2, 3)
        corrected_lt = corrected_lt.reshape(n_time_lt, n_vars, n_lat, n_lon).transpose(1, 0, 2, 3)
        
        # Create data arrays for each variable
        for var_idx, var in enumerate(output_vars):
            # Original forecast
            data_vars[f"{var}_original_lt{lead_time}h"] = xr.DataArray(
                original_lt[var_idx],
                dims=['time', 'latitude', 'longitude'],
                coords={'time': times_lt, 'latitude': lat_values, 'longitude': lon_values}
            )
            
            # Corrected forecast
            data_vars[f"{var}_corrected_lt{lead_time}h"] = xr.DataArray(
                corrected_lt[var_idx],
                dims=['time', 'latitude', 'longitude'],
                coords={'time': times_lt, 'latitude': lat_values, 'longitude': lon_values}
            )
            
            # Mean corrected (if available)
            if training_mean_forecast_error is not None:
                key = f"{var}_lt{lead_time}h"
                if key in training_mean_forecast_error:
                    mean_corrected = original_lt[var_idx] - training_mean_forecast_error[key]
                    data_vars[f"{var}_mean_corrected_lt{lead_time}h"] = xr.DataArray(
                        mean_corrected,
                        dims=['time', 'latitude', 'longitude'],
                        coords={'time': times_lt, 'latitude': lat_values, 'longitude': lon_values}
                    )
            
            # Ground truth (if available)
            if ground_truth_data is not None:
                ground_truth_lt = ground_truth_data[mask]
                ground_truth_lt = ground_truth_lt.reshape(n_time_lt, n_vars, n_lat, n_lon).transpose(1, 0, 2, 3)
                data_vars[f"{var}_ground_truth_lt{lead_time}h"] = xr.DataArray(
                    ground_truth_lt[var_idx],
                    dims=['time', 'latitude', 'longitude'],
                    coords={'time': times_lt, 'latitude': lat_values, 'longitude': lon_values}
                )
    
    # Create dataset
    ds_out = xr.Dataset(data_vars)
    
    # Add metadata
    ds_out.attrs['description'] = f'Original and corrected forecasts from {model_name} using MLP fine-tuning'
    ds_out.attrs['lead_times_hours'] = lead_times
    ds_out.attrs['training_time_minutes'] = training_time_minutes if training_time_minutes is not None else -1
    
    # Save to zarr
    output_path = os.path.expanduser(output_path)
    # ds_out.to_zarr(output_path, mode='w')
    # save to netcdf as well for easier access
    netcdf_path = output_path.replace('.zarr', '.nc')
    ds_out.to_netcdf(netcdf_path)
    print(f"Forecasts saved to {output_path}")


def run_subregion_experiment(lat_vals, lon_vals, output_path, args, data_dir, device, patch_num=None):
    """
    Run experiment with multiple lead times and month encoding.
    """
    start_time = time.time()

    # Load training data
    (fc, fc_output, obs, lead_time_indices, month_indices, train_times, lat_u, lon_u, 
     n_lat, n_lon, n_training_vars, n_output_vars, training_mean_forecast_error) = \
        load_forecasts(data_dir, args, lat_vals, lon_vals, train=True, patch_num=patch_num)
    
    loading_time = time.time()
    print(f"Data loaded in {(loading_time - start_time) / 60:.2f} minutes")

    # Normalize data
    stats_train = {'mean': fc.mean(0), 'std': fc.std(0) + 1e-8}
    stats_out = {'mean': fc_output.mean(0), 'std': fc_output.std(0) + 1e-8}
    fc_norm = (fc - stats_train['mean']) / stats_train['std']
    obs_norm = (obs - stats_out['mean']) / stats_out['std']

    # Split train/validation
    n_train = len(fc)
    idx = np.arange(n_train)
    np.random.shuffle(idx)
    split = int(0.8 * n_train)
    t_idx, v_idx = idx[:split], idx[split:]
    
    train_loader = create_dataloader(fc_norm[t_idx], obs_norm[t_idx], 
                                    lead_time_indices[t_idx], month_indices[t_idx], 
                                    batch_size=128)
    val_loader = create_dataloader(fc_norm[v_idx], obs_norm[v_idx], 
                                  lead_time_indices[v_idx], month_indices[v_idx], 
                                  batch_size=16)

    # Initialize model
    input_dim = n_training_vars * n_lat * n_lon
    output_dim = n_output_vars * n_lat * n_lon
    n_lead_times = len(args.lead_time_hours)

    if hasattr(args, 'model_type') and args.model_type == 'UNet':
        print(f"Using UNet with {n_lead_times} lead times and month encoding")
        model = UNet(input_dim, 32, output_dim, n_lat=n_lat, n_lon=n_lon,
                     n_input_vars=n_training_vars, n_output_vars=n_output_vars,
                     n_lead_times=n_lead_times).to(device)
    else:
        print(f"Using SimpleMLP with {n_lead_times} lead times and month encoding")
        model = SimpleMLP(input_dim = input_dim, 
                          hidden_dim = 1024,
                          output_dim = output_dim, 
                          num_hidden_layers= 2,
                          n_lead_times=n_lead_times,
                          lead_time_embedding_dim=4,
                          month_embedding_dim=16,
                          dropout_rate =0.1097725 
                          ).to(device)

    # Train model
    model, training_time_minutes = train_model(model, train_loader, val_loader,
                                                epochs=1000, lr=4.673747105982307e-05, 
                                                device=device,
                                                weight_decay=2.8276153644203165e-06,
                                                patience=70, min_delta=0.000286450816778278)
    print(f"Training complete in {training_time_minutes:.2f} minutes")

    load_time = time.time()
    # Load test data
    (test_fc, test_fc_output, test_obs, test_lead_time_indices, test_month_indices,
     test_times, _, _, _, _, _, _, _) = \
        load_forecasts(data_dir, args, lat_vals, lon_vals, train=False, patch_num=patch_num)

    # Apply correction
    test_fc_norm = (test_fc - stats_train['mean']) / stats_train['std']
    corrected = apply_correction(model, test_fc_norm, test_lead_time_indices, 
                                test_month_indices, device)
    corrected = (corrected * stats_out['std']) + stats_out['mean']

    # Calculate MSE per lead time and month
    for lt_idx, lead_time in enumerate(args.lead_time_hours):
        mask = test_lead_time_indices == lt_idx
        if mask.sum() > 0:
            mse_original = np.mean((test_fc_output[mask] - test_obs[mask])**2)
            mse_corrected = np.mean((corrected[mask] - test_obs[mask])**2)
            print(f"Lead time {lead_time}h - MSE original: {mse_original:.6f}, MSE corrected: {mse_corrected:.6f}")
    print(f"Test data loaded in {(load_time - loading_time) / 60:.2f} minutes")
            
    save_start_time = time.time()
    # Save results
    save_output(
        output_path=output_path,
        model_name=args.model_name,
        output_vars=args.output_vars,
        lon_values=lon_u,
        lat_values=lat_u,
        time_values=test_times,
        lead_times=args.lead_time_hours,
        original_fc=test_fc_output,
        corrected_fc=corrected,
        lead_time_indices=test_lead_time_indices,
        ground_truth_data=test_obs,
        training_mean_forecast_error=training_mean_forecast_error,
        training_time_minutes=training_time_minutes
    )
    print(f"time to save output: {(time.time() - save_start_time) / 60:.2f} minutes")

    end_time = time.time()
    total_time_minutes = (end_time - start_time) / 60
    print(f"Total experiment time: {total_time_minutes:.2f} minutes")


def main():
    dirs = setup_directories()

    # Parse command line arguments
    args = parse_args()
    
    # Prepare output dir and base path
    args.output_dir = os.path.expanduser(args.output_dir)
    args.data_dir = os.path.expanduser(args.data_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    base_path = generate_output_path(args)

    # Setup device & seeds
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else
                          'cpu')
    torch.manual_seed(58)
    random.seed(58)

    region_lat, region_lon = get_region_grid(args)
    nlat_patch, nlon_patch = get_patch_shape(args)

    # Decide if we're in a climate‐zone region or a geographic one
    if args.region in CLIMATE_ZONE_MAP:
        
        patches_path = os.path.join(dirs["processed"], f"climate_zone_patches_{args.region}_{args.subregion}.npy")
        patches = np.load(patches_path, allow_pickle=True)
        patch_ids = np.arange(1, len(patches) + 1)
        assert len(patches) == 50

        for patch, idx in zip(patches, patch_ids):
            lat_min = patch[0,].min()
            lat_max = patch[0,].max()
            lon_min = patch[1,].min()
            lon_max = patch[1,].max()

            # print lat min and max
            print(f"Processing patch {idx} with lat range ({lat_min}, {lat_max}) and lon range ({lon_min}, {lon_max})")

            print(f"Max and min region_lon: {region_lon.max()}, {region_lon.min()}")

            lat_vals = region_lat[(region_lat >= lat_min) & (region_lat <= lat_max)]
            lon_vals = region_lon[(region_lon >= lon_min) & (region_lon <= lon_max)]

            out_path = base_path.replace('.zarr', f'_{args.region}_bs{idx}.zarr')

            if os.path.exists(out_path):
                print(f"Skipping already existing output: {out_path}")
                continue

            run_subregion_experiment(
                lat_vals, lon_vals, out_path,
                args, os.path.expanduser(args.data_dir), device, patch_num=idx
            )

    elif args.bootstrap:
        # Bootstrap sampling for uniform-grid regions
        for i in range(args.bootstrap):
            si = random.randint(0, len(region_lat) - nlat_patch)
            sj = random.randint(0, len(region_lon) - nlon_patch)
            lat_vals = region_lat[si:si+nlat_patch + 1]
            lon_vals = region_lon[sj:sj+nlon_patch + 1]
            out_path = base_path.replace('.zarr', f'_bs{i+1}.zarr')
            print(f"Running bootstrap sample {i+1}/{args.bootstrap}")
            run_subregion_experiment(lat_vals, lon_vals, out_path, args,
                                     os.path.expanduser(args.data_dir), device)
    else:
        # Central patch
        ci = (len(region_lat) - nlat_patch) // 2
        cj = (len(region_lon) - nlon_patch) // 2
        lat_vals = region_lat[ci:ci+nlat_patch]
        lon_vals = region_lon[cj:cj+nlon_patch]

        run_subregion_experiment(lat_vals, lon_vals, base_path, args,
                                 os.path.expanduser(args.data_dir), device)


if __name__ == "__main__":
    main()