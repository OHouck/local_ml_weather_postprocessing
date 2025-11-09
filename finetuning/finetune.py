#!/usr/bin/env python3
"""
Author: Ozma Houck
Filename: finetune/finetune.py

# Purpose: use a simple MLP to post-process weather forecasts trained on
specific regions and variables. Call this script from command line or with
1_run_experiments.sh script.

# Modified to support multiple lead times training

# NEW: Dynamic data loading - the script now automatically checks if the required
# data exists in the data_dir. If not, it downloads the data on-the-fly from
# weatherbench2 using the prepare_forecasts_and_targets module. This eliminates
# the need to pre-download large global datasets and allows testing different
# lead times and variables without manual data management.

# example call
python3 finetuning/finetune.py \
    --data_dir="~/ai_weather_ag/data/raw/" \
    --output_dir="~/ai_weather_ag/data/fine_tuning_output" \
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
import numcodecs
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import time


import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from helper_funcs import setup_directories
from helper_funcs import generate_output_path

# Import dynamic data loading functions
sys.path.insert(0, str(Path(__file__).parent))
from prepare_forecasts_and_targets import prepare_data_for_finetuning

# print(f"NumPy version: {np.__version__}")
# print(f"PyTorch version: {torch.__version__}")
# print(f"PyTorch built with NumPy: {torch.from_numpy(np.array([1.0])).numpy()}") # test numpy interoperability, have had issues with version mismatches before

# Map the new region strings to Koppen‐Geiger codes:
CLIMATE_ZONE_MAP = {
    'tropical':  1,
    'arid':       2,
    'temperate':  3,
    'cold':       4,
    'polar':      5,
}
# Map topographic zones
TOPO_ZONE_MAP = {
    'flat': 1,
    'hilly': 2,
    'mountainous': 3,
}

# Purpose: save patches of of climate zones to be used for bootstrapping
# ------------------------------
# Simple MLP definition with lead time and month encoding
# ------------------------------
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, output_dim=1, num_hidden_layers=2,
                n_lead_times=1, lead_time_embedding_dim=8, dropout_rate=0.0):
        super(SimpleMLP, self).__init__()

        # Lead time embedding
        self.n_lead_times = n_lead_times
        self.lead_time_embedding = None


        # Calculate actual input dimension
        actual_input_dim = input_dim + 2  # +2 for sin and cos of day of year

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

    def forward(self, x, lead_time_idx=None, day_of_year_features=None):
        # Add day-of-year sin/cos features
        if day_of_year_features is not None:
            x = torch.cat([x, day_of_year_features], dim=-1)

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
    U-Net architecture with FiLM conditioning for weather forecast bias correction.
    Uses Feature-wise Linear Modulation to incorporate lead time and month information.
    """
    
    def __init__(self, input_dim, hidden_dim=128, output_dim=1,
                 n_lat=None, n_lon=None, n_input_vars=None, n_output_vars=None,
                 n_lead_times=1, lead_time_embedding_dim=16, dropout_rate=0.1):
        """
        Initialize U-Net with FiLM conditioning.

        Args:
            input_dim: Flattened input dimension (for compatibility)
            hidden_dim: Base number of channels in first encoder layer
            output_dim: Flattened output dimension
            n_lat, n_lon: Spatial dimensions
            n_input_vars, n_output_vars: Number of input/output variables
            n_lead_times: Number of distinct lead times
            lead_time_embedding_dim: Dimension of lead time embedding
            dropout_rate: Dropout rate for conv blocks
        """
        super(UNet, self).__init__()

        # Store dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_lead_times = n_lead_times
        self.dropout_rate = dropout_rate

        # Spatial dimensions
        if n_lat is None or n_lon is None or n_input_vars is None:
            raise ValueError("Spatial dimensions and n_input_vars must be provided")

        self.height = n_lat
        self.width = n_lon
        self.n_input_vars = n_input_vars
        self.n_output_vars = n_output_vars if n_output_vars is not None else 1

        # Create embeddings
        self.lead_time_embedding = None
        if n_lead_times > 1:
            self.lead_time_embedding = nn.Embedding(n_lead_times, lead_time_embedding_dim)

        # Calculate total conditioning dimension
        # Day-of-year features are 2D (sin and cos)
        self.total_embedding_dim = 2  # sin and cos of day of year
        if n_lead_times > 1:
            self.total_embedding_dim += lead_time_embedding_dim
        
        # Calculate number of encoder/decoder levels
        self.num_levels = self._calculate_num_levels()
        
        # Build encoder and decoder
        self._build_encoder()
        self._build_decoder()
        
        # Build FiLM conditioning layers
        self._build_film_layers()
        
        # Final output layer
        self.final_conv = nn.Conv2d(self.encoder_channels[0], self.n_output_vars, kernel_size=1)
        
    def _calculate_num_levels(self):
        """
        Calculate maximum number of pooling levels based on spatial dimensions.

        Determines how many downsampling operations can be performed while maintaining
        a minimum spatial dimension of 4x4 at the bottleneck. Caps at 5 levels to
        prevent overly deep networks.

        Returns:
            int: Number of encoder/decoder levels (between 1 and 5)
        """
        min_spatial_dim = min(self.height, self.width)
        max_pools = 0
        current_dim = min_spatial_dim
        # Need at least 4x4 to pool down to 2x2
        while current_dim >= 4:
            max_pools += 1
            current_dim = current_dim // 2
        # Cap at 5 levels to avoid too deep networks
        return min(max_pools + 1, 5)
    
    def _make_conv_block(self, in_channels, out_channels):
        """Create a convolutional block with two conv layers, batch norm, and dropout."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), # briefly tried leaky ReLU and got worse results
            nn.Dropout2d(self.dropout_rate),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(self.dropout_rate)
        )
    
    def _make_upconv(self, in_channels, out_channels):
        """Create upsampling layer using transposed convolution."""
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def _build_encoder(self):
        """Build the encoder (downsampling) path."""
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.encoder_channels = []
        
        in_ch = self.n_input_vars
        out_ch = self.hidden_dim
        
        for i in range(self.num_levels):
            # Add encoder block
            self.encoders.append(self._make_conv_block(in_ch, out_ch))
            self.encoder_channels.append(out_ch)
            
            # Add pooling layer (except for last level)
            if i < self.num_levels - 1:
                self.pools.append(nn.MaxPool2d(kernel_size=2))
            
            # Update channels for next level
            in_ch = out_ch
            out_ch = min(out_ch * 2, 512)  # Cap at 512 channels
    
    def _build_decoder(self):
        """Build the decoder (upsampling) path."""
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        
        for i in range(self.num_levels - 1):
            # Calculate channel numbers
            decoder_level = self.num_levels - 1 - i
            skip_level = self.num_levels - 2 - i
            
            in_ch = self.encoder_channels[decoder_level]
            skip_ch = self.encoder_channels[skip_level]
            out_ch = skip_ch
            
            # Add upconvolution
            self.upconvs.append(self._make_upconv(in_ch, out_ch))
            
            # Add decoder block (takes concatenated skip connection)
            combined_ch = out_ch + skip_ch
            self.decoders.append(self._make_conv_block(combined_ch, out_ch))
    
    def _build_film_layers(self):
        """Build FiLM (Feature-wise Linear Modulation) layers for each encoder level."""
        self.film_generators = nn.ModuleList()
        
        for i, channels in enumerate(self.encoder_channels):
            # Each FiLM generator produces scale and shift parameters
            film_generator = nn.Sequential(
                nn.Linear(self.total_embedding_dim, channels * 4),
                nn.ReLU(),
                nn.Linear(channels * 4, channels * 2)  # Output: channels for scale + channels for shift
            )
            self.film_generators.append(film_generator)
    
    def apply_film(self, x, film_params):
        """
        Apply FiLM conditioning to feature maps.
        
        Args:
            x: Feature maps [batch_size, channels, height, width]
            film_params: FiLM parameters [batch_size, channels * 2]
        
        Returns:
            Modulated feature maps
        """
        batch_size, channels, height, width = x.shape
        
        # Split into scale and shift
        scale, shift = torch.chunk(film_params, 2, dim=1)
        
        # Reshape for broadcasting
        scale = scale.view(batch_size, channels, 1, 1)
        shift = shift.view(batch_size, channels, 1, 1)
        
        # Apply affine transformation
        return x * (1 + scale) + shift
    
    def forward(self, x, lead_time_idx=None, day_of_year_features=None):
        """
        Forward pass through U-Net with FiLM conditioning.

        Args:
            x: Input tensor [batch_size, input_dim]
            lead_time_idx: Lead time indices [batch_size]
            day_of_year_features: Day-of-year sin/cos features [batch_size, 2]

        Returns:
            Output tensor [batch_size, output_dim]
        """
        batch_size = x.shape[0]

        # Reshape input to spatial format
        x = x.view(batch_size, self.n_input_vars, self.height, self.width)

        # Prepare conditioning vector from features and embeddings
        conditioning_vectors = []

        if day_of_year_features is not None:
            conditioning_vectors.append(day_of_year_features)

        if self.lead_time_embedding is not None and lead_time_idx is not None:
            lead_time_emb = self.lead_time_embedding(lead_time_idx)
            conditioning_vectors.append(lead_time_emb)

        # Concatenate conditioning vectors
        if conditioning_vectors:
            conditioning = torch.cat(conditioning_vectors, dim=1)
        else:
            # If no conditioning, create zeros
            conditioning = torch.zeros(batch_size, self.total_embedding_dim, device=x.device)
        
        # Generate FiLM parameters for each encoder level
        film_params_list = []
        for film_gen in self.film_generators:
            film_params = film_gen(conditioning)
            film_params_list.append(film_params)
        
        # Encoder path with skip connections and FiLM modulation
        encoder_outputs = []
        
        for i in range(self.num_levels):
            # Apply convolution
            x = self.encoders[i](x)
            
            # Apply FiLM modulation
            # XX Note original paper( https://arxiv.org/abs/1907.01277) applies
            # FiLM in encoder block not after. This is simplier to implement and
            # requires less parameters to estimate.
            x = self.apply_film(x, film_params_list[i])
            
            # Store skip connections (except for bottleneck)
            if i < self.num_levels - 1:
                encoder_outputs.append(x)
                # Apply pooling
                x = self.pools[i](x)
        
        # Decoder path with skip connections
        for i in range(len(self.upconvs)):
            # Upsample
            x = self.upconvs[i](x)
            
            # Get skip connection
            skip_connection = encoder_outputs[-(i+1)]
            
            # Concatenate with skip connection
            x = torch.cat([x, skip_connection], dim=1)
            
            # Apply decoder convolution
            x = self.decoders[i](x)
        
        # Final 1x1 convolution to get output channels
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
    parser.add_argument('--model_name',   type=str, required=True)
    parser.add_argument('--ground_truth_source', type=str, default="")
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
    parser.add_argument('--nn_architecture',   type=str, default='mlp', choices=['mlp', 'unet'])
    parser.add_argument('--bootstrap',      type=int, default=None,
                        help='If set, run N bootstrap samples of subregions')
    parser.add_argument('--growing_season_only', action='store_true',
                        help='Filter data to growing season days only')
    parser.add_argument('--alternate_loss_fn', type=str, default=None, choices=['quantile_loss', 'extreme_heat_loss'])

    # Architecture-specific parameters
    parser.add_argument('--mlp_hidden_dim', type=int, default=1024,
                        help='Hidden dimension for MLP')
    parser.add_argument('--mlp_num_layers', type=int, default=4,
                        help='Number of hidden layers for MLP')
    parser.add_argument('--mlp_dropout', type=float, default=0.2477893381,
                        help='Dropout rate for MLP')
    parser.add_argument('--unet_hidden_dim', type=int, default=32,
                        help='Base hidden dimension for UNet')
    parser.add_argument('--unet_dropout', type=float, default=0.1,
                        help='Dropout rate for UNet')

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
    elif args.region == "corn_belt":
        lat0, lat1 = 36, 46
        lon0, lon1 = -95 + 360, -85 + 360
    elif args.region == "odisha":
        lat0, lat1 = 18, 22
        lon0, lon1 = 82, 86
    elif args.region == "global" or args.region in CLIMATE_ZONE_MAP or args.region in TOPO_ZONE_MAP:
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


def sort_lat_lon(ds):
    # ensure that both lat and lon are sorted ascendingly
    return ds.sortby(['latitude', 'longitude'])

def load_combined_dataset(lat_values, lon_values, time_values, root_dir, data_source, region):
    """
    Finds all files in the subfolders of root_dir matching file_pattern and combines them.
    Uses new file organization: root_dir/data_source/data_source_region_year.zarr
    """

    min_year = min(time_values).astype('datetime64[Y]').astype(int) + 1970
    max_year = max(time_values).astype('datetime64[Y]').astype(int) + 1970

    file_paths = []
    for year in range(min_year, max_year + 1):

        if data_source == "imd":
            file_pattern = f"IMD_0p25deg/data_{year}.nc"
        else:
            # New file organization: data_source/data_source_region_year.zarr
            file_pattern = f"{data_source}/{data_source}_{region}_{year}.zarr"

        file_paths.append(os.path.join(root_dir, file_pattern))
    
    if len(file_paths) == 0:
        raise ValueError(f"No files found matching pattern: {file_pattern}")

    # aifs has different variable names so rename them when loading
    preprocess_fn = lambda ds: ds.sel(latitude = lat_values, longitude = lon_values).sortby('latitude')

    if data_source == "imd":
        preprocess_fn = lambda ds: ds.rename({'TIME': 'time','LONGITUDE': 'longitude', 
                            'LATITUDE': 'latitude', 'RAINFALL':'total_precipitation'}
                            ).sel(latitude = lat_values, longitude = lon_values).sortby('latitude')
    forecast_ds = xr.open_mfdataset(
        file_paths,
        combine="by_coords",
        preprocess=preprocess_fn,
        decode_timedelta=True,
        engine="zarr",
    )

    return(forecast_ds)

def load_forecasts(data_dir, args, lat_values, lon_values, train=True, patch_num=None,):
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

    # only keep growing season dates: 3-15 to 10-31
    if args.growing_season_only:
        time_values= time_values[
            ((time_values.month > 3) | ((time_values.month == 3) & (time_values.day >= 15)) &
            ((time_values.month <= 10)))
        ]

    time_values_np = time_values.to_numpy()

    # Define target dataset name
    if args.ground_truth_source == "":
        if args.model_name == "pangu":
            target = "era5"
        elif args.model_name == "ifs":
            target = "hres_t0"
        elif args.model_name == "aifs":
            target = "era5"
        else:
            raise ValueError(f"Unknown model_name '{args.model_name}' and no ground_truth_source provided")
    else: 
        # For now only IMD is supported as alternative ground truth
        target = args.ground_truth_source
    
    # Load datasets
    forecast_ds = load_combined_dataset(lat_values, lon_values, time_values_np, data_dir, args.model_name, args.region)
    forecast_ds = forecast_ds.rename({'valid_time': 'time'})
    obs_ds = load_combined_dataset(lat_values, lon_values, time_values_np, data_dir, target, args.region)
    
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
    
    # Select lead times and common time range, this prevents off by one errors 
    # from lining up forecasts for days on the edges of the time range
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

    # Create time array
    all_times = np.repeat(common_times, n_lead_times)
    # Create day-of-year sin/cos features
    day_of_year = pd.DatetimeIndex(common_times).dayofyear.to_numpy() # XX maybe should use all_times? I don't think so but flagging
    # Convert to radians: 2*pi*d/365
    day_of_year_rad = 2 * np.pi * day_of_year / 365.0
    day_of_year_sin = np.sin(day_of_year_rad)
    day_of_year_cos = np.cos(day_of_year_rad)
    # Stack to create [n_time, 2] array, then repeat for each lead time
    day_of_year_features = np.stack([day_of_year_sin, day_of_year_cos], axis=1)
    day_of_year_features = np.repeat(day_of_year_features, n_lead_times, axis=0)

    # Remove any samples with NaN
    valid_mask = ~(np.isnan(fc_combined).any(axis=1) | np.isnan(obs_combined).any(axis=1))
    fc_combined = fc_combined[valid_mask]
    fc_output_combined = fc_output_combined[valid_mask]
    obs_combined = obs_combined[valid_mask]
    lead_time_indices_combined = lead_time_indices[valid_mask]
    day_of_year_features_combined = day_of_year_features[valid_mask]
    all_times = all_times[valid_mask]

    # Calculate mean forecast error
    training_mean_forecast_error = {}
    
    for lt_idx, lead_time_hours in enumerate(args.lead_time_hours):
        mask = lead_time_indices_combined == lt_idx
        if not np.any(mask):
            continue
            
        fc_output_lt = fc_output_combined[mask].reshape(-1, n_output_vars, n_lat, n_lon)
        obs_lt = obs_combined[mask].reshape(-1, n_output_vars, n_lat, n_lon)

        # # FORCE COMPUTATION if these are dask arrays
        # if hasattr(fc_output_lt, 'compute'):
        #     fc_output_lt = fc_output_lt.compute()
        # if hasattr(obs_lt, 'compute'):
        #     obs_lt = obs_lt.compute()

        
        mean_error = fc_output_lt.mean(axis=0) - obs_lt.mean(axis=0)
        
        for var_idx, var_name in enumerate(args.output_vars):
            key = f"{var_name}_lt{lead_time_hours}h"
            training_mean_forecast_error[key] = mean_error[var_idx]

    # each sample represents one forecast for one specific lead time and time combination
    return (fc_combined, fc_output_combined, obs_combined, lead_time_indices_combined,
            day_of_year_features_combined, all_times, forecast_ds.latitude.values,
            forecast_ds.longitude.values, n_lat, n_lon,
            n_training_vars, n_output_vars, training_mean_forecast_error)


def create_dataloader(forecast_data, obs_data, lead_time_indices, day_of_year_features, batch_size):
    """
    Create a PyTorch DataLoader from forecast, observation data, lead time indices, and day-of-year features.
    """
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(forecast_data).float(),
        torch.from_numpy(obs_data).float(),
        torch.from_numpy(lead_time_indices).long(),
        torch.from_numpy(day_of_year_features).float()
    )
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True)
    return dataloader


def quantile_loss(preds, targets, quantile=0.95):
    """
    Quantile loss (pinball loss)
    """
    errors = targets - preds
    # if positive error, use quantile * error, else (quantile - 1) * error
    return torch.max((quantile - 1) * errors, quantile*errors).mean()

def extreme_heat_loss(preds, targets, std_out, mean_out):
    """
    up-weight loss for negative errors (under-predictions) above 25C in 
    proportion to coefficents on mortality curve in fatal errors shrader paper

    preds: (batch_size, n_features) in normalized units
    targets: (batch_size, n_features) in normalized units
    """

    # un-normalize 
    preds = preds * std_out + mean_out
    targets = targets * std_out + mean_out

    # convert to Celsius from Kelvin for easier thresholding 
    targets_c = targets - 273.15
    preds_c = preds - 273.15
    errors = targets_c - preds_c
    squared_errors = errors**2
    weights = torch.ones_like(errors)

    weights += ((targets_c > 25) & (targets_c <= 30)).float() * (errors < 0).float() * 2
    weights += (targets_c > 30) * (errors < 0).float() * 10 

    weights = weights / weights.sum()  # sum to 1 for interpretability with MSE
    weighted_mse = (weights * squared_errors).sum()

    return weighted_mse


def train_model(model, train_loader, valid_loader, epochs, lr, device,
                weight_decay=0, 
                stats_out=None, alternate_loss_fn=None, 
                T_0=10, T_mult=3, eta_min=1e-7):
    """
    Train the model over multiple epochs with early stopping.

    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        valid_loader: DataLoader for validation data
        epochs: Maximum number of epochs to train
        lr: Initial learning rate
        device: Device to train on (cpu/cuda)
        weight_decay: L2 regularization weight
        stats_out: Statistics for denormalizing outputs (for custom losses)
        alternate_loss_fn: Name of custom loss function to use
        T_0: Number of epochs for first restart cycle (cosine annealing)
        T_mult: Factor to increase T_0 after each restart (cosine annealing)
        eta_min: Minimum learning rate (cosine annealing)
    """

    loss_functions = {
        "extreme_heat_loss": extreme_heat_loss,
        # Add other loss functions here if needed
    }

    if alternate_loss_fn is None: # use mse if not specified
        use_custom_loss = False
        criterion = nn.MSELoss()
    else:
        use_custom_loss = True
        criterion = loss_functions[alternate_loss_fn]

    # convert stats to torch tensors to un-normalize if needed
    if alternate_loss_fn in {"extreme_heat_loss"} and stats_out is not None:
        mean_out = torch.from_numpy(stats_out['mean']).float().to(device)
        std_out = torch.from_numpy(stats_out['std']).float().to(device)

    optimizer = optim.Adam(model.parameters(),
                           lr=lr,
                           weight_decay=weight_decay)

    # Add cosine annealing with warm restarts scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
    )

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    # Track training time
    train_start_time = time.time()

    for epoch in range(1, epochs + 1):
        # --- training step ---
        model.train()
        train_loss = 0.0
        for x_batch, y_batch, lead_time_batch, doy_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            lead_time_batch, doy_batch = lead_time_batch.to(device), doy_batch.to(device)

            optimizer.zero_grad()

            # Pass lead time and day-of-year features to model
            # Model predicts the error, so add to input forecast
            pred_error = model(x_batch, lead_time_batch, doy_batch)

            # Add predicted error to input forecast to get final prediction
            preds = x_batch + pred_error

            # some custom loss functions need un-normalized values
            if alternate_loss_fn in {"extreme_heat_loss"}:
                loss = criterion(preds, y_batch, std_out, mean_out)

            else:
                loss = criterion(preds, y_batch)

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)
        train_loss /= len(train_loader.dataset)

        # --- validation step ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch, lead_time_batch, doy_batch in valid_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                lead_time_batch, doy_batch = lead_time_batch.to(device), doy_batch.to(device)

                pred_error = model(x_batch, lead_time_batch, doy_batch)

                # Add predicted error to input forecast to get final prediction
                preds = x_batch + pred_error

                if alternate_loss_fn in {"extreme_heat_loss"}:
                    loss = criterion(preds, y_batch, std_out, mean_out)
                else:
                    loss = criterion(preds, y_batch)

                val_loss += loss.item() * x_batch.size(0)
        val_loss /= len(valid_loader.dataset)

        # --- learning rate scheduling ---
        scheduler.step()

        # Print progress every 10 epochs (optional, shows LR changes)
        if epoch % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}, LR: {current_lr:.2e}")

    # Calculate training time in minutes
    train_end_time = time.time()
    training_time_minutes = (train_end_time - train_start_time) / 60.0
    
    # Load best weights
    model.load_state_dict(best_model_wts)
    return model, training_time_minutes


def apply_correction(model, forecast_data, lead_time_indices, day_of_year_features, device):
    """
    Apply the correction to forecast data with lead times and day-of-year features.
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
            doy_batch = torch.from_numpy(day_of_year_features[i:end_idx]).float().to(device)
            predicted_error = model(x_batch, lt_batch, doy_batch).cpu().numpy()
            corrected_batch = x_batch.cpu().numpy() + predicted_error
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
                    mean_error = training_mean_forecast_error[key]
                    if hasattr(mean_error, 'compute'): # force computation to deal with it being slowly loaded
                        mean_error = mean_error.compute()
                    mean_corrected = original_lt[var_idx] - training_mean_forecast_error[key]

                    # Ensure it's a numpy array
                    if hasattr(mean_corrected, 'compute'):
                        mean_corrected = mean_corrected.compute()
                    elif not isinstance(mean_corrected, np.ndarray):
                        mean_corrected = np.array(mean_corrected)

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

    # Create dataset and FORCE COMPUTATION before saving
    ds_out = xr.Dataset(data_vars)
    
    # Compute all dask arrays before saving
    print("Computing all variables before saving...")
    ds_out = ds_out.compute()

    # Add metadata
    ds_out.attrs['description'] = f'Original and corrected forecasts from {model_name} using MLP fine-tuning'
    ds_out.attrs['lead_times_hours'] = lead_times
    ds_out.attrs['training_time_minutes'] = training_time_minutes if training_time_minutes is not None else -1
    
    # Save to zarr with consistent chunking (without custom compression)
    encoding = {}
    for var_name in ds_out.data_vars:
        encoding[var_name] = {
            'chunks': (365, 20, 20)  # Just specify chunks, use default compression
        }
    
    # Save to zarr
    output_path = os.path.expanduser(output_path)
    ds_out.to_zarr(output_path, mode='w')
    print(f"Forecasts saved to {output_path}")


def run_subregion_experiment(lat_vals, lon_vals, output_path, args, data_dir, device, patch_num=None):
    """
    Run experiment with multiple lead times and month encoding.
    """
    start_time = time.time()

    # Load training data
    (fc, fc_output, obs, lead_time_indices, day_of_year_features, train_times, lat_u, lon_u,
     n_lat, n_lon, n_training_vars, n_output_vars, training_mean_forecast_error) = \
        load_forecasts(data_dir, args, lat_vals, lon_vals, train=True,
                       patch_num=patch_num)

    loading_time = time.time()
    print(f"Data loaded in {(loading_time - start_time) / 60:.2f} minutes")

    # Normalize data
    stats_train = {'mean': fc.mean(0), 'std': fc.std(0) + 1e-8}
    stats_out = {'mean': fc_output.mean(0), 'std': fc_output.std(0) + 1e-8}

    fc_norm = (fc - stats_train['mean']) / stats_train['std']

    # normalize target observations using forecasts to be corrected
    obs_norm = (obs - stats_out['mean']) / stats_out['std']

    # Split train/validation
    n_train = len(fc)
    idx = np.arange(n_train)
    np.random.shuffle(idx)
    split = int(0.8 * n_train)
    t_idx, v_idx = idx[:split], idx[split:]

    train_loader = create_dataloader(fc_norm[t_idx], obs_norm[t_idx],
                                    lead_time_indices[t_idx], day_of_year_features[t_idx],
                                    batch_size=128)
    val_loader = create_dataloader(fc_norm[v_idx], obs_norm[v_idx],
                                  lead_time_indices[v_idx], day_of_year_features[v_idx],
                                  batch_size=128)

    # Initialize model
    input_dim = n_training_vars * n_lat * n_lon
    output_dim = n_output_vars * n_lat * n_lon
    n_lead_times = len(args.lead_time_hours)

    if hasattr(args, 'nn_architecture') and args.nn_architecture== 'unet':
        print(f"Using UNet with {n_lead_times} lead times and month encoding")
        print(f"  UNet hidden_dim: {args.unet_hidden_dim}")
        print(f"  UNet dropout: {args.unet_dropout}")
        model = UNet(input_dim, args.unet_hidden_dim, output_dim, n_lat=n_lat, n_lon=n_lon,
                     n_input_vars=n_training_vars, n_output_vars=n_output_vars,
                     n_lead_times=n_lead_times, dropout_rate=args.unet_dropout).to(device)
        num_epochs = 200
    else:
        print(f"Using SimpleMLP with {n_lead_times} lead times and month encoding")
        print(f"  MLP hidden_dim: {args.mlp_hidden_dim}")
        print(f"  MLP num_layers: {args.mlp_num_layers}")
        print(f"  MLP dropout: {args.mlp_dropout}")
        model = SimpleMLP(input_dim = input_dim,
                          hidden_dim = args.mlp_hidden_dim,
                          output_dim = output_dim,
                          num_hidden_layers= args.mlp_num_layers,
                          n_lead_times=n_lead_times,
                          lead_time_embedding_dim=4,
                          dropout_rate=args.mlp_dropout
                          ).to(device)
        num_epochs = 750

    # Train model
    model, training_time_minutes = train_model(model, train_loader, val_loader,
                                                epochs=num_epochs, lr=8.669714431623457e-06,
                                                device=device,
                                                weight_decay=5.210913466175803e-06,
                                                stats_out=stats_out, # used to un-normalize outputs for some loss fns
                                                alternate_loss_fn=args.alternate_loss_fn,
                                                T_0=5,  # First restart after 5 epochs
                                                T_mult=3,  
                                                eta_min=1.4365296743890787e-07
                                              )
    print(f"Training complete in {training_time_minutes:.2f} minutes")

    load_time = time.time()
    # Load test data
    (test_fc, test_fc_output, test_obs, test_lead_time_indices, test_day_of_year_features,
     test_times, _, _, _, _, _, _, _) = \
        load_forecasts(data_dir, args, lat_vals, lon_vals, train=False, patch_num=patch_num)

    # Apply correction
    test_fc_norm = (test_fc - stats_train['mean']) / stats_train['std']
    corrected = apply_correction(model, test_fc_norm, test_lead_time_indices,
                                test_day_of_year_features, device)
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
    base_path = os.path.join(args.output_dir, generate_output_path(args))

    # Setup device & seeds
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else
                          'cpu')
    torch.manual_seed(58)
    random.seed(58)

    region_lat, region_lon = get_region_grid(args)
    nlat_patch, nlon_patch = get_patch_shape(args)

    # Prepare data: check if exists, download if necessary
    print("\nPreparing data for finetuning...")
    prepare_data_for_finetuning(
        data_dir=args.data_dir,
        model_name=args.model_name,
        ground_truth_source=args.ground_truth_source,
        region=args.region,
        training_vars=args.training_vars,
        output_vars=args.output_vars,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        lead_time_hours=args.lead_time_hours,
        region_lat=region_lat,
        region_lon=region_lon
    )
    print("Data preparation complete. Proceeding with finetuning...\n")

    # Decide if we're in a climate‐zone region or a geographic one
    if args.region in CLIMATE_ZONE_MAP or args.region in TOPO_ZONE_MAP:
        
        if args.region in CLIMATE_ZONE_MAP:
            patches_path = os.path.join(dirs["processed"], f"climate_zone_patches_{args.region}_{args.subregion}.npy")
        elif args.region in TOPO_ZONE_MAP:
            patches_path = os.path.join(dirs["processed"], f"topo_zone_patches_{args.region}_{args.subregion}.npy")
        else:
            raise ValueError(f'Unknown file path for region {args.region}')
    
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