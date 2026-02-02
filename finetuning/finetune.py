#!/usr/bin/env python3
"""
Author: Ozma Houck 
Filename: finetune/finetune.py

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
from html import parser
import os
import socket
import random
import glob
import math
import json
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
from finetuning.prepare_forecasts_and_targets import load_forecasts, load_forecasts_classification
from finetuning.custom_loss_fns import (
    mortality_weighted_loss, extreme_heat_loss, quantile_loss,
    heatwave_loss, HeatWaveBatchSampler,
    CLASSIFICATION_LOSS_FNS, N_CLASSES, DEFAULT_CLASS_WEIGHTS, LABEL_GENERATORS
)

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
# Map continents (for 6x6 degree patch-based training)
CONTINENT_MAP = {
    'africa': 1,
    'asia': 2,
    'europe': 3,
    'north_america': 4,
    'south_america': 5,
    'oceania': 6,
}

# Purpose: save patches of of climate zones to be used for bootstrapping
# ------------------------------
# Simple MLP definition with lead time and day-of-year encoding
# ------------------------------
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, output_dim=1, num_hidden_layers=6,
                n_lead_times=1, lead_time_embedding_dim=8, dropout_rate=0.25):
        super(SimpleMLP, self).__init__()

        # Lead time embedding
        self.n_lead_times = n_lead_times
        self.lead_time_embedding = None

        # Day-of-year features (sin/cos) - 2 features
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
# Classifier MLP for classification-based loss functions (e.g., heatwave_loss)
# ------------------------------
class ClassifierMLP(nn.Module):
    """
    Generic MLP classifier for weather event classification.

    Designed to be reusable for any classification task (heatwave duration, drought severity, etc.)
    Unlike SimpleMLP which predicts temperature corrections, this outputs class logits.

    Input: Concatenated forecasts from all lead times for a single pixel
           [batch_size, n_vars × n_lead_times]
           Plus day-of-year features [batch_size, 2]

    Output: Logits for n_classes [batch_size, n_classes]

    For heatwave classification:
        Class 0: No heatwave (0 days above threshold)
        Class 1: Short heatwave (1-2 days)
        Class 2: Medium heatwave (3-5 days)
        Class 3: Long heatwave (6+ days)
    """

    def __init__(self, input_dim, n_classes, hidden_dim=512,
                 num_hidden_layers=4, dropout_rate=0.3):
        """
        Initialize the classifier.

        Args:
            input_dim: Number of input features (n_vars × n_lead_times for a single pixel)
            n_classes: Number of output classes (e.g., 4 for heatwave duration)
            hidden_dim: Hidden layer size (default 512)
            num_hidden_layers: Number of hidden layers (default 4)
            dropout_rate: Dropout probability (default 0.3)
        """
        super(ClassifierMLP, self).__init__()

        self.n_classes = n_classes

        # Day-of-year features (sin/cos) - 2 features added to input
        actual_input_dim = input_dim + 2

        # Build network layers
        layers = [nn.Linear(actual_input_dim, hidden_dim), nn.ReLU()]
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))

        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

        # Output layer produces logits for each class (no softmax - handled by CrossEntropyLoss)
        layers.append(nn.Linear(hidden_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x, day_of_year_features=None):
        """
        Forward pass.

        Args:
            x: Input features [batch_size, n_vars × n_lead_times]
            day_of_year_features: [batch_size, 2] sin/cos encoding of day-of-year

        Returns:
            logits: [batch_size, n_classes] raw scores for each class
        """
        if day_of_year_features is not None:
            x = torch.cat([x, day_of_year_features], dim=-1)
        return self.net(x)


# ------------------------------
# U-Net definition with lead time and day-of-year encoding
# ------------------------------
class UNet(nn.Module):
    """
    U-Net architecture with channel concatenation conditioning for weather forecast bias correction.
    Incorporates lead time and day-of-year information by concatenating them as extra input channels.
    """

    def __init__(self, input_dim, hidden_dim=128, output_dim=1,
                 n_lat=None, n_lon=None, n_input_vars=None, n_output_vars=None,
                 n_lead_times=1, lead_time_embedding_dim=16, dropout_rate=0.1):
        """
        Initialize U-Net with concatenation-based conditioning.

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
        # Note: First encoder block takes n_input_vars + total_embedding_dim channels
        # because we concatenate conditioning as extra channels
        self._build_encoder()
        self._build_decoder()

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

        # First encoder block takes input vars + conditioning channels
        in_ch = self.n_input_vars + self.total_embedding_dim
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
            out_ch = min(out_ch * 2, 128)  # Cap at 128 channels
    
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
    
    def forward(self, x, lead_time_idx=None, day_of_year_features=None):
        """
        Forward pass through U-Net with concatenation-based conditioning.

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
            conditioning = torch.cat(conditioning_vectors, dim=1)  # [batch, total_embedding_dim]
        else:
            # If no conditioning, create zeros
            conditioning = torch.zeros(batch_size, self.total_embedding_dim, device=x.device)

        # Broadcast conditioning to spatial dimensions and concatenate as extra channels
        # conditioning shape: [batch, total_embedding_dim] -> [batch, total_embedding_dim, height, width]
        conditioning_spatial = conditioning.view(batch_size, self.total_embedding_dim, 1, 1)
        conditioning_spatial = conditioning_spatial.expand(batch_size, self.total_embedding_dim, self.height, self.width)

        # Concatenate conditioning as extra input channels
        x = torch.cat([x, conditioning_spatial], dim=1)  # [batch, n_input_vars + total_embedding_dim, height, width]

        # Encoder path with skip connections
        encoder_outputs = []

        for i in range(self.num_levels):
            # Apply convolution
            x = self.encoders[i](x)

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

            # Handle size mismatches due to odd dimensions in pooling
            # This can happen when spatial dims aren't perfectly divisible by 2^n
            if x.shape[2:] != skip_connection.shape[2:]:
                # Center crop the skip connection to match upsampled size
                diff_h = skip_connection.shape[2] - x.shape[2]
                diff_w = skip_connection.shape[3] - x.shape[3]

                # Calculate crop offsets (center crop)
                h_start = diff_h // 2
                w_start = diff_w // 2
                h_end = h_start + x.shape[2]
                w_end = w_start + x.shape[3]

                skip_connection = skip_connection[:, :, h_start:h_end, w_start:w_end]

            # Concatenate with skip connection
            x = torch.cat([x, skip_connection], dim=1)

            # Apply decoder convolution
            x = self.decoders[i](x)
        
        # Final 1x1 convolution to get output channels
        x = self.final_conv(x)

        # Pad output back to original spatial dimensions if needed
        # This handles cases where pooling/upsampling with odd dimensions
        # results in smaller output than input
        current_h, current_w = x.shape[2], x.shape[3]
        if current_h != self.height or current_w != self.width:
            pad_h = self.height - current_h
            pad_w = self.width - current_w
            # Pad symmetrically: (left, right, top, bottom)
            padding = (pad_w // 2, pad_w - pad_w // 2,
                      pad_h // 2, pad_h - pad_h // 2)
            x = F.pad(x, padding, mode='replicate')

        # Reshape back to flat output
        x = x.view(batch_size, -1)

        return x

# ------------------------------
# Load optimal hyperparameters
# ------------------------------
def load_optimal_hyperparameters(architecture, training_vars, output_var):
    """
    Load optimal hyperparameters from hyperopt results.
    
    Args:
        architecture: 'mlp' or 'unet'
        training_vars: List of input variable names
        output_var: Output variable name 
    
    Returns:
        Dictionary of optimal hyperparameters, or None if file not found
    """
    # Get the script's directory
    script_dir = Path(__file__).parent.parent

    # argparse may provide output_vars as a list (nargs='+'); ensure we have a single string
    if isinstance(output_var, (list, tuple)):
        if len(output_var) == 0:
            print("Warning: output_var list is empty")
            return None
        output_var = output_var[0]

    # if using multiple training variables, use those hyperopt results
    if len(training_vars) > 1:
        multi_flag = "_multivar"
    else:
        multi_flag = ""

    if output_var == "2m_temperature":
        results_file = script_dir / f"hyperopt_results{multi_flag}_temperature_{architecture}" / f"optimization_results_{architecture}.json"
    elif output_var == "10m_wind_speed":
        results_file = script_dir / f"hyperopt_results{multi_flag}_wind_{architecture}" / f"optimization_results_{architecture}.json"
    else:
        print(f"Warning: No hyperparameter results available for output variable '{output_var}'")
        return None
    
    if not results_file.exists():
        print(f"Warning: Hyperparameter file not found at {results_file}")
        return None
    
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        print(f"\nLoaded optimal hyperparameters for {architecture} from {results_file}")
        print(f"  Best loss: {results['best_loss']:.6f}")
        print(f"  Evaluations: {results['n_evaluations']}")
        return results['best_hyperparams']
    except Exception as e:
        print(f"Warning: Could not load hyperparameters from {results_file}: {e}")
        return None

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
    parser.add_argument('--alternate_loss_fn', type=str, default=None,
                        choices=['quantile_loss', 'extreme_heat_loss', 'mortality_weighted_loss', 'heatwave_loss'],)

    # Architecture hyperparameters
    parser.add_argument('--mlp_hidden_dim', type=int, default=1024,
                    help='Hidden dimension for MLP (default: 1024, from mlp_moderate)')
    parser.add_argument('--mlp_num_layers', type=int, default=6,
                    help='Number of hidden layers for MLP (default: 6, from mlp_moderate)')
    parser.add_argument('--mlp_dropout', type=float, default=0.25,
                        help='Dropout rate for MLP (default: 0.25, from mlp_moderate)')
    parser.add_argument('--unet_hidden_dim', type=int, default=64,
                        help='Base number of channels for UNet (default: 64, from unet_medium)')
    parser.add_argument('--unet_dropout', type=float, default=0.1,
                        help='Dropout rate for UNet')

    return parser.parse_args()

# ------------------------------
# Region grid and patch helpers
# ------------------------------

# Region center points (lat, lon in degrees)
REGION_CENTERS = {
    'india': (22.0, 77.0),
    'usa_south': (35.0, 260.0),  # -100 + 360
    'amazon': (-5.0, 295.0),  # -65 + 360
    'british_columbia': (53.125, 235.0),  # -125 + 360, using 53.125 to match previous 48.25-58 range
    'pakistan': (29.5, 65.0),
    'ethiopia': (9.0, 39.0),
    'corn_belt': (41.0, 270.0),  # -90 + 360
    'finland': (65.0, 29.0),
}

def get_region_grid(args):
    """
    Return full region latitude and longitude arrays based on center point and subregion size.

    For standard regions, uses the center point defined in REGION_CENTERS and expands based
    on the subregion argument (e.g., '6x6' means 6 degrees in each direction from center).

    For special regions (global, climate zones, etc.), returns full global grid.
    """
    # Handle special regions that use full global grid
    if args.region == "global" or args.region in CLIMATE_ZONE_MAP or args.region in TOPO_ZONE_MAP or args.region in CONTINENT_MAP:
        lat0, lat1 = -90, 90
        lon0, lon1 = 0, 360
        lat_values = np.arange(lat0, lat1, 0.25)
        lon_values = np.arange(lon0, lon1, 0.25)
        return lat_values, lon_values

    # Get center point for region
    if args.region not in REGION_CENTERS:
        raise ValueError(f"Unknown region '{args.region}'. Available regions: {list(REGION_CENTERS.keys())}")

    lat_center, lon_center = REGION_CENTERS[args.region]

    # Parse subregion size (e.g., '6x6' -> 6 degrees lat, 6 degrees lon)
    deg_lat, deg_lon = map(int, args.subregion.split('x'))

    # Calculate bounds: center +/- half the subregion size
    lat0 = lat_center - (deg_lat / 2)
    lat1 = lat_center + (deg_lat / 2)
    lon0 = lon_center - (deg_lon / 2)
    lon1 = lon_center + (deg_lon / 2)

    # Generate lat/lon arrays at 0.25 degree resolution
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


def create_dataloader(forecast_input_data, forecast_output_data, obs_data, lead_time_indices, day_of_year_features, batch_size, device=None):
    """
    Create a PyTorch DataLoader from forecast input, forecast output, observation data, lead time indices, and day-of-year features.

    Args:
        forecast_input_data: Training variables forecast (e.g., 6 vars for UNet input)
        forecast_output_data: Output variables forecast (e.g., 1 var to correct)
        obs_data: Observations for output variables
        device: Device being used (for optimization settings)
    """
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(forecast_input_data).float(),
        torch.from_numpy(forecast_output_data).float(),
        torch.from_numpy(obs_data).float(),
        torch.from_numpy(lead_time_indices).long(),
        torch.from_numpy(day_of_year_features).float()
    )

    # Optimize DataLoader based on device and available CPU cores
    pin_memory = False
    num_workers = 0
    if device is not None and device.type == 'cuda':
        # Auto-detect available CPU cores
        cpu_count = os.cpu_count() or 1
        # Use min(cpu_count - 1, 4) to leave 1 core for main process
        # But if only 1-2 cores available, use 0 workers (main process handles it)
        if cpu_count <= 2:
            num_workers = 0  # Too few cores, use main process
            pin_memory = True
        else:
            num_workers = min(cpu_count - 1, 4)  # Leave 1 core free, cap at 4
            pin_memory = True

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             pin_memory=pin_memory,
                                             num_workers=num_workers,
                                             persistent_workers=num_workers > 0)
    return dataloader


def create_heatwave_dataloader(forecast_input_data, forecast_output_data, obs_data,
                               lead_time_indices, day_of_year_features,
                               n_lead_times, batch_size_timestamps, device=None):
    """
    Create a PyTorch DataLoader with HeatWaveBatchSampler for grouped timestamp batching.
    Required for heatwave_loss since we need all lead times for each timestamp together.

    Filters out incomplete timestamp groups (due to NaN removal) before creating the loader.

    Args:
        forecast_input_data: Training variables forecast [n_samples, n_features]
        forecast_output_data: Output variables forecast [n_samples, n_output_features]
        obs_data: Observations for output variables [n_samples, n_output_features]
        lead_time_indices: Lead time index for each sample [n_samples]
        day_of_year_features: Day-of-year sin/cos features [n_samples, 2]
        n_lead_times: Number of lead times per timestamp
        batch_size_timestamps: Number of timestamps per batch (actual batch size = this * n_lead_times)
        device: Device being used (for optimization settings)

    Returns:
        DataLoader with HeatWaveBatchSampler
    """
    n_samples = forecast_input_data.shape[0]

    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(forecast_input_data).float(),
        torch.from_numpy(forecast_output_data).float(),
        torch.from_numpy(obs_data).float(),
        torch.from_numpy(lead_time_indices).long(),
        torch.from_numpy(day_of_year_features).float()
    )

    # Create batch sampler that ensures complete timestamp groups
    batch_sampler = HeatWaveBatchSampler(
        n_samples=n_samples,
        n_lead_times=n_lead_times,
        lead_time_indices=lead_time_indices,
        batch_size_timestamps=batch_size_timestamps,
        shuffle=True
    )

    # Optimize DataLoader based on device
    pin_memory = False
    num_workers = 0
    if device is not None and device.type == 'cuda':
        cpu_count = os.cpu_count() or 1
        if cpu_count <= 2:
            num_workers = 0
            pin_memory = True
        else:
            num_workers = min(cpu_count - 1, 4)
            pin_memory = True

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=num_workers > 0
    )

    print(f"HeatWave DataLoader: {batch_sampler.n_valid_timestamps} valid timestamps, "
          f"{len(batch_sampler)} batches of {batch_size_timestamps} timestamps each")

    return dataloader


def create_classification_dataloader(forecast_data, labels, day_of_year_features,
                                     batch_size, device=None, shuffle=True):
    """
    Create DataLoader for classification tasks (e.g., heatwave duration classification).

    Unlike create_heatwave_dataloader, this doesn't need HeatWaveBatchSampler because
    each sample is already a complete (timestamp, pixel) unit with all lead times concatenated.

    Args:
        forecast_data: Concatenated forecasts [n_samples, n_vars × n_lead_times]
        labels: Class labels [n_samples] (integer class indices)
        day_of_year_features: Day-of-year sin/cos features [n_samples, 2]
        batch_size: Number of samples per batch
        device: Device being used (for optimization settings)
        shuffle: Whether to shuffle data (default True)

    Returns:
        DataLoader yielding (forecast_data, labels, day_of_year_features) tuples
    """
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(forecast_data).float(),
        torch.from_numpy(labels).long(),
        torch.from_numpy(day_of_year_features).float()
    )

    # Optimize DataLoader based on device
    pin_memory = False
    num_workers = 0
    if device is not None and device.type == 'cuda':
        cpu_count = os.cpu_count() or 1
        if cpu_count <= 2:
            num_workers = 0
            pin_memory = True
        else:
            num_workers = min(cpu_count - 1, 4)
            pin_memory = True

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=num_workers > 0
    )

    return dataloader


def train_model(model, train_loader, valid_loader, epochs, lr, device,
                weight_decay=0,
                stats_out=None, alternate_loss_fn=None,
                patience=50, min_delta=1e-5,
                scheduler_patience=10, scheduler_factor=0.5, min_lr=1e-7,
                n_lead_times=None, lead_time_days=None):
    """
    Train the model over multiple epochs with ReduceLROnPlateau and early stopping.
    Uses mixed precision training for CUDA devices to improve speed.

    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        valid_loader: DataLoader for validation data
        epochs: Maximum number of epochs to train
        lr: Initial learning rate
        device: Device to train on (cpu/cuda/mps)
        weight_decay: L2 regularization weight
        stats_out: Statistics for denormalizing outputs (for custom losses)
        alternate_loss_fn: Name of custom loss function to use
        patience: Early stopping patience (epochs without improvement)
        min_delta: Minimum change in validation loss to qualify as improvement
        scheduler_patience: Number of epochs with no improvement before reducing LR
        scheduler_factor: Factor by which to reduce learning rate (new_lr = lr * factor)
        min_lr: Minimum learning rate (floor for scheduler)
        n_lead_times: Number of lead times (required for heatwave_loss)
        lead_time_days: List of lead time values in days (required for heatwave_loss)
    """

    # Loss functions that require denormalization (is_normalized=True)
    NORMALIZED_LOSS_FNS = {"extreme_heat_loss", "mortality_weighted_loss", "heatwave_loss"}

    loss_functions = {
        "extreme_heat_loss": extreme_heat_loss,
        "mortality_weighted_loss": mortality_weighted_loss,
        "quantile_loss": quantile_loss,
        "heatwave_loss": heatwave_loss
    }

    if alternate_loss_fn is None:  # use mse if not specified
        use_custom_loss = False
        criterion = nn.MSELoss()
    else:
        use_custom_loss = True
        criterion = loss_functions[alternate_loss_fn]

    # convert stats to torch tensors for denormalization if needed
    mean_out = None
    std_out = None
    if alternate_loss_fn in NORMALIZED_LOSS_FNS and stats_out is not None:
        mean_out = torch.from_numpy(stats_out['mean']).float().to(device)
        std_out = torch.from_numpy(stats_out['std']).float().to(device)


    optimizer = optim.Adam(model.parameters(),
                           lr=lr,
                           weight_decay=weight_decay)

    # Add ReduceLROnPlateau scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=scheduler_factor,
        patience=scheduler_patience, min_lr=min_lr
    )

    # Setup mixed precision training for CUDA
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    if use_amp:
        print("Using mixed precision training (AMP) for faster GPU training")

    # Determine if non_blocking transfers should be used
    non_blocking = device.type == 'cuda'

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    # Track training time
    train_start_time = time.time()

    for epoch in range(1, epochs + 1):
        # --- training step ---
        model.train()
        train_loss = 0.0
        for fc_input_batch, fc_output_batch, y_batch, lead_time_batch, doy_batch in train_loader:
            fc_input_batch = fc_input_batch.to(device, non_blocking=non_blocking)
            fc_output_batch = fc_output_batch.to(device, non_blocking=non_blocking)
            y_batch = y_batch.to(device, non_blocking=non_blocking)
            lead_time_batch = lead_time_batch.to(device, non_blocking=non_blocking)
            doy_batch = doy_batch.to(device, non_blocking=non_blocking)

            optimizer.zero_grad()

            # Use automatic mixed precision for CUDA
            if use_amp:
                with torch.amp.autocast("cuda"):
                    # Pass training variables, lead time and day-of-year features to model
                    # Model predicts the error to apply to the output forecast variables
                    pred_error = model(fc_input_batch, lead_time_batch, doy_batch)

                    # Add predicted error to output forecast to get final prediction
                    preds = fc_output_batch + pred_error

                    # Custom loss functions use is_normalized=True since inputs are normalized
                    if alternate_loss_fn == "heatwave_loss":
                        loss = criterion(preds, y_batch, lead_time_batch, n_lead_times,
                                        is_normalized=True, std_out=std_out, mean_out=mean_out,
                                        lead_time_days=lead_time_days)
                    elif alternate_loss_fn in NORMALIZED_LOSS_FNS:
                        loss = criterion(preds, y_batch, is_normalized=True,
                                        std_out=std_out, mean_out=mean_out)
                    else:
                        loss = criterion(preds, y_batch)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training for CPU/MPS
                pred_error = model(fc_input_batch, lead_time_batch, doy_batch)
                preds = fc_output_batch + pred_error

                if alternate_loss_fn == "heatwave_loss":
                    loss = criterion(preds, y_batch, lead_time_batch, n_lead_times,
                                    is_normalized=True, std_out=std_out, mean_out=mean_out,
                                    lead_time_days=lead_time_days)
                elif alternate_loss_fn in NORMALIZED_LOSS_FNS:
                    loss = criterion(preds, y_batch, is_normalized=True,
                                    std_out=std_out, mean_out=mean_out)
                else:
                    loss = criterion(preds, y_batch)

                loss.backward()
                optimizer.step()

            train_loss += loss.item() * fc_output_batch.size(0)
        train_loss /= len(train_loader.dataset)

        # --- validation step ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for fc_input_batch, fc_output_batch, y_batch, lead_time_batch, doy_batch in valid_loader:
                fc_input_batch = fc_input_batch.to(device, non_blocking=non_blocking)
                fc_output_batch = fc_output_batch.to(device, non_blocking=non_blocking)
                y_batch = y_batch.to(device, non_blocking=non_blocking)
                lead_time_batch = lead_time_batch.to(device, non_blocking=non_blocking)
                doy_batch = doy_batch.to(device, non_blocking=non_blocking)

                if use_amp:
                    with torch.amp.autocast('cuda'):
                        pred_error = model(fc_input_batch, lead_time_batch, doy_batch)
                        preds = fc_output_batch + pred_error

                        if alternate_loss_fn == "heatwave_loss":
                            loss = criterion(preds, y_batch, lead_time_batch, n_lead_times,
                                            is_normalized=True, std_out=std_out, mean_out=mean_out,
                                            lead_time_days=lead_time_days)
                        elif alternate_loss_fn in NORMALIZED_LOSS_FNS:
                            loss = criterion(preds, y_batch, is_normalized=True,
                                            std_out=std_out, mean_out=mean_out)
                        else:
                            loss = criterion(preds, y_batch)
                else:
                    pred_error = model(fc_input_batch, lead_time_batch, doy_batch)
                    preds = fc_output_batch + pred_error

                    if alternate_loss_fn == "heatwave_loss":
                        loss = criterion(preds, y_batch, lead_time_batch, n_lead_times,
                                        is_normalized=True, std_out=std_out, mean_out=mean_out,
                                        lead_time_days=lead_time_days)
                    elif alternate_loss_fn in NORMALIZED_LOSS_FNS:
                        loss = criterion(preds, y_batch, is_normalized=True,
                                        std_out=std_out, mean_out=mean_out)
                    else:
                        loss = criterion(preds, y_batch)

                val_loss += loss.item() * fc_output_batch.size(0)
        val_loss /= len(valid_loader.dataset)

        # --- learning rate scheduling (based on validation loss) ---
        scheduler.step(val_loss)

        # --- early stopping check ---
        if val_loss + min_delta < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Print progress every 10 epochs
        if epoch % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}, LR: {current_lr:.2e}, "
                  f"Best Val: {best_val_loss:.6f}, Patience: {epochs_without_improvement}/{patience}")

        # Check for early stopping
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


def train_classifier(model, train_loader, valid_loader, epochs, lr, device,
                     class_weights=None, weight_decay=0,
                     patience=50, min_delta=1e-5,
                     scheduler_patience=10, scheduler_factor=0.5, min_lr=1e-7,
                     loss_type="cross_entropy", focal_gamma=2.0):
    """
    Train classification model (e.g., ClassifierMLP) with configurable loss function.

    Args:
        model: PyTorch classification model to train
        train_loader: DataLoader yielding (features, labels, doy_features)
        valid_loader: DataLoader for validation
        epochs: Maximum number of epochs
        lr: Initial learning rate
        device: Device to train on (cpu/cuda/mps)
        class_weights: Per-class weights for loss function (e.g., [1, 2, 3, 4])
        weight_decay: L2 regularization weight
        patience: Early stopping patience (epochs without improvement)
        min_delta: Minimum change in validation loss to qualify as improvement
        scheduler_patience: Number of epochs with no improvement before reducing LR
        scheduler_factor: Factor by which to reduce learning rate
        min_lr: Minimum learning rate floor
        loss_type: Type of loss function - "cross_entropy" or "focal" (default: "cross_entropy")
        focal_gamma: Focusing parameter for focal loss (default: 2.0). Higher values
                     focus more on hard examples. Only used if loss_type="focal".

    Returns:
        model: Trained model with best weights
        training_time_minutes: Training duration in minutes
        metrics: Dict with final training and validation metrics
    """
    # Setup loss function based on loss_type
    if loss_type == "focal":
        from finetuning.custom_loss_fns import focal_loss
        # For focal loss, class_weights are used as alpha parameter
        alpha = class_weights
        def criterion(logits, labels):
            return focal_loss(logits, labels, alpha=alpha, gamma=focal_gamma, device=device)
        print(f"  Using focal loss with gamma={focal_gamma}, alpha={alpha}")
    else:
        # Standard cross-entropy loss
        if class_weights is not None:
            weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
            criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=scheduler_factor,
        patience=scheduler_patience, min_lr=min_lr
    )

    # Setup mixed precision for CUDA
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    non_blocking = device.type == 'cuda'

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    train_start_time = time.time()

    for epoch in range(1, epochs + 1):
        # --- Training step ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X_batch, labels_batch, doy_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=non_blocking)
            labels_batch = labels_batch.to(device, non_blocking=non_blocking)
            doy_batch = doy_batch.to(device, non_blocking=non_blocking)

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits = model(X_batch, doy_batch)
                    loss = criterion(logits, labels_batch)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(X_batch, doy_batch)
                loss = criterion(logits, labels_batch)
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels_batch).sum().item()
            train_total += X_batch.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # --- Validation step ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, labels_batch, doy_batch in valid_loader:
                X_batch = X_batch.to(device, non_blocking=non_blocking)
                labels_batch = labels_batch.to(device, non_blocking=non_blocking)
                doy_batch = doy_batch.to(device, non_blocking=non_blocking)

                if use_amp:
                    with torch.amp.autocast('cuda'):
                        logits = model(X_batch, doy_batch)
                        loss = criterion(logits, labels_batch)
                else:
                    logits = model(X_batch, doy_batch)
                    loss = criterion(logits, labels_batch)

                val_loss += loss.item() * X_batch.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels_batch).sum().item()
                val_total += X_batch.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        # --- Learning rate scheduling ---
        scheduler.step(val_loss)

        # --- Early stopping check ---
        if val_loss + min_delta < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Print progress every 10 epochs
        if epoch % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}/{epochs} - "
                  f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.4f}, "
                  f"LR: {current_lr:.2e}, Patience: {epochs_without_improvement}/{patience}")

        # Check for early stopping
        if epochs_without_improvement >= patience:
            print(f"→ Early stopping at epoch {epoch}. No improvement in {patience} epochs.")
            break

    # Calculate training time
    train_end_time = time.time()
    training_time_minutes = (train_end_time - train_start_time) / 60.0

    # Load best weights
    model.load_state_dict(best_model_wts)

    metrics = {
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': best_val_loss,
        'val_acc': val_acc,
    }

    return model, training_time_minutes, metrics


def find_optimal_thresholds(model, val_loader, device, n_classes, default_class=0):
    """
    Find optimal probability thresholds for each class using validation data.

    For rare-class over-prediction problems, we want thresholds that:
    - Require higher confidence to predict rare classes
    - Reduce false positives for rare classes

    The optimization uses F0.5 score which favors precision over recall,
    helping to minimize false positives for rare classes.

    Args:
        model: Trained classifier model
        val_loader: Validation DataLoader
        device: Torch device
        n_classes: Number of classes
        default_class: Class to default to when no threshold is met (default: 0)

    Returns:
        thresholds: numpy array [n_classes] of optimal probability thresholds
    """
    model.eval()
    all_probs = []
    all_labels = []

    # Collect all predictions and labels from validation set
    with torch.no_grad():
        for X_batch, labels_batch, doy_batch in val_loader:
            X_batch = X_batch.to(device)
            doy_batch = doy_batch.to(device)

            logits = model(X_batch, doy_batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.extend(labels_batch.numpy())

    all_probs = np.vstack(all_probs)  # [n_samples, n_classes]
    all_labels = np.array(all_labels)

    # Initialize thresholds - default class uses low threshold, rare classes use higher
    thresholds = np.ones(n_classes) * 0.5

    # For the default/majority class, use a lower threshold (easier to predict)
    thresholds[default_class] = 0.3

    print(f"\n  Finding optimal thresholds on validation set...")

    # For each rare class (not the default), find optimal threshold
    for cls in range(n_classes):
        if cls == default_class:
            continue  # Skip default class

        # Count samples of this class
        n_class_samples = (all_labels == cls).sum()
        if n_class_samples == 0:
            print(f"    Class {cls}: No samples in validation set, using default threshold 0.5")
            continue

        best_threshold = 0.5
        best_f_score = -float('inf')

        # Try different thresholds
        for thresh in np.arange(0.30, 0.95, 0.05):
            # Predict: use argmax but only if probability exceeds threshold
            # Otherwise default to the default_class
            preds = np.argmax(all_probs, axis=1).copy()

            # For samples where max prob class is this class but prob < threshold,
            # change prediction to default class
            mask = (preds == cls) & (all_probs[:, cls] < thresh)
            preds[mask] = default_class

            # Compute precision and recall for this class
            true_positives = ((preds == cls) & (all_labels == cls)).sum()
            false_positives = ((preds == cls) & (all_labels != cls)).sum()
            false_negatives = ((preds != cls) & (all_labels == cls)).sum()

            precision = true_positives / (true_positives + false_positives + 1e-8)
            recall = true_positives / (true_positives + false_negatives + 1e-8)

            # F0.5 score - weights precision higher than recall (beta < 1)
            # This favors fewer false positives over catching all true positives
            beta = 0.5
            f_score = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + 1e-8)

            if f_score > best_f_score:
                best_f_score = f_score
                best_threshold = thresh

        thresholds[cls] = best_threshold
        print(f"    Class {cls}: threshold={best_threshold:.2f}, F0.5={best_f_score:.4f}")

    return thresholds


def predict_with_thresholds(model, data_loader, device, thresholds, default_class=0):
    """
    Make predictions using class-specific probability thresholds.

    For rare class problems, this prevents over-prediction by requiring
    higher confidence for rare class predictions.

    Args:
        model: Trained classifier model
        data_loader: DataLoader
        device: Torch device
        thresholds: numpy array [n_classes] of probability thresholds
        default_class: Class to predict when no class exceeds its threshold

    Returns:
        predictions: numpy array [n_samples] of predicted class labels
        probabilities: numpy array [n_samples, n_classes] of class probabilities
    """
    model.eval()
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for X_batch, labels_batch, doy_batch in data_loader:
            X_batch = X_batch.to(device)
            doy_batch = doy_batch.to(device)

            logits = model(X_batch, doy_batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)

            # For each sample, find class with max probability that exceeds threshold
            for i in range(probs.shape[0]):
                # Get argmax prediction
                argmax_cls = np.argmax(probs[i])

                # Check if the argmax class exceeds its threshold
                if probs[i, argmax_cls] >= thresholds[argmax_cls]:
                    all_preds.append(argmax_cls)
                else:
                    # No class confident enough - default to majority class
                    all_preds.append(default_class)

    return np.array(all_preds), np.vstack(all_probs)


def evaluate_classifier(model, test_loader, device, n_classes, class_names=None,
                        thresholds=None, default_class=0):
    """
    Evaluate classification model and print detailed metrics.

    Args:
        model: Trained ClassifierMLP model
        test_loader: DataLoader for test data
        device: Torch device
        n_classes: Number of classes
        class_names: Optional list of class names for display
        thresholds: Optional numpy array [n_classes] of probability thresholds.
                    If provided, uses threshold-based prediction instead of argmax.
        default_class: Class to default to when using thresholds and no class
                       exceeds its threshold (default: 0)

    Returns:
        dict with:
        - overall_accuracy: float
        - per_class_accuracy: [n_classes] array
        - confusion_matrix: [n_classes, n_classes] array
        - class_counts: [n_classes] true label distribution
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, labels_batch, doy_batch in test_loader:
            X_batch = X_batch.to(device)
            doy_batch = doy_batch.to(device)

            logits = model(X_batch, doy_batch)

            if thresholds is not None:
                # Use threshold-based prediction
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                for i in range(probs.shape[0]):
                    argmax_cls = np.argmax(probs[i])
                    if probs[i, argmax_cls] >= thresholds[argmax_cls]:
                        all_preds.append(argmax_cls)
                    else:
                        all_preds.append(default_class)
            else:
                # Simple argmax prediction
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)

            all_labels.extend(labels_batch.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Compute metrics
    overall_acc = (all_preds == all_labels).mean()

    # Confusion matrix
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    for true, pred in zip(all_labels, all_preds):
        confusion[true, pred] += 1

    # Per-class accuracy (recall)
    per_class_acc = confusion.diagonal() / confusion.sum(axis=1).clip(min=1)

    # Per-class precision
    per_class_precision = confusion.diagonal() / confusion.sum(axis=0).clip(min=1)

    # Default class names
    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]

    # Print results
    print(f"\n{'='*60}")
    if thresholds is not None:
        print("CLASSIFICATION RESULTS (with probability thresholds)")
        print(f"  Thresholds: {[f'{t:.2f}' for t in thresholds]}")
    else:
        print("CLASSIFICATION RESULTS")
    print(f"{'='*60}")
    print(f"  Overall Accuracy: {overall_acc:.4f} ({int(overall_acc * len(all_labels))}/{len(all_labels)})")
    print(f"\n  Per-Class Metrics:")
    print(f"    {'Class':<15} {'Recall':>8} {'Precision':>10} {'N':>8}")
    print(f"    {'-'*15} {'-'*8} {'-'*10} {'-'*8}")
    for i, name in enumerate(class_names):
        count = confusion[i].sum()
        print(f"    {name:<15} {per_class_acc[i]:>8.4f} {per_class_precision[i]:>10.4f} {count:>8}")

    print(f"\n  Confusion Matrix:")
    print(f"    {'':15} | " + " | ".join(f"{name:>8}" for name in class_names))
    print(f"    {'-'*15}-+-" + "-+-".join(["-"*8] * n_classes))
    for i, name in enumerate(class_names):
        row = " | ".join(f"{confusion[i, j]:>8}" for j in range(n_classes))
        print(f"    {name:15} | {row}")

    print(f"{'='*60}\n")

    return {
        'overall_accuracy': overall_acc,
        'per_class_accuracy': per_class_acc,
        'per_class_precision': per_class_precision,
        'confusion_matrix': confusion,
        'class_counts': confusion.sum(axis=1),
        'thresholds_used': thresholds
    }


def apply_correction(model, forecast_input_data, forecast_output_data, lead_time_indices, day_of_year_features, device):
    """
    Apply the correction to forecast output data using forecast input data.
    Uses mixed precision for CUDA devices.

    Args:
        forecast_input_data: Training variables (e.g., 6 vars)
        forecast_output_data: Output variables to correct (e.g., 1 var)
        lead_time_indices: Lead time indices
        day_of_year_features: Day-of-year features
        device: Device to run on

    Returns:
        Corrected forecast for output variables
    """
    model.eval()
    corrected_all = []

    # Process in batches to handle memory efficiently
    batch_size = 128
    n_samples = forecast_input_data.shape[0]

    # Use AMP and non_blocking for CUDA
    use_amp = device.type == 'cuda'
    non_blocking = device.type == 'cuda'

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            fc_input_batch = torch.from_numpy(forecast_input_data[i:end_idx]).float().to(device, non_blocking=non_blocking)
            fc_output_batch = torch.from_numpy(forecast_output_data[i:end_idx]).float().to(device, non_blocking=non_blocking)
            lt_batch = torch.from_numpy(lead_time_indices[i:end_idx]).long().to(device, non_blocking=non_blocking)
            doy_batch = torch.from_numpy(day_of_year_features[i:end_idx]).float().to(device, non_blocking=non_blocking)

            if use_amp:
                with torch.cuda.amp.autocast():
                    predicted_error = model(fc_input_batch, lt_batch, doy_batch).cpu().numpy()
                    corrected_batch = fc_output_batch.cpu().numpy() + predicted_error
            else:
                predicted_error = model(fc_input_batch, lt_batch, doy_batch).cpu().numpy()
                corrected_batch = fc_output_batch.cpu().numpy() + predicted_error

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


def run_subregion_experiment(lat_vals, lon_vals, output_path, args, data_dir, device, patch_num=None, use_legacy_global_data=False):
    """
    Run experiment with multiple lead times.

    Automatically detects classification mode when alternate_loss_fn is in CLASSIFICATION_LOSS_FNS
    (e.g., 'heatwave_loss'). In classification mode:
    - Uses ClassifierMLP instead of SimpleMLP
    - Each sample is a (timestamp, pixel) with all lead times concatenated
    - Uses weighted cross-entropy loss instead of MSE
    - Outputs class predictions instead of temperature corrections
    """
    start_time = time.time()

    # Check if this is a classification task
    is_classification = (args.alternate_loss_fn is not None and
                        args.alternate_loss_fn in CLASSIFICATION_LOSS_FNS)

    # ========================================================================
    # CLASSIFICATION MODE (e.g., heatwave_loss)
    # ========================================================================
    if is_classification:
        print(f"\n{'='*70}")
        print(f"CLASSIFICATION MODE: {args.alternate_loss_fn}")
        print(f"{'='*70}")

        # Check for UNet (not supported in classification mode)
        if hasattr(args, 'nn_architecture') and args.nn_architecture == 'unet':
            raise ValueError("UNet is not supported for classification mode. Use MLP instead.")

        # Get classification-specific configuration
        from finetuning.custom_loss_fns import (
            DEFAULT_FOCAL_GAMMA, compute_class_weights
        )

        n_classes = N_CLASSES[args.alternate_loss_fn]
        label_generator = LABEL_GENERATORS[args.alternate_loss_fn]
        lead_time_days = [lt / 24.0 for lt in args.lead_time_hours]

        # Classification defaults: use focal loss and probability thresholds
        # These are the recommended settings for imbalanced classification problems
        # - Focal loss down-weights easy (majority class) examples
        # - Probability thresholds require higher confidence for rare class predictions
        loss_type = "focal"
        focal_gamma = DEFAULT_FOCAL_GAMMA.get(args.alternate_loss_fn, 2.0)
        use_probability_thresholds = True  # Default: use thresholds to reduce false positives

        # Class weights: use defaults from registry
        class_weights = DEFAULT_CLASS_WEIGHTS[args.alternate_loss_fn]

        # Class names for display
        class_names = ["No HW (0d)", "Short (1-2d)", "Medium (3-5d)", "Long (6+d)"]

        # Load training data in classification format
        (fc_concat, labels, day_of_year_features, train_times, lat_u, lon_u,
         n_lat, n_lon, n_vars, n_lead_times) = load_forecasts_classification(
            data_dir, args, lat_vals, lon_vals, train=True,
            label_generator=label_generator,
            threshold_celsius=32.0,
            use_legacy_global_data=use_legacy_global_data
        )

        loading_time = time.time()
        print(f"Data loaded in {(loading_time - start_time) / 60:.2f} minutes")

        print(f"  Using default class weights: {class_weights}")

        # Normalize forecast data
        stats = {'mean': fc_concat.mean(0), 'std': fc_concat.std(0) + 1e-8}
        fc_norm = (fc_concat - stats['mean']) / stats['std']

        # Split train/validation
        n_samples = len(fc_norm)
        idx = np.arange(n_samples)
        np.random.shuffle(idx)
        split = int(0.8 * n_samples)
        t_idx, v_idx = idx[:split], idx[split:]

        # Use optimal batch size if available
        batch_size = args.optimal_batch_size if args.optimal_batch_size else 128
        print(f"Using batch_size: {batch_size}")

        # Create classification dataloaders
        train_loader = create_classification_dataloader(
            fc_norm[t_idx], labels[t_idx], day_of_year_features[t_idx],
            batch_size=batch_size, device=device, shuffle=True
        )
        val_loader = create_classification_dataloader(
            fc_norm[v_idx], labels[v_idx], day_of_year_features[v_idx],
            batch_size=batch_size, device=device, shuffle=False
        )

        print(f"Train samples: {len(t_idx)}, Val samples: {len(v_idx)}")

        # Create classifier model
        input_dim = n_vars * n_lead_times
        print(f"\nUsing ClassifierMLP for {n_classes}-class classification")
        print(f"  Input dim: {input_dim} (n_vars={n_vars} × n_lead_times={n_lead_times})")
        print(f"  Hidden dim: {args.mlp_hidden_dim}")
        print(f"  Num layers: {args.mlp_num_layers}")
        print(f"  Dropout: {args.mlp_dropout}")
        print(f"  Loss type: {loss_type}" + (f" (gamma={focal_gamma})" if loss_type == "focal" else ""))
        print(f"  Class weights: {[f'{w:.2f}' for w in class_weights] if class_weights else 'None'}")
        print(f"  Probability thresholds: {'enabled' if use_probability_thresholds else 'disabled'}")

        model = ClassifierMLP(
            input_dim=input_dim,
            n_classes=n_classes,
            hidden_dim=args.mlp_hidden_dim,
            num_hidden_layers=args.mlp_num_layers,
            dropout_rate=args.mlp_dropout
        ).to(device)

        # Use optimal training hyperparameters if available
        lr = args.optimal_lr if args.optimal_lr else 1e-4
        weight_decay = args.optimal_weight_decay if args.optimal_weight_decay else 1e-5
        patience = args.optimal_patience if args.optimal_patience else 50
        min_delta = args.optimal_min_delta if args.optimal_min_delta else 1e-5

        print(f"\nTraining with:")
        print(f"  lr: {lr}")
        print(f"  weight_decay: {weight_decay}")
        print(f"  patience: {patience}")
        print(f"  loss_type: {loss_type}")

        # Train classifier
        model, training_time_minutes, train_metrics = train_classifier(
            model, train_loader, val_loader,
            epochs=500, lr=lr,
            device=device,
            class_weights=class_weights,
            weight_decay=weight_decay,
            patience=patience,
            min_delta=min_delta,
            loss_type=loss_type,
            focal_gamma=focal_gamma
        )
        print(f"Training complete in {training_time_minutes:.2f} minutes")
        print(f"  Final train accuracy: {train_metrics['train_acc']:.4f}")
        print(f"  Final val accuracy: {train_metrics['val_acc']:.4f}")

        # Load test data
        load_time = time.time()
        (test_fc_concat, test_labels, test_doy_features, test_times, _, _,
         _, _, _, _) = load_forecasts_classification(
            data_dir, args, lat_vals, lon_vals, train=False,
            label_generator=label_generator,
            threshold_celsius=35.0,
            use_legacy_global_data=use_legacy_global_data
        )

        # Normalize test data using training stats
        test_fc_norm = (test_fc_concat - stats['mean']) / stats['std']

        # Create test dataloader
        test_loader = create_classification_dataloader(
            test_fc_norm, test_labels, test_doy_features,
            batch_size=batch_size, device=device, shuffle=False
        )

        print(f"Test data loaded in {(time.time() - load_time) / 60:.2f} minutes")

        # Find optimal thresholds on validation set if enabled
        thresholds = None
        if use_probability_thresholds:
            thresholds = find_optimal_thresholds(
                model, val_loader, device, n_classes, default_class=0
            )
            print(f"\n  Optimal thresholds: {[f'{t:.2f}' for t in thresholds]}")

        # Evaluate classifier (with or without thresholds)
        test_metrics = evaluate_classifier(
            model, test_loader, device, n_classes,
            class_names=class_names,
            thresholds=thresholds,
            default_class=0
        )

        end_time = time.time()
        total_time_minutes = (end_time - start_time) / 60
        print(f"Total experiment time: {total_time_minutes:.2f} minutes")

        # Note: Classification mode doesn't save corrected forecasts (no temperature output)
        # Could save class predictions if needed in future
        return

    # ========================================================================
    # REGRESSION MODE (standard temperature correction)
    # ========================================================================

    # Load training data
    (fc, fc_output, obs, lead_time_indices, day_of_year_features, train_times, lat_u, lon_u,
     n_lat, n_lon, n_training_vars, n_output_vars, training_mean_forecast_error) = \
        load_forecasts(data_dir, args, lat_vals, lon_vals, train=True,
                       patch_num=patch_num, use_legacy_global_data=use_legacy_global_data)

    loading_time = time.time()
    print(f"Data loaded in {(loading_time - start_time) / 60:.2f} minutes")

    # Normalize data
    stats_train = {'mean': fc.mean(0), 'std': fc.std(0) + 1e-8}
    stats_out = {'mean': fc_output.mean(0), 'std': fc_output.std(0) + 1e-8}

    fc_norm = (fc - stats_train['mean']) / stats_train['std']
    fc_output_norm = (fc_output - stats_out['mean']) / stats_out['std']

    # normalize target observations using forecasts to be corrected
    obs_norm = (obs - stats_out['mean']) / stats_out['std']

    # Use optimal batch size if available, otherwise default to 128
    batch_size = args.optimal_batch_size if args.optimal_batch_size else 128
    print(f"Using batch_size: {batch_size}")

    # Compute n_lead_times and lead_time_days
    n_lead_times = len(args.lead_time_hours)
    lead_time_days = [lt / 24 for lt in args.lead_time_hours]

    # Split train/validation
    n_samples = len(fc)

    # Standard random split
    idx = np.arange(n_samples)
    np.random.shuffle(idx)
    split = int(0.8 * n_samples)
    t_idx, v_idx = idx[:split], idx[split:]

    train_loader = create_dataloader(fc_norm[t_idx], fc_output_norm[t_idx], obs_norm[t_idx],
                                    lead_time_indices[t_idx], day_of_year_features[t_idx],
                                    batch_size=batch_size, device=device)
    val_loader = create_dataloader(fc_norm[v_idx], fc_output_norm[v_idx], obs_norm[v_idx],
                                  lead_time_indices[v_idx], day_of_year_features[v_idx],
                                  batch_size=batch_size, device=device)

    # Log DataLoader settings
    print(f"DataLoader settings: num_workers={train_loader.num_workers}, pin_memory={train_loader.pin_memory}")
    if device.type == 'cuda':
        print(f"  CPU cores available: {os.cpu_count()}")

    # Initialize model
    input_dim = n_training_vars * n_lat * n_lon
    output_dim = n_output_vars * n_lat * n_lon

    # Use optimal lead time embedding dimension if available
    lead_time_emb_dim = args.optimal_lead_time_embedding_dim if args.optimal_lead_time_embedding_dim else 4

    if hasattr(args, 'nn_architecture') and args.nn_architecture == 'unet':
        print(f"  UNet hidden_dim: {args.unet_hidden_dim}")
        print(f"  UNet dropout: {args.unet_dropout}")
        print(f"  UNet lead_time_embedding_dim: {lead_time_emb_dim}")
        model = UNet(input_dim, args.unet_hidden_dim, output_dim, n_lat=n_lat, n_lon=n_lon,
                     n_input_vars=n_training_vars, n_output_vars=n_output_vars,
                     n_lead_times=n_lead_times,
                     lead_time_embedding_dim=lead_time_emb_dim,
                     dropout_rate=args.unet_dropout).to(device)
        num_epochs = 500
    else:
        print(f"Using SimpleMLP with {n_lead_times} lead times")
        print(f"  MLP hidden_dim: {args.mlp_hidden_dim}")
        print(f"  MLP num_layers: {args.mlp_num_layers}")
        print(f"  MLP dropout: {args.mlp_dropout}")
        print(f"  MLP lead_time_embedding_dim: {lead_time_emb_dim}")
        model = SimpleMLP(input_dim=input_dim,
                          hidden_dim=args.mlp_hidden_dim,
                          output_dim=output_dim,
                          num_hidden_layers=args.mlp_num_layers,
                          n_lead_times=n_lead_times,
                          lead_time_embedding_dim=lead_time_emb_dim,
                          dropout_rate=args.mlp_dropout
                          ).to(device)
        num_epochs = 750

    # Use optimal training hyperparameters if available
    lr = args.optimal_lr if args.optimal_lr else 8.669714431623457e-06
    weight_decay = args.optimal_weight_decay if args.optimal_weight_decay else 5.210913466175803e-06
    patience = args.optimal_patience if args.optimal_patience else 50
    min_delta = args.optimal_min_delta if args.optimal_min_delta else 1e-5

    print(f"\nTraining with:")
    print(f"  lr: {lr}")
    print(f"  weight_decay: {weight_decay}")
    print(f"  patience: {patience}")
    print(f"  min_delta: {min_delta}")

    # Train model
    model, training_time_minutes = train_model(model, train_loader, val_loader,
                                                epochs=num_epochs, lr=lr,
                                                device=device,
                                                weight_decay=weight_decay,
                                                stats_out=stats_out,
                                                alternate_loss_fn=args.alternate_loss_fn,
                                                patience=patience,
                                                min_delta=min_delta,
                                                scheduler_patience=10,
                                                scheduler_factor=0.5,
                                                min_lr=1e-7,
                                                n_lead_times=n_lead_times,
                                                lead_time_days=lead_time_days
                                              )
    print(f"Training complete in {training_time_minutes:.2f} minutes")

    load_time = time.time()
    # Load test data
    (test_fc, test_fc_output, test_obs, test_lead_time_indices, test_day_of_year_features,
     test_times, _, _, _, _, _, _, _) = \
        load_forecasts(data_dir, args, lat_vals, lon_vals, train=False, patch_num=patch_num, use_legacy_global_data=use_legacy_global_data)

    # Apply correction
    test_fc_norm = (test_fc - stats_train['mean']) / stats_train['std']
    test_fc_output_norm = (test_fc_output - stats_out['mean']) / stats_out['std']
    corrected = apply_correction(model, test_fc_norm, test_fc_output_norm,
                                test_lead_time_indices,
                                test_day_of_year_features, device)
    corrected = (corrected * stats_out['std']) + stats_out['mean']

    # Calculate MSE per lead time
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
    # ========================================================================
    # LEGACY FLAG: Set to True to use global yearly files (legacy format)
    # TO REMOVE: Remove this flag when legacy data is no longer needed
    # ========================================================================
    USE_LEGACY_GLOBAL_DATA = False # <-- EDIT THIS FLAG
    # ========================================================================

    dirs = setup_directories()

    # Parse command line arguments
    args = parse_args()
    
    # Load optimal hyperparameters based on architecture
    optimal_hyperparams = load_optimal_hyperparameters(args.nn_architecture, args.training_vars, args.output_vars)
    if optimal_hyperparams:
        # Override defaults with optimal hyperparameters
        if args.nn_architecture == 'mlp':
            args.mlp_hidden_dim = optimal_hyperparams.get('hidden_dim', args.mlp_hidden_dim)
            args.mlp_num_layers = optimal_hyperparams.get('num_layers', args.mlp_num_layers)
            args.mlp_dropout = optimal_hyperparams.get('dropout_rate', args.mlp_dropout)
        elif args.nn_architecture == 'unet':
            args.unet_hidden_dim = optimal_hyperparams.get('hidden_dim', args.unet_hidden_dim)
            args.unet_dropout = optimal_hyperparams.get('dropout_rate', args.unet_dropout)
        
        # Store training hyperparameters for use later
        args.optimal_lr = optimal_hyperparams.get('learning_rate', None)
        args.optimal_batch_size = optimal_hyperparams.get('batch_size', None)
        args.optimal_weight_decay = optimal_hyperparams.get('weight_decay', None)
        args.optimal_patience = optimal_hyperparams.get('patience', None)
        args.optimal_min_delta = optimal_hyperparams.get('min_delta', None)
        args.optimal_lead_time_embedding_dim = optimal_hyperparams.get('lead_time_embedding_dim', None)
        
        print(f"\nUsing optimal hyperparameters:")
        if args.nn_architecture == 'mlp':
            print(f"  hidden_dim: {args.mlp_hidden_dim}")
            print(f"  num_layers: {args.mlp_num_layers}")
            print(f"  dropout: {args.mlp_dropout}")
        else:
            print(f"  hidden_dim: {args.unet_hidden_dim}")
            print(f"  dropout: {args.unet_dropout}")
        print(f"  learning_rate: {args.optimal_lr}")
        print(f"  batch_size: {args.optimal_batch_size}")
        print(f"  weight_decay: {args.optimal_weight_decay}")
        print(f"  patience: {args.optimal_patience}")
        print(f"  min_delta: {args.optimal_min_delta}")
    else:
        # Set defaults if no optimal hyperparameters found
        args.optimal_lr = None
        args.optimal_batch_size = None
        args.optimal_weight_decay = None
        args.optimal_patience = None
        args.optimal_min_delta = None
        args.optimal_lead_time_embedding_dim = None
        print("\nUsing default hyperparameters (no optimal hyperparameters found)")

    # Prepare output dir and base path
    args.output_dir = os.path.expanduser(args.output_dir)
    args.data_dir = os.path.expanduser(args.data_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    base_path = os.path.join(args.output_dir, generate_output_path(args))

    # Setup device & seeds
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else
                          'cpu')
    print(f"Using device: {device}")

    # Enable cudnn benchmarking for faster training on GPU
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        print("Enabled cudnn benchmarking for faster GPU training")

    torch.manual_seed(58)
    random.seed(58)

    if USE_LEGACY_GLOBAL_DATA:
        print("\n[LEGACY MODE] Will load global yearly files directly")
        print("  Expected files: {data_dir}/{model_name}_YEAR.zarr")
        print(f"  e.g., {args.data_dir}/pangu_2019.zarr\n")

    region_lat, region_lon = get_region_grid(args)
    nlat_patch, nlon_patch = get_patch_shape(args)

    # Decide if we're in a climate, topographic, or continent region or a geographic one
    if args.region in CLIMATE_ZONE_MAP or args.region in TOPO_ZONE_MAP or args.region in CONTINENT_MAP:
        
        if args.region in CLIMATE_ZONE_MAP:
            patches_path = os.path.join(dirs["processed"], f"climate_zone_patches_{args.region}_{args.subregion}.npy")
        elif args.region in TOPO_ZONE_MAP:
            patches_path = os.path.join(dirs["processed"], f"topo_zone_patches_{args.region}_{args.subregion}.npy")
        elif args.region in CONTINENT_MAP:
            patches_path = os.path.join(dirs["processed"], f"{args.region}_patches.npy")
        else:
            raise ValueError(f'Unknown file path for region {args.region}')
    
        patches = np.load(patches_path, allow_pickle=True)
        patch_ids = np.arange(1, len(patches) + 1)

        # Climate and topo zones have 50 patches, continents have variable number
        if args.region in CLIMATE_ZONE_MAP or args.region in TOPO_ZONE_MAP:
            assert len(patches) == 50
        print(f"Loaded {len(patches)} patches for region '{args.region}'")


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


            # Uncomment to skip existing outputs XX
            # if os.path.exists(out_path):
            #     print(f"Skipping already existing output: {out_path}")
            #     continue

            run_subregion_experiment(
                lat_vals, lon_vals, out_path,
                args, os.path.expanduser(args.data_dir), device, patch_num=idx,
                use_legacy_global_data=USE_LEGACY_GLOBAL_DATA
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
                                     os.path.expanduser(args.data_dir), device,
                                     use_legacy_global_data=USE_LEGACY_GLOBAL_DATA)
    else:
        # Central patch
        ci = (len(region_lat) - nlat_patch) // 2
        cj = (len(region_lon) - nlon_patch) // 2
        lat_vals = region_lat[ci:ci+nlat_patch]
        lon_vals = region_lon[cj:cj+nlon_patch]

        run_subregion_experiment(lat_vals, lon_vals, base_path, args,
                                 os.path.expanduser(args.data_dir), device,
                                 use_legacy_global_data=USE_LEGACY_GLOBAL_DATA)


if __name__ == "__main__":
    main()