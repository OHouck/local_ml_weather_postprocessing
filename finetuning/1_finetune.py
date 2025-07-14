#!/usr/bin/env python3
"""
Author: Ozma Houck 

# Purpose: use a simple MLP to post-process weather forecasts trained on
specific regions and variables. Call this script from command line or with 
1_run_experiments.sh script. 

# example call
python3 finetuning/1_finetune.py \
    --data_dir="/Volumes/wd_external_hd/weatherbench" \
    --output_dir="/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/data/fine_tuning_output" \
    --training_vars 2m_temperature \
    --output_vars 2m_temperature \
    --train_start="2018-01-01" --train_end="2021-12-31" \
    --test_start="2022-01-01" --test_end="2022-12-31" \
    --model_name="pangu" \
    --region="global" \
    --subregion="2x2" \
    --lead_time_hours="24" --bootstrap="2"
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
# Simple MLP definition
# ------------------------------
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=1, num_hidden_layers=3):
        super(SimpleMLP, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# ------------------------------
# U-Net definition 
# ------------------------------
class UNet(nn.Module):
    """
    U-Net architecture for weather forecast bias correction.
    Designed as a drop-in replacement for SimpleMLP.
    
    Based on the CU-net architecture from:
    "A Deep Learning Method for Bias Correction of ECMWF 24-240 h Forecasts"
    """
    
    def __init__(self, input_dim, hidden_dim=128, output_dim=1,
                 n_lat=None, n_lon=None, n_input_vars=None, n_output_vars=None):
        """
        Initialize U-Net with spatial dimension information.
        
        Args:
            input_dim: Flattened input dimension (n_input_vars * n_lat * n_lon)
            hidden_dim: Base number of channels (default: 128)
            output_dim: Flattened output dimension (n_output_vars * n_lat * n_lon)
            n_lat: Number of latitude points
            n_lon: Number of longitude points
            n_input_vars: Number of input variables/channels
            n_output_vars: Number of output variables/channels
        """
        super(UNet, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # Use provided spatial dimensions if available
        if n_lat is not None and n_lon is not None and n_input_vars is not None:
            self.height = n_lat
            self.width = n_lon
            self.in_channels = n_input_vars
            self.out_channels = n_output_vars if n_output_vars is not None else 1
            
            # Verify dimensions match
            expected_input = self.in_channels * self.height * self.width
            expected_output = self.out_channels * self.height * self.width
            
            if expected_input != input_dim:
                print(f"Warning: Expected input dim {expected_input} but got {input_dim}")
            if expected_output != output_dim:
                print(f"Warning: Expected output dim {expected_output} but got {output_dim}")
        else:
            raise ValueError("Dimensions not provided")
        
        # Calculate maximum number of levels based on spatial dimensions
        # Each pooling operation halves the spatial dimensions
        # We need at least 2x2 spatial dims before the last pooling
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
        
        print(f"Encoder channels: {self.encoder_channels}")
        
        # Build decoder (upsampling path)
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()

        for i in range(self.num_levels - 1):
            # Current level index in decoder (starting from deepest)
            decoder_level = self.num_levels - 1 - i
            # Corresponding encoder level for skip connection
            skip_level = self.num_levels - 2 - i
            
            # Input channels (from bottleneck or previous decoder layer)
            # Note: decoder_level = skip_level + 1, so this is always the same
            in_ch = self.encoder_channels[decoder_level]
            
            # Skip connection channels
            skip_ch = self.encoder_channels[skip_level]
            
            # Output channels (matching the skip connection level)
            out_ch = skip_ch
            
            print(f"Decoder {i}: in_ch={in_ch}, skip_ch={skip_ch}, out_ch={out_ch}")
            
            # Upsampling layer
            self.upconvs.append(self._make_upconv(in_ch, out_ch))
            
            # Decoder conv block (input = upsampled + skip connection)
            combined_ch = out_ch + skip_ch
            self.decoders.append(self._make_conv_block(combined_ch, out_ch))

        # Final output layer
        if self.decoders:
            final_in_ch = self.encoder_channels[0]  # Same as first encoder level
        else:
            final_in_ch = self.encoder_channels[-1]  # If no decoders, use last encoder
            
        self.final_conv = nn.Conv2d(final_in_ch, self.out_channels, kernel_size=1)
    
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
    
    def forward(self, x):
        """
        Forward pass through U-Net.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        batch_size = x.shape[0]
        
        # Reshape flat input to spatial format: (batch, channels, lat, lon)
        x = x.view(batch_size, self.in_channels, self.height, self.width)
        
        # Encoder path with skip connections
        encoder_outputs = []
        
        for i in range(len(self.encoders)):
            x = self.encoders[i](x)
            if i < len(self.encoders) - 1:  # Don't store the bottleneck as skip connection
                encoder_outputs.append(x)
            if i < len(self.pools):
                x = self.pools[i](x)
        
        # Decoder path with skip connections
        for i, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoders)):
            # Upsample
            x = upconv(x)
            
            # Get skip connection (in reverse order)
            skip_connection = encoder_outputs[-(i+1)]
            
            # Handle size mismatches due to odd dimensions
            if x.shape[2:] != skip_connection.shape[2:]:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=False)
            
            # Concatenate along channel dimension
            x = torch.cat([x, skip_connection], dim=1)
            x = decoder(x)
        
        # Final convolution
        x = self.final_conv(x)
        
        # Reshape back to flat output: (batch, n_output_vars * n_lat * n_lon)
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
    parser.add_argument('--lead_time_hours', type=int, default=24)
    parser.add_argument('--training_vars', type=str, nargs='+', default=["2m_temperature"])
    parser.add_argument('--output_vars',   type=str, nargs='+', default=["2m_temperature"])
    parser.add_argument('--train_start',   type=str, default='2018-01-01')
    parser.add_argument('--train_end',     type=str, default='2019-12-31')
    parser.add_argument('--test_start',    type=str, default='2020-01-01')
    parser.add_argument('--test_end',      type=str, default='2020-12-31')
    parser.add_argument('--model_type',   type=str, default='MLP', choices=['MLP', 'UNet'])
    parser.add_argument('--mlp_hidden_dim', type=int, default=512)
    parser.add_argument('--mlp_layers',     type=int, default=5)
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
    # include the upper bound +0.25 so arange includes endpoint
    lat_values = np.arange(lat0, lat1 + 0.25, 0.25)
    lon_values = np.arange(lon0, lon1 + 0.25, 0.25)

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
        model_str = f"mlp{args.mlp_hidden_dim}x{args.mlp_layers}"
    lead_time = f"leadtime_{args.lead_time_hours}"

    output_path = f"{args.output_dir}/{args.model_name}/{region_str}/train_{training_vars_str}_test_{output_vars_str}_dim{subregion_str}_{lead_time}h_{dates_str}_{model_str}.zarr"
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

def load_combined_dataset(lat_values, lon_values, root_dir, file_pattern):
    """
    Finds all files in the subfolders of root_dir matching file_pattern and combines them.
    """
    file_paths = glob.glob(os.path.join(root_dir, "*", file_pattern))
    file_paths.sort()

    if len(file_paths) == 0:
        raise ValueError(f"No files found matching pattern: {file_pattern}")
    
    # If lat_values or lon_values are None, skip spatial selection (for pre-filtered climate zone patches)
    if lat_values is None or lon_values is None:
        return xr.open_mfdataset(
            file_paths,
            combine="by_coords",
            preprocess=lambda ds: ds.sortby('latitude'), # don't need to select lat/lon for climate patches
            decode_timedelta=True
        )
    else:   
        return xr.open_mfdataset(
            file_paths,
            combine="by_coords",
            preprocess=lambda ds: ds.sel(latitude = lat_values, longitude = lon_values).sortby('latitude'),
            decode_timedelta=True
        )
    
def get_bounds(args):

    # set up lat and lon bounds for each region.
    if args.region == "india":
        lat_min, lat_max = 17, 27
        lon_min, lon_max = 72, 82
    elif args.region == "usa_south":
        lat_min, lat_max = 30, 40
        lon_min, lon_max = (-105 + 360), (-95 + 360)
    elif args.region == "amazon":
        lat_min, lat_max  = -10, 0 
        lon_min, lon_max = (-70 + 360), (-60 + 360)
    elif args.region == "british_columbia":
        lat_min, lat_max = 48.25, 58  # XX note can update this if I fix the inital download
        lon_min, lon_max = (-130 + 360), (-120 + 360)
    elif args.region == "pakistan":
        lat_min, lat_max = 25, 34
        lon_min, lon_max = 60, 70
    else:
        raise ValueError(f"Unknown region '{args.region}'. Please specify a valid region.")
    
    # For each region of 10x10 degrees, there are sub-regions that are
    # add 0.25 degrees to end to properly set the bounds
    # 2x2, 4x4, 6x6, 8x8, and 10x10 degrees.
    if args.subregion == "2x2":
        lat_min, lat_max = lat_min + 4, lat_max - 4 + 0.25
        lon_min, lon_max = lon_min + 4, lon_max - 4 + 0.25
    elif args.subregion == "4x4":
        lat_min, lat_max = lat_min + 3, lat_max - 3 + 0.25
        lon_min, lon_max = lon_min + 3, lon_max - 3 + 0.25
    elif args.subregion == "6x6":
        lat_min, lat_max = lat_min + 2, lat_max - 2 + 0.25
        lon_min, lon_max = lon_min + 2, lon_max - 2 + 0.25
    elif args.subregion == "8x8":
        lat_min, lat_max = lat_min + 1, lat_max - 1 + 0.25
        lon_min, lon_max = lon_min + 1, lon_max - 1 + 0.25
    elif args.subregion == "10x10":
        lat_min, lat_max = lat_min, lat_max + 0.25
        lon_min, lon_max = lon_min, lon_max + 0.25
    else:
        raise ValueError(f"Unknown subregion '{args.subregion}'. Please specify a valid subregion.")
    lat_values = np.arange(lat_min, lat_max, 0.25)
    lon_values = np.arange(lon_min, lon_max, 0.25)
    return lat_values, lon_values

def load_forecasts(data_dir, args, lat_values, lon_values, train=True, patch_num=None): 
    """
    loads forecast data, forecast output data and observation data for training or testing.
    """

    if train:
        ver_str = "train"
    else:
        ver_str = "test"

    # set up time range for training or testing
    time_start = getattr(args, f"{ver_str}_start")
    time_end = getattr(args, f"{ver_str}_end")
    time_values = np.arange(
        np.datetime64(time_start) + np.timedelta64(args.lead_time_hours, 'h'),  # Start at lead time hours after start date
        np.datetime64(time_end) + np.timedelta64(1, 'D'),  # Add 1 day to include end date,
        np.timedelta64(24, 'h')
    )

    n_time = len(time_values)
    n_training_vars = len(args.training_vars)
    n_output_vars = len(args.output_vars)

    fc_dir = os.path.join(data_dir, f"{ver_str}_{args.region}")
    
    if args.region in CLIMATE_ZONE_MAP:
        # if in climate map, then we have already cleaned and combined the patches ahead of time
        fc_pattern = f"{args.model_name}_{ver_str}_forecast_data_{args.region}_{args.subregion}_patch_{patch_num}.nc"
        obs_pattern = f"{args.model_name}_{ver_str}_obs_data_{args.region}_{args.subregion}_patch_{patch_num}.nc"

        forecast_ds = load_combined_dataset(lat_values, lon_values, fc_dir, fc_pattern)
        train_obs_ds = load_combined_dataset(lat_values, lon_values, fc_dir, obs_pattern)


    else:
        fc_pattern = f"{args.model_name}_{ver_str}_forecast_data_*.nc"
        obs_pattern = f"{args.model_name}_{ver_str}_obs_data_*.nc"
        forecast_ds = load_combined_dataset(lat_values, lon_values, fc_dir, fc_pattern)
        train_obs_ds = load_combined_dataset(lat_values, lon_values, fc_dir, obs_pattern)
    
    # rename time init_time in forecast_ds and create a new time variable
    # consistent with the target dataset
    # create new time variable in forecast_ds that is valid time = init_time + lead_time_hours
    forecast_ds = forecast_ds.assign_coords(
        init_time=forecast_ds.time,
        time=forecast_ds.time + np.timedelta64(args.lead_time_hours, 'h')
    ).drop_vars('init_time')

    # create 10m wind speed from u and v components of wind if needed
    if "10m_wind_speed" in args.training_vars:
        fc_u_component = forecast_ds["10m_u_component_of_wind"]
        fc_v_component = forecast_ds["10m_v_component_of_wind"]
        forecast_ds["10m_wind_speed"] = np.sqrt(fc_u_component**2 + fc_v_component**2)
    if "10m_wind_speed" in args.output_vars:
        obs_u_component = train_obs_ds["10m_u_component_of_wind"]
        obs_v_component = train_obs_ds["10m_v_component_of_wind"]
        train_obs_ds["10m_wind_speed"] = np.sqrt(obs_u_component**2 + obs_v_component**2)

    # Now select the desired time, spatial, and (if applicable) prediction_timedelta slices.
    fc_ds = forecast_ds.sel(
        time=time_values,
        prediction_timedelta=np.timedelta64(args.lead_time_hours, 'h')
    )[args.training_vars].drop_vars('prediction_timedelta').compute()
    
    fc_ds_output = forecast_ds.sel(
        time=time_values,
        prediction_timedelta=np.timedelta64(args.lead_time_hours, 'h')
    )[args.output_vars].drop_vars('prediction_timedelta').compute()
    
    # Handle spatial selection for obs_ds - skip if lat/lon are None (climate zone patches)
    if lat_values is None or lon_values is None:
        obs_ds = train_obs_ds.sel(
            time=time_values,
        )[args.output_vars].compute()
    else:
        obs_ds = train_obs_ds.sel(
            time=time_values,
            latitude=lat_values,
            longitude=lon_values,
        )[args.output_vars].compute()

    return fc_ds, fc_ds_output, obs_ds, time_values, n_time, n_training_vars, n_output_vars

def create_dataloader(forecast_data, obs_data, batch_size):
    """
    Create a PyTorch DataLoader from forecast and observation data.
    """
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(forecast_data).float(),
        torch.from_numpy(obs_data).float()
    )
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True)
    return dataloader


def train_model(model, train_loader, valid_loader, epochs, lr, device, patience=50, min_delta=0.0):
    """
    Train the model over multiple epochs. with early stopping
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):

    # --- training step ---
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)
        train_loss /= len(train_loader.dataset)

        # --- validation step ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in valid_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                preds = model(x_batch)
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

    # load best weights
    model.load_state_dict(best_model_wts)
    return model



def apply_correction(model, forecast_data, device):
    """
    Apply the MLP-based correction to forecast data.
    """
    model.eval()
    with torch.no_grad():
        x_tensor = torch.from_numpy(forecast_data).float().to(device)
        corrected = model(x_tensor).cpu().numpy()
    return corrected


def save_output(output_path, model_name, output_vars, lon_values, lat_values,
                time_values, original_fc, corrected_fc, ground_truth_data=None, training_mean_forecast_error=None):
    """
    Save original and corrected forecasts (and optionally ground truth) in Zarr format.
    Supports both single and multiple variables.
    """
    n_vars = len(output_vars)
    n_time = len(time_values)
    n_lon = len(lon_values)
    n_lat = len(lat_values)

    # reshape to be (variable, time, lat, lon)
    original_fc = original_fc.reshape(n_time, n_vars, n_lat, n_lon)
    original_fc = original_fc.transpose(1, 0, 2, 3)
    
    corrected_fc = corrected_fc.reshape(n_time, n_vars, n_lat, n_lon)
    corrected_fc = corrected_fc.transpose(1, 0, 2, 3)

    # convert to xarray DataArray
    original_fc_da = xr.DataArray(
        data=original_fc,
        dims=['variable', 'time', 'latitude', 'longitude'],
        coords={"variable": output_vars,
                "time": time_values, 
                "latitude": lat_values,
                "longitude": lon_values}
        )
    corrected_fc_da = xr.DataArray(
        data=corrected_fc,
        dims=['variable', 'time', 'latitude', 'longitude'],
        coords={"variable": output_vars,
                "time": time_values, 
                "latitude": lat_values,
                "longitude": lon_values}
        )
    

    if ground_truth_data is not None:
        ground_truth_data = ground_truth_data.reshape(n_time, n_vars, n_lat, n_lon)
        ground_truth_data = ground_truth_data.transpose(1, 0, 2, 3)
        ground_truth_da = xr.DataArray(
            data=ground_truth_data,
            dims=['variable', 'time', 'latitude', 'longitude'],
            coords={"variable": output_vars,
                    "time": time_values, 
                    "latitude": lat_values,
                    "longitude": lon_values}
            )
    
    
    # combine 3 DataArrays into a single Dataset. will have time, latitude, and longitude as coords
    # data variables will be of form: {variable}_original, {variable}_corrected, {variable}_ground_truth
    # Create a dictionary to hold each data variable in the final dataset
    data_vars = {}
    for var in output_vars:
        # Select the slice for this variable (dims will then be time, latitude, longitude)
        data_vars[f"{var}_original"] = original_fc_da.sel(variable=var).drop_vars("variable")
        data_vars[f"{var}_corrected"] = corrected_fc_da.sel(variable=var).drop_vars("variable")
        # Simple ANO style correction
        if training_mean_forecast_error is not None:
            data_vars[f"{var}_mean_corrected"]= data_vars[f"{var}_original"] - training_mean_forecast_error[var]
        if ground_truth_data is not None:
            data_vars[f"{var}_ground_truth"] = ground_truth_da.sel(variable=var).drop_vars("variable")

    # Combine data variables into a single dataset
    # print variable names for data_vars
    ds_out = xr.Dataset(data_vars)

    ds_out.attrs['description'] = (f'Original and corrected forecasts from {model_name} '
                                   f'using MLP fine-tuning)')
    output_path = os.path.expanduser(output_path)
    ds_out.to_zarr(output_path, mode='w')
    print(f"Forecasts saved to {output_path}")

def check_2m_temperature(files):
    """
    For each NetCDF file in `files`, attempt to open it without CF time decoding,
    then check whether '2m_temperature' exists and whether it contains any NaNs.
    Returns a dict mapping file path -> info dict.
    """
    results = {}
    for fn in files:
        try:
            ds = xr.open_dataset(fn,
                                 decode_times=False,
                                 decode_timedelta=False)
        except Exception as e:
            # Could not open file at all
            results[fn] = {"error": f"open_dataset failed: {e}"}
            continue

        if "2m_temperature" not in ds:
            results[fn] = {"error": "'2m_temperature' variable not found"}
        else:
            arr = ds["2m_temperature"]
            # Boolean flag for any NaNs
            has_nans = bool(arr.isnull().any().item())
            # Count of NaNs (if you want the exact count)
            n_nans = int(arr.isnull().sum().values) if has_nans else 0
            results[fn] = {
                "has_nans": has_nans,
                "n_nans": n_nans,
                "shape": arr.shape
            }

        ds.close()
    return results

def run_subregion_experiment(lat_vals, lon_vals, output_path, args, data_dir, device, patch_num = None):

    # 1) Load train
    fc_ds, fc_ds_output, obs_ds, train_time_values, n_train_time, n_training_vars, n_output_vars = \
        load_forecasts(data_dir, args, lat_vals, lon_vals, train=True, patch_num = patch_num)

    # save mean forecast error in the training data for mean debiasing correction
    training_mean_forecast_error = fc_ds_output.mean(dim='time') - obs_ds.mean(dim='time')

    # save unique lat/lon
    lat_u = np.unique(fc_ds.latitude.values)
    lon_u = np.unique(fc_ds.longitude.values)
    n_lat, n_lon = len(lat_u), len(lon_u)

    # flatten data and arrange variables in consistent order 
    fc   = fc_ds.to_array().values.transpose(1,0,2,3).reshape(n_train_time, n_training_vars * n_lat * n_lon)
    fc_o = fc_ds_output.to_array().values.transpose(1,0,2,3).reshape(n_train_time, n_output_vars * n_lat * n_lon)
    obs  = obs_ds.to_array().values.transpose(1,0,2,3).reshape(n_train_time, n_output_vars * n_lat * n_lon)

    # normalize
    stats_train = {'mean': fc.mean(0), 'std': fc.std(0) + 1e-8}
    stats_out   = {'mean': fc_o.mean(0), 'std': fc_o.std(0) + 1e-8}
    fc_norm  = (fc  - stats_train['mean']) / stats_train['std']
    obs_norm = (obs - stats_out['mean'])   / stats_out['std']

    # split train/val
    idx = np.arange(n_train_time); np.random.shuffle(idx)
    split = int(0.8 * n_train_time)
    t_idx, v_idx = idx[:split], idx[split:]
    train_loader = create_dataloader(fc_norm[t_idx], obs_norm[t_idx], batch_size=32)
    val_loader   = create_dataloader(fc_norm[v_idx], obs_norm[v_idx], batch_size=32)

    # init & train
    input_dim  = n_training_vars * n_lat * n_lon
    output_dim = n_output_vars    * n_lat * n_lon

    # Choose model type based on args
    if hasattr(args, 'model_type') and args.model_type == 'UNet':
        print(f"Using UNet with spatial dims: {n_lat}x{n_lon}, {n_training_vars} input vars, {n_output_vars} output vars")
        unet_hidden_dim = 32 
        model = UNet(input_dim, unet_hidden_dim, output_dim, n_lat=n_lat, n_lon=n_lon, 
                     n_input_vars=n_training_vars, n_output_vars=n_output_vars).to(device)
    else:
        print("Using SimpleMLP")
        model = SimpleMLP(input_dim, args.mlp_hidden_dim, output_dim, args.mlp_layers).to(device)
    
    model = train_model(model, train_loader, val_loader,
                         epochs=1000, lr=1e-5, device=device,
                         patience=50, min_delta=0.0)
    print("Training complete.")

    # load test
    test_fc_ds, test_fc_o_ds, test_obs_ds, test_times, n_test_time, _, _ = \
        load_forecasts(data_dir, args, lat_vals, lon_vals, train=False, patch_num = patch_num)
    tfc   = test_fc_ds.to_array().values.transpose(1,0,2,3).reshape(n_test_time, -1)
    tfco  = test_fc_o_ds.to_array().values.transpose(1,0,2,3).reshape(n_test_time, -1)
    tobs  = test_obs_ds.to_array().values.transpose(1,0,2,3).reshape(n_test_time, -1)

    # apply correction
    tfc_norm    = (tfc - stats_train['mean']) / stats_train['std']
    corrected   = apply_correction(model, tfc_norm, device)
    corrected   = (corrected * stats_out['std']) + stats_out['mean']


    print(f"MSE original: {np.mean((tfco - tobs)**2):.6f}")
    print(f"MSE corrected: {np.mean((corrected - tobs)**2):.6f}")

    # save
    save_output(
        output_path=output_path,
        model_name=args.model_name,
        output_vars=args.output_vars,
        lon_values=lon_u,
        lat_values=lat_u,
        time_values=test_times,
        original_fc=tfco,
        corrected_fc=corrected,
        ground_truth_data=tobs,
        training_mean_forecast_error=training_mean_forecast_error
    )


def main():

    dirs = setup_directories()

    # Get all prediction and observation files
    file_list = sorted(glob.glob(os.path.join(dirs["raw"], "ethiopia", "predictions_ethiopia_*.nc")))
    # file_list = sorted(glob.glob("/Volumes/wd_external_hd/weatherbench/test_global/**/*pangu*.nc", recursive=True))
    summary = check_2m_temperature(file_list)
    # Print a quick report
    for path, info in summary.items():
        if "error" in info:
            print(f"[ERROR] {path}: {info['error']}")
        else:
            status = "contains NaNs" if info["has_nans"] else "no NaNs"
            if status == "contains NaNs":
                print(f"[WARNING] {path}: {status} (n_nans={info['n_nans']})")
    

    # parse command line arguments
    args = parse_args()
    # prepare output dir and base path
    args.output_dir = os.path.expanduser(args.output_dir)
    args.data_dir = os.path.expanduser(args.data_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    base_path = generate_output_path(args)

    # setup device & seeds
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps'    if torch.backends.mps.is_available() else
                          'cpu')
    torch.manual_seed(58); random.seed(58)

    region_lat, region_lon = get_region_grid(args)
    nlat_patch, nlon_patch = get_patch_shape(args)

    # ---- decide if we're in a climate‐zone region or a geographic one ----
    if args.region in CLIMATE_ZONE_MAP:

        def find_available_patches(region, model_name, train_or_test):
            """
            Find available patches for a given region and model name.
            Returns a list of patch numbers that have data downloaded.
            """
            test_path = os.path.join(args.data_dir, f"{train_or_test}_{region}", model_name)
            patch_files = glob.glob(os.path.join(test_path, "*.nc"))
            patch_nums = [int(os.path.basename(f).split('_')[-1].split('.')[0]) for f in patch_files]
            return sorted(patch_nums)
        
        train_patch_nums = find_available_patches(args.region, args.model_name, 'train')
        test_patch_nums  = find_available_patches(args.region, args.model_name, 'test')

        # save patch numbers that are in both train and test patches
        common_patch_nums = set(train_patch_nums) & set(test_patch_nums)

        # since climate zone patches are precomputed don't need to pass in lon/lat
        # instead just pass in None
        lat_vals = None
        lon_vals = None
        
        for idx in common_patch_nums:
            out_path = base_path.replace(
                '.zarr', f'_{args.region}_bs{idx}.zarr'
            )

            # check if output file already exists XX will have to comment out to redo
            if os.path.exists(out_path):
                print(f"Skipping already existing output: {out_path}")
                continue

            start_time = time.time()
            run_subregion_experiment(
                lat_vals, lon_vals, out_path,
                args, os.path.expanduser(args.data_dir), device, patch_num = idx
            )
            end_time = time.time()
            elapsed_minutes = (end_time - start_time) / 60
            print(f"Saved [{args.region} zone] sample {idx}/{len(common_patch_nums)} in {elapsed_minutes} minutes")
            

    elif args.bootstrap:
        # bootstrap sampling for uniform-grid regions
        for i in range(args.bootstrap):
            si = random.randint(0, len(region_lat) - nlat_patch)
            sj = random.randint(0, len(region_lon) - nlon_patch)
            lat_vals = region_lat[si:si+nlat_patch]
            lon_vals = region_lon[sj:sj+nlon_patch]
            out_path = base_path.replace('.zarr', f'_bs{i+1}.zarr')
            print(f"Running bootstrap sample {i+1}/{args.bootstrap}")
            run_subregion_experiment(lat_vals, lon_vals, out_path, args,
                                     os.path.expanduser(args.data_dir), device)
    else:
        # central patch
        ci = (len(region_lat) - nlat_patch) // 2
        cj = (len(region_lon) - nlon_patch) // 2
        lat_vals = region_lat[ci:ci+nlat_patch]
        lon_vals = region_lon[cj:cj+nlon_patch]
        run_subregion_experiment(lat_vals, lon_vals, base_path, args,
                                 os.path.expanduser(args.data_dir), device)

if __name__ == "__main__":
    main()
