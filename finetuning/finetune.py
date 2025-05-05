#!/usr/bin/env python3
"""
finetuning.py
Author: Ozzy Houck 
Date: 12/20/2024 (modified 2025-03-20)

This script fine-tunes an MLP correction model to a specific region using model
forecasts and corresponding observations from weatherbench2. It now supports fine-tuning
over multiple variables. The model learns a mapping from the concatenated model-forecasted
fields (for all specified variables) to the corresponding observed fields.
----------------------------

Example usage 
python3 finetuning/finetune.py \
    --output_dir="~/wb_finetune_test" \
    --region="north_india" \
    --train_start="2021-01-01" --train_end="2021-01-30" \
    --test_start="2022-01-01" --test_end="2022-01-30" \
    --training_vars 10m_v_component_of_wind 10m_u_component_of_wind \
    --output_vars 10m_v_component_of_wind \
    --lead_time_hours=24 \
    --mlp_hidden_dim=512 \
    --mlp_layers=5 \
"""

import argparse
import os
import random
from datetime import datetime, timedelta
import glob

import numpy as np
import xarray as xr
from xarray.coding.times import CFDatetimeCoder
import dask
import torch
import torch.nn as nn
import torch.optim as optim
import copy


class SimpleMLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) for post-processing weather forecast data.
    """
    def __init__(self, input_dim, hidden_dim=128, output_dim=1, num_hidden_layers=3):
        super(SimpleMLP, self).__init__()
        layers = []
        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the MLP.
        """
        return self.net(x)


def parse_args():
    """
    Parse command-line arguments for fine-tuning the MLP on regional post-processing.
    """
    parser = argparse.ArgumentParser(description='Fine-tune MLP for regional post-processing')
    parser.add_argument('--data_dir', type=str, default="~/weatherbench2_data",
                        help='Directory to save the raw forecasts locally')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the fine-tuned model and corrected forecasts')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name of the base model (e.g. pangu, ifs, neural_gcm)')
    parser.add_argument('--region', type=str, default="india",
                        help='Name of the region')
    parser.add_argument('--subregion', type=str, default="10x10",
                        help='Dimensions of Subregion')
    parser.add_argument('--lead_time_hours', type=int, default=24,
                        help='Lead time in hours for forecast')
    parser.add_argument('--training_vars', type=str, nargs='+', default=["2m_temperature"],
                        help='Variables used to fine-tune (e.g. 2m_temperature precipitation)')
    parser.add_argument('--output_vars', type=str, nargs='+', default=["2m_temperature"],
                        help='subset of training_vars to be predicted (e.g. 2m_temperature precipitation)')
    parser.add_argument('--train_start', type=str, default='2018-01-01',
                        help='Training start date')
    parser.add_argument('--train_end', type=str, default='2019-12-31',
                        help='Training end date')
    parser.add_argument('--test_start', type=str, default='2020-01-01',
                        help='Test start date')
    parser.add_argument('--test_end', type=str, default='2020-12-31',
                        help='Test end date')
    parser.add_argument('--mlp_hidden_dim', type=int, default=512,
                        help='Number of neurons in the hidden layers')
    parser.add_argument('--mlp_layers', type=int, default=5,
                        help='Number of hidden layers in the MLP')
    return parser.parse_args()

def generate_output_path(args):
    region_str = f"{args.region}"
    subregion_str = f"{args.subregion}"
    dates_str = f"train{args.train_start}-{args.train_end}_test{args.test_start}-{args.test_end}"
    training_vars_str = "_".join(args.training_vars)
    output_vars_str = "_".join(args.output_vars)
    mlp_str = f"mlp{args.mlp_hidden_dim}x{args.mlp_layers}"
    lead_time = f"leadtime_{args.lead_time_hours}"

    output_path = f"{args.output_dir}/{args.model_name}/{region_str}/train_{training_vars_str}_test_{output_vars_str}_dim{subregion_str}_{lead_time}h_{dates_str}_{mlp_str}.zarr"
    return output_path 


def load_combined_dataset(root_dir, file_pattern):
    """
    Finds all files in the subfolders of root_dir matching file_pattern and combines them.
    """
    file_paths = glob.glob(os.path.join(root_dir, "*", file_pattern))
    file_paths.sort()

    if len(file_paths) == 0:
        raise ValueError(f"No files found matching pattern: {file_pattern}")
    # print(f"Combining {len(file_paths)} files for pattern: {file_pattern}")
    return xr.open_mfdataset(
        file_paths,
        combine="by_coords",
        decode_timedelta=True     # if you still want your lead-time axis as timedelta
    )



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
                time_values, original_fc, corrected_fc, ground_truth_data=None):
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


def main():

    file_list = sorted(glob.glob("/Users/ohouck/wb_finetune_data/train_india/**/*.nc", recursive=True))
    summary = check_2m_temperature(file_list)
    # Print a quick report
    for path, info in summary.items():
        if "error" in info:
            print(f"[ERROR] {path}: {info['error']}")
        else:
            status = "contains NaNs" if info["has_nans"] else "no NaNs"
            if status == "contains NaNs":
                print(f"[WARNING] {path}: {status} (n_nans={info['n_nans']})")


     # Set up device: prioritize CUDA, then MPS, then CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    # set seed
    torch.manual_seed(58)
    random.seed(58)

    args = parse_args()
    output_path = generate_output_path(args)
    print("output path:", output_path)

    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    data_dir = os.path.expanduser(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)

    # =========================================================================
    # 1) Load training data (for all specified variables)
    # =========================================================================

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
        lat_min, lat_max = 48, 58 
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
    
    train_time_values = np.arange(np.datetime64(args.train_start), np.datetime64(args.train_end), np.timedelta64(24, 'h'))
    test_time_values = np.arange(np.datetime64(args.test_start), np.datetime64(args.test_end), np.timedelta64(24, 'h'))
    lat_values = np.arange(lat_min, lat_max, 0.25)
    lon_values = np.arange(lon_min, lon_max, 0.25)

    n_train_time = len(train_time_values)
    n_test_time = len(test_time_values)
    n_training_vars = len(args.training_vars)
    n_output_vars = len(args.output_vars)

    # ----- Loading combined training data from monthly files -----
    train_dir = os.path.join(data_dir, f"train_{args.region}")
    # For forecast and observation data, we define the patterns for monthly file names.
    fc_pattern = f"{args.model_name}_train_forecast_data_*.nc"
    obs_pattern = f"{args.model_name}_train_obs_data_*.nc"
    train_forecast_ds = load_combined_dataset(train_dir, fc_pattern)
    train_obs_ds = load_combined_dataset(train_dir, obs_pattern)

    # Now select the desired time, spatial, and (if applicable) prediction_timedelta slices.
    fc_ds = train_forecast_ds.sel(
        time=train_time_values,
        latitude=lat_values,
        longitude=lon_values,
        prediction_timedelta=np.timedelta64(args.lead_time_hours, 'h')
    )[args.training_vars].drop_vars('prediction_timedelta').compute()
    
    fc_ds_output = train_forecast_ds.sel(
        time=train_time_values,
        latitude=lat_values,
        longitude=lon_values,
        prediction_timedelta=np.timedelta64(args.lead_time_hours, 'h')
    )[args.output_vars].drop_vars('prediction_timedelta').compute()
    
    obs_ds = train_obs_ds.sel(
        time=train_time_values,
        latitude=lat_values,
        longitude=lon_values,
    )[args.output_vars].compute()

    # # This gives you, for each time t, True if any lat/lon at that time is NaN
    # missing_flag = obs_ds['2m_temperature'].isnull().any(dim=('latitude', 'longitude'))
    # missing_times = obs_ds['time'][missing_flag]
    # print("Times with any NaNs in 2m_temperature:")
    # print(missing_times.values)
    # exit()

    # save the lat and lon values for later use
    lat_values = np.unique(fc_ds.latitude.values)
    lon_values = np.unique(fc_ds.longitude.values)
    n_lat = len(lat_values)
    n_lon = len(lon_values)

    fc = fc_ds.to_array().values  # shape: (variable, time, lat, lon)
    fc = fc.transpose(1, 0, 2, 3)   # shape: (time, variable, lat, lon)
    fc = fc.reshape(n_train_time, n_training_vars * n_lat * n_lon) # shape: (time, variable*lat*lon)

    fc_output = fc_ds_output.to_array().values  # shape: (variable, time, lat, lon)
    fc_output = fc_output.transpose(1, 0, 2, 3)   # shape: (time, variable, lat, lon)
    fc_output = fc_output.reshape(n_train_time, n_output_vars * n_lat * n_lon) # shape: (time, variable*lat*lon)

    obs_fc = obs_ds.to_array().values  # shape: (variable, time, lat, lon)
    obs_fc = obs_fc.transpose(1, 0, 2, 3)   # shape: (time, variable, lat, lon)
    obs = obs_fc.reshape(n_train_time, n_output_vars * n_lat * n_lon) # shape: (time, variable*lat*lon)

    # =========================================================================
    # 4) Normalize training & validation data variable-wise using training stats
    # =========================================================================
    stats_training= {
        'mean': fc.mean(axis=0), # take mean over time
        'std': fc.std(axis=0) + 1e-8, # add small value to avoid division by zero
    }
    stats_output= {
        'mean': fc_output.mean(axis=0), # take mean over time
        'std': fc_output.std(axis=0) + 1e-8, # add small value to avoid division by zero
    }

    # subset fc stats to only include variables in output_vars
    fc_norm = (fc - stats_training['mean']) / stats_training['std']
    obs_norm = (obs - stats_output['mean']) / stats_output['std']

    # =========================================================================
    # 2) Randomly split training data into TRAIN (80%) and VAL (20%)
    # split using the indices of the time dimension
    # =========================================================================
    indices = np.arange(n_train_time)
    np.random.shuffle(indices)
    split_idx = int(0.8 * n_train_time)
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    train_fc_norm = fc_norm[train_idx]
    train_obs_norm =obs_norm[train_idx]
    val_fc_norm = fc_norm[val_idx]
    val_obs_norm = obs_norm[val_idx]

    print(f"Training set size: {fc_norm.shape}")
    print(f"Training data shape: {train_fc_norm.shape}")
    print(f"Validation data shape: {val_fc_norm.shape}")

    # =========================================================================
    # 5) Create PyTorch DataLoaders
    # =========================================================================
    batch_size = 32 
    train_loader = create_dataloader(train_fc_norm, train_obs_norm, batch_size)
    val_loader = create_dataloader(val_fc_norm, val_obs_norm, batch_size)

    # =========================================================================
    # 6) Initialize and train the model
    # =========================================================================
    input_dim = n_training_vars * n_lat * n_lon

    # eventually allow this to be different
    output_dim = n_output_vars * n_lat * n_lon
    model = SimpleMLP(input_dim=input_dim,
                      hidden_dim=args.mlp_hidden_dim,
                      output_dim=output_dim,
                      num_hidden_layers=args.mlp_layers)
    model.to(device)

    epochs = 1000
    learning_rate = 1e-5
    patience = 50
    min_delta = 0.0
    model = train_model(model, train_loader, val_loader, epochs, learning_rate, device, patience, min_delta)

    # =========================================================================
    # 3) Load the test data
    # =========================================================================

    # ----- Loading combined training data from monthly files -----
    test_dir = os.path.join(data_dir, f"test_{args.region}")
    # For forecast and observation data, we define the patterns for monthly file names.
    fc_pattern = f"{args.model_name}_test_forecast_data_*.nc"
    obs_pattern = f"{args.model_name}_test_obs_data_*.nc"
    test_forecast_ds = load_combined_dataset(test_dir, fc_pattern)
    test_obs_ds = load_combined_dataset(test_dir, obs_pattern)

    # Now select the desired time, spatial, and (if applicable) prediction_timedelta slices.
    test_fc_ds = test_forecast_ds.sel(
        time=test_time_values,
        latitude=lat_values,
        longitude=lon_values,
        prediction_timedelta=np.timedelta64(args.lead_time_hours, 'h')
    )[args.training_vars].drop_vars('prediction_timedelta').compute()
    
    test_fc_ds_output = test_forecast_ds.sel(
        time=test_time_values,
        latitude=lat_values,
        longitude=lon_values,
        prediction_timedelta=np.timedelta64(args.lead_time_hours, 'h')
    )[args.output_vars].drop_vars('prediction_timedelta').compute()
    
    test_obs_ds = test_obs_ds.sel(
        time=test_time_values,
        latitude=lat_values,
        longitude=lon_values,
    )[args.output_vars].compute()

    print(test_fc_ds)
    print(test_obs_ds)

    test_fc = test_fc_ds.to_array().values  # shape: (variable, time, lat, lon)
    test_fc = test_fc.transpose(1, 0, 2, 3)   # shape: (time, variable, lat, lon)
    test_fc = test_fc.reshape(n_test_time, n_training_vars * n_lat * n_lon) # shape: (time, variable*lat*lon)

    test_fc_output = test_fc_ds_output.to_array().values  # shape: (variable, time, lat, lon)
    test_fc_output = test_fc_output.transpose(1, 0, 2, 3)   # shape: (time, variable, lat, lon)
    test_fc_output = test_fc_output.reshape(n_test_time, n_output_vars * n_lat * n_lon) # shape: (time, variable*lat*lon)

    test_obs = test_obs_ds.to_array().values  # shape: (variable, time, lat, lon)
    test_obs = test_obs.transpose(1, 0, 2, 3)   # shape: (time, variable, lat, lon)
    test_obs = test_obs.reshape(n_test_time, n_output_vars * n_lat * n_lon) # shape: (time, variable*lat*lon)

    # =========================================================================
    # 8) Apply correction to the test set
    # =========================================================================
    test_fc_norm = (test_fc - stats_training['mean']) / stats_training['std']

    corrected_test_fc_norm = apply_correction(model, test_fc_norm, device)
    # XX should create subset of fc stats for only output vars
    corrected_test_fc = (corrected_test_fc_norm * stats_output['std']) + stats_output['mean']

    print(f"MSE (original forecast, test set): {np.mean((test_fc_output - test_obs) ** 2):.6f}")
    print(f"MSE (corrected forecast, test set): {np.mean((corrected_test_fc - test_obs) ** 2):.6f}")

    # =========================================================================
    # 9) Save outputs for the test set
    # =========================================================================
    save_output(
        output_path = output_path,
        model_name=args.model_name,
        output_vars=args.output_vars,
        lon_values=lon_values,
        lat_values=lat_values,
        time_values=test_time_values,
        original_fc=test_fc_output, # only need output_vars from test_fc
        corrected_fc=corrected_test_fc,
        ground_truth_data=test_obs
    )

if __name__ == "__main__":
    main()