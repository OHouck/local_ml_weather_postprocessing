#!/usr/bin/env python3
"""
weatherbench2_finetuning.py
Author: Ozzy Houck (modified by ChatGPT)
Date: 12/20/2024 (modified 2025-02-25)

This script fine-tunes an MLP correction model to a specific region using model
forecasts and corresponding observations from weatherbench2. It now supports fine-tuning
over multiple variables. The model learns a mapping from the concatenated model-forecasted
fields (for all specified variables) to the corresponding observed fields.
----------------------------

All India: lat_min= 8.68 lat_max = 27.2, lon_min=70.75, lon_max = 87.35
Northern India: lat_min= 21, lat_max = 35.5, lon_min=70.75, lon_max = 87.35
Uttar Pradesh: lat_min= 24.2, lat_max = 26, lon_min=78, lon_max = 87.35

Example usage 
python3 finetuning/finetune.py \
    --forecast_path="gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr" \
    --obs_path="gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr" \
    --output_dir="~/wb_finetune_test" \
    --model_name="pangu" \
    --region="north_india
    --train_start="2021-01-01" --train_end="2021-12-30" \
    --test_start="2022-01-01" --test_end="2022-12-30" \
    --lead_time_hours=48 \
    --training_vars = 10m_v_component_of_wind 10m_u_component_of_wind \
    --output_vars = 10m_v_component_of_wind 10m_u_component_of_wind \
    --epochs=1000 \
    --mlp_hidd_dim=512 \
    --mlp_layers=5

"""

import argparse
import os

import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
import socket



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
    parser.add_argument('--forecast_path', type=str, required=True,
                        help='Path to forecast data (e.g. Zarr or NetCDF)')
    parser.add_argument('--obs_path', type=str, required=True,
                        help='Path to observation data (e.g. ERA5 Zarr or NetCDF)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the fine-tuned model and corrected forecasts')
    parser.add_argument('--model_name', type=str, default="pangu",
                        help='Name of the base model (e.g. pangu, ifs, neural_gcm)')
    parser.add_argument('--region', type=str, default="north_india",
                        help='Region to train over(e.g. full_india, north_india, uttar_pradesh)')
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
    "Helper function to build string identifier for the run."
    # Region string from lat/lon bounds
    region_str = f"{args.region}"
    # Dates string from training and testing periods
    dates_str = f"train{args.train_start}-{args.train_end}_test{args.test_start}-{args.test_end}"
    # Variables string (join with a dash or underscore)
    training_vars_str = "_".join(args.training_vars)
    ouput_vars_str = "_".join(args.output_vars)
    # MLP architecture string
    mlp_str = f"mlp{args.mlp_hidden_dim}x{args.mlp_layers}"
    # Combine all pieces along with the weather model and lead time
    run_id = f"{args.model_name}_{region_str}_{dates_str}_{args.lead_time_hours}h_train_{training_vars_str}_output{ouput_vars_str}_{mlp_str}"
    return run_id

def load_data(forecast_path, obs_path, training_vars, level, lat_slice, lon_slice,
              time_slice, lead_time_hours):
    """
    Load forecast and observation data for the specified variables and region.

    Args:
        forecast_path (str): Path to forecast data.
        obs_path (str): Path to observation data.
        training_vars (list of str): List of variable names used for fine-tuning.
        level (int or None): Pressure level if applicable.
        lat_slice (slice): Latitude slice.
        lon_slice (slice): Longitude slice.
        time_slice (slice): Time slice.
        lead_time_hours (int): Lead time in hours for forecast.

    Returns:
        tuple:
            fc_data (np.ndarray): Flattened forecast data of shape (time, n_vars*lat*lon)
            obs_data (np.ndarray): Flattened observation data of shape (time, n_vars*lat*lon)
            lon_vals (np.ndarray): Longitude values.
            lat_vals (np.ndarray): Latitude values.
            time_vals (np.ndarray): Time values.
            original_shape (tuple): (n_time, n_vars, n_lat, n_lon)
    """
    # Open datasets (supporting Zarr or NetCDF)
    ds_forecast = (
        xr.open_zarr(forecast_path) if forecast_path.endswith('.zarr')
        else xr.open_dataset(forecast_path)
    )
    ds_obs = (
        xr.open_zarr(obs_path) if obs_path.endswith('.zarr')
        else xr.open_dataset(obs_path)
    )
    
    # Ensure consistent ordering of latitude
    ds_forecast = ds_forecast.sortby('latitude')
    ds_obs = ds_obs.sortby('latitude')

    # Rename dims if necessary
    for ds in [ds_forecast, ds_obs]:
        for v in training_vars:
            if v not in ds:
                print(f"Variable '{v}' not found in dataset. Skipping...")
                print(f"Available variables: {list(ds.data_vars)}")
                continue
            dims = ds[v].dims
            if 'latitude' not in dims and 'lat' in dims:
                ds = ds.rename({'lat': 'latitude'})
            if 'longitude' not in dims and 'lon' in dims:
                ds = ds.rename({'lon': 'longitude'})

    # Load each variable separately and store in lists
    fc_vars = []
    obs_vars = []
    for v in training_vars:
        # Select region, time, and level (if applicable)
        if 'level' in ds_forecast[v].dims and level is not None:
            fc_var = ds_forecast[v].sel(time=time_slice,
                                        latitude=lat_slice,
                                        longitude=lon_slice,
                                        level=level)
        else:
            fc_var = ds_forecast[v].sel(time=time_slice,
                                        latitude=lat_slice,
                                        longitude=lon_slice)
        if 'level' in ds_obs[v].dims and level is not None:
            obs_var = ds_obs[v].sel(time=time_slice,
                                    latitude=lat_slice,
                                    longitude=lon_slice,
                                    level=level)
        else:
            obs_var = ds_obs[v].sel(time=time_slice,
                                    latitude=lat_slice,
                                    longitude=lon_slice)
        # For forecast, select the desired lead time
        fc_var = fc_var.sel(prediction_timedelta=np.timedelta64(lead_time_hours, 'h'))
        if 'prediction_timedelta' in fc_var.coords:
            fc_var = fc_var.drop_vars('prediction_timedelta')
        fc_vars.append(fc_var)
        obs_vars.append(obs_var)

    # Concatenate variables along a new dimension called 'variable'
    fc_concat = xr.concat(fc_vars, dim='variable').transpose('time', 'variable', 'latitude', 'longitude')
    obs_concat = xr.concat(obs_vars, dim='variable').transpose('time', 'variable', 'latitude', 'longitude')

    # Align datasets along common coordinates
    fc_concat, obs_concat = xr.align(fc_concat, obs_concat, join='inner')

    # Convert to numpy and record original shape
    original_shape = fc_concat.shape  # (n_time, n_vars, n_lat, n_lon)
    fc_data = fc_concat.values.reshape(original_shape[0], original_shape[1] * original_shape[2] * original_shape[3])
    obs_data = obs_concat.values.reshape(original_shape[0], original_shape[1] * original_shape[2] * original_shape[3])

    lon_vals = fc_concat.longitude.values
    lat_vals = fc_concat.latitude.values
    time_vals = fc_concat.time.values

    print("Forecast data shape (flattened):", fc_data.shape)
    print("Observation data shape (flattened):", obs_data.shape)

    return fc_data, obs_data, lon_vals, lat_vals, time_vals, original_shape


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


def normalize_data(train_fc, val_fc, train_obs, val_obs, n_vars):
    """
    Normalize forecast and observation data variable-wise using training statistics.

    Args:
        train_fc (np.ndarray): Training forecast data of shape (n_time, n_vars*space).
        val_fc (np.ndarray): Validation forecast data.
        train_obs (np.ndarray): Training observation data.
        val_obs (np.ndarray): Validation observation data.
        n_vars (int): Number of variables.

    Returns:
        tuple: Normalized training/validation data and normalization statistics.
    """
    # Determine spatial size (lat*lon)
    space_dim = train_fc.shape[1] // n_vars
    n_time = train_fc.shape[0]

    # Reshape to (n_time, n_vars, space)
    train_fc_reshaped = train_fc.reshape(n_time, n_vars, space_dim)
    val_fc_reshaped = val_fc.reshape(val_fc.shape[0], n_vars, space_dim)
    # Compute stats per variable for forecast
    mean_fc = train_fc_reshaped.mean(axis=(0,2), keepdims=True)  # shape (1, n_vars, 1)
    std_fc = train_fc_reshaped.std(axis=(0,2), keepdims=True)
    train_fc_norm = ((train_fc_reshaped - mean_fc) / (std_fc + 1e-8)).reshape(train_fc.shape)
    val_fc_norm = ((val_fc_reshaped - mean_fc) / (std_fc + 1e-8)).reshape(val_fc.shape)

    # Repeat for observations
    train_obs_reshaped = train_obs.reshape(n_time, n_vars, space_dim)
    val_obs_reshaped = val_obs.reshape(val_obs.shape[0], n_vars, space_dim)
    mean_obs = train_obs_reshaped.mean(axis=(0,2), keepdims=True)
    std_obs = train_obs_reshaped.std(axis=(0,2), keepdims=True)
    train_obs_norm = ((train_obs_reshaped - mean_obs) / (std_obs + 1e-8)).reshape(train_obs.shape)
    val_obs_norm = ((val_obs_reshaped - mean_obs) / (std_obs + 1e-8)).reshape(val_obs.shape)

    stats = {
        'mean_fc': mean_fc.squeeze(),  # shape (n_vars,)
        'std_fc': std_fc.squeeze(),
        'mean_obs': mean_obs.squeeze(),
        'std_obs': std_obs.squeeze(),
        'n_vars': n_vars,
        'space_dim': space_dim
    }
    return train_fc_norm, val_fc_norm, train_obs_norm, val_obs_norm, stats


def unnormalize_data(corrected_norm, stats, is_obs=True):
    """
    Un-normalize corrected data variable-wise using stored statistics.

    Args:
        corrected_norm (np.ndarray): Normalized corrected data (n_time, n_vars*space).
        stats (dict): Contains per-variable stats.
        is_obs (bool): Whether to unnormalize to observation scale.

    Returns:
        np.ndarray: Unnormalized data with same shape as input.
    """
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
    """
    Train the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        predictions = model(x_batch)
        # Extract only the parts of y_batch corresponding to the output variables
        y_batch_subset = y_batch[:, selected_indices]
        loss = criterion(predictions, y_batch_subset)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x_batch.size(0)

    return running_loss / len(dataloader.dataset)


def validate_one_epoch(model, dataloader, criterion, device, selected_indices):
    """
    Validate the model for one epoch.
    """
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
    """
    Train the model over multiple epochs.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, selected_indices)
        valid_loss = validate_one_epoch(model, valid_loader, criterion, device, selected_indices)

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

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


def save_output(output_dir, run_id, output_vars, lon_vals, lat_vals,
                time_vals, original_fc, corrected_fc, original_shape,
                ground_truth_data=None):
    """
    Save original and corrected forecasts (and optionally ground truth) in Zarr format.
    Supports both single and multiple variables.
    """
    # Determine if multiple variables are present
    if len(original_shape) == 4:
        n_time, n_vars, n_lat, n_lon = original_shape
    else:
        # Single variable case (reshape to 4D with n_vars=1)
        n_time, n_lon, n_lat = original_shape
        n_vars = 1
        original_fc = original_fc.reshape(n_time, n_vars * n_lon * n_lat)
        corrected_fc = corrected_fc.reshape(n_time, n_vars * n_lon * n_lat)

    ds_dict = {}
    for i in range(n_vars):
        # Extract slice for variable i
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
    ds_out.attrs['description'] = (f'Original and corrected forecasts for run: {run_id}')
    output_filename = f"{run_id}.zarr"
    output_path = os.path.join(output_dir, output_filename)
    ds_out.to_zarr(output_path, mode='w')
    print(f"Forecasts saved to {output_path} (Zarr format)")


def main():

    # Set up device: prioritize CUDA, then MPS, then CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print("device", device)


    args = parse_args()
    run_id = generate_run_id(args)

    print("run id")
    print(run_id)

    os.makedirs(args.output_dir, exist_ok=True)

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
        print("Invalid region specified. Please choose from 'full_india', 'north_india', or 'uttar_pradesh'.")
        exit()
    n_vars = len(args.training_vars)

    train_time_slice = slice(args.train_start, args.train_end)
    test_time_slice = slice(args.test_start, args.test_end)

    # =========================================================================
    # 1) Load training data (for all specified variables)
    # =========================================================================
    train_fc_full, train_obs_full, lon_vals, lat_vals, train_time_vals, original_shape = load_data(
        args.forecast_path,
        args.obs_path,
        args.training_vars,
        args.level,
        lat_slice,
        lon_slice,
        train_time_slice,
        args.lead_time_hours
    )

    print("Loaded Training Data")

    # =========================================================================
    # 2) Randomly split training data into TRAIN (80%) and VAL (20%)
    # =========================================================================
    n_samples = train_fc_full.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    split_idx = int(0.8 * n_samples)
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    train_fc = train_fc_full[train_idx]
    train_obs = train_obs_full[train_idx]
    val_fc = train_fc_full[val_idx]
    val_obs = train_obs_full[val_idx]
    val_time_subset = train_time_vals[val_idx]

    # =========================================================================
    # 3) Load the test data
    # =========================================================================
    test_fc, test_obs, _, _, test_time_vals, test_original_shape = load_data(
        args.forecast_path,
        args.obs_path,
        args.training_vars,
        args.level,
        lat_slice,
        lon_slice,
        test_time_slice,
        args.lead_time_hours
    )

    print("Loaded Test Data")

    # =========================================================================
    # 4) Normalize training & validation data variable-wise using training stats
    # =========================================================================
    (train_fc_norm,
     val_fc_norm,
     train_obs_norm,
     val_obs_norm,
     stats) = normalize_data(train_fc, val_fc, train_obs, val_obs, n_vars)

    # =========================================================================
    # 5) Create PyTorch DataLoaders
    # =========================================================================
    train_loader = create_dataloader(train_fc_norm, train_obs_norm, args.batch_size)
    val_loader = create_dataloader(val_fc_norm, val_obs_norm, args.batch_size)

    # =========================================================================
    # 6) Initialize and train the model
    # =========================================================================
    input_dim = train_fc.shape[1]  # equals n_vars * lat * lon

    # Compute the spatial dimension based on the training data
    spatial_dim = train_fc.shape[1] // len(args.training_vars)
    # Create a list of indices corresponding to the output variables (which are a subset of training_vars)
    selected_indices = []
    for i, var in enumerate(args.training_vars):
        if var in args.output_vars:
            # For variable i, its flattened data spans from i*spatial_dim to (i+1)*spatial_dim
            selected_indices.extend(list(range(i * spatial_dim, (i + 1) * spatial_dim)))
    # Define the output dimension of the MLP as (number of output vars) * (spatial dimension)
    output_dim = len(selected_indices)

    model = SimpleMLP(input_dim=input_dim,
                      hidden_dim=args.mlp_hidden_dim,
                      output_dim=output_dim,
                      num_hidden_layers=args.mlp_layers)
    model.to(device)
    model = train_model(model, train_loader, val_loader, epochs= args.epochs, 
                        learning_rate = 1e-5, device = device, selected_indices = selected_indices)

    # Save model weights
    model_path = os.path.join(args.output_dir, f"{args.model_name}_mlp_correction.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model weights saved to {model_path}")

    # =========================================================================
    # 7) Apply correction to the validation set
    # =========================================================================
    corrected_val_fc_norm = apply_correction(model, val_fc_norm, device)
    corrected_val_fc = unnormalize_data(corrected_val_fc_norm, stats, is_obs=True)

    # =========================================================================
    # 8) Apply correction to the test set
    # =========================================================================
    test_fc_norm = (test_fc - stats['mean_fc'].mean()) / (stats['std_fc'].mean() + 1e-8)
    corrected_test_fc_norm = apply_correction(model, test_fc_norm, device)
    corrected_test_fc = unnormalize_data(corrected_test_fc_norm, stats, is_obs=True)
    print(f"MSE (original forecast, test set): {np.mean((test_fc - test_obs) ** 2):.6f}")
    print(f"MSE (corrected forecast, test set): {np.mean((corrected_test_fc - test_obs) ** 2):.6f}")

    # =========================================================================
    # 9) Save outputs for the test set
    # =========================================================================
    save_output(
        output_dir=args.output_dir,
        run_id = run_id,
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
