#!/usr/bin/env python3
"""
weatherbench2_finetuning.py
Author: Ozzy Houck
Date: 12/20/2024

This script fine-tunes a MLP to a specific region using model
forecasts and corresponding observations from weatherbench2. It trains a correction model that maps
model-forecasted fields to observed fields over a specified bounding box. After
training, it applies the correction to both a validation split from the training period
and a separate test set of forecasts, then saves the corrected forecasts to disk.
----------------------------

Example usage 
python3 weatherbench2_finetuning.py \
    --forecast_path="gs://weatherbench2/datasets/pangu/2018-2022_0012_64x32_equiangular_conservative.zarr" \
    --obs_path="gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr" \
    --output_dir="~/wb_finetune_test" \
    --model_name="pangu" \
    --lat_min=20 --lat_max=50 --lon_min=60 --lon_max=85 \
    --train_start="2018-03-01" --train_end="2018-06-01" \
    --test_start="2020-03-01" --test_end="2020-06-01" \
    --lead_time_hours=6 \
    --var_name="2m_temperature" \
    --epochs=100 --batch_size=32 --learning_rate=1e-4 

"""

import argparse
import os

import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim


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
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name of the base model (e.g. pangu, ifs, neural_gcm)')
    parser.add_argument('--lat_min', type=float, required=True,
                        help='Minimum latitude for region')
    parser.add_argument('--lat_max', type=float, required=True,
                        help='Maximum latitude for region')
    parser.add_argument('--lon_min', type=float, required=True,
                        help='Minimum longitude for region')
    parser.add_argument('--lon_max', type=float, required=True,
                        help='Maximum longitude for region')
    parser.add_argument('--lead_time_hours', type=int, default=48,
                        help='Lead time in hours for forecast')
    parser.add_argument('--var_name', type=str, default='temperature',
                        help='Variable to fine-tune (e.g. temperature, 2m_temperature)')
    parser.add_argument('--level', type=int, nargs='?', default=None,
                        help='Pressure level if applicable')
    parser.add_argument('--train_start', type=str, default='2018-01-01',
                        help='Training start date')
    parser.add_argument('--train_end', type=str, default='2019-12-31',
                        help='Training end date')
    parser.add_argument('--test_start', type=str, default='2020-01-01',
                        help='Test start date')
    parser.add_argument('--test_end', type=str, default='2020-12-31',
                        help='Test end date')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    return parser.parse_args()


def load_data(forecast_path, obs_path, var_name, level, lat_slice, lon_slice,
              time_slice, lead_time_hours):
    """
    Load forecast and observation data for the specified variable and region.

    Args:
        forecast_path (str): Path to forecast data.
        obs_path (str): Path to observation data.
        var_name (str): Variable name (e.g. 'temperature').
        level (int or None): Pressure level if applicable.
        lat_slice (slice): Latitude slice (min -> max).
        lon_slice (slice): Longitude slice (min -> max).
        time_slice (slice): Time slice (start -> end).
        lead_time_hours (int): Lead time in hours for forecast.

    Returns:
        tuple of np.ndarray: forecast_data, obs_data, longitudes, latitudes, time_values
    """
    # Open forecast and observation datasets
    ds_forecast = (
        xr.open_zarr(forecast_path) if forecast_path.endswith('.zarr')
        else xr.open_dataset(forecast_path)
    )
    ds_obs = (
        xr.open_zarr(obs_path) if obs_path.endswith('.zarr')
        else xr.open_dataset(obs_path)
    )
    
    # to make sure slicing works
    ds_forecast = ds_forecast.sortby('latitude')
    ds_obs = ds_obs.sortby('latitude')

    # some forecasts use lat/lon instead of latitude/longitude so rename them
    if 'latitude' not in ds_forecast[var_name].dims and 'lat' in ds_forecast[var_name].dims:
        ds_forecast = ds_forecast.rename({'lat': 'latitude'})
    if 'longitude' not in ds_forecast[var_name].dims and 'lon' in ds_forecast[var_name].dims:
        ds_forecast = ds_forecast.rename({'lon': 'longitude'})

    # Select region and time for forecast
    if 'level' in ds_forecast[var_name].dims and level is not None:
        fc_var = ds_forecast[var_name].sel(time=time_slice,
                                           latitude=lat_slice,
                                           longitude=lon_slice,
                                           level=level)
    else:
        fc_var = ds_forecast[var_name].sel(time=time_slice,
                                           latitude=lat_slice,
                                           longitude=lon_slice)
    # Select region and time for observations
    if 'level' in ds_obs[var_name].dims and level is not None:
        obs_var = ds_obs[var_name].sel(time=time_slice,
                                       latitude=lat_slice,
                                       longitude=lon_slice,
                                       level=level)
    else:
        obs_var = ds_obs[var_name].sel(time=time_slice,
                                       latitude=lat_slice,
                                       longitude=lon_slice)

    # Select the forecast by the lead time
    fc_var = fc_var.sel(prediction_timedelta=np.timedelta64(lead_time_hours, 'h'))
    if 'prediction_timedelta' in fc_var.coords:
        fc_var = fc_var.drop_vars('prediction_timedelta')

    # Align forecast and obs
    fc_var, obs_var = xr.align(fc_var, obs_var, join='inner')

    # Convert to numpy
    fc_data = fc_var.values
    obs_data = obs_var.values

    print("Forecast data shapes:", fc_data.shape)
    print("Observation data shapes:", obs_data.shape)

    # Flatten time, lat, and lon
    n_time, n_lon, n_lat = fc_data.shape
    fc_data = fc_data.reshape(n_time, n_lon * n_lat)
    obs_data = obs_data.reshape(n_time, n_lon * n_lat)

    # Get coordinate arrays
    lon_values = fc_var.longitude.values
    lat_values = fc_var.latitude.values
    time_values = fc_var.time.values

    return fc_data, obs_data, lon_values, lat_values, time_values


def create_dataloader(forecast_data, obs_data, batch_size):
    """
    Create a PyTorch DataLoader from forecast and observation data.

    Args:
        forecast_data (np.ndarray): 2D array of shape (time, lat*lon) for forecasts.
        obs_data (np.ndarray): 2D array of shape (time, lat*lon) for observations.
        batch_size (int): The batch size for the DataLoader.

    Returns:
        DataLoader: A PyTorch DataLoader for training or validation.
    """
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(forecast_data).float(),
        torch.from_numpy(obs_data).float()
    )
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True)
    return dataloader


def normalize_data(train_fc, val_fc, train_obs, val_obs):
    """
    Normalize forecast and observation data using statistics from the training set.

    Args:
        train_fc (np.ndarray): Training forecast data (2D).
        val_fc (np.ndarray): Validation forecast data (2D).
        train_obs (np.ndarray): Training observation data (2D).
        val_obs (np.ndarray): Validation observation data (2D).

    Returns:
        tuple: Normalized training/validation forecast & observation data,
               along with normalization statistics (means, stds).
    """
    # Forecast normalization
    mean_fc = train_fc.mean()
    std_fc = train_fc.std()
    train_fc_norm = (train_fc - mean_fc) / (std_fc + 1e-8)
    val_fc_norm = (val_fc - mean_fc) / (std_fc + 1e-8)

    # Observation normalization (could also use forecast stats if desired)
    mean_obs = train_obs.mean()
    std_obs = train_obs.std()
    train_obs_norm = (train_obs - mean_obs) / (std_obs + 1e-8)
    val_obs_norm = (val_obs - mean_obs) / (std_obs + 1e-8)

    stats = {
        'mean_fc': mean_fc,
        'std_fc': std_fc,
        'mean_obs': mean_obs,
        'std_obs': std_obs,
    }
    return train_fc_norm, val_fc_norm, train_obs_norm, val_obs_norm, stats


def unnormalize_data(corrected_norm, stats, is_obs=True):
    """
    Un-normalize corrected data using stored statistics.

    Args:
        corrected_norm (np.ndarray): Normalized corrected data.
        stats (dict): Contains 'mean_obs' or 'mean_fc' & 'std_obs' or 'std_fc'.
        is_obs (bool): Whether we should unnormalize to observation scale.

    Returns:
        np.ndarray: Unnormalized data.
    """
    if is_obs:
        return corrected_norm * (stats['std_obs'] + 1e-8) + stats['mean_obs']
    else:
        return corrected_norm * (stats['std_fc'] + 1e-8) + stats['mean_fc']


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train the model for one epoch on the provided DataLoader.
    """
    model.train()
    running_loss = 0.0

    for x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        predictions = model(x_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x_batch.size(0)

    return running_loss / len(dataloader.dataset)


def validate_one_epoch(model, dataloader, criterion, device):
    """
    Validate the model for one epoch on the provided DataLoader.
    """
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)
            running_loss += loss.item() * x_batch.size(0)

    return running_loss / len(dataloader.dataset)


def train_model(model, train_loader, valid_loader, epochs, lr, device):
    """
    Train the model over multiple epochs, including validation after each epoch.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        valid_loss = validate_one_epoch(model, valid_loader, criterion, device)

        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

    return model


def apply_correction(model, forecast_data, device):
    """
    Apply the MLP-based correction to forecast data.

    Args:
        model (nn.Module): The trained MLP model.
        forecast_data (np.ndarray): Forecast data of shape (time, lat*lon).

    Returns:
        np.ndarray: Corrected forecast data of shape (time, lat*lon).
    """
    model.eval()
    with torch.no_grad():
        x_tensor = torch.from_numpy(forecast_data).float().to(device)
        corrected = model(x_tensor).cpu().numpy()

    return corrected


def save_output(output_dir, model_name, var_name, level, lon_vals, lat_vals,
                time_vals, original_fc, corrected_fc, original_shape,
                dataset_label='validation',
                ground_truth_data=None):
    """
    Save original and corrected forecasts (and optionally ground truth) in Zarr format.
    Args:
        output_dir (str): Path to directory where outputs are saved.
        model_name (str): Name of the model (e.g., 'pangu').
        var_name (str): Name of the variable (e.g., 'temperature').
        level (int or None): Pressure level if applicable.
        lon_vals (np.ndarray): Longitude values.
        lat_vals (np.ndarray): Latitude values.
        time_vals (np.ndarray): Time values.
        original_fc (np.ndarray): Original forecast data (time, lat*lon).
        corrected_fc (np.ndarray): Corrected forecast data (time, lat*lon).
        original_shape (tuple): Shape (time, lat, lon) to reshape data back.
        dataset_label (str): Label for the dataset being saved ('validation', 'test', etc.).
        ground_truth_data (np.ndarray, optional): Ground truth data for comparison.
    """
    # Reshape to (time, lat, lon)
    original_fc_reshaped = original_fc.reshape(original_shape)
    corrected_fc_reshaped = corrected_fc.reshape(original_shape)

    print(f"Corrected data shape ({dataset_label}): {corrected_fc_reshaped.shape}")

    # Convert to xarray DataArrays
    da_original = xr.DataArray(
        data=original_fc_reshaped,
        coords=[time_vals, lat_vals, lon_vals],
        dims=['time', 'latitude', 'longitude'],
        name=f"{var_name}_original"
    )
    da_corrected = xr.DataArray(
        data=corrected_fc_reshaped,
        coords=[time_vals, lat_vals, lon_vals],
        dims=['time', 'latitude', 'longitude'],
        name=f"{var_name}_corrected"
    )

    # Combine into a single Dataset
    ds_out = xr.Dataset(
        {
            f'{var_name}_original': da_original,
            f'{var_name}_corrected': da_corrected
        }
    )
    ds_out.attrs['description'] = (f'Original and corrected forecasts from {model_name} '
                                   f'using MLP fine-tuning ({dataset_label} set)')

    # Optionally include ground truth
    if ground_truth_data is not None:
        ground_truth_reshaped = ground_truth_data.reshape(original_shape)
        da_ground_truth = xr.DataArray(
            data=ground_truth_reshaped,
            coords=[time_vals, lat_vals, lon_vals],
            dims=['time', 'latitude', 'longitude'],
            name=f"{var_name}_groundtruth"
        )
        ds_out[f"{var_name}_groundtruth"] = da_ground_truth
        ds_out.attrs["description"] += " (includes sliced ground truth)"

    level_str = f'{level}hPa' if level is not None else ''
    output_filename = f"{model_name}_{dataset_label}_forecasts_{var_name}{level_str}.zarr"
    output_path = os.path.join(output_dir, output_filename)

    ds_out.to_zarr(output_path, mode='w')
    print(f"Original and corrected {dataset_label} forecasts saved to {output_path} (Zarr format)")


def main():
    # Set up device prioritizing CUDA, then MPS, and finally CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare slices
    lat_slice = slice(args.lat_min, args.lat_max)
    lon_slice = slice(args.lon_min, args.lon_max)

    train_time_slice = slice(args.train_start, args.train_end)
    test_time_slice = slice(args.test_start, args.test_end)

    # =========================================================================
    # 1) Load data for the entire "training" time range
    # =========================================================================
    train_fc_full, train_obs_full, lon_vals, lat_vals, train_time_vals = load_data(
        args.forecast_path,
        args.obs_path,
        args.var_name,
        args.level,
        lat_slice,
        lon_slice,
        train_time_slice,
        args.lead_time_hours
    )

    # =========================================================================
    # 2) Randomly split the above training data into TRAIN (80%) and VAL (20%)
    # =========================================================================
    n_samples = train_fc_full.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    split_idx = int(0.8 * n_samples)
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    # Split the data
    train_fc = train_fc_full[train_idx]
    train_obs = train_obs_full[train_idx]
    val_fc = train_fc_full[val_idx]
    val_obs = train_obs_full[val_idx]
    val_time_subset = train_time_vals[val_idx]

    # =========================================================================
    # 3) Load the test data
    # =========================================================================
    test_fc, test_obs, _, _, test_time_vals = load_data(
        args.forecast_path,
        args.obs_path,
        args.var_name,
        args.level,
        lat_slice,
        lon_slice,
        test_time_slice,
        args.lead_time_hours
    )

    # =========================================================================
    # 4) Normalize training & validation data based on training statistics
    # Note: we will normalize the test data using the same training stats later.
    # =========================================================================
    (train_fc_norm,
     val_fc_norm,
     train_obs_norm,
     val_obs_norm,
     stats) = normalize_data(train_fc, val_fc, train_obs, val_obs)


    # =========================================================================
    # 5) Create PyTorch DataLoaders
    # =========================================================================
    train_loader = create_dataloader(train_fc_norm, train_obs_norm, args.batch_size)
    val_loader = create_dataloader(val_fc_norm, val_obs_norm, args.batch_size)

    # =========================================================================
    # 6) Initialize and train the model
    # =========================================================================
    input_dim = train_fc.shape[1]  # lat*lon
    model = SimpleMLP(input_dim=input_dim,
                      hidden_dim=512,
                      output_dim=input_dim,
                      num_hidden_layers=5)

    model.to(device)
    model = train_model(model, train_loader, val_loader, args.epochs, 
                        args.learning_rate, device)

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
    #    (First normalize test using training stats, then unnormalize)
    # =========================================================================
    test_fc_norm = (test_fc - stats['mean_fc']) / (stats['std_fc'] + 1e-8)
    corrected_test_fc_norm = apply_correction(model, test_fc_norm, device)
    corrected_test_fc = unnormalize_data(corrected_test_fc_norm, stats, is_obs=True)
    print(f"MSE (original forecast, test set): {np.mean((test_fc - test_obs) ** 2):.6f}")
    print(f"MSE (corrected forecast, test set): {np.mean((corrected_test_fc - test_obs) ** 2):.6f}")

    # =========================================================================
    # 10) Save outputs for test
    # =========================================================================
    n_lon = len(lon_vals)
    n_lat = len(lat_vals)

    n_time_test = test_fc.shape[0]
    save_output(
        output_dir=args.output_dir,
        model_name=args.model_name,
        var_name=args.var_name,
        level=args.level,
        lon_vals=lon_vals,
        lat_vals=lat_vals,
        time_vals=test_time_vals,
        original_fc=test_fc,
        corrected_fc=corrected_test_fc,
        original_shape=(n_time_test, n_lon, n_lat),
        dataset_label='test',
        ground_truth_data=test_obs
    )

if __name__ == "__main__":
    main()