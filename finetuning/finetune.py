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
    --var_names 2m_temperature precipitation \
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
    parser.add_argument('--model_name', type=str, default="pangu",
                        help='Name of the base model (e.g. pangu, ifs, neural_gcm)')
    parser.add_argument('--region', type=str, default="north_india",
                        help='Name of the base model (e.g. pangu, ifs, neural_gcm)')
    parser.add_argument('--lead_time_hours', type=int, default=48,
                        help='Lead time in hours for forecast')
    parser.add_argument('--var_names', type=str, nargs='+', default=["2m_temperature"],
                        help='Variables to fine-tune (e.g. 2m_temperature precipitation)')
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


def save_data_locally(path, var_names, level, lat_slice, lon_slice,
              time_values, lead_time_hours, output_path):
    # Open datasets (supporting Zarr or NetCDF)
    ds = (
        xr.open_zarr(path) if path.endswith('.zarr')
        else xr.open_dataset(path)
    )
    
    # Ensure consistent ordering of latitude
    ds = ds.sortby('latitude')

    # Rename dims if necessary
    for v in var_names:
        if v not in ds:
            print(f"Variable '{v}' not found in dataset. Skipping...")
            print(f"Available variables: {list(ds.data_vars)}")
            continue
        dims = ds[v].dims
        if 'latitude' not in dims and 'lat' in dims:
            ds = ds.rename({'lat': 'latitude'})
        if 'longitude' not in dims and 'lon' in dims:
            ds = ds.rename({'lon': 'longitude'})

    # Select region, time, and variables
    ds= ds.sel(time=time_values,
                latitude=lat_slice,
                longitude=lon_slice)[var_names]

    if 'prediction_timedelta' in ds.coords:
        ds = ds.sel(prediction_timedelta=np.timedelta64(lead_time_hours, 'h'))
        ds = ds.drop_vars('prediction_timedelta')
    
    #OH note XX: this might not work when mixing surface and levels need to check
    if level is not None:
        ds = ds.sel(level = level)

    # Clear encoding for each data variable
    for var in ds.data_vars:
        ds[var].encoding.clear()

    # save to netcdf
    ds.to_netcdf(output_path, mode='w')


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


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train the model for one epoch.
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
    Validate the model for one epoch.
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
    Train the model over multiple epochs.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        valid_loss = validate_one_epoch(model, valid_loader, criterion, device)

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


def save_output(output_dir, model_name, var_names, level, lon_values, lat_values,
                time_values, original_fc, corrected_fc, ground_truth_data=None):
    """
    Save original and corrected forecasts (and optionally ground truth) in Zarr format.
    Supports both single and multiple variables.
    """
    n_vars = len(var_names)
    n_time = len(time_values)
    n_lon = len(lon_values)
    n_lat = len(lat_values)

    # reshape to be (variable, time, lat, lon)
    original_fc = original_fc.reshape(n_time, n_vars * n_lon * n_lat)
    original_fc = original_fc.transpose(1, 0, 2, 3)
    corrected_fc = corrected_fc.reshape(n_time, n_vars * n_lon * n_lat)
    corrected_fc = corrected_fc.transpose(1, 0, 2, 3)

    # convert to xarray DataArray
    # XX COME BACK HEERE
    original_fc_da = xr.DataArray(
        data=original_fc,
        dims=['variable', 'time', 'latitude', 'longitude'],
        coords={"variable": var_names,
                "time": time_values, 
                "latitude": lat_values,
                "longitude": lon_values}
        )
    corrected_fc_da = xr.DataArray(
        data=corrected_fc,
        dims=['variable', 'time', 'latitude', 'longitude'],
        coords={"variable": var_names,
                "time": time_values, 
                "latitude": lat_values,
                "longitude": lon_values}
        )

    if ground_truth_data is not None:
        ground_truth_data = ground_truth_data.reshape(n_time, n_vars * n_lon * n_lat)
        ground_truth_data = ground_truth_data.transpose(1, 0, 2, 3)
        ground_truth_da = xr.DataArray(
            data=ground_truth_data,
            dims=['variable', 'time', 'latitude', 'longitude'],
            coords={"variable": var_names,
                    "time": time_values, 
                    "latitude": lat_values,
                    "longitude": lon_values}
            )


    ds_out = xr.Dataset(ds_dict)
    ds_out.attrs['description'] = (f'Original and corrected forecasts from {model_name} '
                                   f'using MLP fine-tuning ({dataset_label} set)')
    level_str = f'_{level}hPa' if level is not None else ''
    output_filename = f"{model_name}_{dataset_label}_forecasts{level_str}.zarr"
    output_path = os.path.join(output_dir, output_filename)
    ds_out.to_zarr(output_path, mode='w')
    print(f"Original and corrected {dataset_label} forecasts saved to {output_path} (Zarr format)")


def main():
    # Set up device: prioritize CUDA, then MPS, then CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    args = parse_args()
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Prepare region and time slices
    if args.region == "full_india":
        lat_min, lat_max = 8.75, 27.25
        lon_min, lon_max = 70.75, 87.25
    if args.region == "north_india":
        lat_min, lat_max = 21, 27.25
        lon_min, lon_max = 70.75, 87.25
    if args.region == "uttar_pradesh":
        lat_min, lat_max = 24.25, 26
        lon_min, lon_max = 78, 87.25

    lat_slice = slice(lat_min, lat_max)
    lon_slice = slice(lon_min, lon_max)
    train_time_values = np.arange(np.datetime64(args.train_start), np.datetime64(args.train_end), np.timedelta64(24, 'h'))
    test_time_values = np.arange(np.datetime64(args.train_start), np.datetime64(args.train_end), np.timedelta64(24, 'h'))
    n_vars = len(args.var_names)

    test_forecast_data_path = os.path.join(output_dir, f"{args.model_name}_test_forecast_data.nc")
    test_obs_data_path = os.path.join(output_dir, f"{args.model_name}_test_obs_data.nc")



    # =========================================================================
    # 1) Load training data (for all specified variables)
    # =========================================================================
    full_var_list = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"]
    train_forecast_data_path = os.path.join(output_dir, f"{args.model_name}_train_forecast_data.nc")
    train_obs_data_path = os.path.join(output_dir, f"{args.model_name}_train_obs_data.nc")


    # # filter large datasets to only include variables of interest and save as netcdf
    # save_data_locally(args.forecast_path, full_var_list, args.level, 
    #                   lat_slice, lon_slice, train_time_values,
    #                   args.lead_time_hours, train_forecast_data_path)
    # save_data_locally(args.obs_path, full_var_list, args.level,
    #                    lat_slice, lon_slice, train_time_values,
    #                    args.lead_time_hours, train_obs_data_path)

    # open datasets from saved netcdf and select required variables
    fc_ds = xr.open_dataset(train_forecast_data_path)[args.var_names]
    obs_ds = xr.open_dataset(train_obs_data_path)[args.var_names]

    lon_values = np.unique(fc_ds.longitude.values)
    lat_values = np.unique(fc_ds.latitude.values)

    fc = fc_ds.to_array().values  # shape: (variable, time, lat, lon)
    fc = fc.transpose(1, 0, 2, 3)   # shape: (time, variable, lat, lon)
    n_time, n_vars, n_lat, n_lon = fc.shape
    fc = fc.reshape(n_time, n_vars * n_lat * n_lon) # shape: (time, variable*lat*lon)

    obs_fc = obs_ds.to_array().values  # shape: (variable, time, lat, lon)
    obs_fc = obs_fc.transpose(1, 0, 2, 3)   # shape: (time, variable, lat, lon)
    n_time, n_vars, n_lat, n_lon = obs_fc.shape
    obs = obs_fc.reshape(n_time, n_vars * n_lat * n_lon) # shape: (time, variable*lat*lon)

    # make sure both have same shape
    assert fc.shape == obs.shape, "Forecast and observation data must have the same shape"



    # =========================================================================
    # 4) Normalize training & validation data variable-wise using training stats
    # =========================================================================
    stats = {
        'mean': fc.mean(axis=0), # take mean over time
        'std': fc.std(axis=0) + 1e-8, # add small value to avoid division by zero
    }

    fc_norm = (fc - stats['mean']) / stats['std']
    obs_norm = (obs - stats['mean']) / stats['std']


    # =========================================================================
    # 2) Randomly split training data into TRAIN (80%) and VAL (20%)
    # split using the indices of the time dimension
    # =========================================================================
    indices = np.arange(n_time)
    np.random.shuffle(indices)
    split_idx = int(0.8 * n_time)
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
    train_loader = create_dataloader(train_fc_norm, train_obs_norm, args.batch_size)
    val_loader = create_dataloader(val_fc_norm, val_obs_norm, args.batch_size)

    # =========================================================================
    # 6) Initialize and train the model
    # =========================================================================
    input_dim = n_vars * n_lat * n_lon

    # eventually allow this to be different
    output_dim = n_vars * n_lat * n_lon
    model = SimpleMLP(input_dim=input_dim,
                      hidden_dim=512,
                      output_dim=input_dim,
                      num_hidden_layers=5)
    model.to(device)
    model = train_model(model, train_loader, val_loader, args.epochs, args.learning_rate, device)

    # Save model weights
    model_path = os.path.join(output_dir, f"{args.model_name}_mlp_correction.pt")
    # torch.save(model.state_dict(), model_path)
    print(f"Model weights saved to {model_path}")

    # =========================================================================
    # 3) Load the test data
    # =========================================================================

    test_forecast_data_path = os.path.join(output_dir, f"{args.model_name}_test_forecast_data.nc")
    test_obs_data_path = os.path.join(output_dir, f"{args.model_name}_test_obs_data.nc")

    # save_data_locally(args.forecast_path, full_var_list, args.level,
    #                   lat_slice, lon_slice, test_time_values,
    #                   args.lead_time_hours, test_forecast_data_path)
    # save_data_locally(args.obs_path, full_var_list, args.level,
    #                   lat_slice, lon_slice, test_time_values,
    #                   args.lead_time_hours, test_obs_data_path)

    # open datasets from saved netcdf
    test_fc_ds = xr.open_dataset(test_forecast_data_path)[args.var_names]
    test_obs_ds = xr.open_dataset(test_obs_data_path)[args.var_names]

    test_fc = test_fc_ds.to_array().values  # shape: (variable, time, lat, lon)
    test_fc = test_fc.transpose(1, 0, 2, 3)   # shape: (time, variable, lat, lon)
    n_time, n_vars, n_lat, n_lon = test_fc.shape
    test_fc = test_fc.reshape(n_time, n_vars * n_lat * n_lon) # shape: (time, variable*lat*lon)

    test_obs = test_obs_ds.to_array().values  # shape: (variable, time, lat, lon)
    test_obs = test_obs.transpose(1, 0, 2, 3)   # shape: (time, variable, lat, lon)
    n_time, n_vars, n_lat, n_lon = test_obs.shape
    test_obs = test_obs.reshape(n_time, n_vars * n_lat * n_lon) # shape: (time, variable*lat*lon)


    # =========================================================================
    # 8) Apply correction to the test set
    # =========================================================================
    test_fc_norm = (test_fc - stats['mean'].mean()) / (stats['std'].mean() + 1e-8)
    corrected_test_fc_norm = apply_correction(model, test_fc_norm, device)
    corrected_test_fc = corrected_test_fc_norm * stats['std'] + stats['mean']

    print(f"MSE (original forecast, test set): {np.mean((test_fc - test_obs) ** 2):.6f}")
    print(f"MSE (corrected forecast, test set): {np.mean((corrected_test_fc - test_obs) ** 2):.6f}")


    # =========================================================================
    # 9) Save outputs for the test set
    # =========================================================================
    save_output(
        output_dir=output_dir,
        model_name=args.model_name,
        var_names=args.var_names,
        level=args.level,
        lon_values=lon_values,
        lat_values=lat_values,
        time_values=test_time_values,
        original_fc=test_fc,
        corrected_fc=corrected_test_fc,
        ground_truth_data=test_obs
    )

if __name__ == "__main__":
    main()
