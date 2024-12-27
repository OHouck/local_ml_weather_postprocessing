# weatherbench2_finetuning.py
# Author: Ozzy Houck
# Date: 12/20/2024
#
# early version of this code is a one-shop output from o1 given on 12/20/2024
# This script fine-tunes a simple 5-layer MLP to a specific region using model forecasts and corresponding observations.
# It trains a correction model that maps model-forecasted fields to observed fields over a specified bounding box.
# After training, it applies the correction to a test set of forecasts and saves the corrected forecasts to disk.

# For fine-tuning
# python region_add_on.py \
#   --forecast_path=path/to/base_model_forecasts.zarr \
#   --obs_path=path/to/obs.zarr \
#   --output_dir=path/to/fine_tuned_output \
#   --model_name=pangu \
#   --lat_min=24 --lat_max=37 --lon_min=60 --lon_max=78 \
#   --train_start=2018-01-01 --train_end=2019-12-31 \
#   --valid_start=2020-01-01 --valid_end=2020-12-31 \
#   --var_name=temperature --level=850 \
#   --epochs=10 --batch_size=32

# Then to evaluate the fine-tuned model
# python evaluate.py \
#   --forecast_path=path/to/fine_tuned_output/pangu_corrected_forecasts_temperature.nc \
#   --obs_path=path/to/obs.zarr \
#   --climatology_path=path/to/climatology.zarr \
#   --output_dir=path/to/evaluation_results \
#   --input_chunks=time=1,lead_time=1 \
#   --eval_configs=deterministic \
#   --use_beam=False \
#   --time_start=2020-01-01 \
#   --time_stop=2020-12-31

import argparse
import xarray as xr
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os

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

def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune MLP for regional post-processing')
    parser.add_argument('--forecast_path', type=str, required=True, help='Path to forecast data (e.g. Zarr or NetCDF)')
    parser.add_argument('--obs_path', type=str, required=True, help='Path to observation data (e.g. ERA5 Zarr or NetCDF)')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the fine-tuned model and corrected forecasts')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the base model (e.g. pangu, ifs, neural_gcm)')
    parser.add_argument('--lat_min', type=float, required=True, help='Minimum latitude for region')
    parser.add_argument('--lat_max', type=float, required=True, help='Maximum latitude for region')
    parser.add_argument('--lon_min', type=float, required=True, help='Minimum longitude for region')
    parser.add_argument('--lon_max', type=float, required=True, help='Maximum longitude for region')
    parser.add_argument('--var_name', type=str, default='temperature', help='Variable to fine-tune (e.g. temperature, 2m_temperature)')
    parser.add_argument('--level', type=int, default=850, help='Pressure level if applicable')
    parser.add_argument('--train_start', type=str, default='2018-01-01', help='Training start date')
    parser.add_argument('--train_end', type=str, default='2019-12-31', help='Training end date')
    parser.add_argument('--valid_start', type=str, default='2020-01-01', help='Validation start date')
    parser.add_argument('--valid_end', type=str, default='2020-12-31', help='Validation end date')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    return parser.parse_args()

def load_data(forecast_path, obs_path, var_name, level, lat_slice, lon_slice, time_slice):
    # Load forecast and observation data for a specific variable and region
    # Assumes data is in a compatible format and variable naming is consistent.

    ds_forecast = xr.open_zarr(forecast_path) if forecast_path.endswith('.zarr') else xr.open_dataset(forecast_path)
    ds_obs = xr.open_zarr(obs_path) if obs_path.endswith('.zarr') else xr.open_dataset(obs_path)


    # Select region and time
    if 'level' in ds_forecast[var_name].dims:
        fc_var = ds_forecast[var_name].sel(time=time_slice, latitude=lat_slice, longitude=lon_slice, level=level)
    else:
        fc_var = ds_forecast[var_name].sel(time=time_slice, latitude=lat_slice, longitude=lon_slice)

    if 'level' in ds_obs[var_name].dims:
        obs_var = ds_obs[var_name].sel(time=time_slice, latitude=lat_slice, longitude=lon_slice, level=level)
    else:
        obs_var = ds_obs[var_name].sel(time=time_slice, latitude=lat_slice, longitude=lon_slice)


    # Select the first index of the prediction_timedelta dimension and drop it
    # OH: Kinda of hacky, might not be the best way to do this
    fc_var = fc_var.isel(prediction_timedelta=0).drop('prediction_timedelta')

    # Align forecasts and obs on time, lon, lat 
    fc_var, obs_var = xr.align(fc_var, obs_var, join='inner')

    fc_data = fc_var.values
    obs_data = obs_var.values

    # shape of both forecast and observation data are (time, lon, lat)
    n_time = fc_data.shape[0]
    n_lon = fc_data.shape[1]
    n_lat = fc_data.shape[2]

    # print dimensions
    print("Data shapes of forecast:")
    print(f"Time: {n_time}, Lon: {n_lon}"), print(f"Lat: {n_lat}")
    print("Data shapes of observation:")
    print(f"Time: {obs_data.shape[0]}, Lon: {obs_data.shape[1]}"), print(f"Lat: {obs_data.shape[2]}")

    # explicitly only save time, lat, and lon dimensions
    fc_data = fc_data.reshape(n_time, n_lon, n_lat)
    obs_data = obs_data.reshape(n_time, n_lon, n_lat)

    # make sure the shapes are the same
    assert fc_data.shape == obs_data.shape

    # Flatten spatial dims for MLP
    # new shape: (time, lat*lon)
    fc_data_reshaped = fc_data.reshape(n_time, n_lat*n_lon)
    obs_data_reshaped = obs_data.reshape(n_time, n_lat*n_lon)

    # Extract the time coordinate
    time_values = fc_var['time'].values

    return fc_data_reshaped, obs_data_reshaped, fc_var.longitude.values, fc_var.latitude.values, time_values

def create_dataloader(fc_data, obs_data, batch_size):
    # Creates a PyTorch DataLoader for training/validation
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(fc_data).float(),
        torch.from_numpy(obs_data).float()
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def train_model(model, train_loader, valid_loader, epochs, lr):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    loss_metric = nn.MSELoss() #OH: Place to change the loss function
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_metric(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                loss = loss_metric(y_pred, y)
                valid_loss += loss.item() * x.size(0)
        valid_loss /= len(valid_loader.dataset)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

    return model

def apply_correction(model, fc_data):
    # Apply correction to forecasts
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(fc_data).float().to(device)
        corrected = model(x).cpu().numpy()
    return corrected

def save_corrected_forecasts(output_dir, model_name, var_name, longitude, latitude, time, corrected_data, original_shape):
    # Reshape corrected data back to (time, lat, lon)
    corrected_data = corrected_data.reshape(original_shape)

    # print corrected data shape
    print(f"Corrected data shape: {corrected_data.shape}")

    ds_out = xr.DataArray(corrected_data, coords=[time, longitude, latitude], dims=['time', 'longitude', 'latitude'], name=var_name)
    ds_out = ds_out.to_dataset()
    ds_out.attrs['description'] = f'Corrected forecasts from {model_name} using MLP fine-tuning'
    out_path = os.path.join(output_dir, f"{model_name}_corrected_forecasts_{var_name}.nc")
    ds_out.to_netcdf(out_path)
    print(f"Corrected forecasts saved to {out_path}")

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    lat_slice = slice(args.lat_min, args.lat_max)
    lon_slice = slice(args.lon_min, args.lon_max)
    train_fc, train_obs, lon_vals, lat_vals, train_time = load_data(
        args.forecast_path, args.obs_path, args.var_name, args.level,
        lat_slice, lon_slice, slice(args.train_start, args.train_end)
    )

    valid_fc, valid_obs, _, _, valid_time = load_data(
        args.forecast_path, args.obs_path, args.var_name, args.level,
        lat_slice, lon_slice, slice(args.valid_start, args.valid_end)
    )

    # Create dataloaders
    train_loader = create_dataloader(train_fc, train_obs, args.batch_size)
    valid_loader = create_dataloader(valid_fc, valid_obs, args.batch_size)

    # Model initialization
    input_dim = train_fc.shape[1]  # lat*lon
    output_dim = input_dim
    # 5-layer MLP total: Input->hidden->hidden->hidden->Output (3 hidden layers)
    # Already defined in SimpleMLP: num_hidden_layers=3 means total of 1 input, 3 hidden, 1 output layers
    model = SimpleMLP(input_dim=input_dim, hidden_dim=128, output_dim=output_dim, num_hidden_layers=3)

    # Train model
    model = train_model(model, train_loader, valid_loader, args.epochs, args.learning_rate)

    # Save model weights
    model_path = os.path.join(args.output_dir, f"{args.model_name}_mlp_correction.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model weights saved to {model_path}")

    # Apply correction to validation forecasts
    corrected_valid = apply_correction(model, valid_fc)

    # Save corrected forecasts
    n_time = valid_fc.shape[0]
    n_lat = len(lat_vals)
    n_lon = len(lon_vals)
    save_corrected_forecasts(
        args.output_dir,
        args.model_name,
        args.var_name,
        lon_vals,
        lat_vals,
        valid_time,
        corrected_valid,
        (n_time, n_lat, n_lon)
    )

if __name__ == "__main__":
    main()
