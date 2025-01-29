#!/usr/bin/env python3
"""
weatherbench2_finetuning_chunked.py
Author: Ozzy Houck
Date Created: 12/27/2024

This script fine-tunes a simple MLP to a specific region using model
forecasts and corresponding observations from weatherbench2, but in
a more memory-efficient way. It does the following:

1) Opens the forecast and observation datasets with chunking via xarray/dask.
2) Selects bounding boxes in lat/lon, plus time slices for train/val/test.
3) Uses xarray to compute global mean/std for both forecast and obs in the
   specified region/time, which we will use for normalization.
4) Creates a custom PyTorch Dataset (WeatherBenchChunkedDataset) that:
   - references the xarray objects,
   - loads a single time index (or small subset) on each __getitem__,
   - normalizes on the fly using the precomputed mean/std.
5) Trains an MLP model to map forecast -> observation.
6) Applies corrections to validation/test splits, saves to disk.

"""

import argparse
import os

import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim

# -------------------------
# 1) MLP Definition
# -------------------------
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

# -------------------------
# 2) Argument Parsing
# -------------------------
def parse_args():
    """
    Parse command-line arguments for fine-tuning the MLP on regional post-processing.
    """
    parser = argparse.ArgumentParser(description='Fine-tune MLP for regional post-processing')
    parser.add_argument('--forecast_path', type=str, required=True,
                        help='Path to forecast data (Zarr)')
    parser.add_argument('--obs_path', type=str, required=True,
                        help='Path to observation data (Zarr)')
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

# -------------------------
# 3) Open & Slice Datasets
# -------------------------
def open_and_slice_datasets(forecast_path, obs_path, var_name, level,
                            lat_min, lat_max, lon_min, lon_max,
                            time_start, time_end, lead_time_hours):
    """
    Open the forecast/obs with chunking, select the region/time,
    align them, and return xarray DataArrays.
    
    We'll do this for whichever time range is passed (train vs test).
    """
    # OH: use native chunking (I think) faster than not including chunk argument 
    # Might need to come back to once I understand this better
    ds_forecast = xr.open_zarr(forecast_path, chunks= None)
    ds_obs      = xr.open_zarr(obs_path,      chunks= None)

    # 3b. Build slices
    lat_slice = slice(lat_min, lat_max)
    lon_slice = slice(lon_min, lon_max)
    time_slice = slice(time_start, time_end)

    # 3c. Select var, lat, lon, time from forecast
    if 'level' in ds_forecast[var_name].dims and level is not None:
        fc_var = ds_forecast[var_name].sel(time=time_slice,
                                           latitude=lat_slice,
                                           longitude=lon_slice,
                                           level=level)
    else:
        fc_var = ds_forecast[var_name].sel(time=time_slice,
                                           latitude=lat_slice,
                                           longitude=lon_slice)

    # 3d. Select var, lat, lon, time from obs
    if 'level' in ds_obs[var_name].dims and level is not None:
        obs_var = ds_obs[var_name].sel(time=time_slice,
                                       latitude=lat_slice,
                                       longitude=lon_slice,
                                       level=level)
    else:
        obs_var = ds_obs[var_name].sel(time=time_slice,
                                       latitude=lat_slice,
                                       longitude=lon_slice)

    # 3e. Select forecast lead time
    fc_var = fc_var.sel(prediction_timedelta=np.timedelta64(lead_time_hours, 'h'))
    if 'prediction_timedelta' in fc_var.coords:
        fc_var = fc_var.drop_vars('prediction_timedelta')

    # 3f. Align to ensure same time dimension
    fc_var, obs_var = xr.align(fc_var, obs_var, join='inner')

    return fc_var, obs_var

# -------------------------
# 4) Custom PyTorch Dataset
# -------------------------
class WeatherBenchChunkedDataset(torch.utils.data.Dataset):
    """
    Loads a single time index from the xarray data at each __getitem__ call.
    This avoids loading the entire dataset into memory.
    
    We do forecast -> obs training. Both are normalized using precomputed stats.
    """
    def __init__(self, fc_var, obs_var, time_indices,
                 fc_mean, fc_std, obs_mean, obs_std):
        """
        fc_var, obs_var: xarray DataArrays (time, latitude, longitude).
        time_indices: the list of time indices to sample from (subset for train or val).
        fc_mean, fc_std, obs_mean, obs_std: floats for normalization.
        """
        self.fc_var = fc_var
        self.obs_var = obs_var
        self.time_indices = time_indices

        self.fc_mean = fc_mean
        self.fc_std = fc_std
        self.obs_mean = obs_mean
        self.obs_std = obs_std

    def __len__(self):
        return len(self.time_indices)

    def __getitem__(self, idx):
        t_idx = self.time_indices[idx]

        # We load that time index (lazy load, then compute)
        fc_slice = self.fc_var.isel(time=t_idx).compute()  # shape (lat, lon)
        obs_slice = self.obs_var.isel(time=t_idx).compute()  # shape (lat, lon)

        # Convert to numpy
        fc_np = fc_slice.values
        obs_np = obs_slice.values
        
        # Flatten lat/lon
        fc_np = fc_np.ravel()
        obs_np = obs_np.ravel()

        # Normalize
        fc_norm = (fc_np - self.fc_mean) / (self.fc_std + 1e-8)
        obs_norm = (obs_np - self.obs_mean) / (self.obs_std + 1e-8)

        # Convert to torch tensors
        fc_tensor = torch.from_numpy(fc_norm).float()
        obs_tensor = torch.from_numpy(obs_norm).float()

        return fc_tensor, obs_tensor

# -------------------------
# 5) Mean/Std Computation
# -------------------------
def compute_mean_std(fc_var, obs_var):
    """
    Uses xarray to compute the mean and std across time/lat/lon
    for forecast and observations. Returns scalars for each.
    
    NOTE: This triggers a Dask computation, but doesn't load
    everything in memory at once because it can do distributed
    or chunked reductions.
    """
    # dims = ('time', 'latitude', 'longitude') - your variable might have these names
    fc_mean = fc_var.mean(dim=("time", "latitude", "longitude")).compute().item()
    fc_std  = fc_var.std(dim=("time", "latitude", "longitude")).compute().item()

    obs_mean = obs_var.mean(dim=("time", "latitude", "longitude")).compute().item()
    obs_std  = obs_var.std(dim=("time", "latitude", "longitude")).compute().item()

    return fc_mean, fc_std, obs_mean, obs_std

# -------------------------
# 6) PyTorch Training/Val
# -------------------------
def train_one_epoch(model, dataloader, optimizer, criterion, device):
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

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        valid_loss = validate_one_epoch(model, valid_loader, criterion, device)

        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

    return model

# -------------------------
# 7) Applying Correction
# -------------------------
def apply_correction(model, fc_var, fc_mean, fc_std, obs_mean, obs_std, device):
    """
    Applies the MLP correction to an entire forecast xarray DataArray fc_var.
    We'll do it by time steps, accumulate results, then
    reassemble into a DataArray. 
    """
    # We'll create an array to hold the corrected forecast in memory.
    # If that's too large, we can also chunk this approach, but let's
    # do it step by step in a loop.
    n_time = fc_var.sizes['time']
    lat_size = fc_var.sizes['latitude']
    lon_size = fc_var.sizes['longitude']

    corrected_data = np.zeros((n_time, lat_size, lon_size), dtype=np.float32)

    with torch.no_grad():
        for t_idx in range(n_time):
            fc_slice = fc_var.isel(time=t_idx).compute()  # shape (lat, lon)
            fc_np = fc_slice.values
            fc_np = fc_np.ravel()

            # Normalize
            fc_norm = (fc_np - fc_mean) / (fc_std + 1e-8)
            fc_tensor = torch.from_numpy(fc_norm).float().to(device)

            # Model prediction is in "obs space" (we trained fc -> obs),
            # so output is the normalized obs
            pred_norm = model(fc_tensor)
            # shape (lat*lon, )
            pred_norm = pred_norm.cpu().numpy()

            # Un-normalize to obs space
            pred = pred_norm * (obs_std + 1e-8) + obs_mean

            # Reshape to (lat, lon)
            pred_2d = pred.reshape(lat_size, lon_size)
            corrected_data[t_idx] = pred_2d

    # Convert corrected_data into xarray
    corrected_da = xr.DataArray(
        data=corrected_data,
        coords=[fc_var.time, fc_var.latitude, fc_var.longitude],
        dims=["time", "latitude", "longitude"]
    )
    return corrected_da

# -------------------------
# 8) Saving Output
# -------------------------
def save_output(output_dir, model_name, var_name, level,
                ds_fc, ds_corrected, ds_obs=None,
                dataset_label='validation'):
    """
    Save the original forecast and corrected forecast as a Zarr. Optionally
    include the ground truth obs if provided.
    ds_fc, ds_corrected, ds_obs are xarray DataArrays or Datasets.
    """
    out_ds = xr.Dataset()
    out_ds[f"{var_name}_original"] = ds_fc
    out_ds[f"{var_name}_corrected"] = ds_corrected
    if ds_obs is not None:
        out_ds[f"{var_name}_obs"] = ds_obs

    level_str = f"{level}hPa" if level is not None else ""
    filename = f"{model_name}_{dataset_label}_forecasts_{var_name}{level_str}.zarr"
    out_path = os.path.join(output_dir, filename)
    out_ds.to_zarr(out_path, mode='w')
    print(f"Saved {dataset_label} forecasts to {out_path}")

# -------------------------
# 9) Main Function
# -------------------------
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

    # -----------------------------
    # (A) Open & slice training data
    # -----------------------------
    fc_train, obs_train = open_and_slice_datasets(
        args.forecast_path,
        args.obs_path,
        var_name=args.var_name,
        level=args.level,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        time_start=args.train_start,
        time_end=args.train_end,
        lead_time_hours=args.lead_time_hours
    )

    # Because we do a random split of times for train/val, we get time dimension size:
    n_time_total = fc_train.sizes['time']
    time_indices = np.arange(n_time_total)
    np.random.shuffle(time_indices)

    split_idx = int(0.8 * n_time_total)
    train_indices = time_indices[:split_idx]
    val_indices   = time_indices[split_idx:]

    # -----------------------------
    # (B) Compute mean/std from the ENTIRE training range (train+val times)
    #     so that validation data is also using the same stats.
    # -----------------------------
    fc_mean, fc_std, obs_mean, obs_std = compute_mean_std(fc_train, obs_train)
    print(f"Forecast mean/std: {fc_mean:.4f}, {fc_std:.4f}")
    print(f"Obs mean/std:      {obs_mean:.4f}, {obs_std:.4f}")

    # -----------------------------
    # (C) Create PyTorch Datasets & Loaders for train/val
    # -----------------------------
    train_dataset = WeatherBenchChunkedDataset(fc_train, obs_train,
                                               train_indices,
                                               fc_mean, fc_std,
                                               obs_mean, obs_std)
    val_dataset = WeatherBenchChunkedDataset(fc_train, obs_train,
                                             val_indices,
                                             fc_mean, fc_std,
                                             obs_mean, obs_std)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,  # how many time steps per batch
        shuffle=True, 
        num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers = 0
    )

    # -----------------------------
    # (D) Build MLP
    # Input dimension = lat * lon
    # For that, we check the shapes:
    # But we haven't loaded the data fully, so let's infer from the first sample
    # or from the xarray dims:
    #    lat_size = fc_train.sizes['latitude']
    #    lon_size = fc_train.sizes['longitude']
    # That means input_dim = lat_size * lon_size
    # -----------------------------
    lat_size = fc_train.sizes['latitude']
    lon_size = fc_train.sizes['longitude']
    input_dim = lat_size * lon_size

    model = SimpleMLP(input_dim=input_dim,
                      hidden_dim=128,
                      output_dim=input_dim,  # we want to output shape = lat*lon
                      num_hidden_layers=3)
    model.to(device)

    # -----------------------------
    # (E) Train
    # -----------------------------
    model = train_model(model, train_loader, val_loader,
                        args.epochs, args.learning_rate, device)

    # Save the model
    model_path = os.path.join(args.output_dir,
                              f"{args.model_name}_mlp_correction.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model weights saved to {model_path}")

    # -----------------------------
    # (F) Apply correction to validation subset
    #     We'll do that by building a subset DataArray for just the val times
    # -----------------------------
    val_time_sorted = np.sort(val_indices)
    fc_val = fc_train.isel(time=val_time_sorted)
    obs_val = obs_train.isel(time=val_time_sorted)

    corrected_val = apply_correction(
        model, fc_val, fc_mean, fc_std, obs_mean, obs_std, device
    )

    # Compute MSE for original vs corrected on val:
    # (We'll do it in memory here, be mindful for large data)
    # align again to ensure shapes match
    fc_val, obs_val = xr.align(fc_val, obs_val)
    mse_original_val = ((fc_val - obs_val)**2).mean().compute().item()
    mse_corrected_val = ((corrected_val - obs_val)**2).mean().compute().item()
    print(f"MSE (original forecast, validation set): {mse_original_val:.6f}")
    print(f"MSE (corrected forecast, validation set): {mse_corrected_val:.6f}")

    # Save validation output
    save_output(args.output_dir, args.model_name, args.var_name, args.level,
                ds_fc=fc_val, ds_corrected=corrected_val, ds_obs=obs_val,
                dataset_label='validation')

    # -----------------------------
    # (G) Load & Apply to Test Data
    # -----------------------------
    fc_test, obs_test = open_and_slice_datasets(
        args.forecast_path,
        args.obs_path,
        var_name=args.var_name,
        level=args.level,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        lon_min=args.lon_min,
        lon_max=args.lon_max,
        time_start=args.test_start,
        time_end=args.test_end,
        lead_time_hours=args.lead_time_hours
    )

    corrected_test = apply_correction(
        model, fc_test, fc_mean, fc_std, obs_mean, obs_std, device
    )

    # Optionally compute MSE for test data
    fc_test, obs_test = xr.align(fc_test, obs_test)
    mse_original_test = ((fc_test - obs_test)**2).mean().compute().item()
    mse_corrected_test = ((corrected_test - obs_test)**2).mean().compute().item()
    print(f"MSE (original forecast, test set): {mse_original_test:.6f}")
    print(f"MSE (corrected forecast, test set): {mse_corrected_test:.6f}")

    exit()

    save_output(args.output_dir, args.model_name, args.var_name, args.level,
                ds_fc=fc_test, ds_corrected=corrected_test, ds_obs=obs_test,
                dataset_label='test')


if __name__ == "__main__":
    main()
