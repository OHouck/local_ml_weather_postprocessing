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
    --forecast_path="gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr" \
    --obs_path="gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr" \
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
import dask
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
    parser.add_argument('--data_dir', type=str, default="~/weatherbench2_data",
                        help='Directory to save the raw forecasts locally')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the fine-tuned model and corrected forecasts')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name of the base model (e.g. pangu, ifs, neural_gcm)')
    parser.add_argument('--region', type=str, default="north_india",
                        help='Name of the region')
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

def generate_run_id(args):
    region_str = f"{args.region}"
    dates_str = f"train{args.train_start}-{args.train_end}_test{args.test_start}-{args.test_end}"
    training_vars_str = "_".join(args.training_vars)
    output_vars_str = "_".join(args.output_vars)
    mlp_str = f"mlp{args.mlp_hidden_dim}x{args.mlp_layers}"
    lead_time = f"_leadtime_{args.lead_time_hours}"

    run_id = f"{args.model_name}_{region_str}_{dates_str}_{args.lead_time_hours}h_train_{training_vars_str}_output{output_vars_str}{lead_time}{mlp_str}"
    return run_id


def save_data_locally(path, full_surface_var_list, full_atm_var_list, lat_values, lon_values,
              time_values, lead_time_hours, output_path):
    # Open datasets (supporting Zarr or NetCDF)
    import xarray as xr
    ds = (
        xr.open_zarr(path) if path.endswith('.zarr')
        else xr.open_dataset(path)
    )
    
    # Ensure consistent ordering of latitude
    ds = ds.sortby('latitude')

    # Rename dims if necessary
    for v in full_surface_var_list + full_atm_var_list:
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
    ds_surface = ds.sel(time=time_values,
                latitude=lat_values,
                longitude=lon_values)[full_surface_var_list]
    # select atm vars for 1000 hPa level
    level = 1000
    ds_atm = ds.sel(time=time_values,
                latitude=lat_values,
                longitude=lon_values,
                level = level)[full_atm_var_list].drop_vars('level')
    # rename all atm vars to include the level with hte label "1khPa"
    ds_atm = ds_atm.rename({v: f"{v}_{level}hPa" for v in full_atm_var_list})

    # combine surface and atm datasets
    ds = xr.merge([ds_surface, ds_atm])


    import numpy as np
    import xarray as xr

    if 'prediction_timedelta' in ds.coords:
        selected_datasets = []
        for lead_time in lead_time_hours:
            selected_ds = ds.sel(prediction_timedelta=np.timedelta64(lead_time, 'h'))
            selected_datasets.append(selected_ds)
        ds = xr.concat(selected_datasets, dim='prediction_timedelta')
    
    # save to netcdf
    ds.to_netcdf(output_path, mode='w')


def load_combined_dataset(root_dir, file_pattern):
    """
    Finds all files in the subfolders of root_dir matching file_pattern and combines them.
    """
    file_paths = glob.glob(os.path.join(root_dir, "*", file_pattern))
    file_paths.sort()

    if len(file_paths) == 0:
        raise ValueError(f"No files found matching pattern: {file_pattern}")
    print(f"Combining {len(file_paths)} files for pattern: {file_pattern}")
    return xr.open_mfdataset(file_paths, combine="by_coords", decode_timedelta = True) 


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

        # print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

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


def save_output(output_dir, run_id, model_name, output_vars, lon_values, lat_values,
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

    print(ds_out)

    ds_out.attrs['description'] = (f'Original and corrected forecasts from {model_name} '
                                   f'using MLP fine-tuning)')
    output_filename = f"{run_id}.zarr"
    output_path = os.path.join(output_dir, output_filename)
    ds_out.to_zarr(output_path, mode='w')
    print(f"Forecasts saved to {output_path}")


def get_month_ranges(start_date_str, end_date_str):
    """
    Splits the period between start_date and end_date into a list of (month_start, month_end) tuples.
    """
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    current = start_date.replace(day=1)
    ranges = []
    while current <= end_date:
        # Compute the first day of the next month and then the last day of the current month
        next_month = (current.replace(day=28) + timedelta(days=4)).replace(day=1)
        month_end = next_month - timedelta(days=1)
        mstart = max(current, start_date)
        mend = min(month_end, end_date)
        ranges.append((mstart, mend))
        current = next_month
    return ranges

def main():


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
    run_id = generate_run_id(args)
    print("run id:", run_id)

    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    data_dir = os.path.expanduser(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)


    # Prepare region and time slices
    if args.region == "full_india":
        lat_min, lat_max = 8.75, 27.25
        lon_min, lon_max = 70.75, 87.25
        large_region = "india"
    elif args.region == "north_india":
        lat_min, lat_max = 21.25, 27.25
        lon_min, lon_max = 70.75, 87.25
        large_region = "india"
    elif args.region == "uttar_pradesh":
        lat_min, lat_max = 24.25, 26
        lon_min, lon_max = 78, 87.25
        large_region = "india"
    elif args.region =="pixel":
        lat_min, lat_max = 24.25, 24.5
        lon_min, lon_max = 78, 78.25
        large_region = "india"
    elif args.region == "pakistan":
        lat_min, lat_max = 25, 34
        lon_min, lon_max = 60, 70
        large_region = "pakistan"
    elif args.region == "south_pakistan":
        lat_min, lat_max = 24, 27.25
        lon_min, lon_max = 62, 70
        large_region = "pakistan"
    else:
        raise ValueError(f"Unknown region '{args.region}'. Please specify a valid region.")

    train_time_values = np.arange(np.datetime64(args.train_start), np.datetime64(args.train_end), np.timedelta64(24, 'h'))
    test_time_values = np.arange(np.datetime64(args.test_start), np.datetime64(args.test_end), np.timedelta64(24, 'h'))
    lat_values = np.arange(lat_min, lat_max, 0.25)
    lon_values = np.arange(lon_min, lon_max, 0.25)

    n_train_time = len(train_time_values)
    n_test_time = len(test_time_values)
    n_training_vars = len(args.training_vars)
    n_output_vars = len(args.output_vars)

    full_surface_var_list = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"] 
    full_atm_var_list = ["geopotential", "v_component_of_wind", "u_component_of_wind", "specific_humidity"]
    full_lead_time_hours = [24, 72, 168] # times for 1 day, 3 days, and 7 days ahead
    full_train_start = "2018-01-01"  
    full_train_end = "2021-12-31"  # full range for training data
    full_test_start = "2022-01-01"  # full range for test data
    full_test_end = "2022-12-31"  # full range for test data

    # region to download for india
    if large_region == "india":
        full_lat_values = np.arange(8.5, 28, 0.25) # full_india is max area
        full_lon_values = np.arange(70.5, 87.5, 0.25)
    elif large_region == "pakistan":
        # full lat values are by 0.25 degrees
        full_lat_values = np.arange(23.75, 34.25, 0.25) # pakistan/afganistan
        full_lon_values = np.arange(59.75, 70.25, 0.25) 

    # =========================================================================
    # 0) Download and save the data locally (if needed)
    # =========================================================================

     # ---- Training data ----
    train_months = get_month_ranges(full_train_start, full_train_end)
    train_dir = os.path.join(data_dir, f"train_{large_region}")
    os.makedirs(train_dir, exist_ok=True)

    dask.config.set(scheduler="threads", num_workers=8)
    
    for start_dt, end_dt in train_months:
        month_str = start_dt.strftime("%Y-%m")
        month_folder = os.path.join(train_dir, month_str)
        os.makedirs(month_folder, exist_ok=True)
        print(f"Saving training data for {month_str}...")
        # Create time values for the month (ensuring we include the last day)
        time_values = np.arange(np.datetime64(start_dt.strftime("%Y-%m-%d")),
                                np.datetime64((end_dt + timedelta(days=1)).strftime("%Y-%m-%d")),
                                np.timedelta64(24, 'h'))
        forecast_output_path = os.path.join(month_folder, f"{args.model_name}_train_forecast_data_{month_str}.nc")
        obs_output_path = os.path.join(month_folder, f"{args.model_name}_train_obs_data_{month_str}.nc")

        # delete and uncomment below lines
        # save_data_locally(args.obs_path, full_surface_var_list, full_atm_var_list,
        #                 full_lat_values, full_lon_values, time_values,
        #                 full_lead_time_hours, obs_output_path)

        # check if the files already exist
        if os.path.exists(forecast_output_path) and os.path.exists(obs_output_path):
            print(f"Train files already exist for {month_str}. Skipping...")
            continue
        else:
            save_data_locally(args.forecast_path, full_surface_var_list, full_atm_var_list,
                          full_lat_values, full_lon_values, time_values,
                          full_lead_time_hours, forecast_output_path)
            save_data_locally(args.obs_path, full_surface_var_list, full_atm_var_list,
                            full_lat_values, full_lon_values, time_values,
                            full_lead_time_hours, obs_output_path)

    
    # ---- Test data ----
    test_months = get_month_ranges(full_test_start, full_test_end)
    test_dir = os.path.join(data_dir, f"test_{large_region}")
    os.makedirs(test_dir, exist_ok=True)
    
    for start_dt, end_dt in test_months:
        month_str = start_dt.strftime("%Y-%m")
        month_folder = os.path.join(test_dir, month_str)
        os.makedirs(month_folder, exist_ok=True)
        print(f"Saving test data for {month_str}...")
        time_values = np.arange(np.datetime64(start_dt.strftime("%Y-%m-%d")),
                                np.datetime64((end_dt + timedelta(days=1)).strftime("%Y-%m-%d")),
                                np.timedelta64(24, 'h'))
        forecast_output_path = os.path.join(month_folder, f"{args.model_name}_test_forecast_data_{month_str}.nc")
        obs_output_path = os.path.join(month_folder, f"{args.model_name}_test_obs_data_{month_str}.nc")

        # check if the files already exist
        if os.path.exists(forecast_output_path) and os.path.exists(obs_output_path):
            print(f"Test Files already exist for {month_str}. Skipping...")
            continue
        else:
            save_data_locally(args.forecast_path, full_surface_var_list, full_atm_var_list,
                            full_lat_values, full_lon_values, time_values,
                            full_lead_time_hours, forecast_output_path)
            save_data_locally(args.obs_path, full_surface_var_list, full_atm_var_list,
                            full_lat_values, full_lon_values, time_values,
                            full_lead_time_hours, obs_output_path)
    # =========================================================================
    # 1) Load training data (for all specified variables)
    # =========================================================================

    # ----- Loading combined training data from monthly files -----
    train_dir = os.path.join(data_dir, f"train_{large_region}")
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

    print("fc shape", fc.shape, "obs shape", obs.shape)

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
    model = train_model(model, train_loader, val_loader, epochs, learning_rate, device)

    # Save model weights
    # model_path = os.path.join(output_dir, f"{args.model_name}_mlp_correction.pt")
    # torch.save(model.state_dict(), model_path)
    # print(f"Model weights saved to {model_path}")

    # =========================================================================
    # 3) Load the test data
    # =========================================================================

    # ----- Loading combined training data from monthly files -----
    test_dir = os.path.join(data_dir, f"test_{large_region}")
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
        output_dir=output_dir,
        run_id = run_id,
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
