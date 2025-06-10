#!/usr/bin/env python3
"""
finetuning.py with optional bootstrap sampling of subregions
Author: Ozma Houck 
"""
import argparse
import os
import random
import glob
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
from xarray.coding.times import CFDatetimeCoder
import torch
import torch.nn as nn
import torch.optim as optim
import copy

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
# Argument parsing
# ------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune MLP for regional post-processing')
    parser.add_argument('--data_dir',     type=str, default="~/weatherbench2_data")
    parser.add_argument('--output_dir',   type=str, required=True)
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
        lat0, lat1 = 48.25, 58
        lon0, lon1 = -130 + 360, -120 + 360
    elif args.region == "pakistan":
        lat0, lat1 = 25, 34
        lon0, lon1 = 60, 70
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
    mlp_str = f"mlp{args.mlp_hidden_dim}x{args.mlp_layers}"
    lead_time = f"leadtime_{args.lead_time_hours}"

    output_path = f"{args.output_dir}/{args.model_name}/{region_str}/train_{training_vars_str}_test_{output_vars_str}_dim{subregion_str}_{lead_time}h_{dates_str}_{mlp_str}.zarr"
    return output_path 

def sort_lat_lon(ds):
    # ensure that both lat and lon are sorted ascendingly
    return ds.sortby(['latitude', 'longitude'])
    
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
        preprocess=lambda ds: ds.sortby('latitude'),
        decode_timedelta=True     # if you still want your lead-time axis as timedelta
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

def load_forecasts(data_dir, args, lat_values, lon_values, train=True): 
    """
    loads forecast data, forecast output data and observation data for training or testing.
    """

    if train:
        ver_str = "train"
    else:
        ver_str = "test"

    # set up time range for training or testing
    time_values = np.arange(
        np.datetime64(getattr(args, f"{ver_str}_start")),
        np.datetime64(getattr(args, f"{ver_str}_end")),
        np.timedelta64(24, 'h')
    )

    n_time = len(time_values)
    n_training_vars = len(args.training_vars)
    n_output_vars = len(args.output_vars)

    # ----- Loading combined training data from monthly files -----
    fc_dir = os.path.join(data_dir, f"{ver_str}_{args.region}")
    # For forecast and observation data, we define the patterns for monthly file names.
    fc_pattern = f"{args.model_name}_{ver_str}_forecast_data_*.nc"
    obs_pattern = f"{args.model_name}_{ver_str}_obs_data_*.nc"
    forecast_ds = load_combined_dataset(fc_dir, fc_pattern)

    train_obs_ds = load_combined_dataset(fc_dir, obs_pattern)

    # Now select the desired time, spatial, and (if applicable) prediction_timedelta slices.
    fc_ds = forecast_ds.sel(
        time=time_values,
        latitude=lat_values,
        longitude=lon_values,
        prediction_timedelta=np.timedelta64(args.lead_time_hours, 'h')
    )[args.training_vars].drop_vars('prediction_timedelta').compute()
    
    fc_ds_output = forecast_ds.sel(
        time=time_values,
        latitude=lat_values,
        longitude=lon_values,
        prediction_timedelta=np.timedelta64(args.lead_time_hours, 'h')
    )[args.output_vars].drop_vars('prediction_timedelta').compute()
    
    obs_ds = train_obs_ds.sel(
        time=time_values,
        latitude=lat_values,
        longitude=lon_values,
    )[args.output_vars].compute()

    return fc_ds, fc_ds_output , obs_ds, time_values, n_time, n_training_vars, n_output_vars


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

def run_subregion_experiment(lat_vals, lon_vals, output_path, args, data_dir, device):
    # 1) Load train
    fc_ds, fc_ds_output, obs_ds, train_time_values, n_train_time, n_training_vars, n_output_vars = \
        load_forecasts(data_dir, args, lat_vals, lon_vals, train=True)
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
    model = SimpleMLP(input_dim, args.mlp_hidden_dim, output_dim, args.mlp_layers).to(device)
    model = train_model(model, train_loader, val_loader,
                         epochs=1000, lr=1e-5, device=device,
                         patience=50, min_delta=0.0)

    # load test
    test_fc_ds, test_fc_o_ds, test_obs_ds, test_times, n_test_time, _, _ = \
        load_forecasts(data_dir, args, lat_vals, lon_vals, train=False)
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
        lon_values=lon_vals,
        lat_values=lat_vals,
        time_values=test_times,
        original_fc=tfco,
        corrected_fc=corrected,
        ground_truth_data=tobs
    )


def main():

    # Check for NaNs in 2m_temperature variable across all files, this can happen if download gets interrupted
    file_list = sorted(glob.glob(" /Volumes/wd_external_hd/weatherbench/test_global/**/*.nc", recursive=True))
   
    summary = check_2m_temperature(file_list)
    # Print a quick report
    for path, info in summary.items():
        if "error" in info:
            print(f"[ERROR] {path}: {info['error']}")
        else:
            status = "contains NaNs" if info["has_nans"] else "no NaNs"
            if status == "contains NaNs":
                print(f"[WARNING] {path}: {status} (n_nans={info['n_nans']})")
        args = parse_args()

    exit()

    # prepare output dir and base path
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    base_path = generate_output_path(args)

    # setup device & seeds
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps'    if torch.backends.mps.is_available() else
                          'cpu')
    torch.manual_seed(58); random.seed(58)

    # full region grid
    region_lat, region_lon = get_region_grid(args)
    nlat_patch, nlon_patch = get_patch_shape(args)

    # bootstrap mode
    if args.bootstrap:
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
