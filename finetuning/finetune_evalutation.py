import os
import re
import glob
import socket
import calendar
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature

#######################
# Utility Functions
#######################

def generate_run_id(args):
    region_str = f"{args.region}"
    dates_str = f"train{args.train_start}-{args.train_end}_test{args.test_start}-{args.test_end}"
    training_vars_str = "_".join(args.training_vars)
    output_vars_str = "_".join(args.output_vars)
    mlp_str = f"mlp{args.mlp_hidden_dim}x{args.mlp_layers}"
    lead_time = f"_leadtime_{args.lead_time_hours}"

    run_id = f"{args.model_name}_{region_str}_{dates_str}_{args.lead_time_hours}h_train_{training_vars_str}_output{output_vars_str}{lead_time}{mlp_str}"
    return run_id 

def setup_directories():
    # Determine root directory based on environment.
    nodename = socket.gethostname()
    if nodename == "oMac.local":  # local laptop
        root = os.path.expanduser("~/OneDrive - The University of Chicago/ai_weather_ag/data")
    else:
        raise Exception("Unknown environment, Please specify the root directory")

    dirs = {
        'root': root,
        'raw': os.path.join(root, "raw"),
        'processed': os.path.join(root, "processed"),
        'fig': os.path.join(root, "../figures/finetuning"),
        'input': os.path.join(root, "wb_finetune_test")  # adjusted input directory path
    }

    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs

#######################
# Metrics Function
#######################

def create_metrics(ds_forecasts, var_name):
    """
    Computes various metrics from the forecast dataset.

    Parameters:
      ds_forecasts: xarray dataset containing forecasts and ground truth.
      var_name: variable name (assumes one output variable).

    Returns:
      mse_orig: Monthly MSE for the original forecast.
      mse_corr: Monthly MSE for the corrected forecast.
      raw_spatial_orig: Time-averaged raw values from the original forecast.
      raw_spatial_corr: Time-averaged raw values from the corrected forecast.
      raw_spatial_diff: Difference between corrected and original forecast averages.
      mse_spatial_orig: Spatial MSE map for the original forecast.
      mse_spatial_corr: Spatial MSE map for the corrected forecast.
    """
    # Define variable names.
    var_name_groundtruth = f"{var_name}_ground_truth"
    var_name_original = f"{var_name}_original"
    var_name_corrected = f"{var_name}_corrected"

    # If forecasting wind_speed, compute it from u and v components.
    if var_name == "wind_speed":
        for tag in ["corrected", "original", "groundtruth"]:
            u_component = ds_forecasts[f"10m_u_component_of_wind_{tag}"]
            v_component = ds_forecasts[f"10m_v_component_of_wind_{tag}"]
            wind_speed = np.sqrt(u_component**2 + v_component**2)
            ds_forecasts[f"wind_speed_{tag}"] = wind_speed

    # Extract data arrays.
    ground_truth = ds_forecasts[var_name_groundtruth]
    fc_original = ds_forecasts[var_name_original]
    fc_corrected = ds_forecasts[var_name_corrected]

    # Align forecasts with ground truth along time and spatial dimensions.
    fc_orig_aligned, ground_truth_aligned = xr.align(fc_original, ground_truth, join="inner")
    fc_corr_aligned, _ = xr.align(fc_corrected, ground_truth, join="inner")

    # Compute monthly MSE: average over spatial dimensions and then group by month.
    mse_orig = ((fc_orig_aligned - ground_truth_aligned) ** 2).mean(dim=["longitude", "latitude"]).groupby('time.month').mean(dim='time')
    mse_corr = ((fc_corr_aligned - ground_truth_aligned) ** 2).mean(dim=["longitude", "latitude"]).groupby('time.month').mean(dim='time')

    # Compute spatial raw averages (averaged over time)
    raw_spatial_orig = fc_orig_aligned.mean(dim="time")
    raw_spatial_corr = fc_corr_aligned.mean(dim="time")
    raw_spatial_diff = raw_spatial_corr - raw_spatial_orig

    # Compute spatial MSE maps (averaged over time)
    mse_spatial_orig = ((fc_orig_aligned - ground_truth_aligned) ** 2).mean(dim="time")
    mse_spatial_corr = ((fc_corr_aligned - ground_truth_aligned) ** 2).mean(dim="time")

    return mse_orig, mse_corr, raw_spatial_orig, raw_spatial_corr, raw_spatial_diff, mse_spatial_orig, mse_spatial_corr

#######################
# Plotting Functions (Individual Figures)
#######################

def plot_monthly_mse(mse_orig, mse_corr, model, region, var_name, dirs, training_vars, lead_time):
    """Generates and saves a bar plot of monthly MSE for the original and corrected forecasts."""
    months = [calendar.month_name[i] for i in mse_orig['month'].values]

    plt.figure(figsize=(10, 6))
    plt.bar(months, mse_orig, width=0.4, label='Original MSE', align='center', color='green')
    plt.bar(months, mse_corr, width=0.4, label='Corrected MSE', align='edge', color='lightgreen')
    plt.title(f"Monthly MSE comparison for {model} {var_name}\n(Original vs Corrected)")
    plt.xlabel("Month")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(dirs["fig"], f"mse_time_series_{region}_{var_name}_{model}_trained_with_{'_'.join(training_vars)}_{lead_time}h.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_raw_forecast_original(raw_spatial_orig, model, region, var_name, dirs, training_vars, lead_time):
    """Generates and saves a map for the original forecast values."""
    vmin = float(raw_spatial_orig.min().values)
    vmax = float(raw_spatial_orig.max().values)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    raw_spatial_orig.plot(ax=ax, cmap='viridis', add_colorbar=True, vmin=vmin, vmax=vmax)
    ax.set_title("Original Forecast Values")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    plt.tight_layout()
    save_path = os.path.join(dirs["fig"], f"raw_map_original_{region}_{var_name}_{model}_trained_with_{'_'.join(training_vars)}_{lead_time}h.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_raw_forecast_corrected(raw_spatial_corr, model, region, var_name, dirs, training_vars, lead_time):
    """Generates and saves a map for the corrected forecast values."""
    vmin = float(raw_spatial_corr.min().values)
    vmax = float(raw_spatial_corr.max().values)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    raw_spatial_corr.plot(ax=ax, cmap='viridis', add_colorbar=True, vmin=vmin, vmax=vmax)
    ax.set_title("Corrected Forecast Values")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    plt.tight_layout()
    save_path = os.path.join(dirs["fig"], f"raw_map_corrected_{region}_{var_name}_{model}_trained_with_{'_'.join(training_vars)}_{lead_time}h.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_raw_forecast_diff(raw_spatial_diff, model, region, var_name, dirs, training_vars, lead_time):
    """Generates and saves a map for the difference (corrected - original) of forecast values."""
    vmin = float(raw_spatial_diff.min().values)
    vmax = float(raw_spatial_diff.max().values)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    raw_spatial_diff.plot(ax=ax, cmap='coolwarm', add_colorbar=True, vmin=vmin, vmax=vmax)
    ax.set_title("Difference (Corrected - Original)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    plt.tight_layout()
    save_path = os.path.join(dirs["fig"], f"raw_map_difference_{region}_{var_name}_{model}_trained_with_{'_'.join(training_vars)}_{lead_time}h.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_mse_map_original(mse_spatial_orig, model, region, var_name, dirs, training_vars, lead_time):
    """Generates and saves a spatial map of the original forecast MSE."""
    vmin = float(mse_spatial_orig.min().values)
    vmax = float(mse_spatial_orig.max().values)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    mse_spatial_orig.plot(ax=ax, cmap='viridis', add_colorbar=True, vmin=vmin, vmax=vmax)
    ax.set_title("Original Forecast MSE")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    plt.tight_layout()
    save_path = os.path.join(dirs["fig"], f"mse_map_original_{region}_{var_name}_{model}_trained_with_{'_'.join(training_vars)}_{lead_time}h.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_mse_map_corrected(mse_spatial_corr, model, region, var_name, dirs, training_vars, lead_time):
    """Generates and saves a spatial map of the corrected forecast MSE."""
    vmin = float(mse_spatial_corr.min().values)
    vmax = float(mse_spatial_corr.max().values)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    mse_spatial_corr.plot(ax=ax, cmap='viridis', add_colorbar=True, vmin=vmin, vmax=vmax)
    ax.set_title("Corrected Forecast MSE")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    plt.tight_layout()
    save_path = os.path.join(dirs["fig"], f"mse_map_corrected_{region}_{var_name}_{model}_trained_with_{'_'.join(training_vars)}_{lead_time}h.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_mse_map_diff(mse_spatial_orig, mse_spatial_corr, model, region, var_name, dirs, training_vars, lead_time):
    """Generates and saves a spatial map of the MSE difference (corrected - original)."""
    mse_diff = mse_spatial_corr - mse_spatial_orig
    vmin = float(mse_diff.min().values)
    vmax = float(mse_diff.max().values)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    mse_diff.plot(ax=ax, cmap='coolwarm', add_colorbar=True, vmin=vmin, vmax=vmax)
    ax.set_title("MSE Difference (Corrected - Original)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    plt.tight_layout()
    save_path = os.path.join(dirs["fig"], f"mse_map_difference_{region}_{var_name}_{model}_trained_with_{'_'.join(training_vars)}_{lead_time}h.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

#######################
# Main Generation Function
#######################

def generate_plots(dirs, train_start, train_end, test_start, test_end,
                   model, region, lead_time,
                   training_output_vars, mlp_params):
    """
    Generates individual plots evaluating the performance of corrected weather forecasts.
    
    Parameters:
      train_start, train_end: Strings defining the training period.
      test_start, test_end: Strings defining the test period.
      model: Model name (e.g. "pangu").
      region: Region identifier (e.g. "north_india").
      lead_time: Forecast lead time in hours.
      training_output_vars: A tuple/list of two elements: (training_vars, output_vars).
      mlp_params: A tuple (num_units, num_layers).
    """
    # Unpack training and output variables (ensure they are lists).
    training_vars, output_vars = training_output_vars
    if not isinstance(training_vars, (list, tuple)):
        training_vars = [training_vars]
    if not isinstance(output_vars, (list, tuple)):
        output_vars = [output_vars]
    
    num_units, num_layers = mlp_params

    # Create an arguments object for run_id generation.
    class Args: pass
    args = Args()
    args.region = region
    args.train_start = train_start
    args.train_end = train_end
    args.test_start = test_start
    args.test_end = test_end
    args.training_vars = training_vars
    args.output_vars = output_vars
    args.mlp_hidden_dim = num_units
    args.mlp_layers = num_layers
    args.lead_time_hours = lead_time
    args.model_name = model

    run_id = generate_run_id(args)
    filename = f"{run_id}.zarr"

    # Set up directories and load data.
    forecast_path = os.path.join(dirs['input'], filename)
    print(f"Loading forecast data from {forecast_path}")
    ds_forecasts = xr.open_zarr(forecast_path)

    print(ds_forecasts)


    # OH: have to update this if using more than 1 output variable
    var_name = output_vars[0]

    # Compute metrics.
    mse_orig, mse_corr, raw_spatial_orig, raw_spatial_corr, raw_spatial_diff, mse_spatial_orig, mse_spatial_corr = create_metrics(ds_forecasts, var_name)


    # Print spatial bounds.
    ground_truth = ds_forecasts[f"{var_name}_ground_truth"]
    mse_total_orig = ((ds_forecasts[f"{var_name}_original"] - ground_truth) ** 2).mean()
    mse_total_corr = ((ds_forecasts[f"{var_name}_corrected"] - ground_truth) ** 2).mean()
    print(f"Total MSE for original: {mse_total_orig.values}")
    print(f"Total MSE for corrected: {mse_total_corr.values}")

    # Generate individual plots.
    plot_monthly_mse(mse_orig, mse_corr, model, region, var_name, dirs, training_vars, lead_time)

    # Create maps for all regions besides "pixel".
    if region != "pixel":
        plot_raw_forecast_original(raw_spatial_orig, model, region, var_name, dirs, training_vars, lead_time)
        plot_raw_forecast_corrected(raw_spatial_corr, model, region, var_name, dirs, training_vars, lead_time)
        plot_raw_forecast_diff(raw_spatial_diff, model, region, var_name, dirs, training_vars, lead_time)
        plot_mse_map_original(mse_spatial_orig, model, region, var_name, dirs, training_vars, lead_time)
        plot_mse_map_corrected(mse_spatial_corr, model, region, var_name, dirs, training_vars, lead_time)
        plot_mse_map_diff(mse_spatial_orig, mse_spatial_corr, model, region, var_name, dirs, training_vars, lead_time)

#######################
# New Comparison Function for Multiple Runs
#######################

def compare_runs_mse(dirs, model, training_output_vars, mlp_params):
    """
    Scans the input folder for forecast files matching the given model,
    training/output variables, and MLP parameters, and creates a single bar plot
    that organizes the overall (scalar) MSE by lead time (first level) and region (second level).
    
    For each (lead time, region) combination, the original and corrected MSE are plotted
    as overlapping bars with transparency. The x-axis tick labels are generated dynamically
    (with two lines: lead time on the first line and region on the second) so that the plot
    remains legible regardless of whether there are 3 or 5 regions.
    
    Parameters:
      model: Model name.
      training_output_vars: Tuple/list of (training_vars, output_vars) [each as list or str].
      mlp_params: Tuple (num_units, num_layers).
    """
    dirs = setup_directories()
    input_folder = dirs['input']
    training_vars, output_vars = training_output_vars
    if not isinstance(training_vars, (list, tuple)):
        training_vars = [training_vars]
    if not isinstance(output_vars, (list, tuple)):
        output_vars = [output_vars]
    training_vars_str = "_".join(training_vars)
    output_vars_str = "_".join(output_vars)
    mlp_str = f"mlp{mlp_params[0]}x{mlp_params[1]}"

    # Define the lead times and regions to consider.
    lead_times = [24, 72, 168] # possible lead times
    regions = ["pakistan", "south_pakistan", "north_india", "uttar_pradesh", "pixel"] # possible regions

    # Dictionary to store results keyed by (lead_time, region)
    # Each value is a tuple: (avg_mse_orig, avg_mse_corr)
    results = {}

    # Loop over each combination.
    for lt in lead_times:
        for region in regions:
            pattern = os.path.join(input_folder, f"{model}_{region}_*_{lt}h_train_{training_vars_str}_output{output_vars_str}*{mlp_str}*.zarr")
            files = glob.glob(pattern)
            if not files:
                continue
            mse_orig_list = []
            mse_corr_list = []
            for f in files:
                try:
                    ds = xr.open_zarr(f)
                except Exception as e:
                    print(f"Error opening {f}: {e}")
                    continue
                var_name = output_vars[0]
                ground_truth = ds[f"{var_name}_ground_truth"]
                fc_original = ds[f"{var_name}_original"]
                fc_corrected = ds[f"{var_name}_corrected"]
                mse_total_orig = float(((fc_original - ground_truth) ** 2).mean().values)
                mse_total_corr = float(((fc_corrected - ground_truth) ** 2).mean().values)
                mse_orig_list.append(mse_total_orig)
                mse_corr_list.append(mse_total_corr)
            if mse_orig_list and mse_corr_list:
                avg_mse_orig = np.mean(mse_orig_list)
                avg_mse_corr = np.mean(mse_corr_list)
                results[(lt, region)] = (avg_mse_orig, avg_mse_corr)

    # Prepare data for the single grouped bar plot.
    x_positions = []
    x_labels = []
    mse_orig_vals = []
    mse_corr_vals = []
    pos = 0
    group_gap = 1  # extra gap between different lead time groups
    for region in regions:
        # Collect regions that have results for this lead time.
        for lt in sorted(lead_times):
            regions_with_data = [r for r in regions if (lt, r) in results]
            if not regions_with_data:
                continue
            x_positions.append(pos)
            # Create a two-line label: first line is lead time, second line is region.
            label = f"{lt}h\n{region.replace('_', ' ').title()}"
            x_labels.append(label)
            mse_orig, mse_corr = results[(lt, region)]
            mse_orig_vals.append(mse_orig)
            mse_corr_vals.append(mse_corr)
            pos += 1
        pos += group_gap  # add gap between groups

    # Create the grouped bar plot.
    fig, ax = plt.subplots(figsize=(max(8, len(x_positions)*0.8), 6))
    # Overlap the two bars at the same positions with transparency.
    ax.bar(x_positions, mse_orig_vals, color='blue', width=0.8, alpha=0.5, label='Original MSE')
    ax.bar(x_positions, mse_corr_vals, color='red', width=0.8, alpha=0.5, label='Corrected MSE')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_ylabel("Overall MSE")
    ax.set_title(f"MSE Comparison for {model}\nPredicting {output_vars}")
    ax.legend()
    plt.tight_layout()

    save_path = os.path.join(dirs["fig"], f"mse_comparison_{model}_trained_with_{training_vars_str}_output{output_vars_str}_{mlp_str}.png")
    plt.savefig(save_path, dpi=150)
    print(f"MSE comparison bar chart saved to {save_path}")
    plt.close()

#######################
# Example Usage
#######################

if __name__ == "__main__":
    dirs = setup_directories()

    # Compare multiple runs across lead times and regions in a single plot.
    compare_runs_mse(
        dirs=dirs,
        model="pangu",
        training_output_vars=(["2m_temperature"], ["2m_temperature"]),
        mlp_params=(512, 5)
    )

    regions = ["pakistan", "south_pakistan", "north_india", "uttar_pradesh", "pixel"]
    lead_times = [24, 72, 168]

    for region in regions:
        for lead_time in lead_times:
            print(f"Generating plots for {region} with lead time {lead_time} hours")
            generate_plots(
                dirs=dirs,
                train_start="2018-01-01",
                train_end="2021-12-31",
                test_start="2022-01-01",
                test_end="2022-12-31",
                model="pangu",
                region=region,
                lead_time=lead_time,
                training_output_vars=(["2m_temperature"], ["2m_temperature"]),
                mlp_params=(512, 5)
            )
