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

def generate_output_path(args):
    region_str = f"{args.region}"
    subregion_str = f"{args.subregion}"
    dates_str = f"train{args.train_start}-{args.train_end}_test{args.test_start}-{args.test_end}"
    training_vars_str = "_".join(args.training_vars)
    output_vars_str = "_".join(args.output_vars)
    mlp_str = f"mlp{args.mlp_hidden_dim}x{args.mlp_layers}"
    lead_time = f"leadtime_{args.lead_time_hours}"

    output_path = f"{args.model_name}/{region_str}/train_{training_vars_str}_test_{output_vars_str}_dim{subregion_str}_{lead_time}h_{dates_str}_{mlp_str}.zarr"
    return output_path 

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

def create_metrics(ds_forecasts, prediction_var):
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
    var_name_ground_truth = f"{prediction_var}_ground_truth"
    var_name_original = f"{prediction_var}_original"
    var_name_corrected = f"{prediction_var}_corrected"

    # If forecasting wind_speed, compute it from u and v components.
    if prediction_var== "wind_speed":
        for tag in ["corrected", "original", "ground_truth"]:
            u_component = ds_forecasts[f"10m_u_component_of_wind_{tag}"]
            v_component = ds_forecasts[f"10m_v_component_of_wind_{tag}"]
            wind_speed = np.sqrt(u_component**2 + v_component**2)
            ds_forecasts[f"wind_speed_{tag}"] = wind_speed

    # Extract data arrays.
    ground_truth = ds_forecasts[var_name_ground_truth]
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

def plot_monthly_mse(mse_orig, mse_corr, model, region, subregion, var_name, dirs, training_vars, lead_time):
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

    out_folder = os.path.join(dirs["fig"], model, "time_series", region)
    os.makedirs(out_folder, exist_ok=True)
    save_path = os.path.join(out_folder, f"mse_time_series_{subregion}_{var_name}_trained_with_{'_'.join(training_vars)}_{lead_time}h.png")

    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_raw_forecast_original(raw_spatial_orig, model, region, subregion, var_name, dirs, training_vars, lead_time):
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

    out_folder = os.path.join(dirs["fig"], model, "maps", region)
    os.makedirs(out_folder, exist_ok=True)
    save_path = os.path.join(out_folder, f"{model}/maps/{region}/raw_map_original_{subregion}_{var_name}_trained_with_{'_'.join(training_vars)}_{lead_time}h.png")

    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_raw_forecast_corrected(raw_spatial_corr, model, region, subregion, var_name, dirs, training_vars, lead_time):
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
    out_folder = os.path.join(dirs["fig"], model, "maps", region)
    os.makedirs(out_folder, exist_ok=True)
    save_path = os.path.join(out_folder, f"{model}/maps/{region}/raw_map_corrected_{subregion}_{var_name}_trained_with_{'_'.join(training_vars)}_{lead_time}h.png")

    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_raw_forecast_diff(raw_spatial_diff, model, region, subregion, var_name, dirs, training_vars, lead_time):
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

    out_folder = os.path.join(dirs["fig"], model, "maps", region)
    os.makedirs(out_folder, exist_ok=True)
    save_path = os.path.join(out_folder, f"{model}/maps/{region}/raw_map_difference_{subregion}_{var_name}_trained_with_{'_'.join(training_vars)}_{lead_time}h.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_mse_map_original(mse_spatial_orig, model, region, subregion, var_name, dirs, training_vars, lead_time):
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
    out_folder = os.path.join(dirs["fig"], model, "maps", region)
    os.makedirs(out_folder, exist_ok=True)
    save_path = os.path.join(out_folder, f"{model}/maps/{region}/mse_map_original_{subregion}_{var_name}_trained_with_{'_'.join(training_vars)}_{lead_time}h.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_mse_map_corrected(mse_spatial_corr, model, region, subregion, var_name, dirs, training_vars, lead_time):
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
    out_folder = os.path.join(dirs["fig"], model, "maps", region)
    os.makedirs(out_folder, exist_ok=True)
    save_path = os.path.join(out_folder, f"{model}/maps/{region}/mse_map_corrected_{subregion}_{var_name}_trained_with_{'_'.join(training_vars)}_{lead_time}h.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_mse_map_diff(mse_spatial_orig, mse_spatial_corr, model, region, subregion, var_name, dirs, training_vars, lead_time):
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
    out_folder = os.path.join(dirs["fig"], model, "maps", region)
    os.makedirs(out_folder, exist_ok=True)
    save_path = os.path.join(out_folder, f"{model}/maps/{region}/mse_map_difference_{subregion}_{var_name}_trained_with_{'_'.join(training_vars)}_{lead_time}h.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

#######################
# Main Generation Function
#######################

def generate_plots(dirs, train_start, train_end, test_start, test_end,
                   model, region, subregion, lead_time, 
                   training_output_vars, prediction_var, mlp_params):
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
    args.subregion = subregion
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

    output_path = generate_output_path(args)

    # Set up directories and load data.
    forecast_path = os.path.join(dirs['input'], output_path)
    print(f"Loading forecast data from {forecast_path}")
    ds_forecasts = xr.open_zarr(forecast_path)

    print(ds_forecasts)

    # Compute metrics.
    mse_orig, mse_corr, raw_spatial_orig, raw_spatial_corr, raw_spatial_diff, mse_spatial_orig, mse_spatial_corr = create_metrics(ds_forecasts, prediction_var)


    # Print spatial bounds.
    ground_truth = ds_forecasts[f"{prediction_var}_ground_truth"]
    mse_total_orig = ((ds_forecasts[f"{prediction_var}_original"] - ground_truth) ** 2).mean()
    mse_total_corr = ((ds_forecasts[f"{prediction_var}_corrected"] - ground_truth) ** 2).mean()
    print(f"Total MSE for original: {mse_total_orig.values}")
    print(f"Total MSE for corrected: {mse_total_corr.values}")

    # Generate individual plots.
    plot_monthly_mse(mse_orig, mse_corr, model, region, subregion, prediction_var, dirs, training_vars, lead_time)

    # Create maps for all regions besides "pixel".
    # if region != "pixel":
    #     plot_raw_forecast_original(raw_spatial_orig, model, region,subregion, prediction_var, dirs, training_vars, lead_time)
    #     plot_raw_forecast_corrected(raw_spatial_corr, model, region, subregion, prediction_var, dirs, training_vars, lead_time)
    #     plot_raw_forecast_diff(raw_spatial_diff, model, region, subregion, prediction_var, dirs, training_vars, lead_time)
    #     plot_mse_map_original(mse_spatial_orig, model, region, subregion, prediction_var, dirs, training_vars, lead_time)
    #     plot_mse_map_corrected(mse_spatial_corr, model, region, subregion, prediction_var, dirs, training_vars, lead_time)
    #     plot_mse_map_diff(mse_spatial_orig, mse_spatial_corr, model, region, subregion, prediction_var, dirs, training_vars, lead_time)
def generate_subregion_comparison_plots(dirs, train_start, train_end, test_start, 
    test_end, model, training_output_vars, prediction_var, mlp_params):
    """
    Generates a line plot of MSE improvement vs subregion size
    for different lead times and multiple regions.
    """
    input_folder = dirs['input']
    training_vars, output_vars = training_output_vars
    if not isinstance(training_vars, (list, tuple)):
        training_vars = [training_vars]
    if not isinstance(output_vars, (list, tuple)):
        output_vars = [output_vars]
    training_vars_str = "_".join(training_vars)
    mlp_str = f"mlp{mlp_params[0]}x{mlp_params[1]}"

    regions = ["amazon", "india", "pakistan"]
    subregions = ["2x2", "4x4", "6x6", "8x8", "10x10"]
    lead_times = [24, 72, 168]

    # Nested dict: improvement[region][lead_time] = list of (size, improvement)
    improvement = {region: {lt: [] for lt in lead_times} for region in regions}

    for region in regions:
        for subregion in subregions:
            size = int(subregion.split('x')[0])
            for lt in lead_times:
                # Build args for path generation
                class Args: pass
                args = Args()
                args.model_name = model
                args.region = region
                args.subregion = subregion
                args.train_start = train_start
                args.train_end = train_end
                args.test_start = test_start
                args.test_end = test_end
                args.training_vars = training_vars
                args.output_vars = output_vars
                args.mlp_hidden_dim = mlp_params[0]
                args.mlp_layers = mlp_params[1]
                args.lead_time_hours = lt

                path = generate_output_path(args)
                forecast_path = os.path.join(input_folder, path)
                ds = xr.open_zarr(forecast_path)

                if prediction_var == "wind_speed":
                    ds["wind_speed_ground_truth"] = np.sqrt(
                        ds["10m_u_component_of_wind_ground_truth"]**2 + 
                        ds["10m_v_component_of_wind_ground_truth"]**2
                    )
                    ds["wind_speed_original"] = np.sqrt(
                        ds["10m_u_component_of_wind_original"]**2 + 
                        ds["10m_v_component_of_wind_original"]**2
                    )
                    ds["wind_speed_corrected"] = np.sqrt(
                        ds["10m_u_component_of_wind_corrected"]**2 + 
                        ds["10m_v_component_of_wind_corrected"]**2
                    )

                gt = ds[f"{prediction_var}_ground_truth"]
                orig = ds[f"{prediction_var}_original"]
                corr = ds[f"{prediction_var}_corrected"]
                mse_orig = float(((orig - gt) ** 2).mean().values)
                mse_corr = float(((corr - gt) ** 2).mean().values)

                improvement[region][lt].append((size, mse_orig - mse_corr))

    # Plotting
    plt.figure(figsize=(10, 6))

    # Assign distinct colors and line styles
    cmap = plt.get_cmap('tab10')
    color_map = {region: cmap(i) for i, region in enumerate(regions)}
    linestyle_map = {24: 'solid', 72: '--', 168: ':'}

    for region in regions:
        for lt in lead_times:
            data = improvement[region][lt]
            if not data:
                continue
            data_sorted = sorted(data, key=lambda x: x[0])
            sizes, imps = zip(*data_sorted)
            plt.plot(
                sizes, imps, marker='o',
                color=color_map[region],
                linestyle=linestyle_map[lt],
                label=f"{region.replace('_',' ').title()} {lt}h"
            )

    plt.xlabel("Subregion size (degrees)")
    plt.ylabel("MSE improvement (Original - Corrected)")
    plt.title(f"MSE Improvement vs Subregion Size for {model}")
    sizes_all = [int(s.split('x')[0]) for s in subregions]
    plt.xticks(sizes_all, subregions)
    plt.grid(True)
    plt.legend(title="Region & Lead Time", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Save figure
    out_folder = os.path.join(dirs["fig"], model, "comparison")
    os.makedirs(out_folder, exist_ok=True)
    save_path = os.path.join(
        out_folder,
        f"subregion_mse_improvement_{training_vars_str}_{prediction_var}_{mlp_str}.png"
    )
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved subregion comparison plot to {save_path}")


#######################
# Comparison Function for Multiple Runs
#######################

def compare_runs_mse(dirs, model, training_output_vars, prediction_var, mlp_params):
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
    regions = ["amazon", "usa_south", "india", "pakistan"]  # adjust or extend as needed
    subregion ="10x10"

    # Dictionary to store results keyed by (lead_time, region)
    # Each value is a tuple: (avg_mse_orig, avg_mse_corr)
    results = {}
    ifs_results = {}

    # Loop over each combination to get original and forecast
    for lt in lead_times:
        for region in regions:
            pattern = os.path.join(input_folder, f"{model}/{region}/train_{training_vars_str}_test_{output_vars_str}_dim{subregion}_leadtime_{lt}h*{mlp_str}*.zarr")
            files = glob.glob(pattern)
            ifs_pattern = os.path.join(input_folder, f"ifs/{region}/train_{training_vars_str}_test_{output_vars_str}_dim{subregion}_leadtime_{lt}h*{mlp_str}*.zarr")
            ifs_files = glob.glob(ifs_pattern)
            if not files:
                print(f"No files found for {model} in {region} with lead time {lt}h")
                continue
            mse_orig_list = []
            mse_corr_list = []
            for f in files:
                try:
                    ds = xr.open_zarr(f)
                except Exception as e:
                    print(f"Error opening {f}: {e}")
                    continue
                if prediction_var == "wind_speed":
                    ds["wind_speed_ground_truth"] = np.sqrt(ds["10m_u_component_of_wind_ground_truth"]**2 + ds["10m_v_component_of_wind_ground_truth"]**2)
                    ds["wind_speed_original"] = np.sqrt(ds["10m_u_component_of_wind_original"]**2 + ds["10m_v_component_of_wind_original"]**2)
                    ds["wind_speed_corrected"] = np.sqrt(ds["10m_u_component_of_wind_corrected"]**2 + ds["10m_v_component_of_wind_corrected"]**2)
                    print(f"Mean wind speed original: {ds['wind_speed_original'].mean().values}")
                    print(f"Mean wind speed corrected: {ds['wind_speed_corrected'].mean().values}")
                ground_truth = ds[f"{prediction_var}_ground_truth"]
                fc_original = ds[f"{prediction_var}_original"]
                fc_corrected = ds[f"{prediction_var}_corrected"]
                mse_total_orig = float(((fc_original - ground_truth) ** 2).mean().values)
                mse_total_corr = float(((fc_corrected - ground_truth) ** 2).mean().values)
                mse_orig_list.append(mse_total_orig)
                mse_corr_list.append(mse_total_corr)

            ifs_mse_orig_list = []
            ifs_mse_corr_list = []
            for f in ifs_files:
                try:
                    ds = xr.open_zarr(f)
                except Exception as e:
                    print(f"Error opening {f}: {e}")
                    continue
                if prediction_var == "wind_speed":
                    ds["wind_speed_ground_truth"] = np.sqrt(ds["10m_u_component_of_wind_ground_truth"]**2 + ds["10m_v_component_of_wind_ground_truth"]**2)
                    ds["wind_speed_original"] = np.sqrt(ds["10m_u_component_of_wind_original"]**2 + ds["10m_v_component_of_wind_original"]**2)
                    ds["wind_speed_corrected"] = np.sqrt(ds["10m_u_component_of_wind_corrected"]**2 + ds["10m_v_component_of_wind_corrected"]**2)

                    # print mean wind speed for original and corrected
                    print(f"Mean wind speed original: {ds['wind_speed_original'].mean().values}")
                    print(f"Mean wind speed corrected: {ds['wind_speed_corrected'].mean().values}")
                    print(f"Mean wind speed ground truth: {ds['wind_speed_ground_truth'].mean().values}")
                ifs_ground_truth = ds[f"{prediction_var}_ground_truth"]
                ifs_fc_original = ds[f"{prediction_var}_original"]
                ifs_fc_corrected = ds[f"{prediction_var}_corrected"]
                ifs_mse_total_orig = float(((ifs_fc_original - ifs_ground_truth) ** 2).mean().values)
                ifs_mse_total_corr = float(((ifs_fc_corrected - ifs_ground_truth) ** 2).mean().values)
                ifs_mse_orig_list.append(ifs_mse_total_orig)
                ifs_mse_corr_list.append(ifs_mse_total_corr)

            if mse_orig_list and mse_corr_list:
                avg_mse_orig = np.mean(mse_orig_list)
                avg_mse_corr = np.mean(mse_corr_list)
                results[(lt, region)] = (avg_mse_orig, avg_mse_corr)
            if ifs_mse_orig_list and ifs_mse_corr_list:
                avg_mse_orig_ifs = np.mean(ifs_mse_orig_list)
                avg_mse_corr_ifs = np.mean(ifs_mse_corr_list)
                ifs_results[(lt, region)] = (avg_mse_orig_ifs, avg_mse_corr_ifs)

    # Prepare data for the single grouped bar plot.
    x_positions = []
    x_labels = []
    mse_orig_vals = []
    mse_corr_vals = []
    ifs_mse_orig_vals = []
    ifs_mse_corr_vals = []
    pos = 0
    group_gap = 1  # extra gap between different lead time groups
    for region in regions:
        # Collect regions that have results for this lead time.
        for lt in sorted(lead_times):
            regions_with_data = [r for r in regions if (lt, r) in results]
            regions_with_ifs_data = [r for r in regions if (lt, r) in ifs_results]

            # check if region is available in both results and ifs_results
            if region not in regions_with_data or region not in regions_with_ifs_data:
                print(f"Skipping region {region} for lead time {lt} as no data is available")
                continue

            x_positions.append(pos)
            # Create a two-line label: first line is lead time, second line is region.
            label = f"{lt}h\n{region.replace('_', ' ').title()}"
            x_labels.append(label)
            mse_orig, mse_corr = results[(lt, region)]
            ifs_mse_orig, ifs_mse_corr = ifs_results[(lt, region)]

            mse_orig_vals.append(mse_orig)
            mse_corr_vals.append(mse_corr)
            ifs_mse_orig_vals.append(ifs_mse_orig)
            ifs_mse_corr_vals.append(ifs_mse_corr)
            pos += 1
        pos += group_gap  # add gap between groups

    x_positions_offset = np.array(x_positions) + 0.3  # Offset for IFS bars

    # Create the grouped bar plot.
    fig, ax = plt.subplots(figsize=(max(8, len(x_positions)*0.8), 6))
    # Overlap the two bars at the same positions with transparency.
    ax.bar(x_positions, mse_orig_vals, color='blue', width=0.8, alpha=0.5, label='Original MSE')
    ax.bar(x_positions, mse_corr_vals, color='red', width=0.8, alpha=0.5, label='Corrected MSE')
    ax.bar(x_positions_offset , ifs_mse_orig_vals, color='blue', width=0.1, alpha=.75, label='IFS Baseline MSE')
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_ylabel("Overall MSE")
    ax.set_title(f"MSE Comparison for {model}\nPredicting {prediction_var}")
    ax.legend()
    plt.tight_layout()

    save_path = os.path.join(dirs["fig"], f"mse_comparison_{model}_trained_with_{training_vars_str}_output{prediction_var}_{mlp_str}.png")
    plt.savefig(save_path, dpi=150)
    print(f"MSE comparison bar chart saved to {save_path}")
    plt.close()

def main():

    dirs = setup_directories()
    # three options for training and output variable combinations, uncomment the one you want to use
    training_vars = ["2m_temperature"]
    output_vars = ["2m_temperature"]
    prediction_var = "2m_temperature"

    # training_vars = ["10m_v_component_of_wind", "10m_u_component_of_wind"]
    # output_vars = ["10m_v_component_of_wind", "10m_u_component_of_wind"]
    # prediction_var = "wind_speed"

    # training_vars = ["2m_temperature", "geopotential_1000hPa", "specific_humidity_1000hPa"]
    # output_vars = ["2m_temperature"]
    # prediction_var = "2m_temperature"

    # Compare multiple runs across lead times and regions in a single plot.
    # compare_runs_mse(
    #     dirs=dirs,
    #     model="pangu",
    #     training_output_vars=(training_vars, output_vars),
    #     prediction_var=prediction_var,
    #     mlp_params=(512, 5)
    # )

    generate_subregion_comparison_plots(
        dirs = dirs,
        train_start="2018-01-01",
        train_end="2021-12-31",
        test_start="2022-01-01",
        test_end="2022-12-31",
        model="pangu",
        training_output_vars=(training_vars, output_vars),
        prediction_var=prediction_var,
        mlp_params=(512, 5)
    )

    exit()

    # regions = ["pakistan", "south_pakistan", "full_india", "north_india", "uttar_pradesh", "pixel"]
    # regions = ["usa_south", "amazon", "british_columbia", "india", "pakistan"]
    regions = ["pakistan"]
    subregions = ["2x2", "4x4", "6x6", "8x8", "10x10"]
    lead_times = [24, 72, 168]

    for region in regions:
        for lead_time in lead_times:
            for subregion in subregions:
                print(f"Generating plots for {region} with lead time {lead_time} hours")
                generate_plots(
                    dirs=dirs,
                    train_start="2018-01-01",
                    train_end="2021-12-31",
                    test_start="2022-01-01",
                    test_end="2022-12-31",
                    model="pangu",
                    region=region,
                    subregion=subregion,
                    lead_time=lead_time,
                    training_output_vars=(training_vars, output_vars),
                    prediction_var=prediction_var,
                    mlp_params=(512, 5)
                )

if __name__ == "__main__":
    main()