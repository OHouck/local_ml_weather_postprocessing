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
from types import SimpleNamespace

from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats


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
    if nodename == "oMac.local":
        root = os.path.expanduser("~/OneDrive - The University of Chicago/ai_weather_ag/data")
    else:
        raise Exception(f"Unknown environment, Please specify the root directory. \
                        Nodename found: {nodename}")

    dirs = {
        'root': root,
        'raw': os.path.join(root, "raw"),
        'processed': os.path.join(root, "processed"),
        'fig': os.path.join(root, "../figures/finetuning"),
        'input': os.path.join(root, "fine_tuning_output")  # adjusted input directory path
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

    # Extract data arrays.
    ground_truth = ds_forecasts[f"{prediction_var}_ground_truth"]
    fc_original  = ds_forecasts[f"{prediction_var}_original"]
    fc_corrected = ds_forecasts[f"{prediction_var}_corrected"]

    # compute normalization factors on the ground truth ──
    norm_mean = ground_truth.mean(dim=["time","latitude","longitude"])
    norm_std  = ground_truth.std(dim=["time","latitude","longitude"])
    gt_norm   = (ground_truth - norm_mean) / norm_std
    orig_norm = (fc_original  - norm_mean) / norm_std
    corr_norm = (fc_corrected - norm_mean) / norm_std


    # align the *normalized* arrays
    orig_norm_aligned, gt_norm_aligned = xr.align(orig_norm, gt_norm, join="inner")
    corr_norm_aligned, _            = xr.align(corr_norm, gt_norm, join="inner")

    # ── now compute MSE on the normalized fields ──
    mse_orig = (
        (orig_norm_aligned - gt_norm_aligned) ** 2
    ).mean(dim=["longitude","latitude"]) \
     .groupby("time.month") \
     .mean(dim="time")

    mse_corr = (
        (corr_norm_aligned - gt_norm_aligned) ** 2
    ).mean(dim=["longitude","latitude"]) \
     .groupby("time.month") \
     .mean(dim="time")

    # raw_spatial_* can stay as before (these are *not* MSE)
    raw_spatial_orig = fc_original.mean(dim="time")
    raw_spatial_corr = fc_corrected.mean(dim="time")
    raw_spatial_diff = raw_spatial_corr - raw_spatial_orig

    # spatial MSE maps on normalized data
    mse_spatial_orig = (
        (orig_norm_aligned - gt_norm_aligned) ** 2
    ).mean(dim="time")
    mse_spatial_corr = (
        (corr_norm_aligned - gt_norm_aligned) ** 2
    ).mean(dim="time")

    return (
        mse_orig, mse_corr,
        raw_spatial_orig, raw_spatial_corr, raw_spatial_diff,
        mse_spatial_orig, mse_spatial_corr
    )

def generate_lead_time_plots(
        dirs,
        train_start, train_end,
        test_start, test_end,
        model,
        training_output_vars,
        prediction_var,
        mlp_params,
        regions,
        subregion,
        bootstrap=None,
):
    """
    Generates a single plot showing percent improvement in RMSE by lead time for neural network
    and anomaly correction methods across all specified regions.
    
    Parameters
    ----------
    dirs : dict
        Dictionary containing paths for input and figure directories
    train_start, train_end : str
        Training period boundaries
    test_start, test_end : str
        Testing period boundaries
    model : str
        Model name (e.g., 'pangu', 'ifs')
    training_output_vars : tuple
        Tuple of (training_vars, output_vars)
    prediction_var : str
        Variable to predict
    mlp_params : tuple
        MLP architecture parameters
    regions : list
        List of regions to process
    subregion : str
        Subregion identifier
    bootstrap : bool, optional
        Whether to use bootstrap samples for confidence intervals
    """
    
    # Parse training and output variables
    training_vars, output_vars = training_output_vars
    if not isinstance(training_vars, (list, tuple)):
        training_vars = [training_vars]
    if not isinstance(output_vars, (list, tuple)):
        output_vars = [output_vars]

    # Create string representations for file naming
    training_vars_str = "_".join(training_vars)
    output_vars_str = "_".join(output_vars)
    mlp_str = f"mlp{mlp_params[0]}x{mlp_params[1]}"
    time_str = f"train{train_start}-{train_end}_test{test_start}-{test_end}"

    lead_times = [24, 48, 72, 96, 120, 144, 168]  # Possible lead times in hours

    # Define colors for different regions (can handle up to 10 regions)
    region_colors = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
        '#bcbd22',  # Olive
        '#17becf'   # Cyan
    ]
    
    # Define line styles and markers for different methods
    method_styles = {
        'nn': {'linestyle': '-', 'marker': 'o', 'markersize': 8},
        'ano': {'linestyle': '--', 'marker': 's', 'markersize': 7}
    }
    
    # Store all improvements for all regions
    all_improvements = {}
    
    # Process each region
    for region_idx, region in enumerate(regions):
        # Initialize storage for improvement percentages for this region
        improvements = {
            'pangu_nn': {},
            'pangu_ano': {},
            'ifs_nn': {},
            'ifs_ano': {}
        }
        
        # Process each lead time
        for lt in lead_times:
            # Construct file paths
            if bootstrap:
                pangu_pattern = os.path.join(
                    dirs['input'], 
                    f"{model}/{region}/train_{training_vars_str}_test_{output_vars_str}_"
                    f"dim{subregion}_leadtime_{lt}h_{time_str}_{mlp_str}*bs*.zarr"
                )
                ifs_pattern = os.path.join(
                    dirs['input'],
                    f"ifs/{region}/train_{training_vars_str}_test_{output_vars_str}_"
                    f"dim{subregion}_leadtime_{lt}h_{time_str}_{mlp_str}*bs*.zarr"
                )
            else:
                pangu_pattern = os.path.join(
                    dirs['input'],
                    f"{model}/{region}/train_{training_vars_str}_test_{output_vars_str}_"
                    f"dim{subregion}_leadtime_{lt}h_{time_str}_{mlp_str}.zarr"
                )
                ifs_pattern = os.path.join(
                    dirs['input'],
                    f"ifs/{region}/train_{training_vars_str}_test_{output_vars_str}_"
                    f"dim{subregion}_leadtime_{lt}h_{time_str}_{mlp_str}.zarr"
                )

            # Process Pangu model files
            pangu_files = glob.glob(pangu_pattern)
            if not bootstrap and len(pangu_files) > 1:
                raise ValueError(f"Multiple files found for lead time {lt}h: {pangu_files}")

            for idx, file_path in enumerate(pangu_files):
                try:
                    ds = xr.open_zarr(file_path)
                    
                    # Extract data
                    ground_truth = ds[f"{prediction_var}_ground_truth"]
                    original = ds[f"{prediction_var}_original"]
                    nn_corrected = ds[f"{prediction_var}_corrected"]
                    ano_corrected = ds.get(f"{prediction_var}_mean_corrected", None)

                    # Calculate RMSE
                    rmse_original = float(np.sqrt(((original - ground_truth) ** 2).mean().values))
                    rmse_nn = float(np.sqrt(((nn_corrected - ground_truth) ** 2).mean().values))
                    
                    # Calculate percent improvements
                    pct_improvement_nn = ((rmse_original - rmse_nn) / rmse_original * 100 
                                          if rmse_original != 0 else 0)
                    improvements['pangu_nn'][(lt, idx)] = pct_improvement_nn
                    
                    if ano_corrected is not None:
                        rmse_ano = float(np.sqrt(((ano_corrected - ground_truth) ** 2).mean().values))
                        pct_improvement_ano = ((rmse_original - rmse_ano) / rmse_original * 100 
                                               if rmse_original != 0 else 0)
                        improvements['pangu_ano'][(lt, idx)] = pct_improvement_ano

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    continue

            # Process IFS model files
            ifs_files = glob.glob(ifs_pattern)
            if ifs_files:  # Evaluates to True if the list is empty
                assert len(ifs_files) == len(pangu_files)
                
            for idx, file_path in enumerate(ifs_files):  
                try:
                    ds = xr.open_zarr(file_path)
                    
                    # Extract data
                    ground_truth = ds[f"{prediction_var}_ground_truth"]
                    original = ds[f"{prediction_var}_original"]
                    nn_corrected = ds[f"{prediction_var}_corrected"]
                    ano_corrected = ds.get(f"{prediction_var}_mean_corrected", None)

                    # Calculate RMSE
                    rmse_original = float(np.sqrt(((original - ground_truth) ** 2).mean().values))
                    rmse_nn = float(np.sqrt(((nn_corrected - ground_truth) ** 2).mean().values))
                    
                    # Calculate percent improvements
                    pct_improvement_nn = ((rmse_original - rmse_nn) / rmse_original * 100 
                                          if rmse_original != 0 else 0)
                    improvements['ifs_nn'][(lt, idx)] = pct_improvement_nn
                    
                    if ano_corrected is not None:
                        rmse_ano = float(np.sqrt(((ano_corrected - ground_truth) ** 2).mean().values))
                        pct_improvement_ano = ((rmse_original - rmse_ano) / rmse_original * 100 
                                               if rmse_original != 0 else 0)
                        improvements['ifs_ano'][(lt, idx)] = pct_improvement_ano

                except Exception as e:
                    print(f"Error processing IFS file {file_path}: {e}")
                    continue
        
        # Store improvements for this region
        all_improvements[region] = improvements

    # Create a single plot for all regions
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Track which lead times actually have data across all regions
    all_valid_lead_times = set()
    
    # Plot each region's data
    for region_idx, (region, improvements) in enumerate(all_improvements.items()):
        # Get color for this region
        color = region_colors[region_idx % len(region_colors)]

        # Process each improvement type for this region
        for imp_type, imp_data in improvements.items():
            if not imp_data:
                continue
            
            # Determine if this is NN or anomaly correction
            method_type = 'nn' if 'nn' in imp_type else 'ano'
            style = method_styles[method_type]
            
            # Determine if this is Pangu or IFS (for alpha transparency)
            is_ifs = 'ifs' in imp_type
            line_alpha = 0.6 if is_ifs else 1.0
            
            if bootstrap:
                # Aggregate bootstrap results
                aggregated = {}
                for lt in lead_times:
                    # Extract all values for this lead time
                    lt_values = [imp_data[(lead_time, idx)] 
                                 for (lead_time, idx) in imp_data.keys() 
                                 if lead_time == lt]
                    
                    if lt_values:
                        n = len(lt_values)
                        mean = np.mean(lt_values)
                        std = np.std(lt_values, ddof=1)
                        se = std / np.sqrt(n)
                        
                        # Calculate 95% confidence interval using t-distribution
                        alpha_ci = 0.05
                        t_crit = stats.t.ppf(1 - alpha_ci/2, df=n-1)
                        
                        aggregated[lt] = {
                            'mean': mean,
                            'ci_lower': mean - (t_crit * se),
                            'ci_upper': mean + (t_crit * se),
                            'count': n
                        }
                
                # Extract data for plotting
                valid_lead_times = sorted(aggregated.keys())
                if not valid_lead_times:
                    continue
                
                all_valid_lead_times.update(valid_lead_times)
                means = [aggregated[lt]['mean'] for lt in valid_lead_times]
                ci_lower = [aggregated[lt]['ci_lower'] for lt in valid_lead_times]
                ci_upper = [aggregated[lt]['ci_upper'] for lt in valid_lead_times]
                
                # Create label
                label = f"{region} - {imp_type.replace('_', ' ').title()}"
                
                # Plot line with markers
                x_pos = [lead_times.index(lt) for lt in valid_lead_times]
                ax.plot(x_pos, means, 
                       marker=style['marker'], 
                       linestyle=style['linestyle'], 
                       color=color, 
                       linewidth=2.5, 
                       markersize=style['markersize'],
                       label=label,
                       alpha=line_alpha,
                       zorder=3)
                
                # Add confidence interval bands
                ax.fill_between(x_pos, ci_lower, ci_upper,
                               color=color, 
                               alpha=0.1 * line_alpha, 
                               zorder=1)
                
            else:
                # Non-bootstrap case
                valid_data = [(lt, imp_data[(lt, 0)]) for lt in lead_times 
                             if (lt, 0) in imp_data]
                if not valid_data:
                    continue
                    
                valid_lead_times, values = zip(*valid_data)
                all_valid_lead_times.update(valid_lead_times)
                
                # Create label
                label = f"{region} - {imp_type.replace('_', ' ').title()}"
                
                # Plot line
                x_pos = [lead_times.index(lt) for lt in valid_lead_times]
                ax.plot(x_pos, values, 
                       marker=style['marker'], 
                       linestyle=style['linestyle'],
                       color=color, 
                       linewidth=2.5, 
                       markersize=style['markersize'],
                       label=label,
                       alpha=line_alpha,
                       zorder=3)
    
    # Set x-axis to show all lead times but only label those with data
    ax.set_xticks(range(len(lead_times)))
    ax.set_xticklabels([f"{lt}h" if lt in all_valid_lead_times else "" 
                       for lt in lead_times])
    
    # Customize plot appearance
    ax.set_xlabel("Forecast Lead Time", fontsize=13)
    ax.set_ylabel("RMSE Improvement (%)", fontsize=13)
    
    # Add horizontal line at y=0 for reference
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Title
    regions_str = ", ".join(regions)
    title_parts = [f"RMSE Improvement for {prediction_var.replace('_', ' ').title()}"]
    title_parts.append(f"Regions: {regions_str}, Patch Size: {subregion}")
    if bootstrap:
        title_parts[0] += " (with 95% CI)"
    ax.set_title('\n'.join(title_parts), fontsize=14, pad=15)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Legend - position it outside the plot area if many lines
    n_lines = len([h for h in ax.get_legend_handles_labels()[0]])
    if n_lines > 6:
        legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                          frameon=True, fancybox=True,
                          shadow=True, fontsize=10, ncol=1)
    else:
        legend = ax.legend(loc='best', frameon=True, fancybox=True,
                          shadow=True, fontsize=10, ncol=1)
    
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.95)
    legend.get_frame().set_edgecolor('gray')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    out_folder = os.path.join(dirs["fig"], model, "lead_time", "multi_region", subregion)
    os.makedirs(out_folder, exist_ok=True)
    
    bootstrap_suffix = "_bootstrap" if bootstrap else ""
    regions_file_str = "_".join(regions)
    if "arid" in regions or "tropical" in regions or "temperate" in regions:
        region_type = "climate_zones"
    else:
        region_type = ""
    fname = (f"leadtime_improvement_{prediction_var}_trainedwith_{training_vars_str}_"
            f"{region_type}{bootstrap_suffix}.png")
    save_path = os.path.join(out_folder, fname)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Multi-region lead-time improvement plot saved to: {save_path}")
    
    if bootstrap:
        # Print summary statistics for each region
        for region in regions:
            print(f"\nRegion {region}:")
            for imp_type in ['pangu_nn', 'pangu_ano', 'ifs_nn', 'ifs_ano']:
                if all_improvements[region][imp_type]:
                    # Get sample count from first lead time
                    first_lt = next(iter([lt for (lt, idx) in all_improvements[region][imp_type].keys()]))
                    n_samples = len([v for (lt, idx), v in all_improvements[region][imp_type].items() 
                                   if lt == first_lt])
                    if n_samples > 0:
                        print(f"  {imp_type}: {n_samples} bootstrap samples")

#######################
# plotting functions (individual figures)
#######################

def plot_monthly_mse(mse_orig, mse_corr, model, region, subregion, var_name, dirs, training_vars, lead_time):
    """generates and saves a bar plot of monthly mse for the original and corrected forecasts."""
    months = [calendar.month_name[i] for i in mse_orig['month'].values]

    plt.figure(figsize=(10, 6))
    plt.bar(months, mse_orig, width=0.4, label='original mse', align='center', color='green')
    plt.bar(months, mse_corr, width=0.4, label='corrected mse', align='edge', color='lightgreen')
    plt.title(f"monthly mse comparison for {model} {var_name}\n(original vs corrected)")
    plt.xlabel("month")
    plt.ylabel("mse")
    plt.legend()
    plt.grid(true)
    plt.tight_layout()

    out_folder = os.path.join(dirs["fig"], model, "time_series", region, subregion)
    os.makedirs(out_folder, exist_ok=true)
    save_path = os.path.join(out_folder, f"mse_time_series_{var_name}_trained_with_{'_'.join(training_vars)}_{lead_time}h.png")

    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_raw_forecast_original(raw_spatial_orig, model, region, subregion, var_name, dirs, training_vars, lead_time):
    """generates and saves a map for the original forecast values."""
    vmin = float(raw_spatial_orig.min().values)
    vmax = float(raw_spatial_orig.max().values)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.platecarree())
    raw_spatial_orig.plot(ax=ax, cmap='viridis', add_colorbar=true, vmin=vmin, vmax=vmax)
    ax.set_title("original forecast values")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.coastlines()
    ax.add_feature(cfeature.borders, linestyle=':')
    ax.add_feature(cfeature.land, facecolor='lightgray')
    ax.gridlines(draw_labels=true, dms=true, x_inline=false, y_inline=false)
    plt.tight_layout()

    out_folder = os.path.join(dirs["fig"], model, "maps", region, subregion)
    os.makedirs(out_folder, exist_ok=true)
    save_path = os.path.join(out_folder, f"raw_map_original_{var_name}_trained_with_{'_'.join(training_vars)}_{lead_time}h.png")

    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_raw_forecast_corrected(raw_spatial_corr, model, region, subregion, var_name, dirs, training_vars, lead_time):
    """generates and saves a map for the corrected forecast values."""
    vmin = float(raw_spatial_corr.min().values)
    vmax = float(raw_spatial_corr.max().values)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.platecarree())
    raw_spatial_corr.plot(ax=ax, cmap='viridis', add_colorbar=true, vmin=vmin, vmax=vmax)
    ax.set_title("Corrected Forecast Values")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    plt.tight_layout()
    out_folder = os.path.join(dirs["fig"], model, "maps", region, subregion)
    os.makedirs(out_folder, exist_ok=True)
    save_path = os.path.join(out_folder, f"raw_map_corrected_{var_name}_trained_with_{'_'.join(training_vars)}_{lead_time}h.png")

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

    out_folder = os.path.join(dirs["fig"], model, "maps", region, subregion)
    os.makedirs(out_folder, exist_ok=True)
    save_path = os.path.join(out_folder, f"raw_map_difference_{var_name}_trained_with_{'_'.join(training_vars)}_{lead_time}h.png")
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
    out_folder = os.path.join(dirs["fig"], model, "maps", region, subregion)
    os.makedirs(out_folder, exist_ok=True)
    save_path = os.path.join(out_folder, f"mse_map_original_{var_name}_trained_with_{'_'.join(training_vars)}_{lead_time}h.png")
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
    out_folder = os.path.join(dirs["fig"], model, "maps", region, subregion)
    os.makedirs(out_folder, exist_ok=True)
    save_path = os.path.join(out_folder, f"mse_map_corrected_{var_name}_trained_with_{'_'.join(training_vars)}_{lead_time}h.png")
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_mse_map_diff(mse_spatial_orig, mse_spatial_corr, model, region, subregion, var_name, dirs, training_vars, lead_time):
    """Generates and saves a spatial map of the MSE difference (corrected - original)."""
    mse_diff = mse_spatial_orig - mse_spatial_corr
    vmin = float(mse_diff.min().values)
    vmax = float(mse_diff.max().values)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    mse_diff.plot(ax=ax, cmap='coolwarm', add_colorbar=True, vmin=vmin, vmax=vmax)
    ax.set_title("MSE Improvement (Original - Corrected)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    plt.tight_layout()
    out_folder = os.path.join(dirs["fig"], model, "maps", region, subregion)
    os.makedirs(out_folder, exist_ok=True)
    save_path = os.path.join(out_folder, f"mse_map_difference_{var_name}_trained_with_{'_'.join(training_vars)}_{lead_time}h.png")
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
    ds_forecasts = xr.open_zarr(forecast_path)

    # Compute metrics.
    mse_orig, mse_corr, raw_spatial_orig, raw_spatial_corr, raw_spatial_diff, mse_spatial_orig, mse_spatial_corr = create_metrics(ds_forecasts, prediction_var)


    # Print spatial bounds.
    ground_truth = ds_forecasts[f"{prediction_var}_ground_truth"]
    mse_total_orig = ((ds_forecasts[f"{prediction_var}_original"] - ground_truth) ** 2).mean()
    mse_total_corr = ((ds_forecasts[f"{prediction_var}_corrected"] - ground_truth) ** 2).mean()
    # print(f"Total MSE for original: {mse_total_orig.values}")
    # print(f"Total MSE for corrected: {mse_total_corr.values}")

    # Generate individual plots.
    plot_monthly_mse(mse_orig, mse_corr, model, region, subregion, prediction_var, dirs, training_vars, lead_time)

    # Create maps for all regions besides "pixel".
    if region != "pixel":
        plot_raw_forecast_original(raw_spatial_orig, model, region,subregion, prediction_var, dirs, training_vars, lead_time)
        plot_raw_forecast_corrected(raw_spatial_corr, model, region, subregion, prediction_var, dirs, training_vars, lead_time)
        plot_raw_forecast_diff(raw_spatial_diff, model, region, subregion, prediction_var, dirs, training_vars, lead_time)
        plot_mse_map_original(mse_spatial_orig, model, region, subregion, prediction_var, dirs, training_vars, lead_time)
        plot_mse_map_corrected(mse_spatial_corr, model, region, subregion, prediction_var, dirs, training_vars, lead_time)
        plot_mse_map_diff(mse_spatial_orig, mse_spatial_corr, model, region, subregion, prediction_var, dirs, training_vars, lead_time)

# Function to generate subregion comparison plots that show the MSE improvements of using different subregion sizes.
def generate_subregion_comparison_plots(dirs, train_start, train_end, test_start,
                                        test_end, model, training_output_vars,
                                        prediction_var, mlp_params):
    input_folder = dirs['input']
    training_vars, output_vars = training_output_vars
    training_vars = training_vars if isinstance(training_vars, (list,tuple)) else [training_vars]
    output_vars   = output_vars   if isinstance(output_vars,   (list,tuple)) else [output_vars]
    mlp_str = f"mlp{mlp_params[0]}x{mlp_params[1]}"

    regions   = ["amazon", "india", "usa_south", "british_columbia"]
    subregions = ["2x2","4x4","6x6","8x8","10x10"]
    lead_times = [24,72,168]
    degrees    = [int(s.split('x')[0]) for s in subregions]

    improvement = {r:{lt:[] for lt in lead_times} for r in regions}

    for region in regions:
        # ---- extract central 2×2 bounds and normalization once ----
        central_path = os.path.join(
            input_folder,
            generate_output_path(SimpleNamespace(
                model_name=model, region=region, subregion="2x2",
                train_start=train_start, train_end=train_end,
                test_start=test_start,  test_end=test_end,
                training_vars=training_vars, output_vars=output_vars,
                mlp_hidden_dim=mlp_params[0], mlp_layers=mlp_params[1],
                lead_time_hours=lead_times[0]  # dummy
            ))
        )
        with xr.open_zarr(central_path) as ds2:
            lat_min, lat_max = ds2.latitude.min().item(), ds2.latitude.max().item()
            lon_min, lon_max = ds2.longitude.min().item(), ds2.longitude.max().item()

            gt2 = ds2[f"{prediction_var}_ground_truth"]
            mu, sigma = float(gt2.mean()), float(gt2.std())

        # ---- now loop subregions + lead times, always slicing that 2×2 box ----
        for sub in subregions:
            for lt in lead_times:
                path = os.path.join(
                    input_folder,
                    generate_output_path(SimpleNamespace(
                        model_name=model, region=region, subregion=sub,
                        train_start=train_start, train_end=train_end,
                        test_start=test_start,  test_end=test_end,
                        training_vars=training_vars, output_vars=output_vars,
                        mlp_hidden_dim=mlp_params[0], mlp_layers=mlp_params[1],
                        lead_time_hours=lt
                    ))
                )
                # open and slice to 2x2 grid in the center of the region
                with xr.open_zarr(path) as ds:
                    ds = ds.sel(latitude=slice(lat_min,lat_max),
                                longitude=slice(lon_min,lon_max))
                    
                    # normalize using mu, sigma
                    gt_n   = ds[f"{prediction_var}_ground_truth"] 
                    orig_n = ds[f"{prediction_var}_original"]     
                    corr_n = ds[f"{prediction_var}_corrected"]    

                    rmse_orig = float(np.sqrt(((orig_n - gt_n)**2).mean()))
                    rmse_corr = float(np.sqrt(((corr_n - gt_n)**2).mean()))
                    pct_improvement = (rmse_orig - rmse_corr) / rmse_orig * 100
                    size = int(sub.split('x')[0])
                    improvement[region][lt].append((size, pct_improvement))

    # ---- plotting ----
    cmap = plt.get_cmap('tab10')
    ls_map = {24:'solid',72:'--',168:':'}

    for region in regions:
        plt.figure(figsize=(8,5))
        for idx, lt in enumerate(lead_times):
            data = sorted(improvement[region][lt], key=lambda x: x[0])
            sizes, imps = zip(*data)
            plt.plot(sizes, imps, marker='o',
                     color=cmap(idx), linestyle=ls_map[lt],
                     label=f"{lt}-h lead")
        plt.xticks(degrees, subregions)
        plt.xlabel("Patch size (degrees)")
        plt.ylabel("RMSE pct improvement\n(original − corrected)")
        plt.title(f"{region.replace('_',' ').title()}: RMSE PCT Improvement by Patch Size")
        plt.grid(True)
        plt.legend(title="Lead time")
        plt.tight_layout()

        out_folder = os.path.join(dirs["fig"], model, "subregion")
        os.makedirs(out_folder, exist_ok=True)
        fname = f"subregion_rmse_improvement_{region}_{'_'.join(training_vars)}_{prediction_var}_{mlp_str}.png"
        plt.savefig(os.path.join(out_folder, fname), dpi=150)
        plt.close()


#######################
# Comparison Function for Multiple Runs
#######################

def compare_runs_rmse(dirs, model, training_output_vars, prediction_var, mlp_params):
    """
    Scans the input folder for forecast files matching the given model,
    training/output variables, and MLP parameters, and creates a single bar plot
    that organizes the overall (scalar) RMSE by lead time (first level) and region (second level).
    
    For each (lead time, region) combination, the original and corrected RMSE are plotted
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
    time_str = "train2018-01-01-2021-12-31_test2022-01-01-2022-12-31"

    # Define the lead times and regions to consider.
    lead_times = [24, 72, 168] # possible lead times 
    regions = ["amazon", "usa_south", "india", "british_columbia"]  # adjust or extend as needed
    subregion ="10x10"

    # Dictionary to store results keyed by (lead_time, region)
    # Each value is a tuple: (avg_rmse_orig, avg_rmse_corr)
    results = {}
    ifs_results = {}

    # Loop over each combination to get original and forecast
    for lt in lead_times:
        for region in regions:

            file_path = os.path.join(input_folder, f"{model}/{region}/train_{training_vars_str}_test_{output_vars_str}_dim{subregion}_leadtime_{lt}h_{time_str}_{mlp_str}.zarr")
            ifs_file_path = os.path.join(input_folder, f"ifs/{region}/train_{training_vars_str}_test_{output_vars_str}_dim{subregion}_leadtime_{lt}h_{time_str}_{mlp_str}.zarr")

            try:
                ds = xr.open_zarr(file_path)
            except Exception as e:
                print(f"Error opening {file_path}: {e}")
                continue

            ground_truth = ds[f"{prediction_var}_ground_truth"]
            orig = ds[f"{prediction_var}_original"]
            corr = ds[f"{prediction_var}_corrected"]

            # normalize by test‐set truth
            mean = ground_truth.mean().values
            std  = ground_truth.std().values
            gt_n   = (ground_truth - mean) / std
            orig_n = (orig            - mean) / std
            corr_n = (corr           - mean) / std

            rmse_total_orig = float(np.sqrt(((orig_n - gt_n) ** 2).mean().values))
            rmse_total_corr = float(np.sqrt(((corr_n - gt_n) ** 2).mean().values))
            results[(lt, region)] = (rmse_total_orig, rmse_total_corr)

            # repeat for ifs
            try:
                ifs_ds = xr.open_zarr(ifs_file_path)
            except Exception as e:
                print(f"Error opening {ifs_file_path}: {e}")
                continue
            
            ifs_ground_truth = ifs_ds[f"{prediction_var}_ground_truth"]
            ifs_fc_original = ifs_ds[f"{prediction_var}_original"]
            ifs_fc_corrected = ifs_ds[f"{prediction_var}_corrected"]

            # Normalize by ifs groundtruth
            ifs_mean = ifs_ground_truth.mean().values
            ifs_std  = ifs_ground_truth.std().values
            ifs_gt_n   = (ifs_ground_truth - ifs_mean) / ifs_std
            ifs_orig_n = (ifs_fc_original - ifs_mean) / ifs_std
            ifs_corr_n = (ifs_fc_corrected - ifs_mean) / ifs_std


            ifs_rmse_total_orig = float(np.sqrt(((ifs_orig_n - ifs_gt_n) ** 2).mean().values))
            ifs_rmse_total_corr = float(np.sqrt(((ifs_corr_n - ifs_gt_n) ** 2).mean().values))
            ifs_results[(lt, region)] = (ifs_rmse_total_orig, ifs_rmse_total_corr)

    # Prepare data for the single grouped bar plot.
    x_positions = []
    x_labels = []
    rmse_orig_vals = []
    rmse_corr_vals = []
    ifs_rmse_orig_vals = []
    ifs_rmse_corr_vals = []
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
            rmse_orig, rmse_corr = results[(lt, region)]
            ifs_rmse_orig, ifs_rmse_corr = ifs_results[(lt, region)]

            rmse_orig_vals.append(rmse_orig)
            rmse_corr_vals.append(rmse_corr)
            ifs_rmse_orig_vals.append(ifs_rmse_orig)
            ifs_rmse_corr_vals.append(ifs_rmse_corr)
            pos += 1
        pos += group_gap  # add gap between groups

    x_positions_offset = np.array(x_positions) + 0.3  # Offset for IFS bars

    # # Create the grouped bar plot.
    fig, ax = plt.subplots(figsize=(max(8, len(x_positions)*0.8), 6))
    # Overlap the two bars at the same positions with transparency.
    ax.bar(x_positions, rmse_orig_vals, color='blue', width=0.8, alpha=0.5, label='Original RMSE')
    ax.bar(x_positions, rmse_corr_vals, color='red', width=0.8, alpha=0.5, label='Corrected RMSE')

    ifs_bar_width = 0.18
    # first IFS bar (baseline)
    ax.bar(
        x_positions_offset,
        ifs_rmse_orig_vals,
        width=ifs_bar_width,
        alpha=0.75,
        label='IFS Baseline RMSE'
    )
    # second IFS bar (corrected), shifted over by one bar‐width
    ax.bar(
        x_positions_offset + ifs_bar_width,
        ifs_rmse_corr_vals,
        width=ifs_bar_width,
        alpha=0.75,
        color='#ADD8E6',        # light‐blue
        label='IFS Corrected RMSE'
    )

    # rest stays the same
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_ylabel("Normalied RMSE")
    ax.set_title(f"RMSE Comparison for {model}\nPredicting {prediction_var}")
    ax.legend()
    plt.tight_layout()

    save_path = os.path.join(dirs["fig"], model, "comparison", f"rmse_comparison_{model}_trained_with_{training_vars_str}_output{prediction_var}_{mlp_str}.png")
    plt.savefig(save_path, dpi=150)
    print(f"RMSE comparison bar chart saved to {save_path}")
    plt.close()

def main():

    dirs = setup_directories()

    # three options for training and output variable combinations, uncomment the one you want to use

    training_vars = ["2m_temperature"]
    output_vars = ["2m_temperature"]
    prediction_var = "2m_temperature"

    # training_vars = ["10m_wind_speed"]
    # output_vars = ["10m_wind_speed"]
    # prediction_var = "10m_wind_speed"

    # Compare multiple runs across lead times and regions in a single plot.
    # compare_runs_rmse(
    #     dirs=dirs,
    #     model="pangu",
    #     training_output_vars=(training_vars, output_vars),
    #     prediction_var=prediction_var,
    #     mlp_params=(512, 5)
    # )

    # regions = ["india", "amazon", "british_columbia", "usa_south"]
    # # regions = ["tropical", "arid", "temperate"]
    # generate_lead_time_plots(
    #     dirs = dirs,
    #     train_start="2018-01-01",
    #     train_end="2021-12-31",
    #     test_start="2022-01-01",
    #     test_end="2022-12-31",
    #     model="pangu",
    #     training_output_vars=(training_vars, output_vars),
    #     prediction_var=prediction_var,
    #     mlp_params=(512, 5), 
    #     regions = regions,
    #     subregion="10x10",
    #     bootstrap=False
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

    # regions = ["usa_south", "amazon", "india", "british_columbia"]
    # subregions = ["2x2", "4x4", "6x6", "8x8", "10x10"]
    # lead_times = [24, 72, 168]

    # for region in regions:
    #     for lead_time in lead_times:
    #         for subregion in subregions:
    #             print(f"Generating plots for {region} with lead time {lead_time} hours")
    #             generate_plots(
    #                 dirs=dirs,
    #                 train_start="2018-01-01",
    #                 train_end="2021-12-31",
    #                 test_start="2022-01-01",
    #                 test_end="2022-12-31",
    #                 model="pangu",
    #                 region=region,
    #                 subregion=subregion,
    #                 lead_time=lead_time,
    #                 training_output_vars=(training_vars, output_vars),
    #                 prediction_var=prediction_var,
    #                 mlp_params=(512, 5)
    #             )

if __name__ == "__main__":
    main()