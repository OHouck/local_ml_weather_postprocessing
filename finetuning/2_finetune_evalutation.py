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
    if nodename == "oMac.local":  # local laptop
        root = os.path.expanduser("~/OneDrive - The University of Chicago/ai_weather_ag/data")
    else:
        raise Exception("Unknown environment, Please specify the root directory")

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
        region,
        subregion,
        bootstrap = None,
):
    """
    Generates plot with bars for RMSE by lead time for a given model and region.
    and a line for % improvement relative to original forecast. on the same plot.
    """
    training_vars, output_vars = training_output_vars

    if not isinstance(training_vars, (list, tuple)):
        training_vars = [training_vars]
    if not isinstance(output_vars, (list, tuple)):
        output_vars = [output_vars]

    training_vars_str = "_".join(training_vars)
    output_vars_str = "_".join(output_vars)
    mlp_str = f"mlp{mlp_params[0]}x{mlp_params[1]}"
    time_str = f"train{train_start}-{train_end}_test{test_start}-{test_end}" 

    lead_times = [24, 48, 72, 96, 120, 144, 168]  # possible lead times

    forecast_orig_mse = {}
    forecast_corr_mse = {}
    ifs_orig_mse = {}
    ifs_corr_mse = {}

    forecast_pct_improvement = {}
    ifs_pct_improvement = {}

    for lt in lead_times:

        # if using boostrap look for files with _bs suffix
        if bootstrap:
            file_path = os.path.join(dirs['input'], f"{model}/{region}/train_{training_vars_str}_test_{output_vars_str}_dim{subregion}_leadtime_{lt}h_{time_str}_{mlp_str}*bs*.zarr")
            ifs_file_path = os.path.join(dirs['input'], f"ifs/{region}/train_{training_vars_str}_test_{output_vars_str}_dim{subregion}_leadtime_{lt}h_{time_str}_{mlp_str}*bs*.zarr")
        else:
            file_path = os.path.join(dirs['input'], f"{model}/{region}/train_{training_vars_str}_test_{output_vars_str}_dim{subregion}_leadtime_{lt}h_{time_str}_{mlp_str}.zarr")
            ifs_file_path = os.path.join(dirs['input'], f"ifs/{region}/train_{training_vars_str}_test_{output_vars_str}_dim{subregion}_leadtime_{lt}h_{time_str}_{mlp_str}.zarr")

        file_paths = glob.glob(file_path)

        if not bootstrap and len(file_paths) > 1:
            raise ValueError(f"Multiple files found for lead time {lt}h: {file_paths}")

        for idx, file_path in enumerate(file_paths):

            try:
                ds = xr.open_zarr(file_path)
            except Exception as e:
                print(f"Error opening {file_path}: {e}")
                continue

            ground_truth = ds[f"{prediction_var}_ground_truth"]
            orig = ds[f"{prediction_var}_original"]
            corr = ds[f"{prediction_var}_corrected"]

            mse_total_orig = float(((orig - ground_truth) ** 2).mean().values)
            mse_total_corr = float(((corr - ground_truth) ** 2).mean().values)
            pct_improvement = (mse_total_orig - mse_total_corr) / mse_total_orig * 100 if mse_total_orig != 0 else 0
            forecast_pct_improvement[(lt, idx)] = pct_improvement
            forecast_orig_mse[(lt, idx)] = mse_total_orig
            forecast_corr_mse[(lt, idx)] = mse_total_corr

            # IFS processing (structure ready for when data is available)
            ifs_file_paths = glob.glob(ifs_file_path)
            if ifs_file_paths and idx < len(ifs_file_paths):
                try:
                    ifs_ds = xr.open_zarr(ifs_file_paths[idx])

                    ifs_ground_truth = ifs_ds[f"{prediction_var}_ground_truth"]
                    ifs_fc_original = ifs_ds[f"{prediction_var}_original"]
                    ifs_fc_corrected = ifs_ds[f"{prediction_var}_corrected"]


                    # mse for IFS
                    ifs_mse_total_orig = float(((ifs_fc_original - ifs_ground_truth) ** 2).mean().values)
                    ifs_mse_total_corr = float(((ifs_fc_corrected - ifs_ground_truth) ** 2).mean().values)
                    pct_improvement = (ifs_mse_total_orig - ifs_mse_total_corr) / ifs_mse_total_orig * 100 if ifs_mse_total_orig != 0 else 0
                    ifs_orig_mse[(lt, idx)] = ifs_mse_total_orig
                    ifs_corr_mse[(lt, idx)] = ifs_mse_total_corr
                    ifs_pct_improvement[(lt, idx)] = pct_improvement

                except Exception as e:
                    print(f"Error opening IFS file {ifs_file_paths[idx]}: {e}")
                    continue

    # Helper function to aggregate bootstrap results
    def aggregate_bootstrap_results(mse_dict, lead_times):
        """Aggregate bootstrap results by lead time"""
        results = {}
        for lt in lead_times:
            # Get all MSE values for this lead time
            lt_values = [mse_dict[(lt, idx)] for (lead_time, idx) in mse_dict.keys() if lead_time == lt]
            if lt_values:
                n = len(lt_values)
                mean = np.mean(lt_values)
                std = np.std(lt_values, ddof = 1)
                se = std / np.sqrt(n)
                alpha = 0.05
                t_crit = stats.t.ppf(1 - alpha/2, df=n-1)

                results[lt] = {
                    'mean': mean,
                    'std': std,
                    'ci_lower': mean - (t_crit * se),
                    'ci_upper': mean + (t_crit * se),
                    'count': n,
                    'values': lt_values
                }
        return results

    # Prepare data for plotting
    if bootstrap:
        # Aggregate bootstrap results
        forecast_orig_agg = aggregate_bootstrap_results(forecast_orig_mse, lead_times)
        forecast_corr_agg = aggregate_bootstrap_results(forecast_corr_mse, lead_times)
        forecast_pct_improvement_agg = aggregate_bootstrap_results(forecast_pct_improvement, lead_times)

        # IFS data (if available) XX 6/30/25 don't have downloaded for climate regions
        ifs_orig_agg = aggregate_bootstrap_results(ifs_orig_mse, lead_times) if ifs_orig_mse else {}
        ifs_corr_agg = aggregate_bootstrap_results(ifs_corr_mse, lead_times) if ifs_corr_mse else {}
        ifs_pct_improvement_agg = aggregate_bootstrap_results(ifs_pct_improvement, lead_times) if ifs_pct_improvement else {}
        
        # Extract plotting data
        forecast_orig_mean = [forecast_orig_agg[lt]['mean'] for lt in lead_times if lt in forecast_orig_agg]
        forecast_orig_ci_lower = [forecast_orig_agg[lt]['ci_lower'] for lt in lead_times if lt in forecast_orig_agg]
        forecast_orig_ci_upper = [forecast_orig_agg[lt]['ci_upper'] for lt in lead_times if lt in forecast_orig_agg]
        
        forecast_corr_mean = [forecast_corr_agg[lt]['mean'] for lt in lead_times if lt in forecast_corr_agg]
        forecast_corr_ci_lower = [forecast_corr_agg[lt]['ci_lower'] for lt in lead_times if lt in forecast_corr_agg]
        forecast_corr_ci_upper = [forecast_corr_agg[lt]['ci_upper'] for lt in lead_times if lt in forecast_corr_agg]
        
        forecast_pct_improvement_mean = [forecast_pct_improvement_agg[lt]['mean'] for lt in lead_times if lt in forecast_pct_improvement_agg]
        # XX currently not using CI for percentage improvement
        forecast_pct_improvement_ci_lower = [forecast_pct_improvement_agg[lt]['ci_lower'] for lt in lead_times if lt in forecast_pct_improvement_agg]
        forecast_pct_improvement_ci_upper = [forecast_pct_improvement_agg[lt]['ci_upper'] for lt in lead_times if lt in forecast_pct_improvement_agg]
        
        # IFS data (if available)
        if ifs_orig_agg:
            ifs_orig_mean = [ifs_orig_agg[lt]['mean'] for lt in lead_times if lt in ifs_orig_agg]
            ifs_orig_ci_lower = [ifs_orig_agg[lt]['ci_lower'] for lt in lead_times if lt in ifs_orig_agg]
            ifs_orig_ci_upper = [ifs_orig_agg[lt]['ci_upper'] for lt in lead_times if lt in ifs_orig_agg]
        else:
            ifs_orig_mean = ifs_orig_ci_lower = ifs_orig_ci_upper = []
            
        if ifs_corr_agg:
            ifs_corr_mean = [ifs_corr_agg[lt]['mean'] for lt in lead_times if lt in ifs_corr_agg]
            ifs_corr_ci_lower = [ifs_corr_agg[lt]['ci_lower'] for lt in lead_times if lt in ifs_corr_agg]
            ifs_corr_ci_upper = [ifs_corr_agg[lt]['ci_upper'] for lt in lead_times if lt in ifs_corr_agg]
        else:
            ifs_corr_mean = ifs_corr_ci_lower = ifs_corr_ci_upper = []
        if ifs_pct_improvement_agg:
            ifs_pct_improvement_mean = [ifs_pct_improvement_agg[lt]['mean'] for lt in lead_times if lt in ifs_pct_improvement_agg]
            # XX currently not using CI for percentage improvement
            ifs_pct_improvement_ci_lower = [ifs_pct_improvement_agg[lt]['ci_lower'] for lt in lead_times if lt in ifs_pct_improvement_agg]
            ifs_pct_improvement_ci_upper = [ifs_pct_improvement_agg[lt]['ci_upper'] for lt in lead_times if lt in ifs_pct_improvement_agg]
        
        plot_lead_times = [lt for lt in lead_times if lt in forecast_orig_agg]
        
    else:
        # Single value case
        forecast_orig_mean = [forecast_orig_mse[(lt, 0)] for lt in lead_times if (lt, 0) in forecast_orig_mse]
        forecast_corr_mean = [forecast_corr_mse[(lt, 0)] for lt in lead_times if (lt, 0) in forecast_corr_mse]
        forecast_pct_improvement_mean = [forecast_pct_improvement[(lt, 0)] for lt in lead_times if (lt, 0) in forecast_pct_improvement]

        ifs_orig_mean = [ifs_orig_mse[(lt, 0)] for lt in lead_times if (lt, 0) in ifs_orig_mse]
        ifs_corr_mean = [ifs_corr_mse[(lt, 0)] for lt in lead_times if (lt, 0) in ifs_corr_mse]
        ifs_pct_improvement_mean = [ifs_pct_improvement[(lt, 0)] for lt in lead_times if (lt, 0) in ifs_pct_improvement]
        
        # No confidence intervals for single values
        forecast_orig_ci_lower = forecast_orig_ci_upper = []
        forecast_corr_ci_lower = forecast_corr_ci_upper = []
        ifs_orig_ci_lower = ifs_orig_ci_upper = []
        ifs_corr_ci_lower = ifs_corr_ci_upper = []
        
        plot_lead_times = [lt for lt in lead_times if (lt, 0) in forecast_orig_mse]

        # Replace the existing plotting section with this modified code:

    # Choose colors for consistency with your image style
    color_forecast_orig = '#1f77b4'    # blue for Pangu original
    color_forecast_corr = '#aec7e8'    # light blue for Pangu corrected  
    color_ifs_orig = '#ff7f0e'         # orange for IFS original
    color_ifs_corr = '#ffbb78'         # light orange for IFS corrected

    # Line colors for percentage improvements
    color_forecast_pct = '#d62728'     # red for Pangu improvement line
    color_ifs_pct = '#9467bd'          # purple for IFS improvement line

    # Create the dual-axis plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()  # Create second y-axis

    # Set up bar positions
    n_groups = len(plot_lead_times)
    bar_width = 0.15
    x_pos = np.arange(n_groups)

    # Plot MSE bars on left axis (ax1)
    bars = []

    # Pangu bars
    if forecast_orig_mean:
        bars1 = ax1.bar(x_pos - 1.5*bar_width, forecast_orig_mean, bar_width, 
                        label='Pangu Original', color=color_forecast_orig, alpha=0.8)
        bars.extend(bars1)
        
        # Add error bars for bootstrap if available
        if bootstrap and forecast_orig_ci_lower:
            forecast_orig_errors = [
                [forecast_orig_mean[i] - forecast_orig_ci_lower[i] for i in range(len(forecast_orig_mean))],
                [forecast_orig_ci_upper[i] - forecast_orig_mean[i] for i in range(len(forecast_orig_mean))]
            ]
            ax1.errorbar(x_pos - 1.5*bar_width, forecast_orig_mean, yerr=forecast_orig_errors,
                        fmt='none', ecolor='black', capsize=3, alpha=0.7)

    if forecast_corr_mean:
        bars2 = ax1.bar(x_pos - 0.5*bar_width, forecast_corr_mean, bar_width,
                        label='Pangu Corrected', color=color_forecast_corr, alpha=0.8)
        bars.extend(bars2)
        
        # Add error bars for bootstrap if available
        if bootstrap and forecast_corr_ci_lower:
            forecast_corr_errors = [
                [forecast_corr_mean[i] - forecast_corr_ci_lower[i] for i in range(len(forecast_corr_mean))],
                [forecast_corr_ci_upper[i] - forecast_corr_mean[i] for i in range(len(forecast_corr_mean))]
            ]
            ax1.errorbar(x_pos - 0.5*bar_width, forecast_corr_mean, yerr=forecast_corr_errors,
                        fmt='none', ecolor='black', capsize=3, alpha=0.7)

    # IFS bars (if available)
    if ifs_orig_mean:
        ifs_lead_times_indices = [i for i, lt in enumerate(plot_lead_times) 
                                if (not bootstrap and (lt, 0) in ifs_orig_mse) or 
                                    (bootstrap and lt in ifs_orig_agg)]
        if ifs_lead_times_indices:
            bars3 = ax1.bar(np.array(ifs_lead_times_indices) + 0.5*bar_width, ifs_orig_mean, bar_width,
                            label='IFS Original', color=color_ifs_orig, alpha=0.8)
            bars.extend(bars3)
            
            # Add error bars for bootstrap if available
            if bootstrap and ifs_orig_ci_lower:
                ifs_orig_errors = [
                    [ifs_orig_mean[i] - ifs_orig_ci_lower[i] for i in range(len(ifs_orig_mean))],
                    [ifs_orig_ci_upper[i] - ifs_orig_mean[i] for i in range(len(ifs_orig_mean))]
                ]
                ax1.errorbar(np.array(ifs_lead_times_indices) + 0.5*bar_width, ifs_orig_mean, 
                            yerr=ifs_orig_errors, fmt='none', ecolor='black', capsize=3, alpha=0.7)

    if ifs_corr_mean:
        ifs_lead_times_indices = [i for i, lt in enumerate(plot_lead_times) 
                                if (not bootstrap and (lt, 0) in ifs_corr_mse) or 
                                    (bootstrap and lt in ifs_corr_agg)]
        if ifs_lead_times_indices:
            bars4 = ax1.bar(np.array(ifs_lead_times_indices) + 1.5*bar_width, ifs_corr_mean, bar_width,
                            label='IFS Corrected', color=color_ifs_corr, alpha=0.8)
            bars.extend(bars4)
            
            # Add error bars for bootstrap if available
            if bootstrap and ifs_corr_ci_lower:
                ifs_corr_errors = [
                    [ifs_corr_mean[i] - ifs_corr_ci_lower[i] for i in range(len(ifs_corr_mean))],
                    [ifs_corr_ci_upper[i] - ifs_corr_mean[i] for i in range(len(ifs_corr_mean))]
                ]
                ax1.errorbar(np.array(ifs_lead_times_indices) + 1.5*bar_width, ifs_corr_mean, 
                            yerr=ifs_corr_errors, fmt='none', ecolor='black', capsize=3, alpha=0.7)

    # Plot percentage improvement lines on right axis (ax2)
    if forecast_pct_improvement_mean:
        line1 = ax2.plot(x_pos, forecast_pct_improvement_mean, 'o-', 
                        color=color_forecast_pct, linewidth=2, markersize=6,
                        label='Pangu (%)') 
        
        # Add confidence intervals for percentage improvement if bootstrap
        if bootstrap and forecast_pct_improvement_ci_lower:
            ax2.fill_between(x_pos, forecast_pct_improvement_ci_lower, forecast_pct_improvement_ci_upper,
                            color=color_forecast_pct, alpha=0.2)

    if ifs_pct_improvement_mean:
        ifs_lead_times_indices = [i for i, lt in enumerate(plot_lead_times) 
                                if (not bootstrap and (lt, 0) in ifs_pct_improvement) or 
                                    (bootstrap and lt in ifs_pct_improvement_agg)]
        if ifs_lead_times_indices:
            line2 = ax2.plot(ifs_lead_times_indices, ifs_pct_improvement_mean, 's-',
                            color=color_ifs_pct, linewidth=2, markersize=6,
                            label='IFS (%)')
            
            # Add confidence intervals for IFS percentage improvement if bootstrap
            if bootstrap and ifs_pct_improvement_ci_lower:
                ax2.fill_between(ifs_lead_times_indices, ifs_pct_improvement_ci_lower, ifs_pct_improvement_ci_upper,
                                color=color_ifs_pct, alpha=0.2)

    # Customize the axes
    ax1.set_xlabel("Forecast Lead Time (h)", fontsize=12)
    ax1.set_ylabel("MSE", fontsize=12)
    ax2.set_ylabel("Percentage Decrease", fontsize=12)

    # Set x-axis ticks and labels
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f"{lt}h" for lt in plot_lead_times])

    # Add horizontal line at 0% improvement
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Title
    title_suffix = " (Bootstrap)" if bootstrap else ""
    plt.title(f"MSE and Improvement for {prediction_var.replace('_',' ')}{title_suffix}\n"
            f"Region: {region}, Subregion: {subregion}", fontsize=14)

    # Add bootstrap sample count annotation if bootstrap is used
    if bootstrap:
        n_bootstrap = None
        if forecast_orig_agg:
            first_lt = min(forecast_orig_agg.keys())
            n_bootstrap = forecast_orig_agg[first_lt]['count']
        elif forecast_corr_agg:
            first_lt = min(forecast_corr_agg.keys())
            n_bootstrap = forecast_corr_agg[first_lt]['count']
            
        if n_bootstrap is not None:
            plt.annotate(f'n = {n_bootstrap}', 
                        xy=(0.02, 0.98), xycoords='axes fraction',
                        verticalalignment='top', horizontalalignment='left',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        fontsize=10)

    # Create combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, 
            loc='upper left', bbox_to_anchor=(0.02, 0.88), frameon=True)

    # Grid
    ax1.grid(True, alpha=0.3)
    ax1.set_axisbelow(True)

    # Adjust layout
    plt.tight_layout()

    # Create output folder
    out_folder = os.path.join(dirs["fig"], model, "lead_time", region, subregion)
    os.makedirs(out_folder, exist_ok=True)

    # Generate filename
    bootstrap_suffix = "_bootstrap" if bootstrap else ""
    fname = (f"leadtime_mse_pct_{prediction_var}_trainedwith_{training_vars_str}_"
            f"{mlp_str}{bootstrap_suffix}.png")
    save_path = os.path.join(out_folder, fname)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Lead‐time MSE and percentage improvement plot saved to: {save_path}")

    if bootstrap:
        print(f"Bootstrap aggregation completed with {len(forecast_orig_agg)} lead times")
        # Optionally print some statistics
        for lt in sorted(forecast_orig_agg.keys()):
            n_samples = len(forecast_orig_agg[lt]['values'])
            print(f"  Lead time {lt}h: {n_samples} bootstrap samples")


    

def generate_global_map(
    dirs,
    train_start, train_end,
    test_start,  test_end,
    model,
    training_output_vars,
    prediction_var,
    mlp_params,
    regions,
    subregion="10x10",
    lead_time=168
):
    """
    Generates a world‐map of spatial MSE difference (corrected − original)
    for a single `prediction_var` over the list of `regions`.

    Parameters
    ----------
    dirs : dict
      From setup_directories(), to find dirs['input'] & dirs['fig'].
    train_start, train_end : str
    test_start,  test_end  : str
      e.g. "2018-01-01"
    model : str
      Your model folder name, e.g. "pangu"
    training_output_vars : tuple (training_vars, output_vars)
      Each a list or single‐element list.
    prediction_var : str
      e.g. "2m_temperature" or "10m_wind_speed"
    mlp_params : tuple (hidden_dim, n_layers)
    regions : list of str
      e.g. ["amazon","india","pakistan","usa_south"]
    subregion : str, optional
      e.g. "10x10"
    lead_time : int, optional
      Forecast lead time (hours).
    """
    # unpack and normalize
    training_vars, output_vars = training_output_vars
    if not isinstance(training_vars, (list, tuple)):
        training_vars = [training_vars]
    if not isinstance(output_vars, (list, tuple)):
        output_vars = [output_vars]

    # prepare labels
    mlp_str = f"mlp{mlp_params[0]}x{mlp_params[1]}"

    # 1) compute the spatial‐MSE difference for each region and track min/max
    diffs = {}
    mins, maxs = [], []
    for region in regions:
        # build an Args object just like in your other functions
        class Args: pass
        args = Args()
        args.model_name     = model
        args.region         = region
        args.subregion      = subregion
        args.train_start    = train_start
        args.train_end      = train_end
        args.test_start     = test_start
        args.test_end       = test_end
        args.training_vars  = training_vars
        args.output_vars    = output_vars
        args.mlp_hidden_dim = mlp_params[0]
        args.mlp_layers     = mlp_params[1]
        args.lead_time_hours= lead_time

        path = os.path.join(dirs['input'], generate_output_path(args))
        ds   = xr.open_zarr(path)

        # create_metrics will handle wind_speed from u/v automatically
        mse_o, mse_c, *_ , mse_sp_o, mse_sp_c = create_metrics(ds, prediction_var)
        diff = mse_sp_o - mse_sp_c

        diffs[region] = diff
        mins.append(float(diff.min().values))
        maxs.append(float(diff.max().values))

    vmin, vmax = min(mins), max(maxs)
    # enforce symmetry about zero
    m = max(abs(vmin), abs(vmax))
    vmin, vmax = -m, m


    # 2) plot
    fig, ax = plt.subplots(
        figsize=(12, 6),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )
    for region, diff in diffs.items():
        diff.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap='coolwarm',
            vmin=vmin, vmax=vmax,
            add_colorbar=False
        )

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.set_global()
    ax.set_title(f"Global MSE Improvement (orig − corr) for {prediction_var.replace('_',' ')}")

    # shared colorbar
    mappable = plt.cm.ScalarMappable(cmap='coolwarm')
    mappable.set_clim(vmin, vmax)
    cbar = fig.colorbar(
        mappable,
        ax=ax,
        orientation='horizontal',
        pad=0.05,
        fraction=0.05
    )
    cbar.set_label("Normalized MSE Improvement")

    plt.tight_layout()
    out_dir = os.path.join(dirs['fig'], model, "global_maps")
    os.makedirs(out_dir, exist_ok=True)
    fname = f"global_map_mse_diff_{prediction_var}.png"
    fig.savefig(os.path.join(out_dir, fname), dpi=150)
    plt.close(fig)



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

    out_folder = os.path.join(dirs["fig"], model, "time_series", region, subregion)
    os.makedirs(out_folder, exist_ok=True)
    save_path = os.path.join(out_folder, f"mse_time_series_{var_name}_trained_with_{'_'.join(training_vars)}_{lead_time}h.png")

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

    out_folder = os.path.join(dirs["fig"], model, "maps", region, subregion)
    os.makedirs(out_folder, exist_ok=True)
    save_path = os.path.join(out_folder, f"raw_map_original_{var_name}_trained_with_{'_'.join(training_vars)}_{lead_time}h.png")

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

    regions   = ["amazon", "india", "pakistan", "usa_south", "british_columbia"]
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
                with xr.open_zarr(path) as ds:
                    ds = ds.sel(latitude=slice(lat_min,lat_max),
                                longitude=slice(lon_min,lon_max))
                    
                    # normalize using mu, sigma
                    gt_n   = (ds[f"{prediction_var}_ground_truth"] - mu) / sigma
                    orig_n = (ds[f"{prediction_var}_original"]      - mu) / sigma
                    corr_n = (ds[f"{prediction_var}_corrected"]     - mu) / sigma

                    mse_orig = float(((orig_n - gt_n)**2).mean())
                    mse_corr = float(((corr_n - gt_n)**2).mean())
                    size = int(sub.split('x')[0])
                    improvement[region][lt].append((size, mse_orig - mse_corr))

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
        plt.xlabel("Subregion size (degrees)")
        plt.ylabel("MSE improvement\n(original − corrected)")
        plt.title(f"{region.replace('_',' ').title()}: MSE Improvement")
        plt.grid(True)
        plt.legend(title="Lead time")
        plt.tight_layout()

        out_folder = os.path.join(dirs["fig"], model, "subregion")
        os.makedirs(out_folder, exist_ok=True)
        fname = f"subregion_mse_improvement_{region}_{'_'.join(training_vars)}_{prediction_var}_{mlp_str}.png"
        plt.savefig(os.path.join(out_folder, fname), dpi=150)
        plt.close()


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
    time_str = "train2018-01-01-2021-12-31_test2022-01-01-2022-12-31"

    # Define the lead times and regions to consider.
    lead_times = [24, 72, 168] # possible lead times
    regions = ["amazon", "usa_south", "india", "pakistan", "british_columbia"]  # adjust or extend as needed
    subregion ="10x10"

    # Dictionary to store results keyed by (lead_time, region)
    # Each value is a tuple: (avg_mse_orig, avg_mse_corr)
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

            mse_total_orig = float(((orig_n - gt_n) ** 2).mean().values)
            mse_total_corr = float(((corr_n - gt_n) ** 2).mean().values)
            results[(lt, region)] = (mse_total_orig, mse_total_corr)

            # repeat for ifs
            try:
                ifs_ds = xr.open_zarr(ifs_file_path)
            except Exception as e:
                print(f"Error opening {ifs_file_path}: {e}")
                continue

            ifs_ground_truth = ifs_ds[f"{prediction_var}_ground_truth"]
            ifs_fc_original = ifs_ds[f"{prediction_var}_original"]
            ifs_fc_corrected = ifs_ds[f"{prediction_var}_corrected"]

            ifs_mse_total_orig = float(((ifs_fc_original- ifs_ground_truth) ** 2).mean().values)
            ifs_mse_total_corr = float(((ifs_fc_corrected - ifs_ground_truth) ** 2).mean().values)
            ifs_results[(lt, region)] = (ifs_mse_total_orig, ifs_mse_total_corr)

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

    # # Create the grouped bar plot.
    fig, ax = plt.subplots(figsize=(max(8, len(x_positions)*0.8), 6))
    # Overlap the two bars at the same positions with transparency.
    ax.bar(x_positions, mse_orig_vals, color='blue', width=0.8, alpha=0.5, label='Original MSE')
    ax.bar(x_positions, mse_corr_vals, color='red', width=0.8, alpha=0.5, label='Corrected MSE')

    ifs_bar_width = 0.18
    # first IFS bar (baseline)
    ax.bar(
        x_positions_offset,
        ifs_mse_orig_vals,
        width=ifs_bar_width,
        alpha=0.75,
        label='IFS Baseline MSE'
    )
    # second IFS bar (corrected), shifted over by one bar‐width
    ax.bar(
        x_positions_offset + ifs_bar_width,
        ifs_mse_corr_vals,
        width=ifs_bar_width,
        alpha=0.75,
        color='#ADD8E6',        # light‐blue
        label='IFS Corrected MSE'
    )

    # rest stays the same
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_ylabel("Overall MSE")
    ax.set_title(f"MSE Comparison for {model}\nPredicting {prediction_var}")
    ax.legend()
    plt.tight_layout()

    save_path = os.path.join(dirs["fig"], model, "comparison", f"mse_comparison_{model}_trained_with_{training_vars_str}_output{prediction_var}_{mlp_str}.png")
    plt.savefig(save_path, dpi=150)
    print(f"MSE comparison bar chart saved to {save_path}")
    plt.close()

def main():

    dirs = setup_directories()

    # three options for training and output variable combinations, uncomment the one you want to use

    # training_vars = ["2m_temperature"]
    # output_vars = ["2m_temperature"]
    # prediction_var = "2m_temperature"

    training_vars = ["10m_wind_speed"]
    output_vars = ["10m_wind_speed"]
    prediction_var = "10m_wind_speed"

    generate_lead_time_plots(
        dirs = dirs,
        train_start="2018-01-01",
        train_end="2021-12-31",
        test_start="2022-01-01",
        test_end="2022-12-31",
        model="pangu",
        training_output_vars=(training_vars, output_vars),
        prediction_var=prediction_var,
        mlp_params=(512, 5),
        region = "arid",
        subregion="2x2",
        bootstrap=True
    )

    exit()

    # Compare multiple runs across lead times and regions in a single plot.
    compare_runs_mse(
        dirs=dirs,
        model="pangu",
        training_output_vars=(training_vars, output_vars),
        prediction_var=prediction_var,
        mlp_params=(512, 5)
    )


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



    generate_global_map(
        dirs = dirs,
        train_start = "2018-01-01", train_end = "2021-12-31",
        test_start = "2022-01-01", test_end = "2022-12-31",
        model="pangu",
        training_output_vars=(training_vars, output_vars),
        prediction_var=prediction_var,
        mlp_params=(512,5),
        regions=["amazon","india","pakistan","usa_south", "british_columbia"],
        subregion="10x10",
        lead_time=168
    )

    # regions = ["pakistan", "south_pakistan", "full_india", "north_india", "uttar_pradesh", "pixel"]
    regions = ["usa_south", "amazon", "india", "british_columbia", "pakistan"]
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