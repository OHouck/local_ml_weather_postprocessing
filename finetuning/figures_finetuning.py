# Name: figures_finetuning.py
# Author: Ozma Houck

import os
import glob
import socket
import calendar
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from types import SimpleNamespace
from functools import lru_cache

from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats

import time
from types import SimpleNamespace


#######################
# Utility Functions
#######################
def setup_directories():
    """Set up directory structure based on environment."""
    nodename = socket.gethostname()
    if nodename == "oMac.local":
        root = os.path.expanduser("~/OneDrive - The University of Chicago/ai_weather_ag/data")
    else:
        raise Exception(f"Unknown environment, Please specify the root directory. "
                        f"Nodename found: {nodename}")

    dirs = {
        'root': root,
        'raw': os.path.join(root, "raw"),
        'processed': os.path.join(root, "processed"),
        'fig': os.path.join(root, "../figures/finetuning"),
        'input': os.path.join(root, "fine_tuning_output")
    }

    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def generate_output_path(args):
    """Generate standardized output path for forecast files."""
    region_str = f"{args.region}"
    subregion_str = f"{args.subregion}"
    dates_str = f"train{args.train_start}-{args.train_end}_test{args.test_start}-{args.test_end}"
    training_vars_str = "_".join(args.training_vars)
    output_vars_str = "_".join(args.output_vars)
    
    # Handle different nn architectures
    if args.nn_architecture == 'mlp':
        nn_str = "mlp"
    elif args.nn_architecture == 'unet':
        nn_str = "unet"
    else:
        raise ValueError(f"Unknown nn_architecture: {args.nn_architecture}")
    
    lead_time_str = f"leadtime_{args.lead_time_hours}"

    output_path = (f"{args.model_name}/{region_str}/train_{training_vars_str}_test_{output_vars_str}_"
                   f"dim{subregion_str}_{lead_time_str}h_{dates_str}_{nn_str}.zarr")
    return output_path


@lru_cache(maxsize=256)
def load_zarr_cached(file_path):
    """Cache zarr dataset loading to avoid redundant file reads."""
    return xr.open_zarr(file_path)

def extract_forecast_data(ds, prediction_var, lead_time):
    """Extract forecast data arrays for a specific lead time."""
    var_suffix = f"_lt{lead_time}h"
    
    ground_truth = ds[f"{prediction_var}_ground_truth{var_suffix}"]
    original = ds[f"{prediction_var}_original{var_suffix}"]
    corrected = ds[f"{prediction_var}_corrected{var_suffix}"]
    mean_corrected = ds.get(f"{prediction_var}_mean_corrected{var_suffix}", None)
    
    return ground_truth, original, corrected, mean_corrected


def calculate_rmse(predictions, ground_truth):
    """Calculate RMSE between predictions and ground truth."""
    return float(np.sqrt(((predictions - ground_truth) ** 2).mean().values))


def calculate_improvement_percentage(rmse_original, rmse_corrected):
    """Calculate percentage improvement in RMSE."""
    if rmse_original == 0:
        return 0
    return (rmse_original - rmse_corrected) / rmse_original * 100


def generate_subregion_comparison_plots(dirs, train_start, train_end, test_start,
                                        test_end, model, training_output_vars,
                                        prediction_var, nn_architecture=["mlp"],
                                        lead_time=None, simultaneous=False):
    """
    Creates plot showing how RMSE changes when trained on different sizes of subregions.
    """
    input_folder = dirs['input']
    training_vars, output_vars = training_output_vars
    training_vars = training_vars if isinstance(training_vars, (list, tuple)) else [training_vars]
    output_vars = output_vars if isinstance(output_vars, (list, tuple)) else [output_vars]

    valid_lead_times = [24, 120, 216]
    if lead_time not in valid_lead_times:
        raise ValueError(f"Invalid lead time: {lead_time}. Must be one of {valid_lead_times}.")
    
    regions = ["usa_south", "british_columbia", "ethiopia", "amazon", "india"]
    subregions = ["2x2", "6x6", "10x10"]
    degrees = [int(s.split('x')[0]) for s in subregions]

    # Store improvements for both models and architectures
    improvements = {}
    for arch in nn_architecture:
        improvements[arch] = {
            'pangu': {r: [] for r in regions},
            'ifs': {r: [] for r in regions}
        }

    # Set leadtime arg for file naming
    if simultaneous:
        lead_time_hours = "_".join(str(lt) for lt in valid_lead_times)
    else:
        lead_time_hours = lead_time

    for arch in nn_architecture:
        for region in regions:
            print(f"Processing region: {region}, architecture: {arch}")
            
            # Cache central bounds - compute once per region
            central_bounds = None
            
            for model_name in ['pangu', 'ifs']:
                print(f"  Loading {model_name} data for {region}...")
                
                for sub in subregions:
                    args = SimpleNamespace(
                        model_name=model_name, region=region, subregion=sub,
                        train_start=train_start, train_end=train_end,
                        test_start=test_start, test_end=test_end,
                        training_vars=training_vars, output_vars=output_vars,
                        lead_time_hours=lead_time_hours,
                        nn_architecture=arch
                    )
                    
                    path = os.path.join(input_folder, generate_output_path(args))
                    
                    try:
                        # Use optimized loading with auto chunks
                        with xr.open_zarr(path, chunks='auto') as ds:
                            # Extract central bounds on first successful load
                            if central_bounds is None:
                                if sub == "2x2":
                                    central_bounds = {
                                        'lat_min': float(ds.latitude.min()),
                                        'lat_max': float(ds.latitude.max()),
                                        'lon_min': float(ds.longitude.min()),
                                        'lon_max': float(ds.longitude.max())
                                    }
                                else:
                                    # For larger subregions, extract central 2x2
                                    lat_center = float((ds.latitude.min() + ds.latitude.max()) / 2)
                                    lon_center = float((ds.longitude.min() + ds.longitude.max()) / 2)
                                    central_bounds = {
                                        'lat_min': lat_center - 1.0,
                                        'lat_max': lat_center + 1.0,
                                        'lon_min': lon_center - 1.0,
                                        'lon_max': lon_center + 1.0
                                    }
                            
                            # Extract data using helper function
                            ground_truth, original, corrected, _ = extract_forecast_data(
                                ds, prediction_var, lead_time
                            )
                            
                            # Single spatial slice operation
                            ds_subset = xr.Dataset({
                                'ground_truth': ground_truth,
                                'original': original,
                                'corrected': corrected
                            }).sel(
                                latitude=slice(central_bounds['lat_min'], central_bounds['lat_max']),
                                longitude=slice(central_bounds['lon_min'], central_bounds['lon_max'])
                            )
                            
                            print(f"    Loading {sub} data...")
                            start_time = time.time()
                            data_loaded = ds_subset.load()
                            load_time = time.time() - start_time
                            print(f"    Load time for {sub}: {load_time:.2f}s")
                            
                            try:
                                gt_n = data_loaded['ground_truth']
                                orig_n = data_loaded['original']
                                corr_n = data_loaded['corrected']

                                # Fast numpy operations on loaded arrays
                                rmse_orig = calculate_rmse(orig_n, gt_n)
                                rmse_corr = calculate_rmse(corr_n, gt_n)
                                pct_improvement = calculate_improvement_percentage(rmse_orig, rmse_corr)
                                
                                size = int(sub.split('x')[0])
                                improvements[arch][model_name][region].append((size, pct_improvement))
                                print(f"    {sub}: {pct_improvement:.2f}% improvement")
                                
                            except Exception as e:
                                print(f"    Error computing metrics for {model_name}, {region}, {sub}: {e}")
                    except Exception as e:
                        print(f"    {model_name} data not found for {region}, {sub}: {e}")

    print("Improvements collected:", improvements)
    
    # Plotting code remains the same...
    region_colors = plt.get_cmap('Set1')
    region_color_map = {region: region_colors(i) for i, region in enumerate(regions)}
    ls_map = {24: 'solid', 120: 'solid', 216: 'solid'} # change if I want to plot multiple lead times
    
    model_markers = {'pangu': 'o', 'ifs': '^'}
    arch_fillstyles = {'mlp': 'full', 'unet': 'none'}
    
    plt.figure(figsize=(12, 7))
    
    for arch in nn_architecture:
        for model_name in ['pangu', 'ifs']:
            for region in regions:
                data = sorted(improvements[arch][model_name][region], key=lambda x: x[0])
                if not data:
                    continue
                    
                sizes, imps = zip(*data)
                
                region_label = region.replace('_', ' ').title()
                if len(nn_architecture) > 1:
                    label = f"{region_label} {model_name.upper()} {arch.upper()} ({lead_time}h)"
                else:
                    label = f"{region_label} {model_name.upper()} ({lead_time}h)"
                
                plt.ylim(-18, 35)
                plt.plot(sizes, imps, 
                        marker=model_markers[model_name],
                        fillstyle=arch_fillstyles[arch],
                        color=region_color_map[region], 
                        linestyle=ls_map[lead_time],
                        label=label,
                        linewidth=2, markersize=15)
    
    plt.xticks(degrees, subregions)
    plt.xlabel("Patch size (degrees)", fontsize=15)
    plt.ylabel("RMSE % improvement\n(original − corrected)", fontsize=15)
    
    arch_str = "/".join([a.upper() for a in nn_architecture])
    title = f"RMSE % Improvement by Patch Size ({arch_str}) - {lead_time}h Lead Time"
    
    plt.title(title, fontsize=15)
    plt.grid(True, alpha=0.3)
    
    # Legend creation remains the same...
    from matplotlib.lines import Line2D
    
    region_handles = [Line2D([0], [0], color=region_color_map[region], linewidth=3, 
                            label=region.replace('_', ' ').title()) for region in regions]
    
    model_handles = [
        Line2D([0], [0], color='black', marker='o', linestyle='none', markersize=12, label='Pangu'),
        Line2D([0], [0], color='black', marker='^', linestyle='none', markersize=12, label='IFS')
    ]
    
    if len(nn_architecture) > 1:
        arch_handles = [Line2D([0], [0], color='black', marker='o', fillstyle=arch_fillstyles[arch],
                              markersize=12, linestyle='none', label=arch.upper()) 
                       for arch in nn_architecture]
    
    legend1 = plt.legend(handles=region_handles, title="Region", 
                        loc='lower right', bbox_to_anchor=(1, 0))
    legend2 = plt.legend(handles=model_handles, title="Model", 
                        loc='lower right', bbox_to_anchor=(1, 0.35))
    
    if len(nn_architecture) > 1:
        legend3 = plt.legend(handles=arch_handles, title="Architecture", 
                            loc='lower right', bbox_to_anchor=(1, 0.55))
        plt.gca().add_artist(legend3)
    
    plt.gca().add_artist(legend1)
    plt.gca().add_artist(legend2)
    
    plt.tight_layout()

    # Save figure
    out_folder = os.path.join(dirs["fig"], model, "subregion")
    os.makedirs(out_folder, exist_ok=True)
    
    lead_times_suffix = f"_{lead_time}h"
    arch_suffix = "_".join(nn_architecture)
    fname = f"subregion_rmse_improvement_lt{lead_times_suffix}_{'_'.join(training_vars)}_{prediction_var}_{arch_suffix}.png"
    plt.savefig(os.path.join(out_folder, fname), dpi=150, bbox_inches='tight')
    plt.close()

def generate_map_plots(
        dirs,
        train_start, train_end,
        test_start, test_end,
        model,
        training_output_vars,
        prediction_var,
        nn_architecture="mlp",  
        region="usa_south",
        subregion="2x2",
        lead_time=24,
        simultaneous=False
):
    """
    Generates a figure with 2 maps: original forecast RMSE and percent improvement in RMSE.
    Map extent is always 10x10 degrees, with the subregion determining how much is filled.
    
    Parameters
    ----------
    nn_architecture : str
        Architecture to use: "mlp" or "unet"
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
    time_str = f"train{train_start}-{train_end}_test{test_start}-{test_end}"

    valid_lead_times = [24, 120, 216]
    # Set leadtime arg for file naming
    if simultaneous:
        lead_time_hours = "_".join(str(lt) for lt in valid_lead_times)
    else:
        lead_time_hours = lead_time

    # Set up args for generate_output_path
    args = SimpleNamespace(
        model_name=model,
        region=region,
        subregion=subregion,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        training_vars=training_vars,
        output_vars=output_vars,
        lead_time_hours=lead_time_hours,
        nn_architecture=nn_architecture
    )
    

    # Construct file path
    file_path = os.path.join(dirs['input'], generate_output_path(args))

    # Skip if region is "pixel" (no maps for pixel)
    if region == "pixel":
        print(f"Skipping map generation for region 'pixel'")
        return

    try:
        # Load the forecast data
        ds = load_zarr_cached(file_path)

        # First, get the 10x10 degree extent for this region
        # We need to load the 10x10 file to get the full extent
        args_10x10 = SimpleNamespace(**vars(args))
        args_10x10.subregion = "10x10"
        path_10x10 = os.path.join(dirs['input'], generate_output_path(args_10x10))
        
        try:
            with load_zarr_cached(path_10x10) as ds_10x10:
                lat_min_10x10 = float(ds_10x10.latitude.min())
                lat_max_10x10 = float(ds_10x10.latitude.max())
                lon_min_10x10 = float(ds_10x10.longitude.min())
                lon_max_10x10 = float(ds_10x10.longitude.max())

        except Exception as e:
            print(f"Warning: Could not load 10x10 extent, using current extent: {e}")
            lat_min_10x10 = float(ds.latitude.min())
            lat_max_10x10 = float(ds.latitude.max())
            lon_min_10x10 = float(ds.longitude.min())
            lon_max_10x10 = float(ds.longitude.max())
        
        # Extract data arrays
        ground_truth, original, corrected, _= extract_forecast_data(
            ds, prediction_var, lead_time
        )

        
        # Calculate RMSE for original and corrected forecasts
        mse_spatial_orig = ((original - ground_truth) ** 2).mean(dim="time")
        mse_spatial_corr = ((corrected - ground_truth) ** 2).mean(dim="time")
        rmse_spatial_orig = np.sqrt(mse_spatial_orig)
        rmse_spatial_corr = np.sqrt(mse_spatial_corr)
        # Calculate percent improvement
        pct_improvement = ((rmse_spatial_orig - rmse_spatial_corr) / rmse_spatial_orig * 100)
        
        # Create figure with 2 subplots - optimized for 16:9 slides
        fig = plt.figure(figsize=(14, 4.5))
        
        # Use GridSpec for better control over subplot spacing
        gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.1, hspace=0.5)
        
        # First subplot: Original forecast RMSE
        ax1 = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
        
        # Set the 10x10 degree extent
        ax1.set_extent([lon_min_10x10, lon_max_10x10, lat_min_10x10, lat_max_10x10], 
                       crs=ccrs.PlateCarree())
        
        vmin_orig = float(rmse_spatial_orig.min().values)
        vmax_orig = float(rmse_spatial_orig.max().values)
        
        im1 = rmse_spatial_orig.plot(
            ax=ax1, 
            cmap='viridis', 
            add_colorbar=False,
            vmin=vmin_orig, 
            vmax=vmax_orig
        )
        
        # Add colorbar for first subplot with better positioning
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes("right", size="4%", pad=0.05, axes_class=plt.Axes)
        cbar1 = plt.colorbar(im1, cax=cax1)
        cbar1.set_label('RMSE', fontsize=10)
        cbar1.ax.tick_params(labelsize=20)
        
        ax1.set_title(f"Original {model.upper()} Forecast RMSE\n{prediction_var.replace('_', ' ').title()}", 
                      fontsize=13, pad=2)
        ax1.coastlines(resolution='50m', linewidth=0.5)
        ax1.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
        ax1.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.8, edgecolor='black')
        
        # Add state/province borders based on region
        if region in ['usa_south', 'british_columbia']:
            ax1.add_feature(cfeature.STATES, linestyle=':', linewidth=0.5, edgecolor='gray')
        elif region == 'india':
            ax1.add_feature(cfeature.STATES, linestyle=':', linewidth=0.5, edgecolor='gray')
        elif region == 'amazon':
            ax1.add_feature(cfeature.STATES, linestyle=':', linewidth=0.5, edgecolor='gray')
        
        # Customize gridlines to prevent overlap
        gl1 = ax1.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, 
                           linewidth=0.5, alpha=0.5)
        gl1.right_labels = False
        gl1.top_labels = False
        gl1.xlabel_style = {'size': 9}
        gl1.ylabel_style = {'size': 9}
        
        # Second subplot: Percent improvement
        ax2 = fig.add_subplot(gs[1], projection=ccrs.PlateCarree())
        
        # Set the same 10x10 degree extent
        ax2.set_extent([lon_min_10x10, lon_max_10x10, lat_min_10x10, lat_max_10x10], 
                       crs=ccrs.PlateCarree())
        
        # Use diverging colormap centered at 0
        vmin_pct = float(pct_improvement.min().values)
        vmax_pct = float(pct_improvement.max().values)
        
        # Ensure colormap is centered at 0
        vmax_abs = max(abs(vmin_pct), abs(vmax_pct))
        norm = TwoSlopeNorm(vmin=-vmax_abs, vcenter=0, vmax=vmax_abs)
        
        im2 = pct_improvement.plot(
            ax=ax2, 
            cmap='RdBu', 
            add_colorbar=False,
            norm=norm
        )
        
        # Add colorbar for second subplot with better positioning
        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="4%", pad=0.05, axes_class=plt.Axes)
        cbar2 = plt.colorbar(im2, cax=cax2)
        cbar2.set_label('Improvement (%)', fontsize=10)
        cbar2.ax.tick_params(labelsize=20)
        
        ax2.set_title(f"RMSE Percent Improvement ({nn_architecture.upper()} Corrected)\n{prediction_var.replace('_', ' ').title()}", 
                      fontsize=13, pad=2)
        ax2.coastlines(resolution='50m', linewidth=0.5)
        ax2.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
        ax2.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.8, edgecolor='black')
        
        # Add state/province borders based on region
        if region in ['usa_south', 'british_columbia']:
            ax2.add_feature(cfeature.STATES, linestyle=':', linewidth=0.5, edgecolor='gray')
        elif region == 'india':
            ax2.add_feature(cfeature.STATES, linestyle=':', linewidth=0.5, edgecolor='gray')
        elif region == 'amazon':
            ax2.add_feature(cfeature.STATES, linestyle=':', linewidth=0.5, edgecolor='gray')
        
        # Customize gridlines to prevent overlap
        gl2 = ax2.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                           linewidth=0.5, alpha=0.5)
        gl2.right_labels = False
        gl2.top_labels = False
        gl2.left_labels = False
        gl2.xlabel_style = {'size': 9}
        gl2.ylabel_style = {'size': 9}
        
        # Add overall title - reduce vertical spacing
        fig.suptitle(f"{region.replace('_', ' ').title()} - {lead_time}h Lead Time - Patch Size: {subregion}", 
                     fontsize=13, y=1.00)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        out_folder = os.path.join(dirs["fig"], model, "maps", region, subregion)
        os.makedirs(out_folder, exist_ok=True)
        
        fname = f"rmse_maps_{prediction_var}_trainedwith_{training_vars_str}_{lead_time}h_{nn_architecture}.png"
        save_path = os.path.join(out_folder, fname)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Map plots saved to: {save_path}")
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")


def generate_time_series_plots(
        dirs,
        train_start, train_end,
        test_start, test_end,
        model,
        training_output_vars,
        prediction_var,
        nn_architecture="mlp",  
        region="usa_south",
        subregion="2x2",
        lead_time=24,
        simultaneous=False
):
    """
    Generates a single bar plot showing monthly RMSE for original and corrected forecasts
    for both the main model (e.g., pangu) and IFS.
    
    Parameters
    ----------
    nn_architecture : str
        Architecture to use: "mlp" or "unet"
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
    time_str = f"train{train_start}-{train_end}_test{test_start}-{test_end}"

    valid_lead_times = [24, 120, 216]
    if simultaneous:
        lead_time_hours = "_".join(str(lt) for lt in valid_lead_times)
    else:
        lead_time_hours = lead_time

    # Set up args for generate_output_path
    args = SimpleNamespace(
        model_name=model,
        region=region,
        subregion=subregion,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        training_vars=training_vars,
        output_vars=output_vars,
        lead_time_hours=lead_time_hours,
        nn_architecture=nn_architecture
    )

    # Construct file paths for main model and IFS
    model_file_path = os.path.join(dirs['input'], generate_output_path(args))
    
    args.model_name = 'ifs'
    ifs_file_path = os.path.join(dirs['input'], generate_output_path(args))

    # Initialize storage for monthly RMSE values
    months = []
    model_rmse_orig = []
    model_rmse_corr = []
    ifs_rmse_orig = []
    ifs_rmse_corr = []
    
    try:
        # Load main model data
        ds_model = load_zarr_cached(model_file_path)
        ground_truth, original, corrected, _ = extract_forecast_data(
            ds_model, prediction_var, lead_time
        )
        
        # Calculate monthly RMSE for main model (XX come back and make sure this is correct)
        # First compute MSE, then take mean over spatial dimensions, then group by month
        mse_orig = ((original - ground_truth) ** 2)
        mse_corr = ((corrected - ground_truth) ** 2)
        
        # Average over spatial dimensions first
        mse_orig_spatial_mean = mse_orig.mean(dim=['latitude', 'longitude'])
        mse_corr_spatial_mean = mse_corr.mean(dim=['latitude', 'longitude'])
        
        # Then group by month and average over time
        mse_orig_monthly = mse_orig_spatial_mean.groupby("time.month").mean(dim="time")
        mse_corr_monthly = mse_corr_spatial_mean.groupby("time.month").mean(dim="time")
        
        # Convert to RMSE
        rmse_orig_monthly = np.sqrt(mse_orig_monthly)
        rmse_corr_monthly = np.sqrt(mse_corr_monthly)
        
        # Get month names and values
        months = [calendar.month_name[i] for i in rmse_orig_monthly['month'].values]
        model_rmse_orig = float(rmse_orig_monthly.values) if rmse_orig_monthly.values.ndim == 0 else rmse_orig_monthly.values.flatten()
        model_rmse_corr = float(rmse_corr_monthly.values) if rmse_corr_monthly.values.ndim == 0 else rmse_corr_monthly.values.flatten()
        
        # Ensure we have the correct number of values
        if len(model_rmse_orig) != 12 or len(model_rmse_corr) != 12:
            print(f"Warning: Expected 12 monthly values, but got {len(model_rmse_orig)} for original and {len(model_rmse_corr)} for corrected")
            print(f"Shape of rmse_orig_monthly: {rmse_orig_monthly.shape}")
            print(f"Dimensions: {rmse_orig_monthly.dims}")
        
    except Exception as e:
        print(f"Error loading main model data from {model_file_path}: {e}")
        return
    
    # Try to load IFS data
    has_ifs_data = False
    try:
        ds_ifs = load_zarr_cached(ifs_file_path)
        ifs_ground_truth, ifs_original, ifs_corrected, _ = extract_forecast_data(
            ds_ifs, prediction_var, lead_time
        )
        
        # Calculate monthly RMSE for IFS
        # First compute MSE, then take mean over spatial dimensions, then group by month
        ifs_mse_orig = ((ifs_original - ifs_ground_truth) ** 2)
        ifs_mse_corr = ((ifs_corrected - ifs_ground_truth) ** 2)
        
        # Average over spatial dimensions first
        ifs_mse_orig_spatial_mean = ifs_mse_orig.mean(dim=['latitude', 'longitude'])
        ifs_mse_corr_spatial_mean = ifs_mse_corr.mean(dim=['latitude', 'longitude'])
        
        # Then group by month and average over time
        ifs_mse_orig_monthly = ifs_mse_orig_spatial_mean.groupby("time.month").mean(dim="time")
        ifs_mse_corr_monthly = ifs_mse_corr_spatial_mean.groupby("time.month").mean(dim="time")
        
        # Convert to RMSE
        ifs_rmse_orig_monthly = np.sqrt(ifs_mse_orig_monthly)
        ifs_rmse_corr_monthly = np.sqrt(ifs_mse_corr_monthly)
        
        # Get values
        ifs_rmse_orig = float(ifs_rmse_orig_monthly.values) if ifs_rmse_orig_monthly.values.ndim == 0 else ifs_rmse_orig_monthly.values.flatten()
        ifs_rmse_corr = float(ifs_rmse_corr_monthly.values) if ifs_rmse_corr_monthly.values.ndim == 0 else ifs_rmse_corr_monthly.values.flatten()
        has_ifs_data = True
        
    except Exception as e:
        print(f"IFS data not available for {region}: {e}")
    
    # Create the bar plot - optimized for 16:9 slides
    fig, ax = plt.subplots(figsize=(14, 4.5))
    
    # Set up bar positions
    x = np.arange(len(months))
    bar_width = 0.35
    
    # Plot main model bars (overlapping with transparency)
    ax.bar(x - bar_width/2, model_rmse_orig, bar_width, 
           color='blue', alpha=0.5, label=f'{model.upper()} Original')
    ax.bar(x - bar_width/2, model_rmse_corr, bar_width, 
           color='red', alpha=0.5, label=f'{model.upper()} Corrected ({nn_architecture.upper()})')
    
    # Plot IFS bars if available (offset to the right)
    if has_ifs_data:
        ifs_bar_width = bar_width * 0.5  # Make IFS bars narrower
        offset = bar_width/2 + 0.05  # Small gap between model and IFS bars
        
        ax.bar(x + offset, ifs_rmse_orig, ifs_bar_width, 
               color='darkblue', alpha=0.75, label='IFS Baseline')
        ax.bar(x + offset + ifs_bar_width, ifs_rmse_corr, ifs_bar_width, 
               color='#ADD8E6', alpha=0.75, label=f'IFS Corrected ({nn_architecture.upper()})')
    
    # Customize plot
    ax.set_xlabel('Month', fontsize=15)
    ax.set_ylabel('RMSE', fontsize=15)
    ax.set_title(f'Monthly RMSE Comparison - {region.replace("_", " ").title()}\n'
                 f'{prediction_var.replace("_", " ").title()} - {lead_time}h Lead Time - Patch Size: {subregion}',
                 fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    out_folder = os.path.join(dirs["fig"], model, "time_series", region, subregion)
    os.makedirs(out_folder, exist_ok=True)
    
    fname = f"rmse_monthly_{prediction_var}_trainedwith_{training_vars_str}_{lead_time}h_{nn_architecture}.png"
    save_path = os.path.join(out_folder, fname)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Time series plot saved to: {save_path}")

import os
import numpy as np
import pandas as pd
import xarray as xr
from types import SimpleNamespace


def generate_summary_stat_table(
        dirs,
        train_start, train_end,
        test_start, test_end,
        model,
        training_output_vars,
        prediction_var,
        nn_architecture="mlp",
        regions=None,
        subregion="2x2",
        bootstrap=False,
        lead_times=None,
        simultaneous=False
):
    """
    Generates a single comprehensive summary statistics table for a specific variable.
    
    The table includes:
    - Region and lead time columns
    - Mean ground truth with standard deviation in parentheses
    - RMSE of original and corrected forecasts
    - Percent improvement
    
    Parameters
    ----------
    dirs : dict
        Dictionary of directories
    train_start, train_end, test_start, test_end : str
        Date strings for train/test periods
    model : str
        Model name (e.g., 'pangu')
    variables_config : tuple
        Tuple of (training_vars, output_vars, prediction_var)
    nn_architecture : str
        Architecture type: "mlp" or "unet"
    regions : list
        List of regions to analyze. If None, uses default regions
    subregion : str
        Patch size (e.g., "2x2", "10x10")
    bootstrap : bool
        If True, uses bootstrap samples; otherwise, uses full data
    lead_times : list of int
        Lead times in hours. If None, uses [24, 72, 168]
    simultaneous : bool
        If True, processes all lead times simultaneously; otherwise, processes sequentially
    
    Returns
    -------
    pandas.DataFrame
        DataFrame containing all summary statistics
    """
    
    # Extract variables from config
    training_vars, output_vars = training_output_vars
    
    # Ensure variables are lists
    if not isinstance(training_vars, (list, tuple)):
        training_vars = [training_vars]
    if not isinstance(output_vars, (list, tuple)):
        output_vars = [output_vars]
    
    # Default regions if not specified
    if regions is None:
        regions = ["amazon", "india", "usa_south", "british_columbia", "ethiopia"]
    
    # Default lead times if not specified
    if lead_times is None:
        lead_times = [24, 120, 216]
    
    # Storage for all results
    all_rows = []
    region_ground_truth_stats = {}
    
    # Process each region
    for region in regions:
        region_ground_truth_values = []
        
        for lead_time in lead_times:
            # Set up args for generate_output_path

            if simultaneous:
                # Convert to string for file naming
                lead_time_hours = "_".join(str(lt) for lt in lead_times)
            else:
                lead_time_hours = lead_time


            args = SimpleNamespace(
                model_name=model,
                region=region,
                subregion=subregion,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                training_vars=training_vars,
                output_vars=output_vars,
                lead_time_hours=lead_time_hours,
                nn_architecture=nn_architecture
            )
            
            # Construct file path
            # Construct file paths
            if bootstrap:
                file_pattern = os.path.join(dirs['input'], generate_output_path(args).replace('.zarr', '*bs*.zarr'))
            else:
                file_pattern = os.path.join(dirs['input'], generate_output_path(args))
            
            files = glob.glob(file_pattern)
            if not bootstrap and len(files) > 1:
                raise ValueError(f"Error: Multiple files found for {region} with lead time {lead_time}h. Using first file.")

            # create empty dict to store values
            values = {'rmse_orig': np.nan,
                      'rmse_change': np.nan,
                      'rmse_imp_pct': np.nan}

            for idx, file_path in enumerate(files):
                # Load the data
                ds = load_zarr_cached(file_path)

                ground_truth, original, corrected, _ = extract_forecast_data(
                    ds, prediction_var, lead_time
                )
                
                # Flatten arrays for statistics
                gt_flat = ground_truth.values.flatten()
                orig_flat = original.values.flatten()
                corr_flat = corrected.values.flatten()
                
                # Remove NaN values
                mask = ~(np.isnan(gt_flat) | np.isnan(orig_flat) | np.isnan(corr_flat))
                gt_flat = gt_flat[mask]
                orig_flat = orig_flat[mask]
                corr_flat = corr_flat[mask]
                
                # Collect ground truth values for region-wide statistics
                region_ground_truth_values.extend(gt_flat)
                
                # RMSE calculations
                error_orig = orig_flat - gt_flat
                error_corr = corr_flat - gt_flat
                rmse_orig = np.sqrt(np.mean(error_orig**2))
                rmse_corr = np.sqrt(np.mean(error_corr**2))
                
                # Calculate RMSE change (negative means improvement)
                rmse_change = rmse_corr - rmse_orig
                
                # Percent improvement
                pct_improvement = (rmse_orig - rmse_corr) / rmse_orig * 100

                # save values to dict
                values['rmse_orig'] = rmse_orig
                values['rmse_change'] = rmse_change
                values['rmse_imp_pct'] = pct_improvement

            # take average if multiple bootstrap samples
            rmse_orig = values['rmse_orig'].mean() if isinstance(values['rmse_orig'], (list, np.ndarray)) else values['rmse_orig']
            rmse_change = values['rmse_change'].mean() if isinstance(values['rmse_change'], (list, np.ndarray)) else values['rmse_change']
            pct_improvement = values['rmse_imp_pct'].mean() if isinstance(values['rmse_imp_pct'], (list, np.ndarray)) else values['rmse_imp_pct']

            # Create row data
            row_data = {
                'Region': region.replace('_', ' ').title(),
                'Lead Time': f"{lead_time}h",
                'RMSE (Orig)': rmse_orig,
                'RMSE Change': rmse_change,
                'RMSE Improvement (%)': pct_improvement
            }

            all_rows.append(row_data)
            print(f"Processed {region} - {lead_time}h: RMSE improvement = {pct_improvement:.1f}%")
                    
        
        # Calculate region-wide ground truth statistics
        if region_ground_truth_values:
            region_mean = np.mean(region_ground_truth_values)
            region_std = np.std(region_ground_truth_values)
            region_ground_truth_stats[region.replace('_', ' ').title()] = f"{region_mean:.2f} ({region_std:.2f})"
    
    # Create DataFrame
    if not all_rows:
        print("No data processed successfully.")
        return None
    
    df = pd.DataFrame(all_rows)
    
    # Create LaTeX table
    _create_latex_table(df, prediction_var, nn_architecture, subregion, dirs, model, region_ground_truth_stats)
    
    return df


def _create_latex_table(df, prediction_var, nn_architecture, subregion, dirs, model, region_ground_truth_stats):
    """
    Creates a LaTeX table with proper formatting where region names appear only once
    and ground truth statistics appear in the second row of each region.
    """
    # Prepare output folder
    out_folder = os.path.join(dirs["fig"], model, "summary_stats")
    os.makedirs(out_folder, exist_ok=True)
    
    # Start building the LaTeX table
    latex_lines = []
    
    # Table setup
    latex_lines.append("\\begin{tabular}{llrr}")
    latex_lines.append("\\toprule")
    latex_lines.append("Region & Lead Time & RMSE & Improvement (\\%) \\\\")
    latex_lines.append("\\midrule")
    
    # Group by region to format the table
    current_region = None
    region_rows = []
    
    for _, row in df.iterrows():
        region = row['Region']
        
        # When we encounter a new region, process the previous region's rows
        if region != current_region and current_region is not None:
            # Add the collected rows for the previous region
            for i, region_row in enumerate(region_rows):
                if i == 0:
                    # First row: show region name
                    latex_lines.append(region_row)
                elif i == 1:
                    # Second row: show ground truth stats
                    gt_stats = region_ground_truth_stats.get(current_region, "N/A")
                    latex_lines.append(f"\\textit{{Ground Truth: {gt_stats}}} & {region_row}")
                else:
                    # Subsequent rows: empty first column
                    latex_lines.append(f" & {region_row}")
            region_rows = []
        
        # Format RMSE with change in parentheses
        rmse_display = f"{row['RMSE (Orig)']:.3f} ({row['RMSE Change']:+.3f})"
        
        # Build the data portion of the row (without region name)
        data_portion = f"{row['Lead Time']} & {rmse_display} & {row['RMSE Improvement (%)']:.1f} \\\\"
        
        if region != current_region:
            # First row of new region
            region_rows.append(f"{region} & {data_portion}")
            current_region = region
        else:
            # Additional rows for same region
            region_rows.append(data_portion)
    
    # Don't forget to process the last region
    if region_rows:
        for i, region_row in enumerate(region_rows):
            if i == 0:
                latex_lines.append(region_row)
            elif i == 1:
                gt_stats = region_ground_truth_stats.get(current_region, "N/A")
                latex_lines.append(f"\\textit{{Ground Truth: {gt_stats}}} & {region_row}")
            else:
                latex_lines.append(f" & {region_row}")
    
    # Close the table
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    
    # Join all lines
    latex_str = "\n".join(latex_lines)
    
    # Save the LaTeX file
    variable_name = prediction_var.replace('_', ' ').title()
    filename = f"summary_stats_{prediction_var}_{nn_architecture}_{subregion}.tex"
    filepath = os.path.join(out_folder, filename)
    
    with open(filepath, 'w') as f:
        f.write(latex_str)
    
    print(f"\nSaved LaTeX table to: {filepath}")
    print(f"Table title: Summary Statistics for {variable_name}")

def plot_lead_time_from_csv(
        csv_path,
        dirs,
        variable,
        evaluation_metric,
        regions=None,
        subregion="4x4",
        plot_type="all",
        nn_architectures=["mlp"],
        models=None,
        save_path=None
):
    """
    Generate lead time plots from pre-calculated statistics CSV.
    
    Parameters
    ----------
    csv_path : str
        Path to CSV file containing statistics
    dirs : dict
        Dictionary of directories (for saving plots)
    variable : str
        Variable to plot (must match CSV column)
    evaluation_metric : str
        Type of evaluation metric to plot:
        - "rmse_pct_improvement": Percentage improvement in RMSE
        - "raw_values": Raw forecast values (ground truth, original, corrected)
        - "error_cutoff": Percentage of forecasts exceeding error threshold
    regions : list
        List of regions to include in plot. If None, uses all in CSV
    subregion : str
        Patch size to filter for
    plot_type : str
        Type of plot: "pangu_nn", "pangu_ifs_nn", or "all"
    nn_architectures : list
        List of architectures to include: ["mlp"], ["unet"], or both
    models : list
        List of models to include. If None, determined by plot_type
    save_path : str
        Custom save path. If None, auto-generates based on parameters
    """
    
    
    # Load statistics
    df = pd.read_csv(csv_path)
    
    # Filter by subregion
    df = df[df['subregion'] == subregion]
    
    # Filter by regions if specified
    if regions is not None:
        df = df[df['region'].isin(regions)]
    else:
        regions = df['region'].unique().tolist()
    
    # Filter by architectures
    df = df[df['architecture'].isin(nn_architectures)]
    
    # Determine models based on plot_type
    if models is None:
        if plot_type == "pangu_nn":
            models = ["pangu"]
        elif plot_type == "pangu_ifs_nn":
            models = ["pangu", "ifs"]
        elif plot_type == "all":
            models = df['model'].unique().tolist()
    
    df = df[df['model'].isin(models)]

    # filter to variable of interest
    df = df[df['variable'] == variable]
    
    # Get unique values for plotting
    lead_times = sorted(df['lead_time'].unique())
    prediction_var = df['variable'].iloc[0] if len(df) > 0 else "unknown"

    # Define colors
    region_colors = {
        'india': '#E69F00',
        'usa_south': '#56B4E9',
        'british_columbia': '#009E73',
        'amazon': '#CC79A7',
        'ethiopia': '#D55E00',
    }
    
    climate_region_colors = {
        'tropical': '#228b22',
        'arid': '#FFFF00',
        'temperate': '#90EE90',
        'cold': '#6495ED',
        'polar': '#ADD8E6'
    }
    
    model_markers = {
        'pangu': 'o',
        'ifs': '^'
    }
    
    architecture_fillstyles = {
        'mlp': 'full',
        'unet': 'none'
    }
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Extract error cutoff information if available
    error_cutoff_value = None
    error_cutoff_units = None
    if 'metadata' in df.columns and evaluation_metric == "error_cutoff":
        metadata_str = df['metadata'].iloc[0]
        if pd.notna(metadata_str) and 'Error cutoff:' in metadata_str:
            # Parse metadata string like "Error cutoff: >2.5 K"
            parts = metadata_str.split(':')[1].strip().split()
            if len(parts) >= 2:
                error_cutoff_value = parts[0].replace('>', '')
                error_cutoff_units = ' '.join(parts[1:])
    
    # Plot based on evaluation metric
    if evaluation_metric == "rmse_pct_improvement":
        # Original percentage improvement plot
        for region in regions:
            # Get color for region
            if region in climate_region_colors:
                color = climate_region_colors[region]
            else:
                color = region_colors.get(region, '#1f77b4')
            
            region_df = df[df['region'] == region]
            
            for model in models:
                model_df = region_df[region_df['model'] == model]
                
                for arch in nn_architectures:
                    arch_df = model_df[model_df['architecture'] == arch]
                    
                    if len(arch_df) == 0:
                        continue
                    
                    # Sort by lead time
                    arch_df = arch_df.sort_values('lead_time')
                    
                    # Get styles
                    marker = model_markers.get(model, 'o')
                    fillstyle = architecture_fillstyles.get(arch, 'full')
                    
                    # Plot neural network correction
                    if 'rmse_pct_improvement' in arch_df.columns:
                        x_pos = [lead_times.index(lt) for lt in arch_df['lead_time']]
                        y_values = arch_df['rmse_pct_improvement'].values
                        
                        ax.plot(x_pos, y_values,
                               marker=marker,
                               fillstyle=fillstyle,
                               linestyle='-',
                               color=color,
                               linewidth=2.5,
                               markersize=15,
                               alpha=0.75,
                               zorder=3)
                        
                        # Add confidence intervals if available
                        if 'rmse_pct_improvement_ci_lower' in arch_df.columns:
                            ci_lower = arch_df['rmse_pct_improvement_ci_lower'].values
                            ci_upper = arch_df['rmse_pct_improvement_ci_upper'].values
                            ax.fill_between(x_pos, ci_lower, ci_upper,
                                           color=color,
                                           alpha=0.1,
                                           zorder=1)
                    
                    # Plot mean corrected if available and requested
                    if plot_type == "all" and 'pct_improvement_mean_corrected' in arch_df.columns:
                        if not arch_df['pct_improvement_mean_corrected'].isna().all():
                            x_pos = [lead_times.index(lt) for lt in arch_df['lead_time']]
                            y_values = arch_df['pct_improvement_mean_corrected'].values
                            
                            ax.plot(x_pos, y_values,
                                   marker=marker,
                                   fillstyle=fillstyle,
                                   linestyle='--',
                                   color=color,
                                   linewidth=2.5,
                                   markersize=15,
                                   alpha=0.75,
                                   zorder=3)
        
        # Set axes
        ax.set_ylim(-18, 35)
        ax.set_ylabel("RMSE Improvement (%)", fontsize=20)
        
        # Add reference line
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
    elif evaluation_metric == "raw_values":
        # Plot raw forecast values as deviations from ground truth mean
        
        # Track which forecast types we've added to legend
        legend_added = {'ground_truth': False, 'original': False, 'corrected': False}
        
        # Store mean values for annotation
        region_means = {}
        
        for region in regions:
            # Get color for region
            if region in climate_region_colors:
                color = climate_region_colors[region]
            else:
                color = region_colors.get(region, '#1f77b4')
            
            region_df = df[df['region'] == region]
            
            for model in models:
                model_df = region_df[region_df['model'] == model]
                
                for arch in nn_architectures:
                    arch_df = model_df[model_df['architecture'] == arch]
                    
                    if len(arch_df) == 0:
                        continue
                    
                    # Sort by lead time
                    arch_df = arch_df.sort_values('lead_time')
                    
                    # Get styles
                    marker = model_markers.get(model, 'o')
                    fillstyle = architecture_fillstyles.get(arch, 'full')
                    
                    x_pos = [lead_times.index(lt) for lt in arch_df['lead_time']]
                    
                    # Calculate overall mean of ground truth for this region/model/arch
                    if 'ground_truth_mean' in arch_df.columns:
                        ground_truth_values = arch_df['ground_truth_mean'].values
                        if not np.all(np.isnan(ground_truth_values)):
                            overall_gt_mean = np.nanmean(ground_truth_values)
                            
                            # Store mean for annotation (convert K to C for temperature)
                            region_key = f"{region}_{model}_{arch}"
                            if variable == '2m_temperature':
                                region_means[region_key] = (region, overall_gt_mean - 273.15, 'C')  # Convert K to C
                            elif variable == '10m_wind_speed':
                                region_means[region_key] = (region, overall_gt_mean, 'm/s')
                            elif variable == 'total_precipitation':
                                region_means[region_key] = (region, overall_gt_mean, 'mm')
                            else:
                                region_means[region_key] = (region, overall_gt_mean, '')
                            
                            # Plot ground truth deviations
                            y_values_dev = ground_truth_values - overall_gt_mean
                            
                            # Only add label for first ground truth line
                            label = 'Ground Truth' if not legend_added['ground_truth'] else None
                            if label:
                                legend_added['ground_truth'] = True
                            
                            ax.plot(x_pos, y_values_dev,
                                   marker='s',
                                   fillstyle=fillstyle,
                                   linestyle=':',
                                   color=color,
                                   linewidth=2,
                                   markersize=10,
                                   alpha=0.9,
                                   label=label,
                                   zorder=4)
                            
                            # Plot original forecast deviations
                            if 'mean_original_forecast' in arch_df.columns:
                                y_values = arch_df['mean_original_forecast'].values
                                if not np.all(np.isnan(y_values)):
                                    y_values_dev = y_values - overall_gt_mean
                                    
                                    # Only add label for first original line
                                    label = 'Original Forecast' if not legend_added['original'] else None
                                    if label:
                                        legend_added['original'] = True
                                    
                                    ax.plot(x_pos, y_values_dev,
                                           marker=marker,
                                           fillstyle=fillstyle,
                                           linestyle='--',
                                           color=color,
                                           linewidth=2,
                                           markersize=12,
                                           alpha=0.6,
                                           label=label,
                                           zorder=2)
                            
                            # Plot corrected forecast deviations
                            if 'mean_corrected_forecast' in arch_df.columns:
                                y_values = arch_df['mean_corrected_forecast'].values
                                if not np.all(np.isnan(y_values)):
                                    y_values_dev = y_values - overall_gt_mean
                                    
                                    # Only add label for first corrected line
                                    label = 'Corrected Forecast' if not legend_added['corrected'] else None
                                    if label:
                                        legend_added['corrected'] = True
                                    
                                    ax.plot(x_pos, y_values_dev,
                                           marker=marker,
                                           fillstyle=fillstyle,
                                           linestyle='-',
                                           color=color,
                                           linewidth=2.5,
                                           markersize=15,
                                           alpha=0.75,
                                           label=label,
                                           zorder=3)
        
        # Set ylabel based on variable (now for deviations)
        if variable == '2m_temperature':
            ax.set_ylabel("Temperature Deviation (K)", fontsize=20)
        elif variable == '10m_wind_speed':
            ax.set_ylabel("Wind Speed Deviation (m/s)", fontsize=20)
        elif variable == 'total_precipitation':
            ax.set_ylabel("Precipitation Deviation (mm)", fontsize=20)
        else:
            ax.set_ylabel(f"{variable.replace('_', ' ').title()} Deviation", fontsize=20)
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
        
        # Add annotations for mean values
        if region_means:
            # Create annotation text
            annotation_lines = ["Ground Truth Means:"]
            
            # Group by region for cleaner display
            region_annotations = {}
            for key, (region, mean_val, units) in region_means.items():
                if region not in region_annotations:
                    region_annotations[region] = []
                region_annotations[region].append((mean_val, units))
            
            # Format annotations by region
            for region in sorted(region_annotations.keys()):
                values = region_annotations[region]
                # Average if multiple models/architectures for same region
                avg_mean = np.mean([v[0] for v in values])
                units = values[0][1]
                
                # Get region color for the annotation
                if region in climate_region_colors:
                    color = climate_region_colors[region]
                else:
                    color = region_colors.get(region, '#1f77b4')
                
                # Format the mean value
                if variable == '2m_temperature':
                    annotation_lines.append(f"  {region.replace('_', ' ').title()}: {avg_mean:.1f}°{units}")
                else:
                    annotation_lines.append(f"  {region.replace('_', ' ').title()}: {avg_mean:.2f} {units}")
            
            # Place annotation box
            annotation_text = '\n'.join(annotation_lines)
            ax.text(0.02, 0.98, annotation_text,
                   transform=ax.transAxes,
                   fontsize=11,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9),
                   family='monospace')
        
        # Create custom legend that combines line styles and regions
        # First add the forecast type legend
        forecast_handles = []
        if legend_added['ground_truth']:
            forecast_handles.append(Line2D([0], [0], color='gray', marker='s', 
                                          linestyle=':', linewidth=2, markersize=10,
                                          label='Ground Truth'))
        if legend_added['original']:
            forecast_handles.append(Line2D([0], [0], color='gray', marker='o',
                                          linestyle='--', linewidth=2, markersize=12,
                                          label='Original Forecast'))
        if legend_added['corrected']:
            forecast_handles.append(Line2D([0], [0], color='gray', marker='o',
                                          linestyle='-', linewidth=2.5, markersize=15,
                                          label='Corrected Forecast'))
        
    elif evaluation_metric == "error_cutoff":
        # Plot percentage of forecasts exceeding error threshold
        
        # Track if we've plotted anything
        has_data = False
        
        for region in regions:
            # Get color for region
            if region in climate_region_colors:
                color = climate_region_colors[region]
            else:
                color = region_colors.get(region, '#1f77b4')
            
            region_df = df[df['region'] == region]
            
            for model in models:
                model_df = region_df[region_df['model'] == model]
                
                for arch in nn_architectures:
                    arch_df = model_df[model_df['architecture'] == arch]
                    
                    if len(arch_df) == 0:
                        continue
                    
                    # Sort by lead time
                    arch_df = arch_df.sort_values('lead_time')
                    
                    # Get styles
                    marker = model_markers.get(model, 'o')
                    fillstyle = architecture_fillstyles.get(arch, 'full')
                    
                    # Create label only for the first plot of each type (to avoid duplicate legends)
                    label_original = None
                    label_corrected = None
                    if not has_data:
                        label_original = 'Original'
                        label_corrected = 'Corrected'
                        has_data = True
                    
                    # Plot original error rate (dashed line)
                    if 'pct_error_cutoff_original' in arch_df.columns:
                        x_pos = [lead_times.index(lt) for lt in arch_df['lead_time']]
                        y_values = arch_df['pct_error_cutoff_original'].values
                        
                        # Only plot if we have valid data
                        if not np.all(np.isnan(y_values)):
                            ax.plot(x_pos, y_values,
                                   marker=marker,
                                   fillstyle=fillstyle,
                                   linestyle='--',
                                   color=color,
                                   linewidth=2,
                                   markersize=12,
                                   alpha=0.6,
                                   label=label_original,
                                   zorder=2)
                    
                    # Plot corrected error rate (solid line)
                    if 'pct_error_cutoff_corrected' in arch_df.columns:
                        x_pos = [lead_times.index(lt) for lt in arch_df['lead_time']]
                        y_values = arch_df['pct_error_cutoff_corrected'].values
                        
                        # Only plot if we have valid data
                        if not np.all(np.isnan(y_values)):
                            ax.plot(x_pos, y_values,
                                   marker=marker,
                                   fillstyle=fillstyle,
                                   linestyle='-',
                                   color=color,
                                   linewidth=2.5,
                                   markersize=15,
                                   alpha=0.75,
                                   label=label_corrected,
                                   zorder=3)
        
        # Set y-axis label
        ax.set_ylabel("Forecasts Exceeding Error Threshold (%)", fontsize=20)
        
        # Add annotation for cutoff value
        if error_cutoff_value and error_cutoff_units:
            annotation_text = f"Error threshold: >{error_cutoff_value} {error_cutoff_units}"
            ax.text(0.02, 0.98, annotation_text,
                   transform=ax.transAxes,
                   fontsize=14,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
        
        # Set y-axis limits based on data
        ax.set_ylim(bottom=0)
        
        # Add a subtle grid for better readability
        ax.yaxis.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
    
    # Common settings for all plot types
    ax.set_xticks(range(len(lead_times)))
    ax.set_xticklabels([f"{lt}h" for lt in lead_times])
    ax.set_xlabel("Forecast Lead Time", fontsize=20)
    
    # Title
    arch_str = "/".join([a.upper() for a in nn_architectures])
    regions_str = ", ".join(regions)
    is_bootstrap = df['bootstrap'].iloc[0] if len(df) > 0 and 'bootstrap' in df.columns else False
    
    if evaluation_metric == "rmse_pct_improvement":
        title_main = f"RMSE Improvement for {prediction_var.replace('_', ' ').title()} ({arch_str})"
    elif evaluation_metric == "raw_values":
        title_main = f"Forecast Values for {prediction_var.replace('_', ' ').title()} ({arch_str})"
    elif evaluation_metric == "error_cutoff":
        title_main = f"Error Frequency for {prediction_var.replace('_', ' ').title()} ({arch_str})"
    
    title_parts = [title_main]
    title_parts.append(f"Regions: {regions_str}, Patch Size: {subregion}")
    if is_bootstrap and evaluation_metric == "rmse_pct_improvement":
        title_parts[0] += " (with 95% CI)"
    ax.set_title('\n'.join(title_parts), fontsize=20, pad=15)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    ax.tick_params(axis='both', labelsize=20)
    
    # Create legends (for all evaluation metrics)
    if evaluation_metric == "raw_values":
        # For raw_values, we need both forecast type and region legends
        # Forecast type legend (already created above as forecast_handles)
        legend1 = ax.legend(handles=forecast_handles, title="Forecast Type",
                           loc='upper left', fontsize=12)
        
        # Region legend
        region_handles = []
        for region in regions:
            if region in climate_region_colors:
                color = climate_region_colors[region]
            else:
                color = region_colors.get(region, '#1f77b4')
            region_handles.append(Line2D([0], [0], color=color, linewidth=3,
                                        label=region.replace('_', ' ').title()))
        
        legend2 = ax.legend(handles=region_handles, title="Region",
                           loc='upper left', bbox_to_anchor=(0, 0.75), fontsize=12)
        
        # Model legend if multiple models
        if len(models) > 1:
            model_handles = []
            for model in models:
                marker = model_markers.get(model, 'o')
                model_handles.append(Line2D([0], [0], color='black', marker=marker,
                                           linestyle='none', markersize=15,
                                           label=model.upper()))
            legend3 = ax.legend(handles=model_handles, title="Model",
                               loc='upper left', bbox_to_anchor=(0, 0.5), fontsize=12)
            ax.add_artist(legend3)
        
        ax.add_artist(legend1)
        ax.add_artist(legend2)
        
        # Style legends
        for legend in [legend1, legend2]:
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_alpha(0.95)
            legend.get_frame().set_edgecolor('gray')
            
    elif evaluation_metric in ["rmse_pct_improvement", "error_cutoff"]:
        region_handles = []
        for region in regions:
            if region in climate_region_colors:
                color = climate_region_colors[region]
            else:
                color = region_colors.get(region, '#1f77b4')
            region_handles.append(Line2D([0], [0], color=color, linewidth=3,
                                        label=region.replace('_', ' ').title()))
        
        model_handles = []
        for model in models:
            marker = model_markers.get(model, 'o')
            model_handles.append(Line2D([0], [0], color='black', marker=marker,
                                       linestyle='none', markersize=15,
                                       label=model.upper()))
        
        arch_handles = []
        for arch in nn_architectures:
            fillstyle = architecture_fillstyles.get(arch, 'full')
            arch_handles.append(Line2D([0], [0], color='black', marker='o',
                                      fillstyle=fillstyle, markersize=15,
                                      linestyle='none', label=arch.upper()))
        
        # Add line style legend for error_cutoff
        if evaluation_metric == "error_cutoff":
            line_handles = [
                Line2D([0], [0], color='gray', linestyle='--', linewidth=2,
                      label='Original', alpha=0.6),
                Line2D([0], [0], color='gray', linestyle='-', linewidth=2.5,
                      label='Corrected', alpha=0.75)
            ]
            
            # Adjust legend positions for error_cutoff
            legend1 = ax.legend(handles=region_handles, title="Region",
                               loc='upper left', fontsize=12)
            legend2 = ax.legend(handles=model_handles, title="Model",
                               loc='upper left', bbox_to_anchor=(0, 0.75), fontsize=12)
            legend3 = ax.legend(handles=arch_handles, title="Architecture",
                               loc='upper left', bbox_to_anchor=(0, 0.55), fontsize=12)
            legend4 = ax.legend(handles=line_handles, title="Forecast Type",
                               loc='upper left', bbox_to_anchor=(0, 0.35), fontsize=12)
            
            ax.add_artist(legend1)
            ax.add_artist(legend2)
            ax.add_artist(legend3)
            ax.add_artist(legend4)  # This was missing!
        else:
            # Original legend positions for rmse_pct_improvement
            legend1 = ax.legend(handles=region_handles, title="Region",
                               loc='lower right', bbox_to_anchor=(1, 0), fontsize=12)
            legend2 = ax.legend(handles=model_handles, title="Model",
                               loc='lower right', bbox_to_anchor=(1, 0.45), fontsize=12)
            legend3 = ax.legend(handles=arch_handles, title="Architecture",
                               loc='lower right', bbox_to_anchor=(1, 0.3), fontsize=12)
            
            ax.add_artist(legend1)
            ax.add_artist(legend2)
        
        # Style legends
        for legend in [legend1, legend2, legend3]:
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_alpha(0.95)
            legend.get_frame().set_edgecolor('gray')
    
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save figure
    if save_path is None:
        out_folder = os.path.join(dirs["fig"], models[0], "lead_time", "multi_region", subregion)
        os.makedirs(out_folder, exist_ok=True)
        
        bootstrap_suffix = "_bootstrap" if is_bootstrap else ""
        arch_suffix = "_".join(nn_architectures)
        
        if any(r in climate_region_colors for r in regions):
            region_type = "climate_zones"
        else:
            region_type = "geographic"
        
        training_vars = df['training_vars'].iloc[0] if len(df) > 0 else "unknown"
        
        # Add evaluation metric to filename
        metric_suffix = evaluation_metric
        fname = (f"leadtime_{metric_suffix}_{prediction_var}_trainedwith_{training_vars}_"
                f"{region_type}_{plot_type}_{arch_suffix}{bootstrap_suffix}.png")
        save_path = os.path.join(out_folder, fname)
    plt.show()
    exit() 
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Lead-time plot saved to: {save_path}")


def generate_summary_table_from_csv(
        csv_path,
        dirs,
        regions=None,
        subregion="2x2",
        nn_architecture="mlp",
        lead_times=None,
        save_latex=True
):
    """
    Generate summary statistics table from pre-calculated CSV.
    
    Parameters
    ----------
    csv_path : str
        Path to CSV file containing statistics
    dirs : dict
        Dictionary of directories (for saving table)
    regions : list
        List of regions to include. If None, uses all in CSV
    subregion : str
        Patch size to filter for
    nn_architecture : str
        Architecture to filter for
    lead_times : list
        Lead times to include. If None, uses all in CSV
    save_latex : bool
        Whether to save LaTeX table
        
    Returns
    -------
    pd.DataFrame
        Formatted summary table
    """
    
    # Load statistics
    df = pd.read_csv(csv_path)
    
    # Filter data
    df = df[df['subregion'] == subregion]
    df = df[df['architecture'] == nn_architecture]
    
    if regions is not None:
        df = df[df['region'].isin(regions)]
    
    if lead_times is not None:
        df = df[df['lead_time'].isin(lead_times)]
    
    # Get the first model in the dataframe (assuming we want consistent model)
    model = df['model'].iloc[0] if len(df) > 0 else "unknown"
    prediction_var = df['variable'].iloc[0] if len(df) > 0 else "unknown"
    
    # Create summary table
    summary_rows = []
    region_ground_truth_stats = {}
    
    for region in df['region'].unique():
        region_df = df[df['region'] == region].sort_values('lead_time')
        
        # Calculate region-wide ground truth statistics
        if 'ground_truth_mean' in region_df.columns and 'ground_truth_std' in region_df.columns:
            gt_mean = region_df['ground_truth_mean'].mean()  # Average across lead times
            gt_std = region_df['ground_truth_std'].mean()
            region_display = region.replace('_', ' ').title()
            region_ground_truth_stats[region_display] = f"{gt_mean:.2f} ({gt_std:.2f})"
        
        for _, row in region_df.iterrows():
            summary_row = {
                'Region': region.replace('_', ' ').title(),
                'Lead Time': f"{row['lead_time']}h",
                'RMSE (Orig)': row['rmse_original'],
                'RMSE Change': row['rmse_corrected'] - row['rmse_original'],
                'RMSE Improvement (%)': row['pct_improvement']
            }
            summary_rows.append(summary_row)
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Save LaTeX table if requested
    if save_latex:
        _create_latex_table_from_df(summary_df, prediction_var, nn_architecture, 
                                   subregion, dirs, model, region_ground_truth_stats)
    
    return summary_df


def _create_latex_table_from_df(df, prediction_var, nn_architecture, subregion, 
                               dirs, model, region_ground_truth_stats):
    """
    Creates a LaTeX table from DataFrame.
    """
    out_folder = os.path.join(dirs["fig"], model, "summary_stats")
    os.makedirs(out_folder, exist_ok=True)
    
    latex_lines = []
    latex_lines.append("\\begin{tabular}{llrr}")
    latex_lines.append("\\toprule")
    latex_lines.append("Region & Lead Time & RMSE & Improvement (\\%) \\\\")
    latex_lines.append("\\midrule")
    
    current_region = None
    region_rows = []
    
    for _, row in df.iterrows():
        region = row['Region']
        
        if region != current_region and current_region is not None:
            for i, region_row in enumerate(region_rows):
                if i == 0:
                    latex_lines.append(region_row)
                elif i == 1:
                    gt_stats = region_ground_truth_stats.get(current_region, "N/A")
                    latex_lines.append(f"\\textit{{Ground Truth: {gt_stats}}} & {region_row}")
                else:
                    latex_lines.append(f" & {region_row}")
            region_rows = []
        
        rmse_display = f"{row['RMSE (Orig)']:.3f} ({row['RMSE Change']:+.3f})"
        data_portion = f"{row['Lead Time']} & {rmse_display} & {row['RMSE Improvement (%)']:.1f} \\\\"
        
        if region != current_region:
            region_rows.append(f"{region} & {data_portion}")
            current_region = region
        else:
            region_rows.append(data_portion)
    
    if region_rows:
        for i, region_row in enumerate(region_rows):
            if i == 0:
                latex_lines.append(region_row)
            elif i == 1:
                gt_stats = region_ground_truth_stats.get(current_region, "N/A")
                latex_lines.append(f"\\textit{{Ground Truth: {gt_stats}}} & {region_row}")
            else:
                latex_lines.append(f" & {region_row}")
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    
    latex_str = "\n".join(latex_lines)
    
    variable_name = prediction_var.replace('_', ' ').title()
    filename = f"summary_stats_{prediction_var}_{nn_architecture}_{subregion}.tex"
    filepath = os.path.join(out_folder, filename)
    
    with open(filepath, 'w') as f:
        f.write(latex_str)
    
    print(f"\nSaved LaTeX table to: {filepath}")
    print(f"Table title: Summary Statistics for {variable_name}")



def main():
    dirs = setup_directories()

    # Two options for training and output variable combinations, uncomment the one you want to use
    # training_vars = ["2m_temperature"]
    # output_vars = ["2m_temperature"]
    # prediction_var = "2m_temperature"

    training_vars = ["10m_wind_speed"]
    output_vars = ["10m_wind_speed"]
    prediction_var = "10m_wind_speed"


    stat_path = "/Users/ohouck/globus/forecast_data/processed/forecast_improvement_stats.csv"

    plot_lead_time_from_csv(csv_path = stat_path,
        dirs=dirs,
        evaluation_metric = "raw_values",
        variable="2m_temperature",
        regions=["india", "amazon", "ethiopia", "british_columbia", "usa_south"],
        subregion="6x6",
        plot_type="pangu_nn",
        nn_architectures=["mlp"]
    )
    exit()
    plot_lead_time_from_csv(csv_path = stat_path,
        dirs=dirs,
        variable="10m_wind_speed",
        evaluation_metric = "error_cutoff",
        regions=["india", "amazon", "ethiopia", "british_columbia", "usa_south"],
        subregion="6x6",
        plot_type="pangu_nn",
        nn_architectures=["mlp"]
    )




    
    #=============================================
    # Subregion Comparison Plots
    #=============================================
    start = time.time()    
    generate_subregion_comparison_plots(
        dirs = dirs,
        train_start="2018-01-01",
        train_end="2021-12-31",
        test_start="2022-01-01",
        test_end="2022-12-31",
        model="pangu",
        training_output_vars=(training_vars, output_vars),
        prediction_var=prediction_var,
        nn_architecture=["mlp"],
        lead_time=216,
        simultaneous=True
    )
    end = time.time()
    time_minutes = (end - start) / 60
    print(f"Subregion comparison plots completed in {time_minutes:.2f} minutes.")

    #==============================================
    # Generate Maps
    #============================================== 
    regions = ["usa_south", "amazon", "india", "british_columbia", "ethiopia"]
    for region in regions:
        # MLP maps
        generate_map_plots(
            dirs=dirs,
            train_start="2018-01-01",
            train_end="2021-12-31",
            test_start="2022-01-01",
            test_end="2022-12-31",
            model="pangu",
            training_output_vars=(training_vars, output_vars),
            prediction_var=prediction_var,
            nn_architecture="mlp",
            region=region,
            subregion="2x2",
            lead_time=24,
            simultaneous=True
        )

        #===========================================
        # Generate time series plots
        #===========================================
        generate_time_series_plots(
            dirs=dirs,
            train_start="2018-01-01",
            train_end="2021-12-31",
            test_start="2022-01-01",
            test_end="2022-12-31",
            model="pangu",
            training_output_vars=(training_vars, output_vars),
            prediction_var=prediction_var,
            nn_architecture="mlp",
            region=region,
            subregion="2x2",
            lead_time=24,
            simultaneous=True
        )

    #============================================
    # summary stat tables
    #============================================
    print("Generating climate zone figure and table...")
    # clear cache
    load_zarr_cached.cache_clear()

    generate_summary_stat_table(
        dirs=dirs,
        train_start="2018-01-01",
        train_end="2021-12-31",
        test_start="2022-01-01",
        test_end="2022-12-31",
        model="pangu",
        training_output_vars=(training_vars, output_vars),
        prediction_var=prediction_var,
        nn_architecture="mlp",
        regions = ["tropical", "arid", "temperate"],
        subregion="2x2",
        bootstrap=True,
        lead_times=[24, 120, 216],  
        simultaneous=True
    )
    generate_summary_stat_table(
        dirs=dirs,
        train_start="2018-01-01",
        train_end="2021-12-31",
        test_start="2022-01-01",
        test_end="2022-12-31",
        model="pangu",
        training_output_vars=(training_vars, output_vars),
        prediction_var=prediction_var,
        nn_architecture="mlp",
        regions = ["india", "amazon", "ethiopia"],
        subregion="2x2",
        lead_times=[24, 120, 216],  # Multiple lead times
        simultaneous=True
    )
        

if __name__ == "__main__":
    main()