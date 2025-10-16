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

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from helper_funcs import setup_directories
from helper_funcs import generate_output_path

#######################
# Utility Functions
#######################

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
                                        lead_time=None, simultaneous=False,
                                        growing_season_only = False, alternative_loss_fn = None):
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
                        nn_architecture=arch,
                        growing_season_only = growing_season_only,
                        alternative_loss_fn = alternative_loss_fn
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
        simultaneous=False,
        growing_season_only = False,
        alternative_loss_fn = None
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
        nn_architecture=nn_architecture,
        growing_season_only=growing_season_only,
        alternative_loss_fn= alternative_loss_fn
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
        simultaneous=False,
        growing_season_only=False,
        alternative_loss_fn=None
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
        nn_architecture=nn_architecture,
        growing_season_only=growing_season_only,
        alternative_loss_fn=alternative_loss_fn


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

def _prepare_dataframe(csv_path, variable, regions, subregion, nn_architectures, 
                       model, growing_season_only = False, loss_type="rmse"):
    """
    Common data preparation for all plot types.
    """
    df = pd.read_csv(csv_path)
    
    # Filter by subregion
    df = df[df['subregion'] == subregion]

    # Filter by growing season flag
    df = df[df['growing_season_only'] == growing_season_only]

    # Filter by regions if specified
    if regions is not None:
        df = df[df['region'].isin(regions)]
    else:
        regions = df['region'].unique().tolist()
    
    if loss_type == "rmse":
        df = df[df['loss_fn'] == 'mse']
    elif loss_type == "extreme_heat":
        df = df[df['loss_fn'] == 'extreme_heat_loss']
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    
    # Filter by architectures
    df = df[df['architecture'].isin(nn_architectures)]
    # print number of rows
    
    # Filter by model
    df = df[df['model'] == model]
    
    # Filter to variable of interest
    df = df[df['variable'] == variable]
    
    return df, regions


def _get_color_schemes():
    """Return color schemes for regions, models, and architectures."""
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
    
    return region_colors, climate_region_colors, model_markers, architecture_fillstyles


def plot_rmse_improvement(csv_path, dirs, variable, model="pangu", 
                         regions=None, subregion="4x4", 
                         nn_architectures=["mlp"], growing_season_only=False,
                         loss_type="rmse", save_path=None):
    """
    Generate RMSE percentage improvement plots from pre-calculated statistics.
    
    Parameters
    ----------
    csv_path : str
        Path to CSV file containing statistics
    dirs : dict
        Dictionary of directories (for saving plots)
    variable : str
        Variable to plot (must match CSV column)
    model : str
        Model to plot: "pangu" or "ifs"
    regions : list
        List of regions to include in plot. If None, uses all in CSV
    subregion : str
        Patch size to filter for (default: "4x4")
    nn_architectures : list
        List of architectures to include: ["mlp"], ["unet"], or both
    growing_season_only : bool
        Whether to use results on model trained only on growing season
    loss_type : str
        Loss type to filter for (default: "rmse")
    save_path : str
        Custom save path. If None, auto-generates based on parameters
    """
    # Prepare data
    df, regions = _prepare_dataframe(csv_path, variable, regions, subregion, 
                                    nn_architectures, model, growing_season_only)
    
    if len(df) == 0:
        print(f"No data found for specified filters")
        return
    
    # Get unique values for plotting
    lead_times = sorted(df['lead_time'].unique())
    prediction_var = df['variable'].iloc[0]
    
    # Get color schemes
    region_colors, climate_region_colors, model_markers, architecture_fillstyles = _get_color_schemes()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot each region/architecture combination
    for region in regions:
        # Get color for region
        if region in climate_region_colors:
            color = climate_region_colors[region]
        else:
            color = region_colors.get(region, '#1f77b4')
        
        region_df = df[df['region'] == region]
        
        for arch in nn_architectures:
            arch_df = region_df[region_df['architecture'] == arch]
            
            if len(arch_df) == 0:
                continue
            
            # Sort by lead time
            arch_df = arch_df.sort_values('lead_time')
            
            # Get styles
            marker = model_markers.get(model, 'o')
            fillstyle = architecture_fillstyles.get(arch, 'full')

            if loss_type == "rmse":
                outcome_str = "rmse_pct_improvement"
            elif loss_type == "extreme_heat":
                outcome_str = "rmse_pct_improvement_extreme_heat"
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")
            
            # Plot neural network correction
            if outcome_str in arch_df.columns:
                x_pos = [lead_times.index(lt) for lt in arch_df['lead_time']]
                y_values = arch_df[outcome_str].values

                
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
                if f'{outcome_str}_ci_lower' in arch_df.columns:
                    ci_lower = arch_df[f'{outcome_str}_ci_lower'].values
                    ci_upper = arch_df[f'{outcome_str}_ci_upper'].values
                    ax.fill_between(x_pos, ci_lower, ci_upper,
                                   color=color,
                                   alpha=0.1,
                                   zorder=1)
    
    # Set axes
    ax.set_ylim(-28, 35)
    if loss_type == "rmse":
        ax.set_ylabel("RMSE Improvement (%)", fontsize=20)
    elif loss_type == "extreme_heat":
        ax.set_ylabel("RMSE Improvement for Extreme Heat (%)", fontsize=20)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Common x-axis settings
    ax.set_xticks(range(len(lead_times)))
    ax.set_xticklabels([f"{lt}h" for lt in lead_times])
    ax.set_xlabel("Forecast Lead Time", fontsize=20)
    
    # Title
    arch_str = "/".join([a.upper() for a in nn_architectures])
    regions_str = ", ".join(regions)
    is_bootstrap = df['bootstrap'].iloc[0] if 'bootstrap' in df.columns else False

    if loss_type == "rmse": 
        title_main = f"RMSE Improvement for {prediction_var.replace('_', ' ').title()} ({arch_str})"
    elif loss_type == "extreme_heat":
        title_main = f"RMSE Improvement for Extreme Heat {prediction_var.replace('_', ' ').title()} ({arch_str})"
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    if is_bootstrap:
        title_main += " (with 95% CI)"
    title_parts = [title_main, f"Model: {model.upper()}, Regions: {regions_str}, Patch Size: {subregion}"]
    ax.set_title('\n'.join(title_parts), fontsize=20, pad=15)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', labelsize=20)
    
    # Create legends
    region_handles = []
    for region in regions:
        if region in climate_region_colors:
            color = climate_region_colors[region]
        else:
            color = region_colors.get(region, '#1f77b4')
        region_handles.append(Line2D([0], [0], color=color, linewidth=3,
                                    label=region.replace('_', ' ').title()))
    
    arch_handles = []
    for arch in nn_architectures:
        fillstyle = architecture_fillstyles.get(arch, 'full')
        arch_handles.append(Line2D([0], [0], color='black', marker='o',
                                  fillstyle=fillstyle, markersize=15,
                                  linestyle='none', label=arch.upper()))
    
    legend1 = ax.legend(handles=region_handles, title="Region",
                       loc='lower right', bbox_to_anchor=(1, 0), fontsize=12)
    
    if len(nn_architectures) > 1:
        legend2 = ax.legend(handles=arch_handles, title="Architecture",
                           loc='lower right', bbox_to_anchor=(1, 0.3), fontsize=12)
        ax.add_artist(legend1)
    
    # Style legends
    for legend in ax.get_legend_handles_labels():
        if ax.get_legend():
            ax.get_legend().get_frame().set_facecolor('white')
            ax.get_legend().get_frame().set_alpha(0.95)
            ax.get_legend().get_frame().set_edgecolor('gray')
    
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save figure
    if save_path is None:
        out_folder = os.path.join(dirs["fig"], model, "lead_time", "multi_region", subregion)
        os.makedirs(out_folder, exist_ok=True)
        
        bootstrap_suffix = "_bootstrap" if is_bootstrap else ""
        arch_suffix = "_".join(nn_architectures)
        
        if any(r in climate_region_colors for r in regions):
            region_type = "climate_zones"
        else:
            region_type = "geographic"
        
        training_vars = df['training_vars'].iloc[0] if 'training_vars' in df.columns else "unknown"

        if growing_season_only:
            grow_flag = "_growing_season"
        else:
            grow_flag = ""
        if loss_type == "rmse":
            model_str = model
        elif loss_type == "extreme_heat":
            model_str = f"{model}_extreme_heat"
        
        fname = (f"leadtime_rmse_improvement_{prediction_var}_trainedwith_{training_vars}_"
                f"{region_type}_{model_str}_{arch_suffix}{bootstrap_suffix}{grow_flag}.png")
        save_path = os.path.join(out_folder, fname)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"RMSE improvement plot saved to: {save_path}")


def plot_raw_forecast_values(csv_path, dirs, variable, model="pangu",
                            regions=None, subregion="4x4",
                            nn_architectures=["mlp"], growing_season_only = False, save_path=None):
    """
    Plot raw forecast values as deviations from ground truth mean.
    
    Parameters
    ----------
    csv_path : str
        Path to CSV file containing statistics
    dirs : dict
        Dictionary of directories (for saving plots)
    variable : str
        Variable to plot (must match CSV column)
    model : str
        Model to plot: "pangu" or "ifs"
    regions : list
        List of regions to include in plot. If None, uses all in CSV
    subregion : str
        Patch size to filter for (default: "4x4")
    nn_architectures : list
        List of architectures to include: ["mlp"], ["unet"], or both
    growing_season_only : bool
        Whether to use results on model trained only on growing season
    save_path : str
        Custom save path. If None, auto-generates based on parameters
    """
    # Prepare data
    df, regions = _prepare_dataframe(csv_path, variable, regions, subregion,
                                    nn_architectures, model, growing_season_only=growing_season_only)
    
    if len(df) == 0:
        print(f"No data found for specified filters")
        return
    
    # Get unique values for plotting
    lead_times = sorted(df['lead_time'].unique())
    prediction_var = df['variable'].iloc[0]
    
    # Get color schemes
    region_colors, climate_region_colors, model_markers, architecture_fillstyles = _get_color_schemes()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
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
        
        for arch in nn_architectures:
            arch_df = region_df[region_df['architecture'] == arch]
            
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
                    
                    # Store mean for annotation
                    region_key = f"{region}_{model}_{arch}"
                    if variable == '2m_temperature':
                        region_means[region_key] = (region, overall_gt_mean - 273.15, 'C')
                    elif variable == '10m_wind_speed':
                        region_means[region_key] = (region, overall_gt_mean, 'm/s')
                    elif variable == 'total_precipitation':
                        region_means[region_key] = (region, overall_gt_mean, 'mm')
                    else:
                        region_means[region_key] = (region, overall_gt_mean, '')
                    
                    # Plot original forecast raw errors 
                    if 'mean_original_forecast' in arch_df.columns:
                        y_values = arch_df['mean_original_forecast'].values
                        if not np.all(np.isnan(y_values)):
                            y_values_error = y_values - ground_truth_values 
                            
                            label = 'Original Forecast Error' if not legend_added['original'] else None
                            if label:
                                legend_added['original'] = True
                            
                            ax.plot(x_pos, y_values_error,
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
                            y_values_dev = y_values - ground_truth_values
                            
                            label = 'Corrected Forecast Error' if not legend_added['corrected'] else None
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
    
    # Set ylabel based on variable
    if variable == '2m_temperature':
        ax.set_ylabel("Mean Temperature Error (C)", fontsize=20)
    elif variable == '10m_wind_speed':
        ax.set_ylabel("Mean Wind Speed Error (m/s)", fontsize=20)
    elif variable == 'total_precipitation':
        ax.set_ylabel("Mean Precipitation Error (mm)", fontsize=20)
    else:
        ax.set_ylabel(f"{variable.replace('_', ' ').title()} Deviation", fontsize=20)
    
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
    
    # Add annotations for mean values
    if region_means:
        annotation_lines = ["Ground Truth Means:"]
        
        region_annotations = {}
        for key, (region, mean_val, units) in region_means.items():
            if region not in region_annotations:
                region_annotations[region] = []
            region_annotations[region].append((mean_val, units))
        
        for region in sorted(region_annotations.keys()):
            values = region_annotations[region]
            avg_mean = np.mean([v[0] for v in values])
            units = values[0][1]
            
            if variable == '2m_temperature':
                annotation_lines.append(f"  {region.replace('_', ' ').title()}: {avg_mean:.1f}°{units}")
            else:
                annotation_lines.append(f"  {region.replace('_', ' ').title()}: {avg_mean:.2f} {units}")
        
        annotation_text = '\n'.join(annotation_lines)
        ax.text(0.02, 0.48, annotation_text,
               transform=ax.transAxes,
               fontsize=11,
               verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                        edgecolor='gray', alpha=0.9),
               family='monospace')
    
    # Common x-axis settings
    ax.set_xticks(range(len(lead_times)))
    ax.set_xticklabels([f"{lt}h" for lt in lead_times])
    ax.set_xlabel("Forecast Lead Time", fontsize=20)
    
    # Title
    arch_str = "/".join([a.upper() for a in nn_architectures])
    regions_str = ", ".join(regions)
    
    title_main = f"Forecast Values for {prediction_var.replace('_', ' ').title()} ({arch_str})"
    title_parts = [title_main, f"Model: {model.upper()}, Regions: {regions_str}, Patch Size: {subregion}"]
    ax.set_title('\n'.join(title_parts), fontsize=20, pad=15)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', labelsize=20)
    
    # Create legends
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
    
    legend1 = ax.legend(handles=forecast_handles, title="Forecast Type",
                       loc='lower left', bbox_to_anchor=(0.2,0), fontsize=12)
    
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
                       loc='lower left', bbox_to_anchor=(0, 0), fontsize=12)
    
    ax.add_artist(legend1)
    ax.add_artist(legend2)
    
    # Style legends
    for legend in [legend1, legend2]:
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.95)
        legend.get_frame().set_edgecolor('gray')
    
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save figure
    if save_path is None:
        out_folder = os.path.join(dirs["fig"], model, "lead_time", "multi_region", subregion)
        os.makedirs(out_folder, exist_ok=True)
        
        arch_suffix = "_".join(nn_architectures)
        
        if any(r in climate_region_colors for r in regions):
            region_type = "climate_zones"
        else:
            region_type = "geographic"
        
        if growing_season_only:
            grow_flag = "_growing_season"
        else:
            grow_flag = ""
        training_vars = df['training_vars'].iloc[0] if 'training_vars' in df.columns else "unknown"
        
        fname = (f"leadtime_raw_values_{prediction_var}_trainedwith_{training_vars}_"
                f"{region_type}_{model}_{arch_suffix}{grow_flag}.png")
        save_path = os.path.join(out_folder, fname)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Raw forecast values plot saved to: {save_path}")


def plot_error_cutoff(csv_path, dirs, variable, model="pangu",
                     regions=None, subregion="4x4",
                     nn_architectures=["mlp"], save_path=None, growing_season_only=False):
    """
    Plot percentage of forecasts exceeding error threshold.
    
    Parameters
    ----------
    csv_path : str
        Path to CSV file containing statistics
    dirs : dict
        Dictionary of directories (for saving plots)
    variable : str
        Variable to plot (must match CSV column)
    model : str
        Model to plot: "pangu" or "ifs"
    regions : list
        List of regions to include in plot. If None, uses all in CSV
    subregion : str
        Patch size to filter for (default: "4x4")
    nn_architectures : list
        List of architectures to include: ["mlp"], ["unet"], or both
    growing_season_only : bool
        Whether to use results on model trained only on growing season
    save_path : str
        Custom save path. If None, auto-generates based on parameters
    """
    # Prepare data
    df, regions = _prepare_dataframe(csv_path, variable, regions, subregion,
                                    nn_architectures, model, growing_season_only=growing_season_only)
    
    if len(df) == 0:
        print(f"No data found for specified filters")
        return
    
    # Get unique values for plotting
    lead_times = sorted(df['lead_time'].unique())
    prediction_var = df['variable'].iloc[0]
    
    # Get color schemes
    region_colors, climate_region_colors, model_markers, architecture_fillstyles = _get_color_schemes()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Extract error cutoff information if available
    error_cutoff_value = None
    error_cutoff_units = None
    if 'metadata' in df.columns:
        metadata_str = df['metadata'].iloc[0]
        if pd.notna(metadata_str) and 'Error cutoff:' in metadata_str:
            parts = metadata_str.split(':')[1].strip().split()
            if len(parts) >= 2:
                error_cutoff_value = parts[0].replace('>', '')
                error_cutoff_units = ' '.join(parts[1:])
    
    # Track if we've plotted anything
    has_data = False
    
    for region in regions:
        # Get color for region
        if region in climate_region_colors:
            color = climate_region_colors[region]
        else:
            color = region_colors.get(region, '#1f77b4')
        
        region_df = df[df['region'] == region]
        
        for arch in nn_architectures:
            arch_df = region_df[region_df['architecture'] == arch]
            
            if len(arch_df) == 0:
                continue
            
            # Sort by lead time
            arch_df = arch_df.sort_values('lead_time')
            
            # Get styles
            marker = model_markers.get(model, 'o')
            fillstyle = architecture_fillstyles.get(arch, 'full')
            
            # Create label only for the first plot of each type
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
    
    # Set y-axis
    ax.set_ylabel("Forecasts Exceeding Error Threshold (%)", fontsize=20)
    ax.set_ylim(bottom=0)
    
    # Add annotation for cutoff value
    if error_cutoff_value and error_cutoff_units:
        annotation_text = f"Error Threshold: >{error_cutoff_value} {error_cutoff_units}"
        ax.text(0.02, 0.98, annotation_text,
               transform=ax.transAxes,
               fontsize=14,
               verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
    
    # Add subtle grid for y-axis
    ax.yaxis.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
    
    # Common x-axis settings
    ax.set_xticks(range(len(lead_times)))
    ax.set_xticklabels([f"{lt}h" for lt in lead_times])
    ax.set_xlabel("Forecast Lead Time", fontsize=20)
    
    # Title
    arch_str = "/".join([a.upper() for a in nn_architectures])
    regions_str = ", ".join(regions)
    
    title_main = f"Error Frequency for {prediction_var.replace('_', ' ').title()} ({arch_str})"
    title_parts = [title_main, f"Model: {model.upper()}, Regions: {regions_str}, Patch Size: {subregion}"]
    ax.set_title('\n'.join(title_parts), fontsize=20, pad=15)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', labelsize=20)
    
    # Create legends
    region_handles = []
    for region in regions:
        if region in climate_region_colors:
            color = climate_region_colors[region]
        else:
            color = region_colors.get(region, '#1f77b4')
        region_handles.append(Line2D([0], [0], color=color, linewidth=3,
                                    label=region.replace('_', ' ').title()))
    
    arch_handles = []
    for arch in nn_architectures:
        fillstyle = architecture_fillstyles.get(arch, 'full')
        arch_handles.append(Line2D([0], [0], color='black', marker='o',
                                  fillstyle=fillstyle, markersize=15,
                                  linestyle='none', label=arch.upper()))
    
    line_handles = [
        Line2D([0], [0], color='gray', linestyle='--', linewidth=2,
              label='Original', alpha=0.6),
        Line2D([0], [0], color='gray', linestyle='-', linewidth=2.5,
              label='Corrected', alpha=0.75)
    ]
    
    # Position legends
    legend1 = ax.legend(handles=region_handles, title="Region",
                       loc='upper left', bbox_to_anchor=(0, 0.85), fontsize=12)
    
    legend2 = ax.legend(handles=line_handles, title="Forecast Type",
                       loc='upper left', bbox_to_anchor=(0, 0.6), fontsize=12)
    
    if len(nn_architectures) > 1:
        legend3 = ax.legend(handles=arch_handles, title="Architecture",
                           loc='upper left', bbox_to_anchor=(0, 0.25), fontsize=12)
        ax.add_artist(legend3)
    
    ax.add_artist(legend1)
    ax.add_artist(legend2)
    
    # Style legends
    legends = [legend1, legend2]
    if len(nn_architectures) > 1:
        legends.append(legend3)
    
    for legend in legends:
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.95)
        legend.get_frame().set_edgecolor('gray')
    
    # Remove spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save figure
    if save_path is None:
        out_folder = os.path.join(dirs["fig"], model, "lead_time", "multi_region", subregion)
        os.makedirs(out_folder, exist_ok=True)
        
        arch_suffix = "_".join(nn_architectures)
        
        if any(r in climate_region_colors for r in regions):
            region_type = "climate_zones"
        else:
            region_type = "geographic"
        
        if growing_season_only:
            grow_flag = "_growing_season"
        else:
            grow_flag = ""
        
        training_vars = df['training_vars'].iloc[0] if 'training_vars' in df.columns else "unknown"
        
        fname = (f"leadtime_error_cutoff_{prediction_var}_trainedwith_{training_vars}_"
                f"{region_type}_{model}_{arch_suffix}{grow_flag}.png")
        save_path = os.path.join(out_folder, fname)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Error cutoff plot saved to: {save_path}")


def main():
    dirs = setup_directories()

    stat_path = "/Users/ohouck/globus/forecast_data/processed/forecast_improvement_stats.csv"

    nn_architectures = ["mlp"]
    variable_list = ["2m_temperature", "10m_wind_speed", "total_precipitation"]
    model_list = ["pangu", "ifs", "aifs"]
    geo_regions = ["india", "amazon", "ethiopia", "british_columbia", "usa_south"]
    climate_regions = ["tropical", "arid", "temperate"]
    growing_season_flags = [True, False]
    for var in variable_list:
        for model in model_list:
            for gs_flag in growing_season_flags:

                if model == "aifs" and not gs_flag:
                    # aifs results are only for growing season
                    continue

                plot_rmse_improvement(csv_path = stat_path,
                    dirs=dirs,
                    variable=var,
                    model=model,
                    regions=geo_regions,
                    subregion="6x6",
                    nn_architectures=nn_architectures,
                    growing_season_only=gs_flag,
                    loss_type="extreme_heat" # options: "rmse", "extreme_heat"
                )
                # plot_raw_forecast_values(csv_path = stat_path,
                #     dirs=dirs,
                #     variable=var,
                #     model=model,
                #     regions=climate_regions,
                #     subregion="2x2",
                #     nn_architectures=nn_architectures,
                #     growing_season_only=gs_flag
                # )
                # plot_error_cutoff(csv_path = stat_path,
                #     dirs=dirs,
                #     variable=var,
                #     model=model,
                #     regions=climate_regions,
                #     subregion="2x2",
                #     nn_architectures=nn_architectures,
                #     growing_season_only=gs_flag
                # )
    exit()
    for var in variable_list:
        for model in model_list:
            for gs_flag in growing_season_flags:

                if model == "aifs" and not gs_flag:
                    # aifs results are only for growing season
                    continue

                plot_rmse_improvement(csv_path = stat_path,
                    dirs=dirs,
                    variable=var,
                    model=model,
                    regions=["temperate", "arid", "tropical"],
                    subregion="2x2",
                    nn_architectures=nn_architectures,
                    growing_season_only=gs_flag
                )
                plot_raw_forecast_values(csv_path = stat_path,
                    dirs=dirs,
                    variable=var,
                    model=model,
                    regions=["temperate", "arid", "tropical"],
                    subregion="2x2",
                    nn_architectures=nn_architectures,
                    growing_season_only=gs_flag
                )
                plot_error_cutoff(csv_path = stat_path,
                    dirs=dirs,
                    variable=var,
                    model=model,
                    regions=["temperate", "arid", "tropical"],
                    subregion="2x2",
                    nn_architectures=nn_architectures,
                    growing_season_only=gs_flag
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