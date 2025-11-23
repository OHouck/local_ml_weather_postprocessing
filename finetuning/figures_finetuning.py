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

from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats

import time
from types import SimpleNamespace

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from helper_funcs import setup_directories
from helper_funcs import generate_output_path
from finetuning.process_forecasts import calculate_rmse, calculate_extreme_heat_rmse

# Suppress Zarr warnings (e.g., for .DS_Store files)
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='zarr')

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
    # Handle both numpy arrays and xarray DataArrays
    if hasattr(predictions, 'values'):
        predictions = predictions.values
    if hasattr(ground_truth, 'values'):
        ground_truth = ground_truth.values
    return float(np.sqrt(((predictions - ground_truth) ** 2).mean()))


def calculate_improvement_percentage(rmse_original, rmse_corrected):
    """Calculate percentage improvement in RMSE."""
    if rmse_original == 0:
        return 0
    return (rmse_original - rmse_corrected) / rmse_original * 100


def filter_patch_zarr_files(zone_dir, variable, train_start="2018-01-01", train_end="2021-12-31",
                            test_start="2022-01-01", test_end="2022-12-31",
                            nn_architecture="mlp", subregion="2x2", alternate_loss_fn=None):
    """
    Filter zarr files in a zone directory to match specific model configuration.

    This matches the file naming convention from generate_output_path(), which creates files like:
    train_{training_vars}_test_{output_vars}_dim{subregion}_leadtime_{lead_times}_
    train{train_start}-{train_end}_test{test_start}-{test_end}_{nn_architecture}[_{alternate_loss_fn}]_{zone_type}_bs{batch_num}.zarr

    Parameters
    ----------
    zone_dir : str
        Path to zone directory containing zarr files
    variable : str
        Variable name to search for in filename (e.g., "2m_temperature")
    train_start : str
        Training start date (default: "2018-01-01")
    train_end : str
        Training end date (default: "2021-12-31")
    test_start : str
        Test start date (default: "2022-01-01")
    test_end : str
        Test end date (default: "2022-12-31")
    nn_architecture : str
        Neural network architecture: "mlp" or "unet" (default: "mlp")
    subregion : str
        Subregion size pattern (default: "2x2")
    alternate_loss_fn : str, optional
        Alternate loss function name if used (default: None)

    Returns
    -------
    list
        List of matching zarr file paths
    """
    if not os.path.exists(zone_dir):
        return []

    # Get all zarr files containing the variable
    all_files = glob.glob(os.path.join(zone_dir, f"*{variable}*.zarr"))

    # Build the pattern to match based on generate_output_path structure
    # Pattern: train_{var}_test_{var}_dim{subregion}_leadtime_*_train{dates}_test{dates}_{arch}[_{loss}]_{zone}_bs*.zarr
    dates_str = f"train{train_start}-{train_end}_test{test_start}-{test_end}"
    dim_str = f"dim{subregion}"

    # Build architecture string
    if alternate_loss_fn:
        arch_str = f"{nn_architecture}_{alternate_loss_fn}"
    else:
        arch_str = nn_architecture

    # Filter files that match the model configuration
    matching_files = []
    for file_path in all_files:
        basename = os.path.basename(file_path)

        # Check if file matches the expected pattern
        if (dates_str in basename and
            dim_str in basename and
            f"_{arch_str}_" in basename):
            matching_files.append(file_path)

    return matching_files


def load_region_data(
    dirs,
    model="pangu",
    variable="10m_wind_speed",
    regions=None,
    train_start="2018-01-01",
    train_end="2021-12-31",
    test_start="2022-01-01",
    test_end="2022-12-31",
    nn_architecture="mlp",
    subregion="6x6",
    alternate_loss_fn=None,
    lead_times=None,
    sdor_da=None
):
    """
    Load and process zarr data for multiple regions.

    This function consolidates the data loading logic that was previously duplicated
    in map_global_improvements and plot_scatter_forecast_improvement.

    Parameters
    ----------
    dirs : dict
        Dictionary of directories
    model : str
        Model to use: "pangu" or "ifs"
    variable : str
        Variable to plot: "2m_temperature", "10m_wind_speed", or "total_precipitation"
    regions : list, optional
        List of regions to include. If None, uses:
        ["asia", "africa", "north_america", "south_america", "europe", "oceania"]
    train_start : str
        Training start date
    train_end : str
        Training end date
    test_start : str
        Test start date
    test_end : str
        Test end date
    nn_architecture : str
        Neural network architecture: "mlp" or "unet"
    subregion : str
        Subregion size pattern (e.g., "6x6")
    alternate_loss_fn : str, optional
        Alternate loss function name if used
    lead_times : list, optional
        List of lead times to process. If None, uses [24, 120, 216]
    sdor_da : xarray.DataArray, optional
        Standard deviation of orography data (only needed if calculating sdor)

    Returns
    -------
    dict
        Dictionary keyed by lead time, with each value being a list of patch data dicts.
        Each patch data dict contains:
        - lat_min, lat_max, lon_min, lon_max: spatial bounds
        - improvement: percent improvement in RMSE
        - rmse_original: original forecast RMSE
        - rmse_corrected: corrected forecast RMSE
        - region: region name
        - distance_from_equator: absolute value of center latitude
        - center_lat: center latitude (for hemisphere determination)
        - sdor: mean standard deviation of orography (if sdor_da provided)
        - ds: xarray Dataset (for pixel-level access)
    """

    # Default regions (continents)
    if regions is None:
        regions = ["asia", "africa", "north_america", "south_america", "europe", "oceania"]

    # Default lead times
    if lead_times is None:
        lead_times = [24, 120, 216]

    # Base directory for the model
    base_dir = os.path.join(dirs["raw"], "..", "processed", "finetuning_output", model)

    print(f"\nLoading region data for {model.upper()} - {variable}")
    print(f"Searching in: {base_dir}")
    print(f"Regions: {regions}")
    print(f"Lead times: {lead_times}")
    print(f"Model config: {nn_architecture}, subregion={subregion}, "
          f"train={train_start} to {train_end}, test={test_start} to {test_end}")
    if alternate_loss_fn:
        print(f"Alternate loss function: {alternate_loss_fn}")

    # Collect patch data for each lead time
    all_patch_data = {lt: [] for lt in lead_times}

    for region in regions:
        region_dir = os.path.join(base_dir, region)

        if not os.path.exists(region_dir):
            print(f"Warning: Directory not found: {region_dir}")
            continue

        # Find zarr files matching the model configuration
        zarr_files = filter_patch_zarr_files(
            region_dir, variable, train_start, train_end,
            test_start, test_end, nn_architecture, subregion, alternate_loss_fn
        )

        print(f"\nProcessing {region}: found {len(zarr_files)} matching files")

        for zarr_file in zarr_files:
            try:
                # Load dataset
                ds = xr.open_zarr(zarr_file, consolidated=False)

                # Get spatial bounds
                lat_min = float(ds.latitude.min())
                lat_max = float(ds.latitude.max())
                lon_min = float(ds.longitude.min())
                lon_max = float(ds.longitude.max())

                # Convert longitude to 0-360 if needed
                lon_min_360 = lon_min + 360 if lon_min < 0 else lon_min
                lon_max_360 = lon_max + 360 if lon_max < 0 else lon_max

                # Calculate center latitude for distance from equator
                center_lat = (lat_min + lat_max) / 2
                distance_from_equator = abs(center_lat)

                # Calculate mean sdor for this patch if needed
                mean_sdor = None
                if sdor_da is not None:
                    try:
                        # Ensure longitude is in the same coordinate system as sdor data
                        sdor_lon_min = float(sdor_da.longitude.min())
                        sdor_lon_max = float(sdor_da.longitude.max())

                        # Convert patch longitude to match sdor coordinate system
                        patch_lon_min = lon_min
                        patch_lon_max = lon_max

                        if sdor_lon_min >= 0 and sdor_lon_max > 180:
                            # sdor uses 0-360, convert patch coords if needed
                            if patch_lon_min < 0:
                                patch_lon_min += 360
                            if patch_lon_max < 0:
                                patch_lon_max += 360
                        else:
                            # sdor uses -180 to 180, convert patch coords if needed
                            if patch_lon_min > 180:
                                patch_lon_min -= 360
                            if patch_lon_max > 180:
                                patch_lon_max -= 360

                        # Handle latitude slice order (ERA5 often has descending latitude)
                        sdor_lat = sdor_da.latitude.values
                        if sdor_lat[0] > sdor_lat[-1]:
                            # Descending latitude - use max to min for slice
                            patch_sdor = sdor_da.sel(
                                latitude=slice(lat_max, lat_min),
                                longitude=slice(patch_lon_min, patch_lon_max)
                            )
                        else:
                            # Ascending latitude - use min to max for slice
                            patch_sdor = sdor_da.sel(
                                latitude=slice(lat_min, lat_max),
                                longitude=slice(patch_lon_min, patch_lon_max)
                            )

                        # Calculate mean, ignoring NaN values
                        if patch_sdor.size > 0:
                            mean_sdor = float(patch_sdor.mean(skipna=True))
                            # Check if result is NaN
                            if np.isnan(mean_sdor):
                                print(f"  Warning: sdor calculation returned NaN for patch at "
                                      f"lat=[{lat_min:.2f}, {lat_max:.2f}], lon=[{lon_min:.2f}, {lon_max:.2f}]")
                                mean_sdor = None
                        else:
                            print(f"  Warning: No sdor data found for patch at "
                                  f"lat=[{lat_min:.2f}, {lat_max:.2f}], lon=[{lon_min:.2f}, {lon_max:.2f}]")
                            mean_sdor = None
                    except Exception as e:
                        print(f"  Warning: Could not calculate sdor for patch at "
                              f"lat=[{lat_min:.2f}, {lat_max:.2f}], lon=[{lon_min:.2f}, {lon_max:.2f}]: {e}")
                        mean_sdor = None

                # Process each lead time
                for lead_time in lead_times:
                    var_suffix = f"_lt{lead_time}h"

                    ground_truth = ds[f"{variable}_ground_truth{var_suffix}"]
                    original = ds[f"{variable}_original{var_suffix}"]
                    corrected = ds[f"{variable}_corrected{var_suffix}"]

                    # Flatten arrays and remove NaNs
                    gt_flat = ground_truth.values.flatten()
                    orig_flat = original.values.flatten()
                    corr_flat = corrected.values.flatten()

                    # Remove NaN values
                    mask = ~(np.isnan(gt_flat) | np.isnan(orig_flat) | np.isnan(corr_flat))
                    gt_flat = gt_flat[mask]
                    orig_flat = orig_flat[mask]
                    corr_flat = corr_flat[mask]

                    # Calculate RMSE
                    rmse_original = calculate_rmse(orig_flat, gt_flat)
                    rmse_corrected = calculate_rmse(corr_flat, gt_flat)
                    pct_improvement = calculate_improvement_percentage(rmse_original, rmse_corrected)

                    # Store patch data for this lead time
                    all_patch_data[lead_time].append({
                        'lat_min': lat_min,
                        'lat_max': lat_max,
                        'lon_min': lon_min_360,
                        'lon_max': lon_max_360,
                        'improvement': pct_improvement,
                        'region': region,
                        'rmse_original': rmse_original,
                        'rmse_corrected': rmse_corrected,
                        'distance_from_equator': distance_from_equator,
                        'center_lat': center_lat,
                        'sdor': mean_sdor,
                        'ds': ds,
                        'lead_time': lead_time
                    })

            except Exception as e:
                print(f"  Error processing {os.path.basename(zarr_file)}: {e}")
                continue

    # Check if we have data
    if all(len(all_patch_data[lt]) == 0 for lt in lead_times):
        print("No patch data found for any lead time!")
        return None

    # Print summary for each lead time
    for lt in lead_times:
        if all_patch_data[lt]:
            improvements = [p['improvement'] for p in all_patch_data[lt]]
            print(f"\nLead time {lt}h: {len(all_patch_data[lt])} patches")
            print(f"  Improvement range: {min(improvements):.1f}% to {max(improvements):.1f}%")

    return all_patch_data


def map_global_improvements(
    dirs,
    model="pangu",
    variable="10m_wind_speed",
    regions=None,
    save_dir=None,
    map_type="improvement",
    train_start="2018-01-01",
    train_end="2021-12-31",
    test_start="2022-01-01",
    test_end="2022-12-31",
    nn_architecture="mlp",
    subregion="6x6",
    alternate_loss_fn=None,
    pixel_level=False
):
    """
    Create global maps showing RMSE metrics for all post-processed patches.
    Generates 3 separate maps, one for each lead time (24h, 120h, 216h).

    Only processes zarr files that match the specified model configuration to ensure
    all patches are from the same training/testing setup.

    Parameters
    ----------
    dirs : dict
        Dictionary of directories
    model : str
        Model to use: "pangu" or "ifs"
    variable : str
        Variable to plot: "2m_temperature", "10m_wind_speed", or "total_precipitation"
    regions : list, optional
        List of regions to include. If None, uses:
        ["asia", "africa", "north_america", "south_america", "europe", "oceania"]
    save_dir : str, optional
        Custom save directory. If None, auto-generates based on parameters
    map_type : str, optional
        Type of map to create: "improvement" (percent improvement),
        "original" (original RMSE), or "corrected" (corrected RMSE).
        Default is "improvement".
    train_start : str, optional
        Training start date (default: "2018-01-01")
    train_end : str, optional
        Training end date (default: "2021-12-31")
    test_start : str, optional
        Test start date (default: "2022-01-01")
    test_end : str, optional
        Test end date (default: "2022-12-31")
    nn_architecture : str, optional
        Neural network architecture: "mlp" or "unet" (default: "mlp")
    subregion : str, optional
        Subregion size pattern (default: "6x6")
    alternate_loss_fn : str, optional
        Alternate loss function name if used (default: None)
    pixel_level : bool, optional
        If True, plot RMSE improvement for each quarter-degree pixel.
        If False, plot mean RMSE improvement for each region (default: False)

    Returns
    -------
    figs : dict
        Dictionary of created figures keyed by lead time
    """

    # Lead times to process
    lead_times = [24, 120, 216]

    # Determine what metric to display
    metric_name = {
        "improvement": "improvements",
        "original": "original RMSE",
        "corrected": "corrected RMSE"
    }.get(map_type, "improvements")

    plot_type = "pixel-level" if pixel_level else "region-mean"
    print(f"\nMapping global {metric_name} for {model.upper()} - {variable}")
    print(f"Map type: {map_type}, Plot type: {plot_type}")

    # Load region data using the helper function
    all_patch_data = load_region_data(
        dirs=dirs,
        model=model,
        variable=variable,
        regions=regions,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        nn_architecture=nn_architecture,
        subregion=subregion,
        alternate_loss_fn=alternate_loss_fn,
        lead_times=lead_times,
        sdor_da=None
    )

    if all_patch_data is None:
        return None

    # Create maps for each lead time
    figs = {}

    # Determine output directory
    if save_dir is None:
        out_folder = os.path.join(dirs["fig"], model, "global_maps")
    else:
        out_folder = save_dir
    os.makedirs(out_folder, exist_ok=True)

    # Create a separate map for each lead time
    for lead_time in lead_times:
        patch_data = all_patch_data[lead_time]

        if not patch_data:
            print(f"\nSkipping lead time {lead_time}h - no data available")
            continue

        print(f"\nCreating map for lead time {lead_time}h...")

        # Create the map
        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

        # Set global extent
        ax.set_global()

        # Add map features
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3, zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor='white', zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='black', zorder=2)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.3, edgecolor='gray', zorder=2)

        if pixel_level:
            # Pixel-level plotting: create global gridded dataset and plot as raster
            print(f"  Creating global gridded dataset for pixel-level plotting...")

            # Collect all unique lat/lon coordinates to determine global grid
            # Use numpy concatenate instead of list extend for better performance
            all_lats_list = [patch['ds'].latitude.values for patch in patch_data]
            all_lons_list = [patch['ds'].longitude.values for patch in patch_data]

            all_lats = np.concatenate(all_lats_list)
            all_lons = np.concatenate(all_lons_list)

            # Get unique sorted coordinates
            unique_lats = np.unique(all_lats)
            unique_lons = np.unique(all_lons)

            print(f"  Global grid: {len(unique_lats)} latitudes × {len(unique_lons)} longitudes")

            # Create empty global grid filled with NaN
            global_improvement = np.full((len(unique_lats), len(unique_lons)), np.nan)

            # Helper function to find nearest indices (robust to floating point errors)
            def find_nearest_indices(values, array):
                """Find indices of nearest values in array for each value in values."""
                # For each value, find the index of the closest match in array
                indices = np.empty(len(values), dtype=int)
                for i, val in enumerate(values):
                    indices[i] = np.argmin(np.abs(array - val))
                return indices

            # Pre-allocate arrays for batch processing
            print(f"  Processing {len(patch_data)} patches...")

            # Fill in data from each patch - optimized version
            for i, patch in enumerate(patch_data):
                if (i + 1) % 50 == 0:  # Progress indicator
                    print(f"    Processed {i + 1}/{len(patch_data)} patches...")

                ds = patch['ds']
                var_suffix = f"_lt{lead_time}h"

                ground_truth = ds[f"{variable}_ground_truth{var_suffix}"]
                original = ds[f"{variable}_original{var_suffix}"]
                corrected = ds[f"{variable}_corrected{var_suffix}"]

                # Load data into memory for faster computation (critical for performance)
                if hasattr(ground_truth, 'load'):
                    ground_truth = ground_truth.load()
                    original = original.load()
                    corrected = corrected.load()

                # Compute pixel-wise RMSE over time dimension in one pass
                # Using numpy directly is faster than xarray for simple operations
                gt_data = ground_truth.values  # Shape: (time, lat, lon)
                orig_data = original.values
                corr_data = corrected.values

                # Vectorized RMSE computation
                mse_original = np.mean((orig_data - gt_data) ** 2, axis=0)
                mse_corrected = np.mean((corr_data - gt_data) ** 2, axis=0)

                rmse_original_pixel = np.sqrt(mse_original)
                rmse_corrected_pixel = np.sqrt(mse_corrected)

                # Compute improvement percentage for each pixel
                # Add small epsilon to avoid division by zero
                improvement_pixel = ((rmse_original_pixel - rmse_corrected_pixel) /
                                   (rmse_original_pixel + 1e-10) * 100)

                # Get patch coordinates
                patch_lats = ds.latitude.values
                patch_lons = ds.longitude.values

                # Find nearest indices using robust method (handles floating point precision)
                # This fixes smearing artifacts caused by searchsorted using insertion points
                lat_indices = find_nearest_indices(patch_lats, unique_lats)
                lon_indices = find_nearest_indices(patch_lons, unique_lons)

                # Create meshgrid of indices
                lat_idx_grid, lon_idx_grid = np.meshgrid(lat_indices, lon_indices, indexing='ij')

                # Assign values using fancy indexing (vectorized operation)
                global_improvement[lat_idx_grid, lon_idx_grid] = improvement_pixel

            print(f"  All {len(patch_data)} patches processed.")

            # Calculate statistics using nanXXX functions (faster than masking)
            n_pixels = int(np.count_nonzero(~np.isnan(global_improvement)))

            if n_pixels == 0:
                print(f"  No valid pixel data for lead time {lead_time}h!")
                continue

            # Use nanXXX functions - they're optimized and faster than manual masking
            vmin = float(np.nanmin(global_improvement))
            vmax = float(np.nanmax(global_improvement))
            mean_val = float(np.nanmean(global_improvement))
            median_val = float(np.nanmedian(global_improvement))
            std_val = float(np.nanstd(global_improvement))

            print(f"  Pixel improvement range: {vmin:.1f}% to {vmax:.1f}%")
            print(f"  Valid pixels: {n_pixels}")

            # Create colormap
            if vmin >= 0:
                norm = plt.Normalize(vmin=0, vmax=vmax)
            elif vmax <= 0:
                norm = plt.Normalize(vmin=vmin, vmax=0)
            else:
                norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
            cmap = plt.cm.RdBu  # Red for negative, Blue for positive

            # Plot using pcolormesh - keep original longitude coordinates
            # Let cartopy handle the coordinate transformations
            # This fixes the smearing issue by not manually converting coordinates
            mesh = ax.pcolormesh(
                unique_lons, unique_lats, global_improvement,
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                norm=norm,
                shading='auto',
                rasterized=True,  # Rasterize for better performance with large grids
                zorder=1
            )

            # Add black boxes showing patch boundaries
            print(f"  Adding patch boundary boxes...")
            for patch in patch_data:
                lat_min = patch['lat_min']
                lat_max = patch['lat_max']
                lon_min = patch['lon_min']
                lon_max = patch['lon_max']

                width = lon_max - lon_min
                height = lat_max - lat_min

                # Draw black boundary box
                boundary_rect = Rectangle(
                    (lon_min, lat_min),
                    width,
                    height,
                    facecolor='none',
                    edgecolor='black',
                    linewidth=0.5,
                    alpha=1.0,
                    transform=ccrs.PlateCarree(),
                    zorder=2
                )
                ax.add_patch(boundary_rect)

            # Statistics for title
            title_main = f"Global RMSE Improvement Map (Pixel-Level)"
            title_parts = [
                title_main,
                f"{model.upper()} - {variable.replace('_', ' ').title()} - {lead_time}h Lead Time",
                f"N = {n_pixels} pixels"
            ]

            stats_text = (
                f"Mean: {mean_val:.1f}%\n"
                f"Median: {median_val:.1f}%\n"
                f"Std: {std_val:.1f}%"
            )

        else:
            # Region-mean plotting: plot mean RMSE improvement for each region
            # Extract values based on map type
            if map_type == "improvement":
                values = [p['improvement'] for p in patch_data]
                value_key = 'improvement'
            elif map_type == "original":
                values = [p['rmse_original'] for p in patch_data]
                value_key = 'rmse_original'
            elif map_type == "corrected":
                values = [p['rmse_corrected'] for p in patch_data]
                value_key = 'rmse_corrected'
            else:
                raise ValueError(f"Invalid map_type: {map_type}. Must be 'improvement', 'original', or 'corrected'.")

            vmin = min(values)
            vmax = max(values)

            # Create colormap based on map type
            if map_type == "improvement":
                # For improvements, center at 0 with asymmetric bounds
                if vmin >= 0:
                    norm = plt.Normalize(vmin=0, vmax=vmax)
                elif vmax <= 0:
                    norm = plt.Normalize(vmin=vmin, vmax=0)
                else:
                    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
                cmap = plt.cm.RdBu
            else:
                # For RMSE values, use sequential colormap
                norm = plt.Normalize(vmin=vmin, vmax=vmax)
                cmap = plt.cm.YlOrRd

            # Plot each patch as a colored rectangle
            for patch in patch_data:
                lat_min = patch['lat_min']
                lat_max = patch['lat_max']
                lon_min = patch['lon_min']
                lon_max = patch['lon_max']
                value = patch[value_key]

                color = cmap(norm(value))
                width = lon_max - lon_min
                height = lat_max - lat_min

                rect = Rectangle(
                    (lon_min, lat_min),
                    width,
                    height,
                    facecolor=color,
                    edgecolor='black',
                    linewidth=0.3,
                    alpha=0.8,
                    transform=ccrs.PlateCarree(),
                    zorder=1
                )
                ax.add_patch(rect)

            # Statistics for title
            mean_val = np.mean(values)
            median_val = np.median(values)
            std_val = np.std(values)

            if map_type == "improvement":
                title_main = f"Global RMSE Improvement Map"
                stats_text = (
                    f"Mean: {mean_val:.1f}%\n"
                    f"Median: {median_val:.1f}%\n"
                    f"Std: {std_val:.1f}%"
                )
            elif map_type == "original":
                title_main = f"Global Original Forecast Error Map"
                stats_text = (
                    f"Mean: {mean_val:.3f}\n"
                    f"Median: {median_val:.3f}\n"
                    f"Std: {std_val:.3f}"
                )
            elif map_type == "corrected":
                title_main = f"Global Corrected Forecast Error Map"
                stats_text = (
                    f"Mean: {mean_val:.3f}\n"
                    f"Median: {median_val:.3f}\n"
                    f"Std: {std_val:.3f}"
                )

            title_parts = [
                title_main,
                f"{model.upper()} - {variable.replace('_', ' ').title()} - {lead_time}h Lead Time",
                f"N = {len(patch_data)} regions"
            ]

        # Add gridlines
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                          linewidth=0.5, alpha=0.5, linestyle='--', zorder=3)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size': 10}

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.05, shrink=0.6)

        # Set colorbar label
        if pixel_level or map_type == "improvement":
            cbar.set_label('RMSE Improvement (%)', fontsize=14, weight='bold')
        elif map_type == "original":
            cbar.set_label('Original Forecast RMSE', fontsize=14, weight='bold')
        elif map_type == "corrected":
            cbar.set_label('Corrected Forecast RMSE', fontsize=14, weight='bold')
        cbar.ax.tick_params(labelsize=12)

        # Add title
        ax.set_title('\n'.join(title_parts), fontsize=16, weight='bold', pad=20)

        # Add statistics box
        ax.text(0.02, 0.98, stats_text,
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                         edgecolor='black', alpha=0.8),
                family='monospace',
                zorder=10)

        plt.tight_layout()

        # Save figure
        if pixel_level:
            fname = f"global_improvement_map_pixel_{variable}_{model}_lt{lead_time}.png"
        elif map_type == "improvement":
            fname = f"global_improvement_map_{variable}_{model}_lt{lead_time}.png"
        elif map_type == "original":
            fname = f"global_original_rmse_map_{variable}_{model}_lt{lead_time}.png"
        elif map_type == "corrected":
            fname = f"global_corrected_rmse_map_{variable}_{model}_lt{lead_time}.png"

        save_path = os.path.join(out_folder, fname)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")

        # Store figure
        figs[lead_time] = fig

        # Close to free memory
        plt.close(fig)

    map_type_str = "pixel-level improvement" if pixel_level else {
        "improvement": "improvement",
        "original": "original RMSE",
        "corrected": "corrected RMSE"
    }.get(map_type, map_type)
    print(f"\nAll {len(figs)} global {map_type_str} maps created successfully!")
    return figs


def generate_subregion_comparison_plots(dirs, train_start, train_end, test_start,
                                        test_end, model, training_output_vars,
                                        prediction_var, nn_architecture=["mlp"],
                                        lead_time=None, simultaneous=False,
                                        growing_season_only = False, alternate_loss_fn = None):
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
                        alternate_loss_fn = alternate_loss_fn
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
        alternate_loss_fn = None
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
        alternate_loss_fn=alternate_loss_fn
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
        alternate_loss_fn=None
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
        alternate_loss_fn=alternate_loss_fn


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
                       model, growing_season_only = False, loss_fn="mse"):
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
    
    print(loss_fn)
    df = df[df['loss_fn'] == loss_fn]
    
    # Filter by architectures
    df = df[df['architecture'].isin(nn_architectures)]

    
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
        'corn_belt': '#90EE90', 
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

    topographic_region_colors = {
        'flat': '#2E7D32',
        'hilly': '#FFD54F',
        'mountainous': '#6D4C41'
    }
    
    model_markers = {
        'pangu': 'o',
        'ifs': '^'
    }
    
    architecture_fillstyles = {
        'mlp': 'full',
        'unet': 'none'
    }
    
    return region_colors, climate_region_colors, topographic_region_colors, model_markers, architecture_fillstyles


def plot_rmse_improvement(csv_path, dirs, variable, model="pangu", 
                         regions=None, subregion="6x6", 
                         nn_architectures=["mlp"], growing_season_only=False,
                         loss_trained_on="mse", evaluation_loss = "rmse", save_path=None):
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
    loss_trained_on: str
        Loss function used to train the model: "mse", "extreme_heat"
    evaluation_loss: str
        Loss function used for evaluation/plotting: "rmse", "extreme_heat",
    save_path : str
        Custom save path. If None, auto-generates based on parameters
    """
    # choose loss function used to train model
    if loss_trained_on == "mse":
        loss_fn = "mse"
    elif loss_trained_on == "extreme_heat":
        loss_fn = "extreme_heat_loss"
    else:
        raise ValueError(f"Unknown loss_trained_on: {loss_trained_on}")

    # Prepare data
    df, regions = _prepare_dataframe(csv_path, variable, regions, subregion, 
                                    nn_architectures, model, growing_season_only,
                                    loss_fn=loss_fn)
    # metric on which to plot improvement
    if evaluation_loss == "rmse":
        outcome_str = "rmse_pct_improvement"
    elif evaluation_loss == "extreme_heat":
        outcome_str = "rmse_pct_improvement_extreme_heat"

   # helpful debug print 
    # cols_to_keep = ['region', 'lead_time', 'architecture', outcome_str, 'subregion', 'model', 'output_vars', "loss_fn", "growing_season_only"] 
    # print(df[cols_to_keep])
    
    if len(df) == 0:
        print(f"No data found for specified filters")
        return
    
    
    # Get unique values for plotting
    lead_times = sorted(df['lead_time'].unique())
    prediction_var = df['variable'].iloc[0]
    
    # Get color schemes
    region_colors, climate_region_colors, topographic_region_colors, model_markers, architecture_fillstyles = _get_color_schemes()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Calculate bar width and positions
    n_regions = len(regions)
    n_architectures = len(nn_architectures)
    n_groups_per_leadtime = n_regions * n_architectures
    bar_width = 0.6 / n_groups_per_leadtime

    # Plot each region/architecture combination
    for region_idx, region in enumerate(regions):
        # Get color for region
        if region in climate_region_colors:
            color = climate_region_colors[region]
        elif region in topographic_region_colors:
            color = topographic_region_colors[region]
        else:
            color = region_colors.get(region, '#1f77b4')

        region_df = df[df['region'] == region]

        for arch_idx, arch in enumerate(nn_architectures):
            arch_df = region_df[region_df['architecture'] == arch]

            if len(arch_df) == 0:
                continue

            # Sort by lead time
            arch_df = arch_df.sort_values('lead_time')

            # Get styles
            fillstyle = architecture_fillstyles.get(arch, 'full')

            # Calculate alpha based on architecture (for visual distinction)
            alpha = 0.9 if fillstyle == 'full' else 0.6


            # Plot neural network correction as bars
            if outcome_str in arch_df.columns:
                y_values = arch_df[outcome_str].values

                # Calculate x positions for this region/arch combination
                group_offset = (region_idx * n_architectures + arch_idx) * bar_width
                x_pos = np.arange(len(lead_times)) + group_offset - (n_groups_per_leadtime * bar_width) / 2 + bar_width / 2

                # Use hatching pattern for unet to distinguish from mlp
                hatch = None if fillstyle == 'full' else '//'

                bars = ax.bar(x_pos, y_values,
                             width=bar_width,
                             color=color,
                             alpha=alpha,
                             edgecolor='black',
                             linewidth=0.5,
                             hatch=hatch,
                             zorder=3)

                # Add lighter, thinner bar for mean bias correction improvement (only for RMSE evaluation)
                if evaluation_loss == "rmse" and 'pct_improvement_mean_corrected' in arch_df.columns:
                    mean_corrected_values = arch_df['pct_improvement_mean_corrected'].values
                    # Make the bar thinner (60% of original width) and lighter
                    mean_bar_width = bar_width * 0.6
                    ax.bar(x_pos, mean_corrected_values,
                          width=mean_bar_width,
                          color=color,
                          alpha=alpha * 0.4,  # Lighter by reducing alpha
                          edgecolor='black',
                          linewidth=0.3,
                          hatch=hatch,
                          zorder=4)

                # Add error bars if confidence intervals are available
                if f'{outcome_str}_ci_lower' in arch_df.columns:
                    ci_lower = arch_df[f'{outcome_str}_ci_lower'].values
                    ci_upper = arch_df[f'{outcome_str}_ci_upper'].values
                    # Calculate error bar lengths
                    yerr_lower = y_values - ci_lower
                    yerr_upper = ci_upper - y_values
                    ax.errorbar(x_pos, y_values,
                               yerr=[yerr_lower, yerr_upper],
                               fmt='none',
                               ecolor='black',
                               elinewidth=1,
                               capsize=3,
                               capthick=1,
                               alpha=0.7,
                               zorder=5)
    
    # Set axes
    ax.set_ylim(-15, 35)
    if evaluation_loss == "rmse":
        ax.set_ylabel("RMSE Improvement (%)", fontsize=20)
    elif evaluation_loss == "extreme_heat":
        ax.set_ylabel("RMSE Improvement for Extreme Heat (%)", fontsize=20)
    else:
        raise ValueError(f"Unknown evaluation_loss: {evaluation_loss}")

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Common x-axis settings
    ax.set_xticks(range(len(lead_times)))
    ax.set_xticklabels([f"{lt}h" for lt in lead_times])
    ax.set_xlabel("Forecast Lead Time", fontsize=20)
    
    # Title
    arch_str = "/".join([a.upper() for a in nn_architectures])
    regions_str = ", ".join(regions)
    is_bootstrap = df['bootstrap'].iloc[0] if 'bootstrap' in df.columns else False

    if evaluation_loss == "rmse":
        title_main = f"RMSE Improvement for {prediction_var.replace('_', ' ').title()} ({arch_str})"
    elif evaluation_loss == "extreme_heat":
        title_main = f"RMSE Improvement for Extreme Heat {prediction_var.replace('_', ' ').title()} ({arch_str})"
    else:
        raise ValueError(f"Unknown evaluation_loss: {evaluation_loss}")

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
        elif region in topographic_region_colors:
            color = topographic_region_colors[region]
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
                       loc='lower right', bbox_to_anchor=(1, 0), fontsize=16, title_fontsize=16)
    
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

    # Add annotation explaining the transparent bars (only for RMSE evaluation)
    if evaluation_loss == "rmse":
        annotation_text = "Note: Lighter inner bars show improvement from simple mean debiasing"
        ax.text(0.02, 0.98, annotation_text,
               transform=ax.transAxes,
               fontsize=16,
               verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                        edgecolor='gray', alpha=0.8))

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
        elif any(r in topographic_region_colors for r in regions):
            region_type = "topographic_zones"
        else:
            region_type = "geographic"
        
        training_vars = df['training_vars'].iloc[0] if 'training_vars' in df.columns else "unknown"

        if growing_season_only:
            grow_flag = "_growing_season"
        else:
            grow_flag = ""

        if loss_trained_on == "mse" and evaluation_loss == "rmse":
            model_str = model
        elif loss_trained_on == "extreme_heat" and evaluation_loss == "extreme_heat":
            model_str = f"{model}_extreme_heat"
        elif loss_trained_on == "extreme_heat" and evaluation_loss == "rmse":   
            model_str = f"{model}_extreme_heat_train_rmse_eval"
        elif loss_trained_on == "mse" and evaluation_loss == "extreme_heat":
            model_str = f"{model}_mse_train_extreme_heat_eval"
        
        fname = (f"leadtime_rmse_improvement_{prediction_var}_trainedwith_{training_vars}_"
                f"{region_type}_{model_str}_{arch_suffix}{bootstrap_suffix}{grow_flag}.png")
        save_path = os.path.join(out_folder, fname)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"RMSE improvement plot saved to: {save_path}")


def plot_rmse_improvement_by_weather_bin(dirs, train_start, train_end, test_start, test_end,
                                         model, training_output_vars, variable, lead_time,
                                         regions=None, subregion="6x6",
                                         nn_architecture="mlp", loss_trained_on="mse",
                                         evaluation_loss="rmse",
                                         n_bins=10, growing_season_only=False,
                                         ground_truth_source="",
                                         save_path=None):
    """
    Plot RMSE improvement (or other metrics) binned by weather variable values.

    Creates a line plot with RMSE improvement (or other metric) on y-axis and
    binned weather variable values on x-axis. Each region gets its own line.
    Includes a translucent histogram at the bottom showing data density.

    Parameters
    ----------
    dirs : dict
        Dictionary of directories from setup_directories()
    train_start : str
        Training start date (YYYY-MM-DD)
    train_end : str
        Training end date (YYYY-MM-DD)
    test_start : str
        Test start date (YYYY-MM-DD)
    test_end : str
        Test end date (YYYY-MM-DD)
    model : str
        Model name: "pangu", "ifs", "aifs", etc.
    training_output_vars : tuple
        Tuple of (training_vars, output_vars) where each is a list
        Example: (['2m_temperature'], ['2m_temperature'])
    variable : str
        Variable to plot (e.g., "2m_temperature", "10m_wind_speed")
    lead_time : int
        Specific lead time in hours (e.g., 24, 72, 144)
    regions : list or None
        List of regions to include. If None, defaults to standard regions
    subregion : str
        Patch size identifier (default: "6x6")
    nn_architecture : str
        Architecture used: "mlp" or "unet"
    evaluation_loss : str
        Evaluation loss function. Options:
        - "rmse_pct_improvement": RMSE percentage improvement (default)
        - "extreme_heat": Extreme heat RMSE percentage improvement (uses weighted loss function)
        Additional metrics can be added as needed
    n_bins : int
        Number of bins for weather variable (default: 10)
    growing_season_only : bool
        Whether to use results from model trained only on growing season
    alternate_loss_fn : str or None
        Alternate loss function used (e.g., "extreme_heat_loss")
    ground_truth_source : str
        Ground truth source identifier (default: "")
    save_path : str
        Custom save path. If None, auto-generates based on parameters

    Returns
    -------
    None
        Saves plot to file
    """
    input_folder = dirs['input']
    training_vars, output_vars = training_output_vars
    training_vars = training_vars if isinstance(training_vars, (list, tuple)) else [training_vars]
    output_vars = output_vars if isinstance(output_vars, (list, tuple)) else [output_vars]

    # If regions not specified, use standard regions
    if regions is None:
        regions = ["usa_south", "british_columbia", "ethiopia", "amazon", "india"]

    # Get color schemes
    region_colors, climate_region_colors, topographic_region_colors, _, _ = _get_color_schemes()

    # Create figure with gridspec for main plot and histogram
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax_main = fig.add_subplot(gs[0])
    ax_hist = fig.add_subplot(gs[1], sharex=ax_main)

    # Storage for histogram data per region
    region_histogram_data = {}

    # Determine if temperature conversion is needed
    is_temperature = 'temperature' in variable.lower()

    # First pass: Load all data and calculate global min/max
    global_min, global_max = float('inf'), float('-inf')
    loaded_data = {}  # Cache loaded data to avoid redundant loading

    for region in regions:

        # Build path using generate_output_path
        if loss_trained_on == "extreme_heat":
            alternate_loss_fn= "extreme_heat_loss"
        else:
            alternate_loss_fn = None
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
            alternate_loss_fn=alternate_loss_fn,
            lead_time_hours=[24, 120, 216], # XX change if adding more lead times
            nn_architecture=nn_architecture,
            growing_season_only=growing_season_only,
            ground_truth_source=ground_truth_source
        )

        zarr_path = os.path.join(input_folder, generate_output_path(args))
        print(f"Loading data from: {zarr_path}")

        try:
            ds = load_zarr_cached(zarr_path)
            ground_truth, original, corrected, mean_corrected = extract_forecast_data(ds, variable, lead_time)
        except Exception as e:
            print(f"Warning: Could not load data for region {region}: {e}")
            continue

        # Flatten arrays
        gt_flat = ground_truth.values.flatten()
        orig_flat = original.values.flatten()
        corr_flat = corrected.values.flatten()

        # Remove NaN values
        valid_mask = ~(np.isnan(gt_flat) | np.isnan(orig_flat) | np.isnan(corr_flat))
        gt_flat = gt_flat[valid_mask]
        orig_flat = orig_flat[valid_mask]
        corr_flat = corr_flat[valid_mask]

        # Convert temperature from Kelvin to Celsius if needed
        if is_temperature:
            gt_flat = gt_flat - 273.15
            orig_flat = orig_flat - 273.15
            corr_flat = corr_flat - 273.15

        # Store data for this region (for calculating global bins and histogram)
        loaded_data[region] = {
            'gt': gt_flat,
            'orig': orig_flat,
            'corr': corr_flat
        }

        # Update global min/max
        global_min = min(global_min, gt_flat.min())
        global_max = max(global_max, gt_flat.max())

    # Create bins with equal spacing based on global min/max
    bin_edges = np.linspace(global_min, global_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Second pass: Calculate metrics and plot using cached data
    for region in regions:
        if region not in loaded_data:
            continue

        gt_flat = loaded_data[region]['gt']
        orig_flat = loaded_data[region]['orig']
        corr_flat = loaded_data[region]['corr']

        # Store for histogram (per region)
        region_histogram_data[region] = gt_flat

        # Bin the data
        bin_indices = np.digitize(gt_flat, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)  # Handle edge cases

        # Calculate metric for each bin
        metric_values = []
        for i in range(n_bins):
            bin_mask = bin_indices == i

            if np.sum(bin_mask) == 0:
                metric_values.append(np.nan)
                continue

            gt_bin = gt_flat[bin_mask]
            orig_bin = orig_flat[bin_mask]
            corr_bin = corr_flat[bin_mask]

            if evaluation_loss == "rmse":
                rmse_orig = np.sqrt(np.mean((orig_bin - gt_bin) ** 2))
                rmse_corr = np.sqrt(np.mean((corr_bin - gt_bin) ** 2))

                if rmse_orig == 0:
                    improvement = 0
                else:
                    improvement = (rmse_orig - rmse_corr) / rmse_orig * 100
                metric_values.append(improvement)

            elif evaluation_loss == "extreme_heat":
                # Calculate extreme heat RMSE using the weighted loss function
                rmse_orig_extreme = calculate_extreme_heat_rmse(orig_bin, gt_bin)
                rmse_corr_extreme = calculate_extreme_heat_rmse(corr_bin, gt_bin)

                if rmse_orig_extreme == 0:
                    improvement = 0
                else:
                    improvement = (rmse_orig_extreme - rmse_corr_extreme) / rmse_orig_extreme * 100
                metric_values.append(improvement)
            else:
                raise ValueError(f"Unknown metric: {evaluation_loss}")

        # Get color for region
        if region in climate_region_colors:
            color = climate_region_colors[region]
        elif region in topographic_region_colors:
            color = topographic_region_colors[region]
        else:
            color = region_colors.get(region, '#1f77b4')

        # Plot line
        ax_main.plot(bin_centers, metric_values, marker='o', linewidth=2.5,
                    markersize=8, color=color, label=region.replace('_', ' ').title(),
                    alpha=0.9)

    # Set up main axis
    if evaluation_loss == "rmse":
        ax_main.set_ylabel("RMSE Improvement (%)", fontsize=18)
        ax_main.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    elif evaluation_loss == "extreme_heat":
        ax_main.set_ylabel("Extreme Heat RMSE Improvement (%)", fontsize=18)
        ax_main.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    ax_main.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax_main.set_axisbelow(True)
    ax_main.tick_params(axis='both', labelsize=16)

    # Title
    regions_str = ", ".join([r.replace('_', ' ').title() for r in regions])
    var_display = variable.replace('_', ' ').title()
    title = f"{evaluation_loss.replace('_', ' ').title()} by {var_display} Value\n"
    title += f"Model: {model.upper()}, Lead Time: {lead_time}h, Architecture: {nn_architecture.upper()}"
    if len(regions) <= 3:
        title += f"\nRegions: {regions_str}"
    ax_main.set_title(title, fontsize=18, pad=15)

    # Legend
    ax_main.legend(loc='best', fontsize=14, framealpha=0.95, edgecolor='gray')

    # Remove x-axis labels from main plot (shared with histogram)
    plt.setp(ax_main.get_xticklabels(), visible=False)

    # Create overlapping histograms for each region (density/percentage)
    for region in regions:
        if region not in region_histogram_data:
            continue

        # Get color matching the line plot
        if region in climate_region_colors:
            color = climate_region_colors[region]
        elif region in topographic_region_colors:
            color = topographic_region_colors[region]
        else:
            color = region_colors.get(region, '#1f77b4')

        # Plot histogram with density=True to get probability density
        ax_hist.hist(region_histogram_data[region], bins=bin_edges,
                    color=color, alpha=0.3, edgecolor=color, linewidth=1.5,
                    density=True, label=region.replace('_', ' ').title())

    # Convert y-axis to percentages
    ax_hist.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y*100:.1f}%'))
    ax_hist.set_ylabel('Probability Density (%)', fontsize=16)
    ax_hist.tick_params(axis='both', labelsize=14)
    ax_hist.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    ax_hist.set_axisbelow(True)

    # Add legend to histogram if multiple regions
    if len(regions) > 1:
        ax_hist.legend(loc='upper right', fontsize=10, framealpha=0.9, ncol=min(3, len(regions)))

    # Set x-axis label
    if is_temperature:
        unit = "°C"
    elif "wind" in variable:
        unit = "m/s"
    else:
        unit = ""

    xlabel = f"{var_display} {unit}".strip()
    ax_hist.set_xlabel(xlabel, fontsize=18)

    # Remove top spine from histogram
    ax_hist.spines['top'].set_visible(False)
    ax_main.spines['bottom'].set_visible(False)

    # Remove other unnecessary spines
    for ax in [ax_main, ax_hist]:
        ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # Save figure
    if save_path is None:
        out_folder = os.path.join(dirs["fig"], model, "binned_analysis", subregion)
        os.makedirs(out_folder, exist_ok=True)

        if len(regions) == 1:
            region_str = regions[0]
        elif any(r in climate_region_colors for r in regions):
            region_str = "climate_zones"
        elif any(r in topographic_region_colors for r in regions):
            region_str = "topographic_zones"
        else:
            region_str = "multi_region"

        fname = (f"binned_{evaluation_loss}_trained_on_{loss_trained_on}_{variable}_lt{lead_time}h_{model}_"
                f"{nn_architecture}_{region_str}_{n_bins}bins.png")
        save_path = os.path.join(out_folder, fname)

    plt.show()
    exit()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Binned RMSE improvement plot saved to: {save_path}")


def plot_raw_forecast_values(csv_path, dirs, variable, model="pangu",
                            regions=None, subregion="6x6",
                            nn_architectures=["mlp"], growing_season_only = False, 
                            loss_trained_on="mse",save_path=None):
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
    loss_trained_on: str
        Loss function used to train the model: "mse", "extreme_heat"
    save_path : str
        Custom save path. If None, auto-generates based on parameters
    """
    # choose loss function used to train model
    if loss_trained_on == "mse":
        loss_fn = "mse"
    elif loss_trained_on == "extreme_heat":
        loss_fn = "extreme_heat_loss"
    else:
        raise ValueError(f"Unknown loss_trained_on: {loss_trained_on}")

    # Prepare data
    df, regions = _prepare_dataframe(csv_path, variable, regions, subregion, 
                                    nn_architectures, model, growing_season_only,
                                    loss_fn=loss_fn)
    if len(df) == 0:
        print(f"No data found for specified filters")
        return
    
    # Get unique values for plotting
    lead_times = sorted(df['lead_time'].unique())
    prediction_var = df['variable'].iloc[0]
    
    # Get color schemes
    region_colors, climate_region_colors, topographic_region_colors, model_markers, architecture_fillstyles = _get_color_schemes()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Calculate bar width and positions
    n_regions = len(regions)
    n_architectures = len(nn_architectures)
    n_groups_per_leadtime = n_regions * n_architectures * 2  # 2 for original and corrected
    bar_width = 0.6 / n_groups_per_leadtime
    region_gap = bar_width * 0.25  # Small gap between regions

    # Track which forecast types we've added to legend
    legend_added = {'original': False, 'corrected': False}

    # Store mean values for annotation
    region_means = {}

    for region_idx, region in enumerate(regions):
        # Get color for region
        if region in climate_region_colors:
            color = climate_region_colors[region]
        else:
            color = region_colors.get(region, '#1f77b4')

        region_df = df[df['region'] == region]

        for arch_idx, arch in enumerate(nn_architectures):
            arch_df = region_df[region_df['architecture'] == arch]

            if len(arch_df) == 0:
                continue

            # Sort by lead time
            arch_df = arch_df.sort_values('lead_time')

            # Get styles
            fillstyle = architecture_fillstyles.get(arch, 'full')
            alpha = 0.9 if fillstyle == 'full' else 0.6
            hatch = None if fillstyle == 'full' else '//'

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

                    # Plot original forecast raw errors as bars
                    if 'mean_original_forecast' in arch_df.columns:
                        y_values = arch_df['mean_original_forecast'].values
                        if not np.all(np.isnan(y_values)):
                            y_values_error = y_values - ground_truth_values

                            label = 'Original Forecast Error' if not legend_added['original'] else None
                            if label:
                                legend_added['original'] = True

                            # Calculate x positions for original bars with region gaps
                            group_offset = (region_idx * n_architectures * 2 + arch_idx * 2) * bar_width + region_idx * region_gap
                            x_pos = np.arange(len(lead_times)) + group_offset - (n_groups_per_leadtime * bar_width + (n_regions - 1) * region_gap) / 2 + bar_width / 2

                            ax.bar(x_pos, y_values_error,
                                   width=bar_width,
                                   color=color,
                                   alpha=alpha * 0.6,
                                   edgecolor='black',
                                   linewidth=0.5,
                                   hatch=hatch,
                                   label=label,
                                   zorder=2)

                    # Plot corrected forecast deviations as bars
                    if 'mean_corrected_forecast' in arch_df.columns:
                        y_values = arch_df['mean_corrected_forecast'].values
                        if not np.all(np.isnan(y_values)):
                            y_values_dev = y_values - ground_truth_values

                            label = 'Corrected Forecast Error' if not legend_added['corrected'] else None
                            if label:
                                legend_added['corrected'] = True

                            # Calculate x positions for corrected bars (offset from original) with region gaps
                            group_offset = (region_idx * n_architectures * 2 + arch_idx * 2 + 1) * bar_width + region_idx * region_gap
                            x_pos = np.arange(len(lead_times)) + group_offset - (n_groups_per_leadtime * bar_width + (n_regions - 1) * region_gap) / 2 + bar_width / 2

                            ax.bar(x_pos, y_values_dev,
                                   width=bar_width,
                                   color=color,
                                   alpha=alpha,
                                   edgecolor='black',
                                   linewidth=0.5,
                                   hatch=hatch,
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
        ax.text(0.22, 0.35, annotation_text,
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
    from matplotlib.patches import Patch
    forecast_handles = []
    if legend_added['original']:
        forecast_handles.append(Patch(facecolor='gray', edgecolor='black',
                                      alpha=0.6, label='Original Forecast Error'))
    if legend_added['corrected']:
        forecast_handles.append(Patch(facecolor='gray', edgecolor='black',
                                      alpha=0.9, label='Corrected Forecast Error'))

    legend1 = ax.legend(handles=forecast_handles, title="Forecast Type",
                       loc='lower left', bbox_to_anchor=(0.2, 0), fontsize=12)
    
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
                       loc='lower left', bbox_to_anchor=(0, 0), fontsize=16, title_fontsize=16)
    
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

        if loss_trained_on == "mse":
            model_str = model
        elif loss_trained_on == "extreme_heat":
            model_str = f"{model}_extreme_heat"

        fname = (f"leadtime_raw_values_{prediction_var}_trainedwith_{training_vars}_"
                f"{region_type}_{model_str}_{arch_suffix}{grow_flag}.png")
        
        save_path = os.path.join(out_folder, fname)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Raw forecast values plot saved to: {save_path}")


def plot_error_cutoff(csv_path, dirs, variable, model="pangu",
                     regions=None, subregion="4x4",
                     nn_architectures=["mlp"], save_path=None, 
                     loss_trained_on="mse", growing_season_only=False):
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
    loss_trained_on: str
        Loss function used to train the model: "mse", "extreme_heat"
    save_path : str
        Custom save path. If None, auto-generates based on parameters
    """
    if loss_trained_on == "mse":
        loss_fn = "mse"
    elif loss_trained_on == "extreme_heat":
        loss_fn = "extreme_heat_loss"
    else:
        raise ValueError(f"Unknown loss_trained_on: {loss_trained_on}")

    # Prepare data
    df, regions = _prepare_dataframe(csv_path, variable, regions, subregion, 
                                    nn_architectures, model, growing_season_only,
                                    loss_fn=loss_fn)
    
    if len(df) == 0:
        print(f"No data found for specified filters")
        return
    
    # Get unique values for plotting
    lead_times = sorted(df['lead_time'].unique())
    prediction_var = df['variable'].iloc[0]
    
    # Get color schemes
    region_colors, climate_region_colors, topographic_region_colors, model_markers, architecture_fillstyles = _get_color_schemes()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Calculate bar width and positions
    n_regions = len(regions)
    n_architectures = len(nn_architectures)
    n_groups_per_leadtime = n_regions * n_architectures * 2  # 2 for original and corrected
    bar_width = 0.6 / n_groups_per_leadtime
    region_gap = bar_width * 0.25  # Small gap between regions

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
    legend_added = {'original': False, 'corrected': False}

    for region_idx, region in enumerate(regions):
        # Get color for region
        if region in climate_region_colors:
            color = climate_region_colors[region]
        else:
            color = region_colors.get(region, '#1f77b4')

        region_df = df[df['region'] == region]

        for arch_idx, arch in enumerate(nn_architectures):
            arch_df = region_df[region_df['architecture'] == arch]

            if len(arch_df) == 0:
                continue

            # Sort by lead time
            arch_df = arch_df.sort_values('lead_time')

            # Get styles
            fillstyle = architecture_fillstyles.get(arch, 'full')
            alpha = 0.9 if fillstyle == 'full' else 0.6
            hatch = None if fillstyle == 'full' else '//'

            # Plot original error rate as bars
            if 'pct_error_cutoff_original' in arch_df.columns:
                y_values = arch_df['pct_error_cutoff_original'].values

                if not np.all(np.isnan(y_values)):
                    label = 'Original' if not legend_added['original'] else None
                    if label:
                        legend_added['original'] = True

                    # Calculate x positions for original bars with region gaps
                    group_offset = (region_idx * n_architectures * 2 + arch_idx * 2) * bar_width + region_idx * region_gap
                    x_pos = np.arange(len(lead_times)) + group_offset - (n_groups_per_leadtime * bar_width + (n_regions - 1) * region_gap) / 2 + bar_width / 2

                    ax.bar(x_pos, y_values,
                           width=bar_width,
                           color=color,
                           alpha=alpha * 0.6,
                           edgecolor='black',
                           linewidth=0.5,
                           hatch=hatch,
                           label=label,
                           zorder=2)

            # Plot corrected error rate as bars
            if 'pct_error_cutoff_corrected' in arch_df.columns:
                y_values = arch_df['pct_error_cutoff_corrected'].values

                if not np.all(np.isnan(y_values)):
                    label = 'Corrected' if not legend_added['corrected'] else None
                    if label:
                        legend_added['corrected'] = True

                    # Calculate x positions for corrected bars (offset from original) with region gaps
                    group_offset = (region_idx * n_architectures * 2 + arch_idx * 2 + 1) * bar_width + region_idx * region_gap
                    x_pos = np.arange(len(lead_times)) + group_offset - (n_groups_per_leadtime * bar_width + (n_regions - 1) * region_gap) / 2 + bar_width / 2

                    ax.bar(x_pos, y_values,
                           width=bar_width,
                           color=color,
                           alpha=alpha,
                           edgecolor='black',
                           linewidth=0.5,
                           hatch=hatch,
                           label=label,
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

    from matplotlib.patches import Patch
    bar_handles = [
        Patch(facecolor='gray', edgecolor='black', alpha=0.6, label='Original'),
        Patch(facecolor='gray', edgecolor='black', alpha=0.9, label='Corrected')
    ]

    # Position legends
    legend1 = ax.legend(handles=region_handles, title="Region",
                       loc='upper left', bbox_to_anchor=(0, 0.85), fontsize=16, title_fontsize=16)

    legend2 = ax.legend(handles=bar_handles, title="Forecast Type",
                       loc='upper left', bbox_to_anchor=(0, 0.50), fontsize=12)
    
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

        if loss_trained_on == "mse":
            model_str = model
        elif loss_trained_on == "extreme_heat":
            model_str = f"{model}_extreme_heat"

        fname = (f"leadtime_error_cutoff_{prediction_var}_trainedwith_{training_vars}_"
                f"{region_type}_{model_str}_{arch_suffix}{grow_flag}.png")
        save_path = os.path.join(out_folder, fname)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Error cutoff plot saved to: {save_path}")


def main():
    dirs = setup_directories()


#=============================================
# Global Improvement Plots
#=============================================
    for map_type in ["original", "improvement"]:
        for variable in ["2m_temperature", "10m_wind_speed"]:
            for model in ["pangu"]:
                # plot_scatter_forecast_improvement(dirs=dirs, model=model, 
                #                                 variable=variable, y_metric=map_type, 
                #                                 x_metric="equator_distance")
                # plot_scatter_forecast_improvement(dirs=dirs, model=model, 
                #                                 variable=variable, y_metric=map_type, 
                #                                 x_metric="sdor")
                for pixel_flag in [True]:
                    map_global_improvements(dirs=dirs, model=model, 
                                            variable=variable, map_type=map_type,
                                            pixel_level=pixel_flag)
    exit()

#=============================================
# Binned RMSE Improvement Plots
#=============================================
    dirs = setup_directories()
    var = "2m_temperature"
    loss_train_on = "mse"
    evaluation_loss = "rmse"
    training_outptut_vars = ([var], [var])
    variable = var
    plot_rmse_improvement_by_weather_bin(dirs = dirs, train_start="2018-01-01", train_end ="2021-12-31",
                                test_start="2022-01-01", test_end="2022-12-31",
                                model="pangu",
                                training_output_vars=training_outptut_vars,
                                variable=variable,
                                nn_architecture="mlp",
                                lead_time=216,
                                regions=["india", "ethiopia", "usa_south", "amazon"],
                                subregion="6x6",
                                n_bins=10,
                                loss_trained_on=loss_train_on,
                                evaluation_loss=evaluation_loss
                                )

    exit()

#=============================================
# Lead Time Plots (by region and lead time)
#=============================================

    nn_architectures = ['mlp'] # can be ['mlp'], ['unet'], or ['mlp', 'unet'] which plots both at once
    variable_list = ["2m_temperature", "10m_wind_speed"]
    model_list = ["ifs", "pangu"]
    geo_regions = ["india", "amazon", "ethiopia", "usa_south", "corn_belt"]
    climate_regions = ["tropical", "arid", "temperate"]
    topo_regions = ["flat", "hilly", "mountainous"]
    growing_season_flags = [False]
    stat_path = os.path.join(dirs["processed"], "forecast_improvement_stats.csv")
    for var in variable_list:
        for model in model_list:
            for gs_flag in growing_season_flags:
                for regions in [geo_regions, climate_regions, topo_regions]:
                    if regions == climate_regions or regions == topo_regions:
                        subregion = "2x2"
                    elif regions == geo_regions:
                        subregion = "6x6"
                    for loss_train_on in ["mse", "extreme_heat"]:
                        if model == "aifs" and not gs_flag:
                            # aifs results are only for growing season
                            continue

                        for evaluation_loss in ["rmse", "extreme_heat"]:
                            plot_rmse_improvement(csv_path = stat_path,
                                dirs=dirs,
                                variable=var,
                                model=model,
                                regions=regions,
                                subregion=subregion,
                                nn_architectures=nn_architectures,
                                growing_season_only=gs_flag,
                                loss_trained_on=loss_train_on,
                                evaluation_loss=evaluation_loss
                            )
                        plot_raw_forecast_values(csv_path = stat_path,
                            dirs=dirs,
                            variable=var,
                            model=model,
                            regions=regions,
                            subregion=subregion,
                            nn_architectures=nn_architectures,
                            growing_season_only=gs_flag,
                            loss_trained_on=loss_train_on
                        )
                        plot_error_cutoff(csv_path = stat_path,
                            dirs=dirs,
                            variable=var,
                            model=model,
                            regions=regions,
                            subregion=subregion,
                            nn_architectures=nn_architectures,
                            growing_season_only=gs_flag,
                            loss_trained_on=loss_train_on
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
        

def plot_scatter_forecast_improvement(
    dirs,
    model="pangu",
    variable="10m_wind_speed",
    regions=None,
    save_dir=None,
    x_metric="equator_distance",
    y_metric="improvement",
    lead_times=None,
    train_start="2018-01-01",
    train_end="2021-12-31",
    test_start="2022-01-01",
    test_end="2022-12-31",
    nn_architecture="mlp",
    subregion="6x6",
    alternate_loss_fn=None,
):
    """
    Create scatter plots showing relationship between geographic/topographic features and RMSE metrics.
    Each point represents either a region 6x6 patch mean 

    Only processes zarr files that match the specified model configuration to ensure
    all patches are from the same training/testing setup.

    Parameters
    ----------
    dirs : dict
        Dictionary of directories
    model : str
        Model to use: "pangu" or "ifs"
    variable : str
        Variable to plot: "2m_temperature", "10m_wind_speed", or "total_precipitation"
    regions : list, optional
        List of regions to include. If None, uses:
        ["asia", "africa", "north_america", "south_america", "europe", "oceania"]
    save_dir : str, optional
        Custom save directory. If None, auto-generates based on parameters
    x_metric : str, optional
        Metric to plot on x-axis: "equator_distance" (distance from equator in degrees)
        or "sdor" (standard deviation of orography). Default is "equator_distance".
    y_metric : str, optional
        Metric to plot on y-axis: "improvement" (percent improvement in RMSE)
        or "original" (original forecast RMSE). Default is "improvement".
    lead_times : list, optional
        List of lead times to plot. If None, uses [24, 120, 216]
    train_start : str, optional
        Training start date (default: "2018-01-01")
    train_end : str, optional
        Training end date (default: "2021-12-31")
    test_start : str, optional
        Test start date (default: "2022-01-01")
    test_end : str, optional
        Test end date (default: "2022-12-31")
    nn_architecture : str, optional
        Neural network architecture: "mlp" or "unet" (default: "mlp")
    subregion : str, optional
        Subregion size pattern (default: "6x6")
    alternate_loss_fn : str, optional
        Alternate loss function name if used (default: None)

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure
    """

    # Default lead times
    if lead_times is None:
        lead_times = [24, 120, 216]

    # Determine axis labels
    x_label = {
        "equator_distance": "Distance from Equator (degrees)",
        "sdor": "Standard Deviation of Orography (m)"
    }.get(x_metric, "Distance from Equator (degrees)")

    y_label = {
        "improvement": "RMSE Improvement (%)",
        "original": "Original Forecast RMSE"
    }.get(y_metric, "RMSE Improvement (%)")

    print(f"\nCreating forecast improvement scatter plot for {model.upper()} - {variable}")
    print(f"X-axis metric: {x_metric}")
    print(f"Y-axis metric: {y_metric}")

    # Load sdor data if needed
    sdor_da = None
    if x_metric == "sdor":
        era5_static_path = os.path.join(dirs["raw"], "era5_static.nc")
        sdor_da = xr.open_dataset(era5_static_path, engine="netcdf4")["sdor"]
        print(f"Loaded sdor data from {era5_static_path}")

    # Load region data using the helper function
    all_patch_data = load_region_data(
        dirs=dirs,
        model=model,
        variable=variable,
        regions=regions,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        nn_architecture=nn_architecture,
        subregion=subregion,
        alternate_loss_fn=alternate_loss_fn,
        lead_times=lead_times,
        sdor_da=sdor_da
    )

    if all_patch_data is None:
        return None

    # Print summary for each lead time (sdor-specific if needed)
    if x_metric == "sdor":
        for lt in lead_times:
            if all_patch_data[lt]:
                total_patches = len(all_patch_data[lt])
                patches_with_sdor = sum(1 for p in all_patch_data[lt] if p['sdor'] is not None)
                print(f"\nLead time {lt}h: {total_patches} patches total, "
                      f"{patches_with_sdor} with valid sdor values")

    # Determine output directory
    if save_dir is None:
        out_folder = os.path.join(dirs["fig"], model, "scatter_plots")
    else:
        out_folder = save_dir
    os.makedirs(out_folder, exist_ok=True)

    # Create scatter plot
    fig, axes = plt.subplots(1, len(lead_times), figsize=(6 * len(lead_times), 5))
    if len(lead_times) == 1:
        axes = [axes]

    # Color map for continents
    continent_colors = {
        'asia': '#FF6B6B',           # Red
        'africa': '#4ECDC4',         # Turquoise
        'north_america': '#45B7D1',  # Blue
        'south_america': '#FFA07A',  # Light Salmon
        'europe': '#98D8C8',         # Mint
        'oceania': '#F7DC6F'         # Yellow
    }

    for idx, lead_time in enumerate(lead_times):
        ax = axes[idx]
        patch_data = all_patch_data[lead_time]

        if not patch_data:
            print(f"\nSkipping lead time {lead_time}h - no data available")
            continue

        # Set marker size based on patch-level plotting
        marker_size = 20
        marker_edge = 0.2

        # Region-mean scatter: use existing patch-level data
        # Determine x and y values based on metrics
        if x_metric == "equator_distance":
            x_values = [p['distance_from_equator'] for p in patch_data]
        elif x_metric == "sdor":
            x_values = [p['sdor'] for p in patch_data if p['sdor'] is not None]
            # Filter patch_data to only include patches with sdor values
            patch_data = [p for p in patch_data if p['sdor'] is not None]
            if not patch_data:
                print(f"\nSkipping lead time {lead_time}h - no patches with valid sdor values")
                continue
        else:
            raise ValueError(f"Invalid x_metric: {x_metric}. Must be 'equator_distance' or 'sdor'.")

        if y_metric == "improvement":
            y_values = [p['improvement'] for p in patch_data]
        elif y_metric == "original":
            y_values = [p['rmse_original'] for p in patch_data]
        else:
            raise ValueError(f"Invalid y_metric: {y_metric}. Must be 'improvement' or 'original'.")

        # Group patches by continent
        patches_by_continent = {}
        for p in patch_data:
            continent = p.get('region', 'unknown')
            if continent not in patches_by_continent:
                patches_by_continent[continent] = []
            patches_by_continent[continent].append(p)

        # Plot each continent separately
        for continent, continent_patches in patches_by_continent.items():
            # Extract x values based on metric
            if x_metric == "equator_distance":
                continent_x = [p['distance_from_equator'] for p in continent_patches]
            else:
                continent_x = [p['sdor'] for p in continent_patches if p['sdor'] is not None]
                continent_patches = [p for p in continent_patches if p['sdor'] is not None]

            # Extract y values based on metric
            if y_metric == "improvement":
                continent_y = [p['improvement'] for p in continent_patches]
            else:
                continent_y = [p['rmse_original'] for p in continent_patches]

            # Skip if no data for this continent
            if not continent_x or not continent_y:
                continue

            # Get color for this continent
            color = continent_colors.get(continent, '#808080')  # Gray as default
            label = continent.replace('_', ' ').title()

            # Plot this continent
            ax.scatter(continent_x, continent_y,
                      c=color,
                      label=label,
                      alpha=0.6, s=marker_size, edgecolors='black', linewidth=marker_edge)

        # Add horizontal line at y=0 for improvement plots
        if y_metric == "improvement":
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

        # Calculate and display correlation
        if len(x_values) > 1 and len(y_values) > 1:
            correlation = np.corrcoef(x_values, y_values)[0, 1]
            ax.text(0.02, 0.98, f'r = {correlation:.3f}',
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Labels and title
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(f'Lead Time: {lead_time}h', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')

        # Add legend only to the first subplot
        if idx == 0:
            ax.legend(loc='best', fontsize=9, framealpha=0.9)

    # Add overall title
    title = f'{model.upper()} - {variable.replace("_", " ").title()}\n'
    title += f'{x_label} vs {y_label}'
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    x_suffix = "equator" if x_metric == "equator_distance" else "sdor"
    y_suffix = y_metric
    filename = f"scatter_{x_suffix}_{variable}_{y_suffix}.png"
    save_path = os.path.join(out_folder, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved scatter plot to: {save_path}")

    return fig


if __name__ == "__main__":
    main()