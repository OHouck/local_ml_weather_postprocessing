"""
Stand alone version of map_global_improvements from figures_finetuning.py
to create global maps of forecast improvements.

imports directory set up from helper_funcs.py

Data inputs:

Post-processed zarr files from finetuning output directory structure. 
These are all created by finetune.py with different runs managed by run_experiments.sh

The helper function all_patch_data is used
as a data processing function but i have saved the output and commented it out 
so it shouldn't have to be used. 

"""

import os
import glob
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path
from binsreg import binsregselect, binsreg, binsqreg, binsglm, binstest, binspwc
from figures_finetuning import _extract_pixel_level_data
sys.path.insert(0, str(Path(__file__).parent.parent))
from helper_funcs import setup_directories, generate_output_path

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

    This function consolidates the data loading logic for map_global_improvements
    and plot_scatter_forecast_improvement. It returns a dictionary of patch data

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

def validate_non_overlapping_patches(patch_data, tolerance=1e-6):
    """
    Validate that patches are non-overlapping tiles.

    Checks that no two patches have overlapping spatial regions by testing
    whether their bounding boxes intersect.

    Parameters
    ----------
    patch_data : list of dict
        List of patch dictionaries with 'lat_min', 'lat_max', 'lon_min', 'lon_max'
    tolerance : float, optional
        Numerical tolerance for floating point comparisons (default: 1e-6)
        Patches that share edges within this tolerance are considered non-overlapping

    Returns
    -------
    bool
        True if all patches are non-overlapping

    Raises
    ------
    ValueError
        If overlapping patches are detected, with details about which patches overlap
    """
    n_patches = len(patch_data)

    for i in range(n_patches):
        patch_i = patch_data[i]
        lat_min_i, lat_max_i = patch_i['lat_min'], patch_i['lat_max']
        lon_min_i, lon_max_i = patch_i['lon_min'], patch_i['lon_max']

        for j in range(i + 1, n_patches):
            patch_j = patch_data[j]
            lat_min_j, lat_max_j = patch_j['lat_min'], patch_j['lat_max']
            lon_min_j, lon_max_j = patch_j['lon_min'], patch_j['lon_max']

            # Check for overlap in latitude
            # Patches overlap if: not (i is completely above j OR i is completely below j)
            lat_overlap = not (lat_min_i >= lat_max_j - tolerance or lat_max_i <= lat_min_j + tolerance)

            # Check for overlap in longitude
            lon_overlap = not (lon_min_i >= lon_max_j - tolerance or lon_max_i <= lon_min_j + tolerance)

            # Patches overlap if they overlap in BOTH dimensions
            if lat_overlap and lon_overlap:
                # Calculate overlap area for detailed error message
                overlap_lat_min = max(lat_min_i, lat_min_j)
                overlap_lat_max = min(lat_max_i, lat_max_j)
                overlap_lon_min = max(lon_min_i, lon_min_j)
                overlap_lon_max = min(lon_max_i, lon_max_j)

                overlap_area = (overlap_lat_max - overlap_lat_min) * (overlap_lon_max - overlap_lon_min)

                raise ValueError(
                    f"Overlapping patches detected!\n"
                    f"  Patch {i}: lat=[{lat_min_i:.3f}, {lat_max_i:.3f}], lon=[{lon_min_i:.3f}, {lon_max_i:.3f}]\n"
                    f"  Patch {j}: lat=[{lat_min_j:.3f}, {lat_max_j:.3f}], lon=[{lon_min_j:.3f}, {lon_max_j:.3f}]\n"
                    f"  Overlap region: lat=[{overlap_lat_min:.3f}, {overlap_lat_max:.3f}], "
                    f"lon=[{overlap_lon_min:.3f}, {overlap_lon_max:.3f}]\n"
                    f"  Overlap area: {overlap_area:.6f} square degrees"
                )

    return True

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

    # # Load region data using the helper function
    # all_patch_data = load_region_data(
    #     dirs=dirs,
    #     model=model,
    #     variable=variable,
    #     regions=regions,
    #     train_start=train_start,
    #     train_end=train_end,
    #     test_start=test_start,
    #     test_end=test_end,
    #     nn_architecture=nn_architecture,
    #     subregion=subregion,
    #     alternate_loss_fn=alternate_loss_fn,
    #     lead_times=lead_times,
    #     sdor_da=None
    # )

    import pickle
    with open(os.path.join(dirs["processed"], f"{model}_{variable}_patch_data.pkl"), "rb") as f:
        all_patch_data = pickle.load(f)

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

            # Validate that patches are non-overlapping
            print(f"  Validating that patches are non-overlapping tiles...")
            try:
                validate_non_overlapping_patches(patch_data)
                print(f"  ✓ All {len(patch_data)} patches are non-overlapping")
            except ValueError as e:
                print(f"  ✗ Patch validation failed: {e}")
                raise

            # Step 1: Collect all patch datasets
            print(f"  Concatenating {len(patch_data)} patches into global dataset...")
            patch_datasets = []

            for patch in patch_data:
                ds = patch['ds']
                var_suffix = f"_lt{lead_time}h"

                # Extract only the variables needed for this lead time
                ds_subset = ds[[
                    f"{variable}_ground_truth{var_suffix}",
                    f"{variable}_original{var_suffix}",
                    f"{variable}_corrected{var_suffix}"
                ]]

                patch_datasets.append(ds_subset)

            # Step 2: Combine all patches into a single global dataset
            # combine_by_coords automatically handles non-overlapping spatial tiles
            global_ds = xr.combine_by_coords(patch_datasets, combine_attrs='drop_conflicts')

            print(f"  Global grid: {len(global_ds.latitude)} latitudes × {len(global_ds.longitude)} longitudes")

            # Step 3: Single vectorized RMSE calculation for all pixels at once
            print(f"  Computing RMSE for all pixels in single operation...")
            var_suffix = f"_lt{lead_time}h"

            ground_truth = global_ds[f"{variable}_ground_truth{var_suffix}"]
            original = global_ds[f"{variable}_original{var_suffix}"]
            corrected = global_ds[f"{variable}_corrected{var_suffix}"]

            # Compute pixel-wise RMSE over time dimension in one vectorized operation
            # Shape: (time, lat, lon) -> (lat, lon)
            rmse_original_pixel = np.sqrt(((original - ground_truth) ** 2).mean(dim='time'))
            rmse_corrected_pixel = np.sqrt(((corrected - ground_truth) ** 2).mean(dim='time'))

            # Select the appropriate data to plot based on map_type
            if map_type == "improvement":
                # Compute improvement percentage for each pixel (single operation)
                plot_data = ((rmse_original_pixel - rmse_corrected_pixel) / rmse_original_pixel * 100)
            elif map_type == "original":
                plot_data = rmse_original_pixel
            elif map_type == "corrected":
                plot_data = rmse_corrected_pixel
            else:
                raise ValueError(f"Invalid map_type: {map_type}. Must be 'improvement', 'original', or 'corrected'.")

            # Extract coordinates for plotting
            unique_lats = global_ds.latitude.values
            unique_lons = global_ds.longitude.values

            # Calculate statistics using nanXXX functions (faster than masking)
            n_pixels = int(np.count_nonzero(~np.isnan(plot_data.values)))

            if n_pixels == 0:
                print(f"  No valid pixel data for lead time {lead_time}h!")
                continue

            # Use nanXXX functions - they're optimized and faster than manual masking
            vmin = float(np.nanmin(plot_data.values))
            vmax = float(np.nanmax(plot_data.values))
            mean_val = float(np.nanmean(plot_data.values))
            median_val = float(np.nanmedian(plot_data.values))
            std_val = float(np.nanstd(plot_data.values))

            print(f"  Pixel range: {vmin:.1f} to {vmax:.1f}")
            print(f"  Valid pixels: {n_pixels}")

            # Create colormap based on map type
            if map_type == "improvement":
                norm = plt.Normalize(vmin=vmin, vmax=vmax)
                cmap = plt.cm.Blues# Red for negative, Blue for positive
            else:
                # For RMSE values (original or corrected), use single-color gradient
                norm = plt.Normalize(vmin=vmin, vmax=vmax)
                cmap = plt.cm.YlOrRd  # Yellow to Red for RMSE values

            # Step 4: Single plot call using pcolormesh (much faster than per-pixel plotting)
            print(f"  Plotting global {metric_name} raster...")

            # Use pcolormesh for proper georeferencing - it handles DataArrays directly
            mesh = ax.pcolormesh(
                plot_data.longitude,
                plot_data.latitude,
                plot_data.values,
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                norm=norm,
                shading='auto',
                zorder=1
            )

            # Add black boxes showing patch boundaries (optimized with LineCollection)
            print(f"  Adding patch boundary boxes...")
            from matplotlib.collections import LineCollection

            boundary_segments = []
            for patch in patch_data:
                lat_min, lat_max = patch['lat_min'], patch['lat_max']
                lon_min, lon_max = patch['lon_min'], patch['lon_max']

                # Create rectangle segments (4 lines per rectangle)
                boundary_segments.extend([
                    [(lon_min, lat_min), (lon_max, lat_min)],  # Bottom
                    [(lon_max, lat_min), (lon_max, lat_max)],  # Right
                    [(lon_max, lat_max), (lon_min, lat_max)],  # Top
                    [(lon_min, lat_max), (lon_min, lat_min)]   # Left
                ])

            # Add all boundaries at once (much faster than individual patches)
            lc = LineCollection(
                boundary_segments,
                colors='black',
                linewidths=0.5,
                alpha=1.0,
                transform=ccrs.PlateCarree(),
                zorder=2
            )
            ax.add_collection(lc)

            # Statistics for title
            if map_type == "improvement":
                title_main = f"Global RMSE Improvement Map (Pixel-Level)"
                unit = "%"
            elif map_type == "original":
                title_main = f"Global Original RMSE Map (Pixel-Level)"
                unit = ""
            else:  # corrected
                title_main = f"Global Corrected RMSE Map (Pixel-Level)"
                unit = ""

            title_parts = [
                title_main,
                f"{model.upper()} - {variable.replace('_', ' ').title()} - {lead_time}h Lead Time",
                f"N = {n_pixels} pixels"
            ]

            stats_text = (
                f"Mean: {mean_val:.1f}{unit}\n"
                f"Median: {median_val:.1f}{unit}\n"
                f"Std: {std_val:.1f}{unit}"
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
            # XX where color ramp is set. This could be improved!
            if map_type == "improvement":
                norm = plt.Normalize(vmin=vmin, vmax=vmax)
                cmap = plt.cm.Blues
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
        if map_type == "improvement":
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
            if map_type == "improvement":
                fname = f"global_improvement_map_pixel_{variable}_{model}_lt{lead_time}.png"
            elif map_type == "original":
                fname = f"global_original_rmse_map_pixel_{variable}_{model}_lt{lead_time}.png"
            elif map_type == "corrected":
                fname = f"global_corrected_rmse_map_pixel_{variable}_{model}_lt{lead_time}.png"
        else:
            if map_type == "improvement":
                fname = f"global_improvement_map_{variable}_{model}_lt{lead_time}.png"
            elif map_type == "original":
                fname = f"global_original_rmse_map_{variable}_{model}_lt{lead_time}.png"
            elif map_type == "corrected":
                fname = f"global_corrected_rmse_map_{variable}_{model}_lt{lead_time}.png"

        save_path = os.path.join(out_folder, fname)
        plt.show()
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



def lead_time_compare_binscatter(
    dirs,
    model="pangu",
    x_metric="equator_distance",
    regions=None,
    save_dir=None,
    train_start="2018-01-01",
    train_end="2021-12-31",
    test_start="2022-01-01",
    test_end="2022-12-31",
    nn_architecture="mlp",
    subregion="6x6",
    alternate_loss_fn=None
):
    """
    Create 2x2 binscatter comparison across lead times.

    Creates a figure with 4 panels arranged in a 2x2 grid:
    - Row 1: 2m_temperature
    - Row 2: 10m_wind_speed
    - Column 1: Original RMSE vs x_metric
    - Column 2: RMSE percent improvement vs x_metric

    Each panel overlays 3 lead times (1, 5, 9 days) with different colors,
    each with its own binscatter dots and linear regression line.

    Parameters
    ----------
    dirs : dict
        Dictionary of directories
    model : str
        Model to use: "pangu" or "ifs"
    x_metric : str
        Metric to plot on x-axis: "equator_distance" or "sdor"
    regions : list, optional
        List of regions to include. If None, uses default continents
    save_dir : str, optional
        Custom save directory. If None, auto-generates based on parameters
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

    # Lead times in hours and their labels in days
    lead_times = [24, 120, 216]
    lead_time_labels = {24: "1 day", 120: "5 days", 216: "9 days"}

    # Variables for each row
    variables = ["2m_temperature", "10m_wind_speed"]

    # Colors for each lead time
    colors = {24: '#1f77b4', 120: '#ff7f0e', 216: '#2ca02c'}  # Blue, Orange, Green

    # Determine axis labels
    x_label = {
        "equator_distance": "Distance from Equator (degrees)",
        "sdor": "Standard Deviation of Orography (m)"
    }.get(x_metric, "Distance from Equator (degrees)")

    print(f"\nCreating lead time comparison binscatter for {model.upper()}")
    print(f"X-axis metric: {x_metric}")
    print(f"Lead times: {lead_times} hours = {[lead_time_labels[lt] for lt in lead_times]}")

    # Load sdor data if needed
    sdor_da = None
    if x_metric == "sdor":
        era5_static_path = os.path.join(dirs["raw"], "era5_static.nc")
        sdor_da = xr.open_dataset(era5_static_path, engine="netcdf4")["sdor"]
        print(f"Loaded sdor data from {era5_static_path}")

    # Create figure with 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Process each variable (row)
    for row_idx, variable in enumerate(variables):
        print(f"\n{'='*60}")
        print(f"Processing {variable}")
        print(f"{'='*60}")

        # Load region data for this variable
        # all_patch_data = load_region_data(
        #     dirs=dirs,
        #     model=model,
        #     variable=variable,
        #     regions=regions,
        #     train_start=train_start,
        #     train_end=train_end,
        #     test_start=test_start,
        #     test_end=test_end,
        #     nn_architecture=nn_architecture,
        #     subregion=subregion,
        #     alternate_loss_fn=alternate_loss_fn,
        #     lead_times=lead_times,
        #     sdor_da=sdor_da
        # )

        import pickle
        with open(os.path.join(dirs["processed"], f"{model}_{variable}_patch_data.pkl"), "rb") as f:
            all_patch_data = pickle.load(f)


        if all_patch_data is None:
            print(f"Warning: No data available for {variable}")
            continue

        # Process each lead time
        for lead_time in lead_times:
            patch_data = all_patch_data[lead_time]

            if not patch_data:
                print(f"\nSkipping lead time {lead_time}h - no data available")
                continue

            # Extract pixel-level data for binscatter. (dataloading function from figures_finetuning.py)
            pixel_data = _extract_pixel_level_data(
                patch_data, variable, lead_time, x_metric, sdor_da
            )

            if pixel_data is None or len(pixel_data['x']) == 0:
                print(f"\nSkipping lead time {lead_time}h - no pixel data available")
                continue

            # Plot Original RMSE (Column 0)
            ax_original = axes[row_idx, 0]
            _plot_binscatter(
                ax=ax_original,
                x=pixel_data['x'],
                y=pixel_data['rmse_original'],
                color=colors[lead_time],
                label=lead_time_labels[lead_time],
                lead_time=lead_time,
                is_first=(lead_time == lead_times[0])
            )

            # Plot RMSE Improvement (Column 1)
            ax_improvement = axes[row_idx, 1]
            _plot_binscatter(
                ax=ax_improvement,
                x=pixel_data['x'],
                y=pixel_data['improvement'],
                color=colors[lead_time],
                label=lead_time_labels[lead_time],
                lead_time=lead_time,
                is_first=(lead_time == lead_times[0]),
                add_zero_line=True
            )

        # Set labels and titles for this row
        variable_title = variable.replace("_", " ").title()

        # Original RMSE panel
        axes[row_idx, 0].set_xlabel(x_label, fontsize=12)
        axes[row_idx, 0].set_ylabel("Original Forecast RMSE", fontsize=12)
        axes[row_idx, 0].grid(True, alpha=0.3, linestyle='--')
        if row_idx == 0:
            axes[row_idx, 0].set_title("Original RMSE", fontsize=13, fontweight='bold', pad=10)
        axes[row_idx, 0].text(-0.12, 0.5, variable_title,
                             transform=axes[row_idx, 0].transAxes, fontsize=12, fontweight='bold',
                             rotation=90, verticalalignment='center')

        # Improvement panel
        axes[row_idx, 1].set_xlabel(x_label, fontsize=12)
        axes[row_idx, 1].set_ylabel("RMSE Improvement (%)", fontsize=12)
        axes[row_idx, 1].grid(True, alpha=0.3, linestyle='--')
        if row_idx == 0:
            axes[row_idx, 1].set_title("RMSE Improvement", fontsize=13, fontweight='bold', pad=10)

        # Add legend to first row
        if row_idx == 0:
            axes[row_idx, 1].legend(loc='upper right', fontsize=10, framealpha=0.9)

    # Add overall title
    title = f'{model.upper()} - Lead Time Comparison\n'
    title += f'{x_label} vs RMSE Metrics'
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Determine output directory
    if save_dir is None:
        out_folder = os.path.join(dirs["fig"], model, "lead_time_comparison")
    else:
        out_folder = save_dir
    os.makedirs(out_folder, exist_ok=True)

    # Save figure
    x_suffix = "equator" if x_metric == "equator_distance" else "sdor"
    filename = f"lead_time_compare_binscatter_{x_suffix}.png"
    save_path = os.path.join(out_folder, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved lead time comparison plot to: {save_path}")

    return fig


def model_compare_binscatter(
    dirs,
    variable="2m_temperature",
    x_metric="equator_distance",
    regions=None,
    save_dir=None,
    train_start="2018-01-01",
    train_end="2021-12-31",
    test_start="2022-01-01",
    test_end="2022-12-31",
    nn_architecture="mlp",
    subregion="6x6",
    alternate_loss_fn=None
):
    """
    Create 1x3 binscatter comparison across models (Pangu original, IFS original, Pangu corrected).

    Creates a figure with 3 panels arranged in a 1x3 grid:
    - Column 1: 1 day lead time (24h)
    - Column 2: 5 days lead time (120h)
    - Column 3: 9 days lead time (216h)

    Each panel shows RMSE vs x_metric with three binscatters:
    - Original Pangu forecast
    - Original IFS forecast
    - Corrected Pangu forecast

    Parameters
    ----------
    dirs : dict
        Dictionary of directories
    variable : str
        Variable to plot: "2m_temperature" or "10m_wind_speed"
    x_metric : str
        Metric to plot on x-axis: "equator_distance" or "sdor"
    regions : list, optional
        List of regions to include. If None, uses default continents
    save_dir : str, optional
        Custom save directory. If None, auto-generates based on parameters
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

    # Lead times in hours and their labels in days
    lead_times = [24, 120, 216]
    lead_time_labels = {24: "1 day", 120: "5 days", 216: "9 days"}

    # Models and their colors
    model_info = [
        {'model': 'pangu', 'type': 'original', 'label': 'Pangu Original', 'color': '#1f77b4'},  # Blue
        {'model': 'ifs', 'type': 'original', 'label': 'IFS Original', 'color': '#ff7f0e'},      # Orange
        {'model': 'pangu', 'type': 'corrected', 'label': 'Pangu Corrected', 'color': '#2ca02c'} # Green
    ]

    # Determine axis labels
    x_label = {
        "equator_distance": "Distance from Equator (degrees)",
        "sdor": "Standard Deviation of Orography (m)"
    }.get(x_metric, "Distance from Equator (degrees)")

    variable_title = variable.replace("_", " ").title()

    print(f"\nCreating model comparison binscatter for {variable}")
    print(f"X-axis metric: {x_metric}")
    print(f"Models: Pangu Original, IFS Original, Pangu Corrected")

    # Load sdor data if needed
    sdor_da = None
    if x_metric == "sdor":
        era5_static_path = os.path.join(dirs["raw"], "era5_static.nc")
        sdor_da = xr.open_dataset(era5_static_path, engine="netcdf4")["sdor"]
        print(f"Loaded sdor data from {era5_static_path}")

    # Create figure with 1x3 grid
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Load data for both models
    print(f"\n{'='*60}")
    print(f"Loading data for Pangu and IFS models")
    print(f"{'='*60}")

    # Load Pangu data
    # pangu_data = load_region_data(
    #     dirs=dirs,
    #     model='pangu',
    #     variable=variable,
    #     regions=regions,
    #     train_start=train_start,
    #     train_end=train_end,
    #     test_start=test_start,
    #     test_end=test_end,
    #     nn_architecture=nn_architecture,
    #     subregion=subregion,
    #     alternate_loss_fn=alternate_loss_fn,
    #     lead_times=lead_times,
    #     sdor_da=sdor_da
    # )

    

    # # Load IFS data
    # ifs_data = load_region_data(
    #     dirs=dirs,
    #     model='ifs',
    #     variable=variable,
    #     regions=regions,
    #     train_start=train_start,
    #     train_end=train_end,
    #     test_start=test_start,
    #     test_end=test_end,
    #     nn_architecture=nn_architecture,
    #     subregion=subregion,
    #     alternate_loss_fn=alternate_loss_fn,
    #     lead_times=lead_times,
    #     sdor_da=sdor_da
    # )

    import pickle
    with open(os.path.join(dirs["processed"], f"pangu_{variable}_patch_data.pkl"), "rb") as f:
        pangu_data = pickle.load(f)
    with open(os.path.join(dirs["processed"], f"ifs_{variable}_patch_data.pkl"), "rb") as f:
        ifs_data = pickle.load(f)

    if pangu_data is None or ifs_data is None:
        print(f"Error: Could not load data for both models")
        return None

    # Process each lead time (column)
    for col_idx, lead_time in enumerate(lead_times):
        print(f"\n{'='*60}")
        print(f"Processing lead time: {lead_time_labels[lead_time]}")
        print(f"{'='*60}")

        ax = axes[col_idx]

        # Plot each model type
        for idx, model_spec in enumerate(model_info):
            model_name = model_spec['model']
            model_type = model_spec['type']
            label = model_spec['label']
            color = model_spec['color']

            # Get the appropriate data
            if model_name == 'pangu':
                patch_data = pangu_data[lead_time]
            else:  # ifs
                patch_data = ifs_data[lead_time]

            if not patch_data:
                print(f"\nSkipping {label} - no data available")
                continue

            # Extract pixel-level data for binscatter
            pixel_data = _extract_pixel_level_data(
                patch_data, variable, lead_time, x_metric, sdor_da
            )

            if pixel_data is None or len(pixel_data['x']) == 0:
                print(f"\nSkipping {label} - no pixel data available")
                continue

            # Select the appropriate y values based on model type
            if model_type == 'original':
                y_values = pixel_data['rmse_original']
            else:  # corrected
                y_values = pixel_data['rmse_corrected']

            # Plot binscatter
            _plot_binscatter(
                ax=ax,
                x=pixel_data['x'],
                y=y_values,
                color=color,
                label=label,
                position_idx=idx,
                is_first=(idx == 0)
            )

        # Set labels and title for this panel
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel("RMSE", fontsize=12)
        ax.set_title(lead_time_labels[lead_time], fontsize=13, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Add legend to last panel
        if col_idx == 2:
            ax.legend(loc='upper right', fontsize=10, framealpha=0.9)

    # Add overall title
    title = f'{variable_title} - Model Comparison\n'
    title += f'{x_label} vs RMSE'
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.00)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Determine output directory
    if save_dir is None:
        out_folder = os.path.join(dirs["fig"], "model_comparison")
    else:
        out_folder = save_dir
    os.makedirs(out_folder, exist_ok=True)

    # Save figure
    x_suffix = "equator" if x_metric == "equator_distance" else "sdor"
    filename = f"model_compare_binscatter_{variable}_{x_suffix}.png"
    save_path = os.path.join(out_folder, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved model comparison plot to: {save_path}")

    return fig


def _plot_binscatter(ax, x, y, color, label,
                     nbins=None,
                     x_label=None,
                     y_label=None,
                     marker_size=6,
                     position_idx=None,
                     lead_time=None,
                     is_first=False,
                     add_zero_line=False,
                     add_styling=False):
    """
    Create a binscatter plot with cubic polynomial fit.

    Uses quantile-based binning (via binsreg), confidence intervals around bin means,
    and fits a cubic polynomial to the bin means to capture non-linear relationships.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    x : np.ndarray
        X values (independent variable)
    y : np.ndarray
        Y values (dependent variable)
    color : str
        Color for this series
    label : str
        Label for this series
    nbins : int or None, default=None
        Number of bins. If None, uses automatic IMSE-optimal selection
    x_label : str, optional
        Label for x-axis (only used if add_styling=True)
    y_label : str, optional
        Label for y-axis (only used if add_styling=True)
    marker_size : float, default=6
        Size of markers (for add_styling=True, will be square-rooted)
    position_idx : int, optional
        Index for positioning text boxes (for overlay plots, currently unused)
    lead_time : int, optional
        Lead time in hours (for overlay plots, currently unused)
    is_first : bool, default=False
        Whether this is the first series being plotted
    add_zero_line : bool, default=False
        Whether to add horizontal line at y=0 (only if is_first=True)
    add_styling : bool, default=False
        Whether to add axis labels, grid, and legend (for standalone plots)

    Returns
    -------
    None
    """
    # Remove any remaining NaN values
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    x = x[valid_mask]
    y = y[valid_mask]

    if len(x) == 0:
        print(f"  Warning: No valid data for {label}")
        return

    print(f"  Processing {label}: {len(x):,} observations")

    # Create a DataFrame for binsreg
    df = pd.DataFrame({'x': x, 'y': y})

    # Run binsreg to create bins and confidence intervals
    print(f"  Running binsreg for binning...")

    try:
        # Try with automatic or specified bin selection
        est = binsreg(
            y='y',
            x='x',
            data=df,
            nbins=nbins,       # Automatic IMSE-optimal if None, else fixed
            binspos='qs',      # Quantile-spaced bins
            dots=(0, 0),       # Point estimates at bin means
            ci=(3, 3),         # Confidence intervals (cubic)
            noplot=True        # Don't create automatic plot
        )

        # Check if dots were actually created
        if est.data_plot is None or len(est.data_plot) == 0 or est.data_plot[0].dots is None:
            raise ValueError("binsreg did not create dots")

    except (ValueError, Exception) as e:
        print(f"  Using fixed 20 bins...")
        est = binsreg(
            y='y',
            x='x',
            data=df,
            nbins=20,          # Fixed 20 bins
            binspos='qs',      # Quantile-spaced bins
            dots=(0, 0),       # Point estimates at bin means
            ci=(3, 3),         # Confidence intervals (cubic)
            noplot=True        # Don't create automatic plot
        )

    # Get the data object
    data_obj = est.data_plot[0]

    # Get binned points from .dots DataFrame
    dots_df = data_obj.dots

    if dots_df is None:
        # Manual binning fallback
        print(f"  Using manual binning fallback...")

        if data_obj.data_bin is not None:
            bin_info = data_obj.data_bin
            n_bins = len(bin_info)

            bin_x = np.zeros(n_bins)
            bin_y = np.zeros(n_bins)
            ci_l = np.zeros(n_bins)
            ci_r = np.zeros(n_bins)

            for i in range(n_bins):
                left = bin_info.iloc[i]['left_endpoint']
                right = bin_info.iloc[i]['right.endpoint']

                if i == n_bins - 1:
                    mask = (x >= left) & (x <= right)
                else:
                    mask = (x >= left) & (x < right)

                if np.sum(mask) > 0:
                    x_bin = x[mask]
                    y_bin = y[mask]

                    bin_x[i] = np.mean(x_bin)
                    bin_y[i] = np.mean(y_bin)

                    if len(y_bin) > 1:
                        se = np.std(y_bin, ddof=1) / np.sqrt(len(y_bin))
                        ci_l[i] = bin_y[i] - 1.96 * se
                        ci_r[i] = bin_y[i] + 1.96 * se
                    else:
                        ci_l[i] = bin_y[i]
                        ci_r[i] = bin_y[i]
                else:
                    bin_x[i] = (left + right) / 2
                    bin_y[i] = np.nan
                    ci_l[i] = np.nan
                    ci_r[i] = np.nan

            # Remove empty bins
            valid_mask = ~np.isnan(bin_y)
            bin_x = bin_x[valid_mask]
            bin_y = bin_y[valid_mask]
            ci_l = ci_l[valid_mask]
            ci_r = ci_r[valid_mask]
        else:
            print(f"  ERROR: Cannot create binscatter")
            return
    else:
        # Normal path: dots is available
        bin_x = dots_df['x'].values
        bin_y = dots_df['fit'].values

        # Get confidence intervals from .ci DataFrame
        ci_df = data_obj.ci
        ci_l = ci_df['ci_l'].values
        ci_r = ci_df['ci_r'].values

    # Calculate error bars
    yerr_lower = np.abs(bin_y - ci_l)
    yerr_upper = np.abs(ci_r - bin_y)
    yerr = np.array([yerr_lower, yerr_upper])

    # Plot binscatter points with error bars
    scatter_label = label if not add_styling else 'Bin means (95% CI)'
    actual_marker_size = np.sqrt(marker_size) if add_styling else marker_size
    ax.errorbar(bin_x, bin_y, yerr=yerr,
                fmt='o', color=color,
                markersize=actual_marker_size,
                ecolor=color, alpha=0.7, capsize=3, capthick=1.5,
                label=scatter_label)

    # Fit a cubic polynomial to the bin means
    if len(bin_x) >= 4:  # Need at least 4 points to fit a cubic
        # Fit cubic polynomial: y = a*x^3 + b*x^2 + c*x + d
        coeffs = np.polyfit(bin_x, bin_y, deg=3)

        # Create smooth x values for plotting the curve
        x_smooth = np.linspace(bin_x.min(), bin_x.max(), 200)
        y_smooth = np.polyval(coeffs, x_smooth)

        # Plot cubic fit line
        line_label = 'Cubic fit' if add_styling else None
        line_style = 'r--' if add_styling else '--'
        line_color = None if add_styling else color
        ax.plot(x_smooth, y_smooth, line_style, color=line_color,
                linewidth=2, alpha=0.8, label=line_label)

        print(f"  Plotted cubic polynomial fit to {len(bin_x)} bin means")
    else:
        print(f"  Warning: Need at least 4 bins to fit cubic polynomial, only have {len(bin_x)}")

    # Add zero line if requested
    if add_zero_line and is_first:
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Add styling if requested (for standalone plots)
    if add_styling:
        if x_label:
            ax.set_xlabel(x_label, fontsize=12)
        if y_label:
            ax.set_ylabel(y_label, fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='lower right', fontsize=8, framealpha=0.9)

    print(f"  Completed {label}")


def main():


    dirs = setup_directories()

    # load in forecast data and save as pickle files
    # only needs to be run once
    # for model in ["pangu", "ifs"]:
    #     for variable in ["2m_temperature", "10m_wind_speed"]:
    #         patch_data = load_region_data(
    #             dirs=dirs,
    #             model=model,
    #             variable=variable
    #         )
    #         # save to pickle for use in plotting

    #         import pickle
    #         with open(os.path.join(dirs["processed"], f"{model}_{variable}_patch_data.pkl"), "wb") as f:
    #             pickle.dump(patch_data, f)


    #=============================================
    # Global Improvement Map Plots
    #=============================================
    for model in ["pangu", "ifs"]:
        for variable in ["2m_temperature", "10m_wind_speed"]:
            for map_type in ["original", "improvement"]:
                for pixel_flag in [False, True]: # if the map is pixel-level or region-mean
                    map_global_improvements(dirs=dirs, model=model, 
                                            variable=variable, map_type=map_type,
                                            pixel_level=pixel_flag)
    #=============================================
    # Binscatter Plots
    #=============================================
    # overlaying lead times for single model: Currently plot used for main paper 
    for model in ["pangu", "ifs"]:
        for x_metric in ["sdor", "equator_distance"]:
            _ = lead_time_compare_binscatter(
                dirs=dirs,
                model=model,
                x_metric=x_metric
            )
    
    # Create plot comparing model binscatters: Currently in Appendix
    for x_metric in ["sdor", "equator_distance"]:
        for variable in ["2m_temperature", "10m_wind_speed"]:
            _ = model_compare_binscatter(
                dirs=dirs,
                variable=variable,
                x_metric=x_metric
            )

if __name__ == "__main__":
    main()