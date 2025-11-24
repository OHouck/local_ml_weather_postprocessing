#!/usr/bin/env python3
"""
Diagnostic script to understand pixel-level mapping issues.
Run this after generating a global map to check for potential smearing causes.
"""

import numpy as np
import xarray as xr
from pathlib import Path
import sys


def diagnose_coordinate_distribution(zarr_paths):
    """
    Analyze coordinate spacing and distribution from patch files.

    Args:
        zarr_paths: List of paths to patch zarr files
    """
    print("=" * 80)
    print("COORDINATE DISTRIBUTION ANALYSIS")
    print("=" * 80)

    all_lats = []
    all_lons = []

    for path in zarr_paths:
        if not Path(path).exists():
            print(f"Warning: {path} does not exist, skipping...")
            continue

        ds = xr.open_zarr(path)
        all_lats.extend(ds.latitude.values.tolist())
        all_lons.extend(ds.longitude.values.tolist())

    all_lats = np.array(all_lats)
    all_lons = np.array(all_lons)

    unique_lats = np.unique(all_lats)
    unique_lons = np.unique(all_lons)

    print(f"\nTotal coordinates: {len(all_lats)} lat values, {len(all_lons)} lon values")
    print(f"Unique coordinates: {len(unique_lats)} latitudes, {len(unique_lons)} longitudes")

    # Check latitude spacing
    if len(unique_lats) > 1:
        lat_diffs = np.diff(unique_lats)
        print(f"\nLatitude spacing:")
        print(f"  Min: {lat_diffs.min():.6f}°")
        print(f"  Max: {lat_diffs.max():.6f}°")
        print(f"  Mean: {lat_diffs.mean():.6f}°")
        print(f"  Std: {lat_diffs.std():.6f}°")

        # Check for non-uniform spacing
        if lat_diffs.std() > 0.001:
            print(f"  ⚠️  NON-UNIFORM latitude spacing detected!")
            print(f"     This is normal if combining different regional patches")

    # Check longitude spacing
    if len(unique_lons) > 1:
        lon_diffs = np.diff(unique_lons)
        print(f"\nLongitude spacing:")
        print(f"  Min: {lon_diffs.min():.6f}°")
        print(f"  Max: {lon_diffs.max():.6f}°")
        print(f"  Mean: {lon_diffs.mean():.6f}°")
        print(f"  Std: {lon_diffs.std():.6f}°")

        if lon_diffs.std() > 0.001:
            print(f"  ⚠️  NON-UNIFORM longitude spacing detected!")
            print(f"     This is normal if combining different regional patches")

    # Check for coordinate duplicates at different precision levels
    print(f"\nCoordinate precision analysis:")
    for precision in [3, 4, 5, 6, 7]:
        lats_rounded = np.round(all_lats, precision)
        lons_rounded = np.round(all_lons, precision)

        unique_lats_rounded = np.unique(lats_rounded)
        unique_lons_rounded = np.unique(lons_rounded)

        lat_collision_rate = (len(all_lats) - len(unique_lats_rounded)) / len(all_lats) * 100
        lon_collision_rate = (len(all_lons) - len(unique_lons_rounded)) / len(all_lons) * 100

        print(f"  At {precision} decimals: {lat_collision_rate:.2f}% lat collisions, "
              f"{lon_collision_rate:.2f}% lon collisions")

        if lat_collision_rate > 1 or lon_collision_rate > 1:
            print(f"    ⚠️  Significant coordinate collisions at {precision} decimal places!")
            print(f"       This could cause smearing if using this precision")

    # Check longitude wrapping
    print(f"\nLongitude range analysis:")
    print(f"  Min longitude: {unique_lons.min():.2f}°")
    print(f"  Max longitude: {unique_lons.max():.2f}°")
    print(f"  Span: {unique_lons.max() - unique_lons.min():.2f}°")

    # Check if data crosses 0° meridian or 180° meridian
    crosses_zero = unique_lons.min() < 0 and unique_lons.max() > 0
    crosses_180 = unique_lons.max() > 170 and unique_lons.min() < -170

    if crosses_zero:
        print(f"  ✓ Data crosses 0° meridian (prime meridian)")
    if crosses_180:
        print(f"  ✓ Data crosses 180° meridian (date line)")

    # Check for suspicious gaps that might cause rendering artifacts
    print(f"\nLarge gaps in longitude (potential rendering issues):")
    large_gaps = lon_diffs > 2 * np.median(lon_diffs)
    if np.any(large_gaps):
        gap_indices = np.where(large_gaps)[0]
        for idx in gap_indices[:10]:  # Show first 10
            print(f"  Gap at index {idx}: {unique_lons[idx]:.2f}° to {unique_lons[idx+1]:.2f}° "
                  f"(Δ = {lon_diffs[idx]:.2f}°)")
        if len(gap_indices) > 10:
            print(f"  ... and {len(gap_indices) - 10} more gaps")
    else:
        print(f"  No large gaps detected")


def diagnose_data_values(zarr_path, variable='2m_temperature', lead_time=24):
    """
    Analyze the actual data values to check for anomalies.

    Args:
        zarr_path: Path to a sample zarr file
        variable: Variable name to check
        lead_time: Lead time to examine
    """
    print("\n" + "=" * 80)
    print(f"DATA VALUE ANALYSIS: {variable} at {lead_time}h lead time")
    print("=" * 80)

    if not Path(zarr_path).exists():
        print(f"File {zarr_path} does not exist, skipping...")
        return

    ds = xr.open_zarr(zarr_path)
    var_suffix = f"_lt{lead_time}h"

    try:
        ground_truth = ds[f"{variable}_ground_truth{var_suffix}"]
        original = ds[f"{variable}_original{var_suffix}"]
        corrected = ds[f"{variable}_corrected{var_suffix}"]
    except KeyError as e:
        print(f"Variable not found: {e}")
        return

    # Load data
    gt_data = ground_truth.values
    orig_data = original.values
    corr_data = corrected.values

    # Compute RMSE
    mse_orig = np.mean((orig_data - gt_data) ** 2, axis=0)
    mse_corr = np.mean((corr_data - gt_data) ** 2, axis=0)

    rmse_orig = np.sqrt(mse_orig)
    rmse_corr = np.sqrt(mse_corr)

    improvement = ((rmse_orig - rmse_corr) / (rmse_orig + 1e-10) * 100)

    print(f"\nOriginal RMSE:")
    print(f"  Min: {np.nanmin(rmse_orig):.2f}")
    print(f"  Max: {np.nanmax(rmse_orig):.2f}")
    print(f"  Mean: {np.nanmean(rmse_orig):.2f}")

    print(f"\nCorrected RMSE:")
    print(f"  Min: {np.nanmin(rmse_corr):.2f}")
    print(f"  Max: {np.nanmax(rmse_corr):.2f}")
    print(f"  Mean: {np.nanmean(rmse_corr):.2f}")

    print(f"\nImprovement (%):")
    print(f"  Min: {np.nanmin(improvement):.1f}%")
    print(f"  Max: {np.nanmax(improvement):.1f}%")
    print(f"  Mean: {np.nanmean(improvement):.1f}%")
    print(f"  Median: {np.nanmedian(improvement):.1f}%")

    # Check for spatial patterns that might cause rendering artifacts
    print(f"\nSpatial gradient analysis (looking for sharp transitions):")

    # Compute gradients
    grad_lat = np.abs(np.diff(improvement, axis=0))
    grad_lon = np.abs(np.diff(improvement, axis=1))

    print(f"  Latitude gradient (max): {np.nanmax(grad_lat):.1f}% per pixel")
    print(f"  Longitude gradient (max): {np.nanmax(grad_lon):.1f}% per pixel")

    # Find largest gradients
    large_lat_grad = grad_lat > 20  # More than 20% change per pixel
    large_lon_grad = grad_lon > 20

    if np.any(large_lat_grad):
        print(f"  ⚠️  Found {np.sum(large_lat_grad)} pixels with large latitudinal gradients (>20%/pixel)")
    if np.any(large_lon_grad):
        print(f"  ⚠️  Found {np.sum(large_lon_grad)} pixels with large longitudinal gradients (>20%/pixel)")
        print(f"     These could appear as visible bands/streaks on the map")


if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                 PIXEL-LEVEL MAPPING DIAGNOSTIC TOOL                        ║
║                                                                            ║
║  This tool helps diagnose the cause of smearing/artifacts in global maps  ║
╚════════════════════════════════════════════════════════════════════════════╝
""")

    # Example usage - modify these paths to match your data
    data_dir = Path.home() / "ai_weather_ag" / "data" / "fine_tuning_output"
    data_dir = Path("/Users/ohouck/globus/forecast_data/processed/finetuning_output/pangu/africa")

    print("Looking for zarr files in:", data_dir)
    zarr_files = list(data_dir.glob("*2m_temperature*.zarr"))

    if not zarr_files:
        print("\nNo zarr files found. Please specify paths manually:")
        print("\nUsage:")
        print("  python diagnose_smearing.py")
        print("\nOr edit the script to set the correct data_dir path")
        sys.exit(1)

    print(f"\nFound {len(zarr_files)} zarr files")
    for f in zarr_files[:5]:
        print(f"  - {f.name}")
    if len(zarr_files) > 5:
        print(f"  ... and {len(zarr_files) - 5} more")

    # Run coordinate analysis on all files
    diagnose_coordinate_distribution(zarr_files)

    # Run data analysis on first file as example
    if zarr_files:
        print(f"\n\nAnalyzing data from: {zarr_files[0].name}")
        diagnose_data_values(zarr_files[0])

    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)
    print("""
Key things to look for:
  1. Coordinate collisions at your chosen precision level
  2. Non-uniform grid spacing (normal, but pcolormesh needed, not imshow)
  3. Large gaps in longitude (can cause rendering artifacts)
  4. Large spatial gradients (can appear as visible bands/streaks)

If you see coordinate collisions > 1% at precision 6, that was causing smearing.
The new exact-matching approach should eliminate this issue.
""")
