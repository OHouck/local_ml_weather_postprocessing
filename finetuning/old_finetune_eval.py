# eval_weatherbench2_finetuning.py
# Author: Ozzy Houck (modified by GPT)
# Date Created: 06/27/2024
#
# Purpose: take .zarr outputs from weatherbench2_finetuning.py (which include
# both original and corrected forecasts) and create figures showing the MSE
# improvement compared to observational data.

import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import calendar
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def main():

    # ==========================================================================
    # User Configuration
    # ==========================================================================

    # Where the Zarr outputs from weatherbench2_finetuning.py are stored
    # Each model (pangu, ifs, neural_gcm, etc.) should have a .zarr directory
    # containing the ground truth, original forecast, and corrected forecast.
    input_dir = "~/wb_finetune_test" 

    # Where to save output figures
    fig_dir = "/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/figures/finetuning_weatherbench2"

    # OH: eventually would like to select region using lat/lon bounds
    # region = 

    # Variable name(s) we want to compare. Adjust as needed.
    # For example: "temperature" at level=850 hPa or "2m_temperature"
    var_name = "10m_u_component_of_wind"  

    # OH: eventually would like to be able to add additional models
    model = "pangu_test"

    # If you need a specific level (e.g., 850 hPa), set this to an integer or None
    level = "" # or None if not needed

    var_name_original = f"{var_name}_original"
    var_name_corrected = f"{var_name}_corrected"
    var_name_groundtruth = f"{var_name}_groundtruth"

    # ==========================================================================
    # Load Data
    # ==========================================================================

    forecast_path = os.path.join(input_dir, f"{model}_forecasts_{var_name}{level}.zarr")
    print(f"Loading forecast data from {forecast_path}")
    ds_forecasts = xr.open_zarr(forecast_path) if forecast_path.endswith(".zarr") else xr.open_dataset(forecast_path)

    ground_truth = ds_forecasts[var_name_groundtruth]
    fc_original = ds_forecasts[var_name_original]
    fc_corrected = ds_forecasts[var_name_corrected]

    # print lat/lon bounds
    print(f"Latitude bounds: {ground_truth.latitude.min().values} to {ground_truth.latitude.max().values}")
    print(f"Longitude bounds: {ground_truth.longitude.min().values} to {ground_truth.longitude.max().values}")

    # for original and corrected forecasts, print lat and long bounds
    print(f"Latitude bounds for original forecast: {fc_original.latitude.min().values} to {fc_original.latitude.max().values}")
    print(f"Longitude bounds for original forecast: {fc_original.longitude.min().values} to {fc_original.longitude.max().values}")

    print(f"Latitude bounds for corrected forecast: {fc_corrected.latitude.min().values} to {fc_corrected.latitude.max().values}")
    print(f"Longitude bounds for corrected forecast: {fc_corrected.longitude.min().values} to {fc_corrected.longitude.max().values}")


    # Create a plot for each model
    plt.figure(figsize=(8, 5))
    
    # Align with obs (ensures time, longitude, latitude match)
    fc_orig_aligned, ground_truth_aligned = xr.align(fc_original, ground_truth, join="inner")
    fc_corr_aligned, ground_truth_aligned = xr.align(fc_corrected, ground_truth, join="inner")
    
    # Compute MSE over lat/lon, then group by month to see evolution over time
    mse_orig = ((fc_orig_aligned - ground_truth_aligned) ** 2).mean(dim=["longitude","latitude"]).groupby('time.month').mean(dim='time')
    mse_corr = ((fc_corr_aligned - ground_truth_aligned) ** 2).mean(dim=["longitude","latitude"]).groupby('time.month').mean(dim='time')

    # Compute spatial MSE (averaged over time) for mapping across space
    mse_spatial_orig = ((fc_orig_aligned - ground_truth_aligned) ** 2).mean(dim=["time"])
    mse_spatial_corr = ((fc_corr_aligned - ground_truth_aligned) ** 2).mean(dim=["time"])

    # Compute overall MSE (scalar value)
    mse_total_orig = ((fc_orig_aligned - ground_truth_aligned) ** 2).mean()
    mse_total_corr = ((fc_corr_aligned - ground_truth_aligned) ** 2).mean()

    print(f"Total MSE for original: {mse_total_orig.values}")
    print(f"Total MSE for corrected: {mse_total_corr.values}")
    
    # 5. Plot the time-series of MSE
    label_orig = f"{model}_original"
    label_corr = f"{model}_corrected"
    
    mse_orig.plot(label=label_orig, color="green")
    mse_corr.plot(label=label_corr, color="lightgreen")

    # Convert month numbers to month names
    months = [calendar.month_name[i] for i in mse_orig['month'].values]

    plt.figure(figsize=(10, 6))
    plt.bar(months, mse_orig, width=0.4, label='Original MSE', align='center', color='green')
    plt.bar(months, mse_corr, width=0.4, label='Corrected MSE', align='edge', color='lightgreen')

    plt.title(f"Monthly MSE comparison for Pangu using 2022 data \n(Original vs Corrected)")
    plt.xlabel("Month")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the time-series figure
    save_path = os.path.join(fig_dir, f"mse_time_series_{var_name}_{model}.png")
    plt.savefig(save_path, dpi=150)
    print(f"Time-series plot saved to {save_path}")
    plt.close()

    # =============================================================================
    # Create and Save Spatial MSE Maps with Base Map (Country Outlines)
    # =============================================================================
     # Determine common color scale limits for spatial MSE maps
    vmin = float(min(mse_spatial_orig.min().values, mse_spatial_corr.min().values))
    vmax = float(max(mse_spatial_orig.max().values, mse_spatial_corr.max().values))

    # Create a new figure with two subplots (using a PlateCarree projection)
    fig, axes = plt.subplots(1, 3, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(14, 6))
    
    # Plot spatial MSE for the original forecast
    mse_spatial_orig.plot(ax=axes[0], cmap='viridis', add_colorbar=True, vmin=vmin, vmax=vmax)
    axes[0].set_title("Original Forecast MSE")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    axes[0].coastlines()
    axes[0].add_feature(cfeature.BORDERS, linestyle=':')
    axes[0].add_feature(cfeature.LAND, facecolor='lightgray')
    axes[0].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    # Plot spatial MSE for the corrected forecast
    mse_spatial_corr.plot(ax=axes[1], cmap='viridis', add_colorbar=True, vmin=vmin, vmax=vmax)
    axes[1].set_title("Corrected Forecast MSE")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    axes[1].coastlines()
    axes[1].add_feature(cfeature.BORDERS, linestyle=':')
    axes[1].add_feature(cfeature.LAND, facecolor='lightgray')
    axes[1].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    # Plot differences in MSE between the original and corrected forecasts
    mse_diff = mse_spatial_corr - mse_spatial_orig
    mse_diff.plot(ax=axes[2], cmap='coolwarm', add_colorbar=True)
    axes[2].set_title("MSE Difference (Corrected - Original)")
    axes[2].set_xlabel("Longitude")
    axes[2].set_ylabel("Latitude")
    axes[2].coastlines()
    axes[2].add_feature(cfeature.BORDERS, linestyle=':')
    axes[2].add_feature(cfeature.LAND, facecolor='lightgray')
    axes[2].gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    
    plt.suptitle(f"Spatial MSE Map for {model} - {var_name}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the spatial MSE figure with base map
    save_path_spatial = os.path.join(fig_dir, f"mse_spatial_{var_name}_{model}.png")
    plt.savefig(save_path_spatial, dpi=150)
    print(f"Spatial MSE plot with base map saved to {save_path_spatial}")
    plt.close()


if __name__ == "__main__":
    main()