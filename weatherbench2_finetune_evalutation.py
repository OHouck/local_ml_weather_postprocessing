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
import matplotlib.gridspec as gridspec

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

    raw_data_dir = "/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/data/raw"

    # OH: eventually would like to select region using lat/lon bounds
    # region = pakistan
    

    # variables used to finetune the model. Uncomment the one you want to use
    # vars_trained_using = "2m_temperature"
    # vars_trained_using = "10m_u_component_of_wind"
    vars_trained_using = "10m_u_component_of_wind_10m_v_component_of_wind"

    # variable to plot
    # var_name = "2m_temperature"
    # var_name = "10m_u_component_of_wind"
    var_name = "wind_speed"
    

    # OH: eventually would like to be able to add additional models
    model = "pangu"

    # If you need a specific level (e.g., 850 hPa), set this to an integer or None
    level = "" # or None if not needed

    var_name_original = f"{var_name}_original"
    var_name_corrected = f"{var_name}_corrected"
    var_name_groundtruth = f"{var_name}_groundtruth"

    # ==========================================================================
    # Load Data
    # ==========================================================================

    forecast_path = os.path.join(input_dir, f"{model}_forecasts_{vars_trained_using}{level}.zarr")
    print(f"Loading forecast data from {forecast_path}")
    ds_forecasts = xr.open_zarr(forecast_path) if forecast_path.endswith(".zarr") else xr.open_dataset(forecast_path)

    # if want to plot windspeed, need to compute it from u and v components
    if var_name == "wind_speed":

        for string in "corrected", "original", "groundtruth":
            u_component = ds_forecasts[f"10m_u_component_of_wind_{string}"]
            v_component = ds_forecasts[f"10m_v_component_of_wind_{string}"]
            wind_speed = np.sqrt(u_component**2 + v_component**2)
            ds_forecasts[f"wind_speed_{string}"] = wind_speed

    ground_truth = ds_forecasts[var_name_groundtruth]
    fc_original = ds_forecasts[var_name_original]
    fc_corrected = ds_forecasts[var_name_corrected]

    # print lat/lon bounds
    print(f"Latitude bounds: {ground_truth.latitude.min().values} to {ground_truth.latitude.max().values}")
    print(f"Longitude bounds: {ground_truth.longitude.min().values} to {ground_truth.longitude.max().values}")

    # ==========================================================================
    # Plot MSE by month
    # ==========================================================================

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

    plt.title(f"Monthly MSE comparison for Pangu {var_name} using 2022 data \n(Original vs Corrected)")
    plt.xlabel("Month")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the time-series figure
    save_path = os.path.join(fig_dir, f"mse_time_series_{var_name}_{model}_traind_with_{vars_trained_using}.png")
    plt.savefig(save_path, dpi=150)
    print(f"Time-series plot saved to {save_path}")
    plt.close()

    # =============================================================================
    # Create and Save Spatial MSE Maps with Base Map (Country Outlines)
    # =============================================================================
        # Determine common color scale limits for the original and corrected forecasts
    vmin = float(min(mse_spatial_orig.min().values, mse_spatial_corr.min().values))
    vmax = float(max(mse_spatial_orig.max().values, mse_spatial_corr.max().values))

    # Create a figure with 2 rows:
    #   Row 1: two subplots (original and corrected forecasts, sharing one colorbar)
    #   Row 2: one subplot for the MSE difference, spanning both columns.
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
    
    # First row subplots
    ax0 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax1 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
    # Second row: difference plot spanning both columns
    ax2 = fig.add_subplot(gs[1, :], projection=ccrs.PlateCarree())

    # Plot original forecast spatial MSE with colorbar
    im0 = mse_spatial_orig.plot(ax=ax0, cmap='viridis', add_colorbar=False,
                                vmin=vmin, vmax=vmax)
    ax0.set_title("Original Forecast MSE")
    ax0.set_xlabel("Longitude")
    ax0.set_ylabel("Latitude")
    ax0.coastlines()
    ax0.add_feature(cfeature.BORDERS, linestyle=':')
    ax0.add_feature(cfeature.LAND, facecolor='lightgray')
    ax0.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    # Plot corrected forecast spatial MSE without an extra colorbar
    mse_spatial_corr.plot(ax=ax1, cmap='viridis', add_colorbar=False,
                          vmin=vmin, vmax=vmax)
    ax1.set_title("Corrected Forecast MSE")
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.coastlines()
    ax1.add_feature(cfeature.BORDERS, linestyle=':')
    ax1.add_feature(cfeature.LAND, facecolor='lightgray')
    ax1.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    # Plot MSE difference (corrected - original) in the second row with its own colorbar
    mse_diff = mse_spatial_corr - mse_spatial_orig
    im2 = mse_diff.plot(ax=ax2, cmap='coolwarm', add_colorbar=False)
    ax2.set_title("MSE Difference (Corrected - Original)")
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    ax2.coastlines()
    ax2.add_feature(cfeature.BORDERS, linestyle=':')
    ax2.add_feature(cfeature.LAND, facecolor='lightgray')
    ax2.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    # Manually add a colorbar for the MSE plots.
    # Get the current position of ax2 and create a new axis to the right.
    pos = ax0.get_position()
    cax2 = fig.add_axes([pos.x1 + 0.02, pos.y0, 0.02, pos.height])
    fig.colorbar(im0, cax=cax2, label='MSE')


    # Manually add a colorbar for the difference plot.
    # Get the current position of ax2 and create a new axis to the right.
    pos = ax2.get_position()
    cax2 = fig.add_axes([pos.x1 + 0.02, pos.y0, 0.02, pos.height])
    fig.colorbar(im2, cax=cax2, label='MSE Difference')

    plt.suptitle(f"Spatial MSE Map for {model} - {var_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    save_path_spatial = os.path.join(fig_dir, f"mse_spatial_{var_name}_{model}_trained_with_{vars_trained_using}.png")
    plt.savefig(save_path_spatial, dpi=150)
    print(f"Spatial MSE plot with base map and elevation gradients saved to {save_path_spatial}")
    plt.close()


if __name__ == "__main__":
    main()
