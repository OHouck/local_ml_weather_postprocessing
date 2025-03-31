# eval_weatherbench2_finetuning.py
# Author: Ozzy Houck 
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
import socket

def setup_directories():
    # check if we are on the server or local
    nodename = socket.gethostname()
    if nodename == "oMac.local": # local laptop
        root = os.path.expanduser("~/OneDrive - The University of Chicago/ai_weather_ag/data")
    else:
        raise Exception("Unknown environment, Please specify the root directory")

    dirs = {
        'root': root,
        'raw': os.path.join(root, "raw"),
        'processed': os.path.join(root, "processed"),
        'fig': os.path.join(root, "../figures/finetuning"),
        'input': "~/wb_finetune_test"
    }

    for path in dirs.values():
        os.makedirs(path, exist_ok=True)

    return dirs

def main():


    # ==========================================================================
    # User Configuration
    # ==========================================================================

    # Where the Zarr outputs from weatherbench2_finetuning.py are stored
    # Each model (pangu, ifs, neural_gcm, etc.) should have a .zarr directory
    # containing the ground truth, original forecast, and corrected forecast.

    # OH: eventually would like to select region using lat/lon bounds
    # region = pakistan
    

    # variables used to finetune the model. Uncomment the one you want to use
    vars_trained_using = "2m_temperature"
    # vars_trained_using = "10m_u_component_of_wind"
    # vars_trained_using = "10m_u_component_of_wind_10m_v_component_of_wind"


    model = "pangu"
    region = "north_india"
    train_start_time = "2018-01-01"
    train_end_time = "2021-12-31"
    test_start_time = "2022-01-01"
    test_end_time = "2022-12-31"
    lead_time = 24
    training_vars = "2m_temperature" 
    output_vars = "2m_temperature" 
    num_layers = 5
    num_units = 512


    var_name = output_vars # change if there are more than 1
    var_name_original = f"{var_name}_original"
    var_name_corrected = f"{var_name}_corrected"
    var_name_groundtruth = f"{var_name}_ground_truth"

    # ==========================================================================
    # Load Data
    # ==========================================================================

    dir = setup_directories()

    filename = f"{model}_{region}_train{train_start_time}-{train_end_time}_test{test_start_time}-{test_end_time}_{lead_time}h_train_{training_vars}_output{output_vars}_mlp{num_units}x{num_layers}.zarr"
    # filename = "pangu_north_india_train2018-01-01-2021-12-30_test2022-01-01-2022-12-30_48h_train_2m_temperature_output2m_temperature_mlp512x5.zarr"

    # forecast_path = os.path.join(dir['input'], f"{model}_forecasts_{vars_trained_using}{level}.zarr")
    forecast_path = os.path.join(dir['input'], filename)
    print(f"Loading forecast data from {forecast_path}")
    ds_forecasts = xr.open_zarr(forecast_path) if forecast_path.endswith(".zarr") else xr.open_dataset(forecast_path)
    print(ds_forecasts)


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

    # Compute spatial raw average over time for mapping
    raw_spartial_orig = fc_orig_aligned.mean(dim=["time"])
    raw_spartial_corr = fc_corr_aligned.mean(dim=["time"])
    raw_spatial_diff = raw_spartial_corr - raw_spartial_orig

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
    plt.show()

    # Save the time-series figure
    save_path = os.path.join(dir["fig"], f"mse_time_series_{region}_{var_name}_{model}_traind_with_{vars_trained_using}.png")
    # plt.savefig(save_path, dpi=150)
    print(f"Time-series plot saved to {save_path}")
    plt.close()

    # =============================================================================
    # Create and Save Spatial Raw Maps with Base Map (Country Outlines)
    # =============================================================================
        # Determine common color scale limits for the original and corrected forecasts
    vmin = float(min(raw_spartial_orig.min().values, raw_spartial_corr.min().values))
    vmax = float(max(raw_spartial_orig.max().values, raw_spartial_corr.max().values))

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
    im0 = raw_spartial_orig.plot(ax=ax0, cmap='viridis', add_colorbar=False,
                                vmin=vmin, vmax=vmax)
    ax0.set_title("Original Forecast Values")
    ax0.set_xlabel("Longitude")
    ax0.set_ylabel("Latitude")
    ax0.coastlines()
    ax0.add_feature(cfeature.BORDERS, linestyle=':')
    ax0.add_feature(cfeature.LAND, facecolor='lightgray')
    ax0.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    # Plot corrected forecast spatial MSE without an extra colorbar
    raw_spartial_corr.plot(ax=ax1, cmap='viridis', add_colorbar=False,
                          vmin=vmin, vmax=vmax)
    ax1.set_title("Corrected Forecast Values")
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.coastlines()
    ax1.add_feature(cfeature.BORDERS, linestyle=':')
    ax1.add_feature(cfeature.LAND, facecolor='lightgray')
    ax1.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    # Plot MSE difference (corrected - original) in the second row with its own colorbar
    im2 = raw_spatial_diff.plot(ax=ax2, cmap='coolwarm', add_colorbar=False)
    ax2.set_title("Difference (Corrected - Original)")
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
    fig.colorbar(im0, cax=cax2, label='Degrees K')


    # Manually add a colorbar for the difference plot.
    # Get the current position of ax2 and create a new axis to the right.
    pos = ax2.get_position()
    cax2 = fig.add_axes([pos.x1 + 0.02, pos.y0, 0.02, pos.height])
    fig.colorbar(im2, cax=cax2, label='MSE Difference')

    plt.suptitle(f"Spatial MSE Map for {model} - {var_name}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    save_path_spatial = os.path.join(dir["fig"], f"mse_spatial_{region}_{var_name}_{model}_trained_with_{vars_trained_using}.png")
    # plt.savefig(save_path_spatial, dpi=150)
    print(f"Spatial MSE plot with base map and elevation gradients saved to {save_path_spatial}")
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
    plt.show()

    save_path_spatial = os.path.join(dir["fig"], f"mse_spatial_{region}_{var_name}_{model}_trained_with_{vars_trained_using}.png")
    # plt.savefig(save_path_spatial, dpi=150)
    print(f"Spatial MSE plot with base map and elevation gradients saved to {save_path_spatial}")
    plt.close()


if __name__ == "__main__":
    main()
