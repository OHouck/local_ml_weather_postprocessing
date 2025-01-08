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



# =============================================================================
# Helper Functions
# =============================================================================


# =============================================================================
# Main Script
# =============================================================================

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
    var_name = "2m_temperature"  

    # OH: eventaully would like to be able to add additional models
    model = "pangu_test"

    # If you need a specific level (e.g., 850 hPa), set this to an integer or None
    level = ""# or None if not needed

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

    print(f"Ground truth shape: {ground_truth.shape}")
    print(f"Original forecast shape: {fc_original.shape}")
    print(f"Corrected forecast shape: {fc_corrected.shape}")
    
    # Create a plot for each model
    plt.figure(figsize=(8, 5))
    
    # Align with obs (ensures time, longitude, latitude match)
    # If the time range is the same, it should align easily. If not, adjust 'join' arg
    fc_orig_aligned, ground_truth_aligned = xr.align(fc_original, ground_truth, join="inner")
    fc_corr_aligned, ground_truth_aligned = xr.align(fc_corrected, ground_truth, join="inner")
    
    # 4. Compute MSE over lat/lon, keep time dimension to see how MSE evolves
    # If you want just a single numeric MSE, include 'time' in the mean dimension below.
    mse_orig = ((fc_orig_aligned - ground_truth_aligned) ** 2).mean(dim=["longitude","latitude"])
    mse_corr = ((fc_corr_aligned - ground_truth_aligned) ** 2).mean(dim=["longitude","latitude"])

    # take mse over all dimensions
    mse_total_orig = ((fc_orig_aligned - ground_truth_aligned) ** 2).mean()
    mse_total_corr = ((fc_corr_aligned - ground_truth_aligned) ** 2).mean()

    print(f"Total MSE for original: {mse_total_orig.values}")
    print(f"Total MSE for corrected: {mse_total_corr.values}")
    
    # 5. Plot the time-series of MSE
    # We'll put them all on the same figure for easy comparison
    label_orig = f"{model}_original"
    label_corr = f"{model}_corrected"
    
    mse_orig.plot(label=label_orig, color="green")
    mse_corr.plot(label=label_corr, color="lightgreen")

    plt.title(f"Time-series MSE comparison for {model}\n(Original vs Corrected)")
    plt.xlabel("Time")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 6. Save the figure
    save_path = os.path.join(fig_dir, f"mse_time_series_{var_name}_{model}.png")
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    main()
