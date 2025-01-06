# eval_weatherbench2_finetuning.py
# Author: Ozzy Houck (modified by GPT)
# Date Created: 06/27/2024
#
# Purpose: take .zarr outputs from weatherbench2_finetuning.py (which include
# both original and corrected forecasts) and create figures showing the RMSE
# improvement compared to observational data.

import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt



# =============================================================================
# Helper Functions
# =============================================================================

def compute_rmse(forecast: xr.DataArray, obs: xr.DataArray, dims=('time','longitude','latitude')) -> xr.DataArray:
    """
    Compute the RMSE over the given dimensions. 
    By default, we take a mean over time, lat, and lon, or whatever is provided.
    If you want a time-series of RMSE, remove 'time' from dims in the .mean(...) call.
    """
    return np.sqrt(((forecast - obs) ** 2).mean(dim=dims))

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
    var_name = "temperature"

    # OH: eventaully would like to be able to add additional models
    model = "pangu_test"

    # If you need a specific level (e.g., 850 hPa), set this to an integer or None
    level = 850  # or None if not needed

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
    
    # 4. Compute RMSE over lat/lon, keep time dimension to see how RMSE evolves
    # If you want just a single numeric RMSE, include 'time' in the mean dimension below.
    rmse_orig = np.sqrt(((fc_orig_aligned - ground_truth_aligned) ** 2).mean(dim=["longitude","latitude"]))
    rmse_corr = np.sqrt(((fc_corr_aligned - ground_truth_aligned) ** 2).mean(dim=["longitude","latitude"]))
    
    # 5. Plot the time-series of RMSE
    # We'll put them all on the same figure for easy comparison
    label_orig = f"{model}_original"
    label_corr = f"{model}_corrected"
    
    rmse_orig.plot(label=label_orig, color="green")
    rmse_corr.plot(label=label_corr, color="lightgreen")

    plt.title(f"Time-series RMSE comparison for {model}\n(Original vs Corrected)")
    plt.xlabel("Time")
    plt.ylabel("RMSE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 6. Save the figure
    save_path = os.path.join(fig_dir, f"rmse_time_series_{var_name}_{model}.png")
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    main()
