#!/usr/bin/env python3
"""
Script to generate architecture comparison plots from run_custom_loss_experiments.sh outputs.

Usage:
    python3 finetuning/plot_custom_loss.py

Available options for loss_trained_on and metrics:

    loss_trained_on (loss function used during training):
        - "mse": Standard mean squared error loss (default)
        - "extreme_heat_loss": Weighted MSE that penalizes errors at extreme temps more
            - T <= 25C: weight = 1.0, 25C < T <= 30C: weight = 6.0, T > 30C: weight = 11.0
        - "mortality_weighted_loss": MSE in mortality dose-response space (Carleton et al. 2022)
        - "heatwave_loss": Duration-weighted MSE based on consecutive days above threshold
            - Requires --growing_season_only flag and all lead times per sample

    metrics (evaluation metric for plotting):
        - "rmse": Standard RMSE percentage improvement
        - "extreme_heat_rmse": Extreme heat weighted RMSE percentage improvement
        - "mortality_weighted_rmse": Mortality weighted RMSE percentage improvement
        - "raw_error": Raw error values (orig and corrected on same plot)
        - "error_difference": Difference between corrected and original errors
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from helper_funcs import setup_directories
from finetuning.figures_finetuning import plot_improvement_by_weather_bin, plot_rmse_improvement 


def main():
    """Generate architecture comparison plots."""
    # Setup directories
    dirs = setup_directories()

    train_start = "2018-01-01"
    train_end = "2021-12-31"
    test_start = "2022-01-01"
    test_end = "2022-12-31"
    model = "pangu"
    regions = ["corn_belt"]  # List of regions
    subregion = "6x6"
    training_output_vars = (["2m_temperature"], ["2m_temperature"])  # Tuple of (training_vars, output_vars)
    variable = "2m_temperature"
    lead_times = [24, 120, 216]  # Lead time in hours
    loss_trained_on = ["extreme_heat_loss", "mse"]
    metrics = ["rmse", "extreme_heat_rmse"]

    for metric in metrics:
        plot_improvement_by_weather_bin(
            dirs=dirs,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            model=model,
            training_output_vars=training_output_vars,
            variable=variable,
            lead_times=lead_times,
            regions=regions,
            subregion=subregion,
            loss_trained_on=loss_trained_on,
            metric=metric,
        )


if __name__ == "__main__":
    main()
