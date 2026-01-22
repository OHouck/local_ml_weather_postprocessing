#!/usr/bin/env python3
"""
Script to generate architecture comparison plots from run_custom_loss_experiments.sh outputs.

Usage:
    python3 finetuning/plot_custom_loss.py
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
    regions = ["india"]  # List of regions
    subregion = "6x6"
    training_output_vars = (["2m_temperature"], ["2m_temperature"])  # Tuple of (training_vars, output_vars)
    variable = "2m_temperature"
    lead_times = [24, 120, 216]  # Lead time in hours
    loss_trained_on = ["mortality_weighted_loss", "mse"]
    metrics = ["rmse", "mortality_weighted_rmse"]

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
