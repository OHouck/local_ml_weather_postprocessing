#!/usr/bin/env python3
"""
Example script demonstrating how to use the plot_improvement_by_weather_bin function.

The function supports four metrics via the 'metric' parameter:
1. "rmse" - RMSE percentage improvement (default)
2. "extreme_heat_rmse" - Extreme heat weighted RMSE percentage improvement
3. "raw_error" - Mean raw prediction error (shows bias)
4. "error_difference" - Difference between original and corrected error

Visual encoding:
- Color = Region
- Line style = Loss function trained on (solid, dashed, etc.)
- Filled markers = Corrected forecasts (or only line for improvement metrics)
- Hollow markers = Original forecasts (only for raw_error metric)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from helper_funcs import setup_directories
from finetuning.figures_finetuning import plot_improvement_by_weather_bin

# Setup directories
dirs = setup_directories(
    output_dir="~/ai_weather_ag/data/fine_tuning_output",
    fig_dir="~/ai_weather_ag/figures",
    log_dir="~/ai_weather_ag/logs"
)

training_output_vars = (
    ['2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind',
     'temperature_1000hPa', 'specific_humidity_1000hPa', 'geopotential_1000hPa'],
    ['2m_temperature']
)


def example_1_rmse_improvement():
    """
    Example 1: RMSE Improvement by Temperature Bin

    Shows how much the correction improved RMSE across different temperature ranges.
    Compares models trained with MSE vs extreme heat loss.
    """
    print("\n" + "="*80)
    print("Example 1: RMSE Improvement")
    print("="*80)

    plot_improvement_by_weather_bin(
        dirs=dirs,
        train_start="2020-01-01",
        train_end="2020-12-31",
        test_start="2021-01-01",
        test_end="2021-06-30",
        model="pangu",
        training_output_vars=training_output_vars,
        variable="2m_temperature",
        lead_time=24,
        regions=["india", "corn_belt"],
        loss_trained_on=["mse", "extreme_heat"],
        metric="rmse",  # RMSE improvement %
        n_bins=10
    )

    print("\nLegend interpretation:")
    print("- Colors = Regions (India, Corn Belt)")
    print("- Solid line = Trained on MSE")
    print("- Dashed line = Trained on extreme heat")
    print("- Y-axis: Positive values = improvement")


def example_2_raw_error():
    """
    Example 2: Raw Prediction Errors by Temperature Bin

    Shows the mean prediction error (bias) for both original and corrected forecasts.
    Useful for understanding systematic bias patterns.
    """
    print("\n" + "="*80)
    print("Example 2: Raw Prediction Errors (Bias Analysis)")
    print("="*80)

    plot_improvement_by_weather_bin(
        dirs=dirs,
        train_start="2020-01-01",
        train_end="2020-12-31",
        test_start="2021-01-01",
        test_end="2021-06-30",
        model="pangu",
        training_output_vars=training_output_vars,
        variable="2m_temperature",
        lead_time=24,
        regions=["india", "corn_belt"],
        loss_trained_on=["mse", "extreme_heat"],
        metric="raw_error",  # Show mean prediction error
        n_bins=10
    )

    print("\nLegend interpretation:")
    print("- Colors = Regions")
    print("- Solid line = Trained on MSE, Dashed = Trained on extreme heat")
    print("- Hollow circles = Original forecasts")
    print("- Filled circles = Corrected forecasts")
    print("- Y-axis: Positive = over-prediction, Negative = under-prediction")


def example_3_error_difference():
    """
    Example 3: Error Difference by Temperature Bin

    Shows the difference between original and corrected absolute errors.
    Positive values mean correction reduced the error.
    """
    print("\n" + "="*80)
    print("Example 3: Error Difference (Error Reduction)")
    print("="*80)

    plot_improvement_by_weather_bin(
        dirs=dirs,
        train_start="2020-01-01",
        train_end="2020-12-31",
        test_start="2021-01-01",
        test_end="2021-06-30",
        model="pangu",
        training_output_vars=training_output_vars,
        variable="2m_temperature",
        lead_time=24,
        regions=["india"],
        loss_trained_on=["mse", "extreme_heat"],
        metric="error_difference",  # Show error reduction
        n_bins=10
    )

    print("\nLegend interpretation:")
    print("- Colors = Regions")
    print("- Line styles = Loss functions")
    print("- Y-axis: Positive = error was reduced, Negative = error increased")


def example_4_extreme_heat_rmse():
    """
    Example 4: Extreme Heat Weighted RMSE Improvement

    Uses extreme heat weighted RMSE for evaluation.
    Useful for seeing if extreme heat loss training improves high-temp predictions.
    """
    print("\n" + "="*80)
    print("Example 4: Extreme Heat RMSE Improvement")
    print("="*80)

    plot_improvement_by_weather_bin(
        dirs=dirs,
        train_start="2020-01-01",
        train_end="2020-12-31",
        test_start="2021-01-01",
        test_end="2021-06-30",
        model="pangu",
        training_output_vars=training_output_vars,
        variable="2m_temperature",
        lead_time=72,
        regions=["india", "corn_belt"],
        loss_trained_on=["mse", "extreme_heat"],
        metric="extreme_heat_rmse",  # Extreme heat weighted RMSE
        n_bins=10
    )

    print("\nLegend interpretation:")
    print("- Similar to RMSE but with extreme heat weighting")
    print("- Should show higher improvement at high temperatures")
    print("- if extreme heat loss training is effective")


def example_5_comparison():
    """
    Example 5: Compare Different Metrics

    Generate multiple plots to see different perspectives.
    """
    print("\n" + "="*80)
    print("Example 5: Generate Multiple Metrics")
    print("="*80)

    common_params = {
        "dirs": dirs,
        "train_start": "2020-01-01",
        "train_end": "2020-12-31",
        "test_start": "2021-01-01",
        "test_end": "2021-06-30",
        "model": "pangu",
        "training_output_vars": training_output_vars,
        "variable": "2m_temperature",
        "lead_time": 24,
        "regions": ["india", "corn_belt"],
        "loss_trained_on": ["mse", "extreme_heat"],
        "n_bins": 10
    }

    # RMSE improvement
    print("\nGenerating RMSE improvement plot...")
    plot_improvement_by_weather_bin(**common_params, metric="rmse")

    # Raw errors
    print("\nGenerating raw errors plot...")
    plot_improvement_by_weather_bin(**common_params, metric="raw_error")

    # Error difference
    print("\nGenerating error difference plot...")
    plot_improvement_by_weather_bin(**common_params, metric="error_difference")

    print("\nAll plots generated!")


if __name__ == "__main__":
    print("="*80)
    print("plot_improvement_by_weather_bin Examples")
    print("="*80)
    print("\nThis script demonstrates the four metric types:")
    print("1. rmse - RMSE percentage improvement")
    print("2. extreme_heat_rmse - Extreme heat weighted RMSE improvement")
    print("3. raw_error - Mean prediction error (bias)")
    print("4. error_difference - Error reduction")
    print("\nVisual encoding:")
    print("- Color = Region")
    print("- Line style = Loss function (solid/dashed)")
    print("- Hollow circles = Original (raw_error only)")
    print("- Filled circles = Corrected")
    print("\nUncomment examples below to run them:")
    print("="*80)

    # Uncomment to run examples:
    # example_1_rmse_improvement()
    # example_2_raw_error()
    # example_3_error_difference()
    # example_4_extreme_heat_rmse()
    # example_5_comparison()

    print("\nExamples ready. Uncomment function calls to run.")
