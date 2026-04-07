#!/usr/bin/env python3
"""
Script to generate architecture comparison plots from run_arch_experiments.sh outputs.

This script plots RMSE improvement for 6 architecture/training configurations:
- MLP with single variable (minimal)
- MLP with 3 variables (partial: temp + temp_1000hPa + specific_humidity_1000hPa)
- MLP Snapshot Ensemble ×3 with single variable (minimal)
- MLP Snapshot Ensemble ×3 with 3 variables (partial)
- UNet with single variable (minimal)
- UNet with 3 variables (partial)

Usage:
    python3 finetuning/plot_arch_experiment_results.py
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from helper_funcs import setup_directories
from finetuning.figures_finetuning import plot_arch_experiment_results


def main():
    """Generate architecture comparison plots."""
    # Setup directories
    dirs = setup_directories()

    # Configuration matching run_arch_experiments.sh
    train_start = "2018-01-01"
    train_end = "2021-12-31"
    test_start = "2022-01-01"
    test_end = "2022-12-31"
    model = "pangu"
    region = "india"
    subregion = "6x6"
    variable = "2m_temperature"

    print("=" * 80)
    print("ARCHITECTURE COMPARISON PLOT GENERATION")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Model: {model}")
    print(f"  Region: {region}")
    print(f"  Subregion: {subregion}")
    print(f"  Variable: {variable}")
    print(f"  Training: {train_start} to {train_end}")
    print(f"  Testing: {test_start} to {test_end}")
    print("\nExperiments:")
    print("  1. MLP (Minimal) - Single variable")
    print("  2. MLP (Partial) - 3 variables")
    print("  3. MLP Snapshot Ensemble x3 (Minimal) - Single variable")
    print("  4. MLP Snapshot Ensemble x3 (Partial) - 3 variables")
    print("  5. UNet (Minimal) - Single variable")
    print("  6. UNet (Partial) - 3 variables")
    print("  7. Block LTHO Ensemble k=3 (Minimal) - NEW BEST")
    print("\n" + "=" * 80)

    # Generate plot
    plot_arch_experiment_results(
        dirs=dirs,
        model=model,
        region=region,
        subregion=subregion,
        variable=variable,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end
    )

    print("\n" + "=" * 80)
    print("Plot generation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
