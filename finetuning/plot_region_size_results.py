#!/usr/bin/env python3
"""
Script to generate region size comparison plots from run_region_size_experiments.sh outputs.

Usage:
    python3 finetuning/plot_region_size_results.py
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from helper_funcs import setup_directories, generate_output_path
from finetuning.figures_finetuning import generate_subregion_comparison_plots


def main():
    """Generate region size comparison plots."""
    # Setup directories
    dirs = setup_directories()

    # Configuration matching run_region_size_experiments.sh
    train_start = "2018-01-01"
    train_end = "2021-12-31"
    test_start = "2022-01-01"
    test_end = "2022-12-31"
    model = "pangu"
    nn_architecture = "mlp"

    # Generate plots
    generate_subregion_comparison_plots(
        dirs=dirs,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        model=model,
        nn_architecture=nn_architecture,
        growing_season_only=False,
        alternate_loss_fn=None
    )

    print("\n" + "=" * 80)
    print("Plot generation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
