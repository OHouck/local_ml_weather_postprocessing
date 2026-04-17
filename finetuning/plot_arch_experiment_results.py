#!/usr/bin/env python3
"""
Script to generate architecture comparison plots from run_arch_experiments.sh outputs.

All experiments are evaluated on a 5% eval sample of continent 6x6 cells (disjoint
from the hyperopt split) and results are averaged across cells for a geographically
diverse comparison.

Usage:
    python3 finetuning/plot_arch_experiment_results.py
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from helper_funcs import setup_directories, sample_continent_patches
from finetuning.figures_finetuning import (
    plot_arch_experiment_results,
    map_arch_exeriment_regions,
)


VAR_CONFIGS = [
    {
        'label': '2m Temperature',
        'training_vars': ['2m_temperature'],
        'output_vars': ['2m_temperature'],
    },
    {
        'label': '10m Wind Speed',
        'training_vars': ['10m_wind_speed'],
        'output_vars': ['10m_wind_speed'],
    },
    {
        'label': '2m Temperature + 1000hPa T & q',
        'training_vars': ['2m_temperature', 'temperature_1000hPa', 'specific_humidity_1000hPa'],
        'output_vars': ['2m_temperature'],
    },
]


def main():
    """Generate one architecture comparison plot per variable config."""
    dirs = setup_directories()

    train_start = "2018-01-01"
    train_end = "2021-12-31"
    test_start = "2022-01-01"
    test_end = "2022-12-31"
    model = "pangu"
    subregion = "6x6"

    print("=" * 80)
    print("ARCHITECTURE COMPARISON PLOT GENERATION")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Model: {model}")
    print(f"  Eval: 5% continent cell sample (global, averaged)")
    print(f"  Subregion: {subregion}")
    print(f"  Training: {train_start} to {train_end}")
    print(f"  Testing: {test_start} to {test_end}")
    print(f"\nGenerating {len(VAR_CONFIGS)} plots:")
    for i, vc in enumerate(VAR_CONFIGS, 1):
        print(f"  {i}. {vc['label']}")
    print("\n" + "=" * 80)

    eval_cells = sample_continent_patches(dirs['processed'], fraction=0.05, seed=42, split='eval')
    map_arch_exeriment_regions(
        dirs=dirs,
        eval_cells=eval_cells,
        fraction=0.05,
        seed=42,
        split='eval',
        model=model,
        subregion=subregion,
    )

    shared_kwargs = dict(
        dirs=dirs, model=model, subregion=subregion,
        train_start=train_start, train_end=train_end,
        test_start=test_start, test_end=test_end,
        eval_cells=eval_cells,
    )
    for vc in VAR_CONFIGS:
        plot_arch_experiment_results(
            label=vc['label'],
            training_vars=vc['training_vars'],
            output_vars=vc['output_vars'],
            **shared_kwargs,
        )

    print("\n" + "=" * 80)
    print("Plot generation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
