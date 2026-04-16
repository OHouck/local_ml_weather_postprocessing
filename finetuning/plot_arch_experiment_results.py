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
    subregion = "6x6"
    variable = "2m_temperature"

    print("=" * 80)
    print("ARCHITECTURE COMPARISON PLOT GENERATION")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Model: {model}")
    print(f"  Eval: 5% continent cell sample (global, averaged)")
    print(f"  Subregion: {subregion}")
    print(f"  Variable: {variable}")
    print(f"  Training: {train_start} to {train_end}")
    print(f"  Testing: {test_start} to {test_end}")
    print("\nExperiments (all averaged across global eval cells):")
    print("  1. MLP — single variable")
    print("  2. MLP — 3 variables (+ 1000hPa T & q)")
    print("  3. MLP Snapshot Ensemble x3 — single variable")
    print("  4. MLP Snapshot Ensemble x3 — 3 variables")
    print("  5. UNet — single variable")
    print("  6. UNet — 3 variables")
    print("  7. Block LTHO Ensemble k=3 — single variable")
    print("  8. Block LTHO + LT-Weighted — 5x weight on 24h in snapshot training")
    print("  9. Per-LT Block LTHO — separate model per lead time")
    print(" 10. Block LTHO + SmallInit — zero-init final layer")
    print(" 11. Block LTHO + DRN (Gaussian CRPS) — probabilistic head, CRPS loss")
    print(" 12. Block LTHO + BQN d=6 (Bernstein Quantile) — median as point estimate")
    print("\n" + "=" * 80)

    # Generate plot
    plot_arch_experiment_results(
        dirs=dirs,
        model=model,
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
