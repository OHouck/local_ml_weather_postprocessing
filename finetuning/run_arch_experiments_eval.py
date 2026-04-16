#!/usr/bin/env python3
"""
Run architecture experiments on a 10% eval sample of continent 6x6 cells.

This script:
1. Samples a deterministic 10% "eval" subset of continent cells (disjoint from hyperopt subset)
2. Runs each experiment configuration on each cell
3. Saves output zarrs with continent/patch info so plot_arch_experiment_results.py can aggregate

Usage:
    python3 finetuning/run_arch_experiments_eval.py
"""

import os
import sys
import numpy as np
import torch
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from helper_funcs import setup_directories, generate_output_path, sample_continent_patches
from finetuning.finetune import (
    run_subregion_experiment, load_optimal_hyperparameters, parse_args
)

USE_LEGACY_GLOBAL_DATA = False


def make_base_args(data_dir, output_dir, model_name, train_start, train_end,
                   test_start, test_end, lead_time_hours):
    """Create base args namespace with common settings."""
    return SimpleNamespace(
        data_dir=data_dir,
        output_dir=output_dir,
        model_name=model_name,
        subregion="6x6",
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
        training_vars=["2m_temperature"],
        output_vars=["2m_temperature"],
        lead_time_hours=lead_time_hours,
        alternate_loss_fn=None,
        growing_season_only=False,
        bootstrap=None,
        ground_truth_source="",
        pca_components=0,
        # MLP defaults (will be overridden by optimal hyperparams)
        mlp_hidden_dim=1024,
        mlp_num_layers=6,
        mlp_dropout=0.25,
        unet_hidden_dim=64,
        unet_dropout=0.1,
        optimal_lr=None,
        optimal_batch_size=None,
        optimal_weight_decay=None,
        optimal_patience=None,
        optimal_min_delta=None,
        optimal_lead_time_embedding_dim=None,
        optimal_snapshot_T0=None,
        optimal_snapshot_T_mult=None,
        # New features defaults
        lead_time_loss_weights=None,
        cmixup_alpha=0.0,
        mc_dropout_samples=0,
        per_lead_time=False,
        small_output_init=False,
        # Probabilistic head defaults
        probabilistic_head='none',
        bernstein_degree=6,
    )


# Define experiment configurations
EXPERIMENTS = [
    # ---- Baseline architectures (evaluated on eval cells for fair comparison) ----
    {
        'name': 'MLP (2m Temperature)',
        'nn_architecture': 'mlp',
        'training_vars': ['2m_temperature'],
        'block_ensemble': False,
        'block_holdout': 1,
        'snapshot_ensemble': None,
        'snapshot_epochs': 210,
        'snapshot_T0': 10,
        'snapshot_T_mult': 1,
        'ensemble': None,
        'swa_ensemble': None,
    },
    {
        'name': 'MLP (2m Temperature + 1000hPa Temperature and Specific Humidity)',
        'nn_architecture': 'mlp',
        'training_vars': ['2m_temperature', 'temperature_1000hPa', 'specific_humidity_1000hPa'],
        'block_ensemble': False,
        'block_holdout': 1,
        'snapshot_ensemble': None,
        'snapshot_epochs': 210,
        'snapshot_T0': 10,
        'snapshot_T_mult': 1,
        'ensemble': None,
        'swa_ensemble': None,
    },
    {
        'name': 'MLP Snapshot Ensemble x3 (2m Temperature)',
        'nn_architecture': 'mlp',
        'training_vars': ['2m_temperature'],
        'block_ensemble': False,
        'block_holdout': 1,
        'snapshot_ensemble': 3,
        'snapshot_epochs': 210,
        'snapshot_T0': 30,
        'snapshot_T_mult': 1,
        'ensemble': None,
        'swa_ensemble': None,
    },
    {
        'name': 'MLP Snapshot Ensemble x3 (2m Temperature + 1000hPa Temperature and Specific Humidity)',
        'nn_architecture': 'mlp',
        'training_vars': ['2m_temperature', 'temperature_1000hPa', 'specific_humidity_1000hPa'],
        'block_ensemble': False,
        'block_holdout': 1,
        'snapshot_ensemble': 3,
        'snapshot_epochs': 210,
        'snapshot_T0': 30,
        'snapshot_T_mult': 1,
        'ensemble': None,
        'swa_ensemble': None,
    },
    {
        'name': 'UNet (2m Temperature)',
        'nn_architecture': 'unet',
        'training_vars': ['2m_temperature'],
        'block_ensemble': False,
        'block_holdout': 1,
        'snapshot_ensemble': None,
        'snapshot_epochs': 210,
        'snapshot_T0': 10,
        'snapshot_T_mult': 1,
        'ensemble': None,
        'swa_ensemble': None,
    },
    {
        'name': 'UNet (2m Temperature + 1000hPa Temperature and Specific Humidity)',
        'nn_architecture': 'unet',
        'training_vars': ['2m_temperature', 'temperature_1000hPa', 'specific_humidity_1000hPa'],
        'block_ensemble': False,
        'block_holdout': 1,
        'snapshot_ensemble': None,
        'snapshot_epochs': 210,
        'snapshot_T0': 10,
        'snapshot_T_mult': 1,
        'ensemble': None,
        'swa_ensemble': None,
    },
    # ---- Block LTHO variants ----
    # Baseline: the current best method
    {
        'name': 'Block LTHO Ensemble',
        'nn_architecture': 'mlp',
        'block_ensemble': True,
        'block_holdout': 3,
        'snapshot_ensemble': 1,
        'snapshot_epochs': 210,
        'snapshot_T0': 10,
        'snapshot_T_mult': 1,
        'ensemble': None,
        'swa_ensemble': None,
    },
    # Experiment 1: Block LTHO + LT-Weighted snapshots
    # Combines the best ensemble method with lead-time weighting INSIDE the
    # snapshot training loop. 5x weight on 24h forces each snapshot to optimize
    # harder for short-range corrections while still benefiting from block diversity.
    {
        'name': 'Block LTHO + LT-Weighted',
        'nn_architecture': 'mlp',
        'block_ensemble': True,
        'block_holdout': 3,
        'snapshot_ensemble': 1,
        'snapshot_epochs': 210,
        'snapshot_T0': 10,
        'snapshot_T_mult': 1,
        'ensemble': None,
        'swa_ensemble': None,
        'lead_time_loss_weights': [5.0, 1.0, 0.5],
    },
    # Experiment 2: Per-Lead-Time Block LTHO
    # Train a SEPARATE block ensemble for each lead time. The 24h model only
    # sees 24h data, so 100% of optimization goes to short-range correction.
    # No gradient competition from 120h/216h. ~3x training time.
    {
        'name': 'Per-LT Block LTHO',
        'nn_architecture': 'mlp',
        'block_ensemble': True,
        'block_holdout': 3,
        'snapshot_ensemble': 1,
        'snapshot_epochs': 210,
        'snapshot_T0': 10,
        'snapshot_T_mult': 1,
        'ensemble': None,
        'swa_ensemble': None,
        'per_lead_time': True,
    },
    # Experiment 3: Block LTHO + Small Output Init
    # Same architecture as baseline but initialize final layer near zero.
    # Model starts predicting ~zero correction, which is closer to the
    # optimal 24h correction (small) than random initialization.
    {
        'name': 'Block LTHO + SmallInit',
        'nn_architecture': 'mlp',
        'block_ensemble': True,
        'block_holdout': 3,
        'snapshot_ensemble': 1,
        'snapshot_epochs': 210,
        'snapshot_T0': 10,
        'snapshot_T_mult': 1,
        'ensemble': None,
        'swa_ensemble': None,
        'small_output_init': True,
    },
    # Experiment 4: Block LTHO + DRN (Gaussian CRPS head)
    # Distributional Regression Network (Rasp & Lerch 2018, MWR 146(11)).
    # Same MLP backbone + block ensemble, but outputs (mu_error, log_sigma) and
    # trains with closed-form Gaussian CRPS after a 20-epoch MSE warm-start.
    # Point-forecast RMSE uses mu (mean of corrected Gaussian).
    # Advantage: learns heteroscedastic uncertainty; CRPS loss gives the network
    # a reason to pull mu toward climatology when sigma is large, helping in
    # high-topography / equatorial regions identified as hardest in the paper.
    {
        'name': 'Block LTHO + DRN (Gaussian CRPS)',
        'nn_architecture': 'mlp',
        'block_ensemble': True,
        'block_holdout': 3,
        'snapshot_ensemble': 1,
        'snapshot_epochs': 210,
        'snapshot_T0': 10,
        'snapshot_T_mult': 1,
        'ensemble': None,
        'swa_ensemble': None,
        'probabilistic_head': 'gaussian',
        'bernstein_degree': 6,
    },
    # Experiment 5: Block LTHO + BQN (Bernstein Quantile Network, degree=6)
    # Bremnes 2020, MWR 148(1); validated SOTA for wind by Schulz & Lerch 2022.
    # Model outputs 7*(output_dim) raw values, transformed to a monotone
    # Bernstein polynomial quantile function via cumsum(softplus(.)).
    # Loss = average pinball loss over 19 quantile levels (tau=0.05..0.95).
    # Point-forecast RMSE uses the median (tau=0.5).
    # Advantage: distribution-free — critical for 10m wind speed (right-skewed).
    {
        'name': 'Block LTHO + BQN (d=6)',
        'nn_architecture': 'mlp',
        'block_ensemble': True,
        'block_holdout': 3,
        'snapshot_ensemble': 1,
        'snapshot_epochs': 210,
        'snapshot_T0': 10,
        'snapshot_T_mult': 1,
        'ensemble': None,
        'swa_ensemble': None,
        'probabilistic_head': 'bernstein',
        'bernstein_degree': 6,
    },
]


def main():
    dirs = setup_directories()
    data_dir = dirs['raw']
    output_dir = dirs['input']

    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )
    print(f"Using device: {device}")

    # Sample the eval 5% of continent cells
    eval_cells = sample_continent_patches(
        dirs['processed'], fraction=0.05, seed=42, split='eval'
    )
    print(f"\nLoaded {len(eval_cells)} eval cells from continent 10% sample")

    # Configuration
    train_start = "2018-01-01"
    train_end = "2021-12-31"
    test_start = "2022-01-01"
    test_end = "2022-12-31"
    model_name = "pangu"
    lead_time_hours = [24, 120, 216]

    # Load optimal hyperparameters for MLP (used by both mlp and gated_mlp)
    optimal_hyperparams = load_optimal_hyperparameters(
        'mlp', ['2m_temperature'], ['2m_temperature'],
        alternate_loss_fn=None, use_snapshot=True, use_block_ltho=True
    )

    for exp in EXPERIMENTS:
        print(f"\n{'#'*70}")
        print(f"EXPERIMENT: {exp['name']}")
        print(f"{'#'*70}")

        for i, (continent, patch_idx, patch_array) in enumerate(eval_cells):
            lat_vals = patch_array[0]
            lon_vals = patch_array[1]

            print(f"\n{'='*70}")
            print(f"[{exp['name']}] Cell {i+1}/{len(eval_cells)}: {continent} patch {patch_idx}")
            print(f"  Lat: {lat_vals.min():.2f} to {lat_vals.max():.2f}")
            print(f"  Lon: {lon_vals.min():.2f} to {lon_vals.max():.2f}")
            print(f"{'='*70}")

            # Build args
            args = make_base_args(data_dir, output_dir, model_name,
                                  train_start, train_end, test_start, test_end,
                                  lead_time_hours)
            args.region = continent
            args.training_vars = exp.get('training_vars', args.training_vars)
            args.output_vars = args.training_vars[:1]  # always correct only the first var
            args.nn_architecture = exp['nn_architecture']
            args.block_ensemble = exp['block_ensemble']
            args.block_holdout = exp['block_holdout']
            args.snapshot_ensemble = exp['snapshot_ensemble']
            args.snapshot_epochs = exp['snapshot_epochs']
            args.snapshot_T0 = exp['snapshot_T0']
            args.snapshot_T_mult = exp['snapshot_T_mult']
            args.ensemble = exp.get('ensemble')
            args.swa_ensemble = exp.get('swa_ensemble')
            args.lead_time_loss_weights = exp.get('lead_time_loss_weights')
            args.cmixup_alpha = exp.get('cmixup_alpha', 0.0)
            args.per_lead_time = exp.get('per_lead_time', False)
            args.small_output_init = exp.get('small_output_init', False)
            args.probabilistic_head = exp.get('probabilistic_head', 'none')
            args.bernstein_degree = exp.get('bernstein_degree', 6)

            # Apply optimal hyperparameters
            if optimal_hyperparams:
                args.mlp_hidden_dim = optimal_hyperparams.get('hidden_dim', args.mlp_hidden_dim)
                args.mlp_num_layers = optimal_hyperparams.get('num_layers', args.mlp_num_layers)
                args.mlp_dropout = optimal_hyperparams.get('dropout_rate', args.mlp_dropout)
                args.optimal_lr = optimal_hyperparams.get('learning_rate')
                args.optimal_batch_size = optimal_hyperparams.get('batch_size')
                args.optimal_weight_decay = optimal_hyperparams.get('weight_decay')
                args.optimal_patience = optimal_hyperparams.get('patience')
                args.optimal_min_delta = optimal_hyperparams.get('min_delta')
                args.optimal_lead_time_embedding_dim = optimal_hyperparams.get('lead_time_embedding_dim')
                args.optimal_snapshot_T0 = optimal_hyperparams.get('snapshot_T0')
                args.optimal_snapshot_T_mult = optimal_hyperparams.get('snapshot_T_mult')

            # Generate output path with eval cell identifier
            base_path = os.path.join(output_dir, generate_output_path(args))
            out_path = base_path.replace('.zarr', f'_{continent}_bs{patch_idx}.zarr')

            print(f"  Output: {out_path}")

            # Skip if already exists
            if os.path.exists(out_path):
                print(f"  Skipping (already exists)")
                continue

            try:
                run_subregion_experiment(
                    lat_vals, lon_vals, out_path,
                    args, data_dir, device, patch_num=patch_idx,
                    use_legacy_global_data=USE_LEGACY_GLOBAL_DATA
                )
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue

    print(f"\n{'='*70}")
    print("All experiments complete.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
