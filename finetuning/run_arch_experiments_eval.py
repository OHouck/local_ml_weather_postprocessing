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

USE_LEGACY_GLOBAL_DATA = True # can be true if only using 2m temp and 10m wind speed


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
    )


# ---------------------------------------------------------------------------
# Variable configurations to loop over.
# Each entry defines what the model trains on and what it predicts.
# Comment out any configs you don't want to run.
# ---------------------------------------------------------------------------
VAR_CONFIGS = [
    {
        'label': '2m Temperature',
        'training_vars': ['2m_temperature'],
        'output_vars':   ['2m_temperature'],
    },
    {
        'label': '10m Wind Speed',
        'training_vars': ['10m_wind_speed'],
        'output_vars':   ['10m_wind_speed'],
    },
    # {
    #     'label': '2m Temperature + 1000hPa T & q',
    #     'training_vars': ['2m_temperature', 'temperature_1000hPa', 'specific_humidity_1000hPa'],
    #     'output_vars':   ['2m_temperature'],
    # },
]

# ---------------------------------------------------------------------------
# Architecture templates — one entry per model variant.
# training_vars / output_vars are filled in from VAR_CONFIGS above.
# ---------------------------------------------------------------------------
_ARCH_TEMPLATES = [
    {
        'name_prefix': 'MLP',
        'nn_architecture': 'mlp',
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
        'name_prefix': 'MLP Snapshot Ensemble x3',
        'nn_architecture': 'mlp',
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
        'name_prefix': 'UNet',
        'nn_architecture': 'unet',
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
        'name_prefix': 'Block LTHO Ensemble',
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
    {
        'name_prefix': 'Per-LT MLP Snapshot x3',
        'nn_architecture': 'mlp',
        'block_ensemble': False,
        'block_holdout': 1,
        'snapshot_ensemble': 3,
        'snapshot_epochs': 210,
        'snapshot_T0': 30,
        'snapshot_T_mult': 1,
        'ensemble': None,
        'swa_ensemble': None,
        'per_lead_time': True,
    },
]

# Build the full experiment list by crossing each architecture template with
# each variable configuration.
EXPERIMENTS = []
for var_cfg in VAR_CONFIGS:
    for tmpl in _ARCH_TEMPLATES:
        exp = dict(tmpl)
        exp['name'] = f"{tmpl['name_prefix']} ({var_cfg['label']})"
        exp['training_vars'] = var_cfg['training_vars']
        exp['output_vars']   = var_cfg['output_vars']
        exp.pop('name_prefix')
        EXPERIMENTS.append(exp)


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

    # Pre-load hyperparameter sets so we don't re-read JSON on every cell.
    # Each experiment picks the set that matches its architecture / training mode.
    _hp_cache = {}

    def get_hyperparams(arch, training_vars, output_vars, use_snapshot, use_block_ltho, use_per_lt):
        key = (arch, tuple(training_vars), tuple(output_vars), use_snapshot, use_block_ltho, use_per_lt)
        if key not in _hp_cache:
            _hp_cache[key] = load_optimal_hyperparameters(
                arch, training_vars, output_vars,
                alternate_loss_fn=None,
                use_snapshot=use_snapshot,
                use_block_ltho=use_block_ltho,
                use_per_lt=use_per_lt,
            )
        return _hp_cache[key]

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

            # Apply optimal hyperparameters — select the right set per experiment.
            # Block LTHO uses its own dedicated hyperopt; per-LT uses per_lt hyperopt;
            # plain MLP/Snapshot use the snapshot-tuned set; UNet uses non-snapshot results.
            use_block = exp['block_ensemble']
            use_snap = exp['snapshot_ensemble'] is not None
            use_per_lt = exp.get('per_lead_time', False)
            hp = get_hyperparams(
                exp['nn_architecture'],
                exp.get('training_vars', ['2m_temperature']),
                exp.get('output_vars', ['2m_temperature']),
                use_snapshot=use_block or use_snap,
                use_block_ltho=use_block,
                use_per_lt=use_per_lt,
            )
            if hp:
                args.mlp_hidden_dim = hp.get('hidden_dim', args.mlp_hidden_dim)
                args.mlp_num_layers = hp.get('num_layers', args.mlp_num_layers)
                args.mlp_dropout = hp.get('dropout_rate', args.mlp_dropout)
                args.unet_hidden_dim = hp.get('hidden_dim', args.unet_hidden_dim)
                args.unet_dropout = hp.get('dropout_rate', args.unet_dropout)
                args.optimal_lr = hp.get('learning_rate')
                args.optimal_batch_size = hp.get('batch_size')
                args.optimal_weight_decay = hp.get('weight_decay')
                args.optimal_patience = hp.get('patience')
                args.optimal_min_delta = hp.get('min_delta')
                args.optimal_lead_time_embedding_dim = hp.get('lead_time_embedding_dim')
                args.optimal_snapshot_T0 = hp.get('snapshot_T0')
                args.optimal_snapshot_T_mult = hp.get('snapshot_T_mult')

            # Generate output path with eval cell identifier
            base_path = os.path.join(output_dir, generate_output_path(args))
            out_path = base_path.replace('.zarr', f'_{continent}_bs{patch_idx}.zarr')

            print(f"  Output: {out_path}")

            # Skip if already exists
            if os.path.exists(out_path):
                print(f"  Skipping (already exists)")
                # continue

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
