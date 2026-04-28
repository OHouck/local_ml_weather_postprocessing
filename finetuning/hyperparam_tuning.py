#!/usr/bin/env python3
"""
filename: hyperparam_tuning.py
Author: Ozma Houck
Date created: 07/17/2025
Date modified: 11/07/2025

Hyperparameter optimization module for weather forecast fine-tuning using hyperopt.
Uses Bayesian optimization with early stopping for both MLP and UNet architectures.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import hyperopt
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval
import json
from datetime import datetime
import copy
import pickle
from types import SimpleNamespace
from typing import Dict, Any

# Import model classes and utilities from finetune.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from finetuning.finetune import SimpleMLP, UNet, load_forecasts, create_dataloader, get_region_grid, train_snapshot_ensemble
from finetuning.custom_loss_fns import (
    mortality_weighted_loss, extreme_heat_loss, quantile_loss,
    heatwave_loss, joint_temp_wind_loss
)
from functools import partial
from helper_funcs import setup_directories, sample_continent_patches


def _year_holdout_split(times):
    """
    Split sample indices by holding out the last year as validation.

    Args:
        times: Array-like of timestamps (numpy datetime64 or pandas-compatible).

    Returns:
        train_idx: Indices for all years except the last.
        val_idx: Indices for the last year.
        holdout_year: The held-out year (int).
    """
    years = pd.DatetimeIndex(times).year.values
    holdout_year = int(np.max(years))
    val_mask = years == holdout_year
    return np.where(~val_mask)[0], np.where(val_mask)[0], holdout_year


def make_eval_dataloader(fc_norm_sub, fc_output_norm_sub, obs_norm_sub,
                         lead_time_indices_sub, day_of_year_features_sub, batch_size):
    """
    Create a non-shuffling DataLoader for evaluation.

    IMPORTANT: Must be used (instead of create_dataloader) whenever predictions from
    multiple snapshots need to be averaged into an ensemble. create_dataloader always
    shuffles, so iterating it multiple times yields different sample orderings — which
    makes per-snapshot predictions misaligned when stacked into an ensemble tensor.

    This loader uses shuffle=False so that every pass over it returns samples in the
    same order, making cross-snapshot prediction tensors correctly aligned.
    """
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(fc_norm_sub).float(),
        torch.from_numpy(fc_output_norm_sub).float(),
        torch.from_numpy(obs_norm_sub).float(),
        torch.from_numpy(lead_time_indices_sub).long(),
        torch.from_numpy(day_of_year_features_sub).float()
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


def create_mlp_search_space():
    """
    Define the hyperparameter search space for MLP architecture with early stopping.

    OPTIMIZED FOR FAST HYPERPARAMETER SEARCH:
    - Higher learning rates (1e-4 to 1e-2) for faster convergence
    - Lower patience (15-30) for quicker trials
    - Removed batch_size=32 (too slow)

    Returns:
        dict: Search space definition for hyperopt
    """
    return {
        # Model architecture
        'hidden_dim': hp.choice('hidden_dim', [64, 128, 256, 512, 1024]),
        # 'num_layers': hp.choice('num_layers', [2, 3, 4, 5, 6]),
        'num_layers': hp.choice('num_layers', [2, 3, 4, 5, 6, 8, 10]),

        # Training parameters - OPTIMIZED: Higher learning rates
        'learning_rate': hp.loguniform('learning_rate', np.log(1e-6), np.log(1e-2)),
        'batch_size': hp.choice('batch_size', [64, 128, 256]),
        'weight_decay': hp.loguniform('weight_decay', np.log(1e-6), np.log(1e-3)),

        # Early stopping parameters - OPTIMIZED: Lower patience
        'patience': hp.choice('patience', [15, 20, 25, 30, 50, 100]),
        'min_delta': hp.loguniform('min_delta', np.log(1e-5), np.log(1e-3)),

        # Embedding and regularization
        'lead_time_embedding_dim': hp.choice('lead_time_embedding_dim', [4, 8, 16]),  
        'dropout_rate': hp.uniform('dropout_rate', 0.1, 0.3),
    }


def create_unet_search_space():
    """
    Define the hyperparameter search space for UNet architecture with early stopping.

    OPTIMIZED FOR FAST HYPERPARAMETER SEARCH:
    - Higher learning rates (1e-4 to 1e-2) for faster convergence
    - Lower patience (15-30) for quicker trials
    - Removed batch_size=32 (too slow)

    Returns:
        dict: Search space definition for hyperopt
    """
    return {
        # Model architecture - centered on unet_medium optimal values
        'hidden_dim': hp.choice('hidden_dim', [64, 128]),

        # Training parameters - OPTIMIZED: Higher learning rates
        'learning_rate': hp.loguniform('learning_rate', np.log(1e-4), np.log(1e-2)),
        'batch_size': hp.choice('batch_size', [64, 128, 256]),
        'weight_decay': hp.loguniform('weight_decay', np.log(1e-6), np.log(1e-3)),

        # Early stopping parameters - OPTIMIZED: Lower patience
        'patience': hp.choice('patience', [15, 20, 25, 30]),
        'min_delta': hp.loguniform('min_delta', np.log(1e-5), np.log(1e-3)),

        # Embedding and regularization - centered on optimal dropout of 0.1
        'lead_time_embedding_dim': hp.choice('lead_time_embedding_dim', [4, 8, 16]),
        'dropout_rate': hp.uniform('dropout_rate', 0.05, 0.20),
    }


def create_mlp_snapshot_search_space():
    """
    Search space for snapshot ensemble MLP hyperparameter tuning.

    Replaces patience/min_delta (not used in snapshot training) with snapshot_T0
    (the cosine cycle period, which determines how many snapshots are saved per run).
    snapshot_epochs is fixed at 210 across all trials so that trial runtime is
    predictable (~0.2 min each on MPS/GPU).

    """
    return {
        # Model architecture
        'hidden_dim': hp.choice('hidden_dim', [32, 64, 128, 256, 512, 1024]),
        'num_layers': hp.choice('num_layers', [2, 3, 4, 5, 6, 8, 10]),
        'lead_time_embedding_dim': hp.choice('lead_time_embedding_dim', [4, 8, 16]),

        # Training parameters
        'learning_rate': hp.loguniform('learning_rate', np.log(1e-4), np.log(1e-2)),
        'batch_size': hp.choice('batch_size', [64, 128, 256]),
        'weight_decay': hp.loguniform('weight_decay', np.log(1e-6), np.log(1e-3)),

        # Snapshot cycle period — determines number of snapshots: 210 // T0
        # T0=20 → 10 snaps, T0=30 → 7 snaps, T0=42 → 5 snaps
        'snapshot_T0': hp.choice('snapshot_T0', [20, 30, 42]),

        # Restart growth multiplier for cosine annealing cycles.
        # T_mult=1 keeps fixed cycle lengths; T_mult=2 doubles cycle lengths.
        'snapshot_T_mult': hp.choice('snapshot_T_mult', [1, 2]),

        'dropout_rate': hp.uniform('dropout_rate', 0.1, 0.3),
    }


def create_block_ltho_search_space():
    """
    Search space for Block Leave-Three-Out (LTHO) ensemble MLP hyperparameter tuning.

    Block LTHO trains one model per available training year (4 models for 2018-2021
    training data), using cosine annealing snapshots within each block. There is no
    patience/min_delta since training is fixed-epoch.

    Key difference from regular snapshot: snapshot_T0 matters more here because each
    block trains on only ~1083 samples (1 year). Shorter cycles converge faster on
    small datasets — empirically T0=10 is optimal (21 snaps/block at 210 epochs).

    The hyperopt objective is leave-one-year-out mega-ensemble CV:
      - Each fold holds out 1 year; trains snapshot ensembles on each remaining year
      - All training-year snapshots are combined into a mega-ensemble (mirrors production)
      - Objective = mean mega-ensemble MSE on held-out year across all folds
    """
    return {
        # Model architecture
        'hidden_dim': hp.choice('hidden_dim', [64, 128, 256, 512, 1024]),
        'num_layers': hp.choice('num_layers', [2, 3, 4, 5, 6]),
        'lead_time_embedding_dim': hp.choice('lead_time_embedding_dim', [4, 8, 16]),  

        # Training parameters
        'learning_rate': hp.loguniform('learning_rate', np.log(5e-5), np.log(5e-3)),
        'batch_size': hp.choice('batch_size', [64, 128, 256]),
        'weight_decay': hp.loguniform('weight_decay', np.log(1e-7), np.log(1e-3)),

        # Cosine annealing cycle — shorter cycles work better for single-year (~1083 samples) blocks.
        # T0=10 is empirically optimal: gives 21 snapshots per block with 210 epochs.
        'snapshot_T0': hp.choice('snapshot_T0', [5, 7, 10, 15, 21]),
        'snapshot_T_mult': hp.choice('snapshot_T_mult', [1, 2]),

        # Regularization
        'dropout_rate': hp.uniform('dropout_rate', 0.1, 0.4),
    }



def train_with_early_stopping(model, train_loader, valid_loader, hyperparams, device,
                               alternate_loss_fn=None, stats_out=None,
                               n_output_vars=1, n_lat=None, n_lon=None,
                               n_lead_times=None, lead_time_days=None):
    """
    Train model with early stopping, mixed precision, and GPU optimizations.

    Args:
        model: The neural network model
        train_loader: Training data loader
        valid_loader: Validation data loader
        hyperparams: Dictionary of hyperparameters
        device: torch device
        alternate_loss_fn: Name of custom loss function to use (None = MSE)
        stats_out: Statistics for denormalizing outputs (for custom losses)
        n_output_vars: Number of output variables (required for joint_temp_wind_loss)
        n_lat: Number of latitude points (required for joint_temp_wind_loss)
        n_lon: Number of longitude points (required for joint_temp_wind_loss)
        n_lead_times: Number of lead times (required for heatwave_loss)
        lead_time_days: List of lead time values in days (required for heatwave_loss)

    Returns:
        tuple: (best_val_loss, num_epochs_trained)
    """
    import time

    # Loss functions that require denormalization (is_normalized=True)
    NORMALIZED_LOSS_FNS = {"extreme_heat_loss", "mortality_weighted_loss", "heatwave_loss",
                           "joint_temp_wind_loss"}

    loss_functions = {
        "extreme_heat_loss": extreme_heat_loss,
        "mortality_weighted_loss": mortality_weighted_loss,
        "quantile_loss": quantile_loss,
        "heatwave_loss": heatwave_loss,
        "joint_temp_wind_loss": joint_temp_wind_loss
    }

    if alternate_loss_fn is None:
        use_custom_loss = False
        criterion = nn.MSELoss()
    else:
        use_custom_loss = True
        criterion = loss_functions[alternate_loss_fn]

        if alternate_loss_fn == "joint_temp_wind_loss":
            # use partial to preserve signature
            criterion = partial(criterion, n_output_vars=n_output_vars,
                                n_lat=n_lat, n_lon=n_lon)

    # Convert stats to torch tensors for denormalization if needed
    mean_out = None
    std_out = None
    if alternate_loss_fn in NORMALIZED_LOSS_FNS and stats_out is not None:
        mean_out = torch.from_numpy(stats_out['mean']).float().to(device)
        std_out = torch.from_numpy(stats_out['std']).float().to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=hyperparams['learning_rate'],
        weight_decay=hyperparams['weight_decay']
    )

    # Match production train_model exactly: scheduler_patience=10, min_lr=1e-7
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        min_lr=1e-7
    )

    patience = hyperparams['patience']
    min_delta = hyperparams['min_delta']
    max_epochs = 750  # Match production num_epochs so trials aren't artificially truncated

    # Setup mixed precision training for CUDA
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # Determine if non_blocking transfers should be used
    non_blocking = device.type == 'cuda'

    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    train_start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        # Training step
        model.train()
        train_loss = 0.0
        for fc_input_batch, fc_output_batch, y_batch, lead_time_batch, doy_batch in train_loader:
            fc_input_batch = fc_input_batch.to(device, non_blocking=non_blocking)
            fc_output_batch = fc_output_batch.to(device, non_blocking=non_blocking)
            y_batch = y_batch.to(device, non_blocking=non_blocking)
            lead_time_batch = lead_time_batch.to(device, non_blocking=non_blocking)
            doy_batch = doy_batch.to(device, non_blocking=non_blocking)

            optimizer.zero_grad()

            # Use automatic mixed precision for CUDA
            if use_amp:
                with torch.amp.autocast("cuda"):
                    # Model takes training inputs and predicts error
                    pred_error = model(fc_input_batch, lead_time_batch, doy_batch)
                    # Apply error to output forecast
                    preds = fc_output_batch + pred_error

                    if alternate_loss_fn == "heatwave_loss":
                        loss = criterion(preds, y_batch, lead_time_batch, n_lead_times,
                                        is_normalized=True, std_out=std_out, mean_out=mean_out,
                                        lead_time_days=lead_time_days)
                    elif alternate_loss_fn in NORMALIZED_LOSS_FNS:
                        loss = criterion(preds, y_batch, is_normalized=True,
                                        std_out=std_out, mean_out=mean_out)
                    else:
                        loss = criterion(preds, y_batch)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training for CPU/MPS
                pred_error = model(fc_input_batch, lead_time_batch, doy_batch)
                preds = fc_output_batch + pred_error

                if alternate_loss_fn == "heatwave_loss":
                    loss = criterion(preds, y_batch, lead_time_batch, n_lead_times,
                                    is_normalized=True, std_out=std_out, mean_out=mean_out,
                                    lead_time_days=lead_time_days)
                elif alternate_loss_fn in NORMALIZED_LOSS_FNS:
                    loss = criterion(preds, y_batch, is_normalized=True,
                                    std_out=std_out, mean_out=mean_out)
                else:
                    loss = criterion(preds, y_batch)

                loss.backward()
                optimizer.step()

            train_loss += loss.item() * fc_output_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for fc_input_batch, fc_output_batch, y_batch, lead_time_batch, doy_batch in valid_loader:
                fc_input_batch = fc_input_batch.to(device, non_blocking=non_blocking)
                fc_output_batch = fc_output_batch.to(device, non_blocking=non_blocking)
                y_batch = y_batch.to(device, non_blocking=non_blocking)
                lead_time_batch = lead_time_batch.to(device, non_blocking=non_blocking)
                doy_batch = doy_batch.to(device, non_blocking=non_blocking)

                if use_amp:
                    with torch.amp.autocast("cuda"):
                        pred_error = model(fc_input_batch, lead_time_batch, doy_batch)
                        preds = fc_output_batch + pred_error

                        if alternate_loss_fn == "heatwave_loss":
                            loss = criterion(preds, y_batch, lead_time_batch, n_lead_times,
                                            is_normalized=True, std_out=std_out, mean_out=mean_out,
                                            lead_time_days=lead_time_days)
                        elif alternate_loss_fn in NORMALIZED_LOSS_FNS:
                            loss = criterion(preds, y_batch, is_normalized=True,
                                            std_out=std_out, mean_out=mean_out)
                        else:
                            loss = criterion(preds, y_batch)
                else:
                    pred_error = model(fc_input_batch, lead_time_batch, doy_batch)
                    preds = fc_output_batch + pred_error

                    if alternate_loss_fn == "heatwave_loss":
                        loss = criterion(preds, y_batch, lead_time_batch, n_lead_times,
                                        is_normalized=True, std_out=std_out, mean_out=mean_out,
                                        lead_time_days=lead_time_days)
                    elif alternate_loss_fn in NORMALIZED_LOSS_FNS:
                        loss = criterion(preds, y_batch, is_normalized=True,
                                        std_out=std_out, mean_out=mean_out)
                    else:
                        loss = criterion(preds, y_batch)

                val_loss += loss.item() * fc_output_batch.size(0)

        val_loss /= len(valid_loader.dataset)

        # Update learning rate scheduler
        scheduler.step(val_loss)

        # Early stopping check
        if val_loss + min_delta < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

    train_time_minutes = (time.time() - train_start_time) / 60.0

    # Load best weights
    model.load_state_dict(best_model_wts)

    return best_val_loss, epoch


def preload_training_data(args: SimpleNamespace,
                          data_dir: str,
                          device: torch.device,
                          use_legacy_global_data: bool = False,
                          split_seed: int = 42) -> Dict[str, Any]:
    """
    Pre-load and cache training data to avoid repeated loading across hyperparameter trials.

    This function loads the training data once, normalizes it, and performs train/val split.
    The cached data can be reused across all hyperparameter trials, dramatically reducing
    data loading overhead. The split holds out the last training year as validation, mirroring
    the year-separated evaluation used in production (train 2018-2021, test 2022).

    Args:
        args: Configuration with region, variables, lead times, etc.
        data_dir: Path to data directory
        device: torch device
        use_legacy_global_data: Whether to use legacy global data format
        split_seed: Unused; kept for API compatibility.

    Returns:
        dict: Cached training data including normalized arrays, indices, and metadata
    """
    print("\n" + "="*70)
    print("PRE-LOADING TRAINING DATA (will be cached for all trials)")
    print("="*70)

    # Get region grid
    lat_vals, lon_vals = get_region_grid(args)

    # Load training data
    (fc, fc_output, obs, lead_time_indices, day_of_year_features, train_times,
     lat_u, lon_u, n_lat, n_lon, n_training_vars, n_output_vars, _) = \
        load_forecasts(data_dir, args, lat_vals, lon_vals, train=True,
                       use_legacy_global_data=use_legacy_global_data)

    print(f"  Loaded {len(fc)} training samples")
    print(f"  Spatial dimensions: {n_lat} x {n_lon}")
    print(f"  Training variables: {n_training_vars}, Output variables: {n_output_vars}")

    # Normalize data
    stats_train = {'mean': fc.mean(0), 'std': fc.std(0) + 1e-8}
    stats_out = {'mean': fc_output.mean(0), 'std': fc_output.std(0) + 1e-8}
    fc_norm = (fc - stats_train['mean']) / stats_train['std']
    fc_output_norm = (fc_output - stats_out['mean']) / stats_out['std']
    obs_norm = (obs - stats_out['mean']) / stats_out['std']

    # Hold out the last training year as validation so tuning objective matches the
    # year-separated production evaluation (train 2018-2021, test 2022).
    train_idx, val_idx, holdout_year = _year_holdout_split(train_times)

    print(f"  Holdout year: {holdout_year} ({(val_idx.size)} val samples)")
    print(f"  Train samples: {len(train_idx)}, Validation samples: {len(val_idx)}")
    print("  Data caching complete!")
    print("="*70 + "\n")

    return {
        'fc_norm': fc_norm,
        'fc_output_norm': fc_output_norm,
        'obs_norm': obs_norm,
        'lead_time_indices': lead_time_indices,
        'day_of_year_features': day_of_year_features,
        'train_idx': train_idx,
        'val_idx': val_idx,
        'n_lat': n_lat,
        'n_lon': n_lon,
        'n_training_vars': n_training_vars,
        'n_output_vars': n_output_vars,
        'stats_train': stats_train,
        'stats_out': stats_out,
        'train_times': train_times,  # Timestamps for year-based block LTHO splits
    }


def preload_multi_cell_data(cell_patches: list,
                            args: SimpleNamespace,
                            data_dir: str,
                            device: torch.device,
                            use_legacy_global_data: bool = False,
                            split_seed: int = 42) -> list:
    """
    Pre-load and cache training data for multiple continent 6x6 cells.

    Each cell is loaded independently and stored as its own cached_data dict,
    so that different hyperopt trials can train on different cells.

    Args:
        cell_patches: List of (continent, patch_idx, patch_array) from sample_continent_patches()
        args: Configuration with variables, lead times, etc. (region/subregion are overridden per cell)
        data_dir: Path to data directory
        device: torch device
        use_legacy_global_data: Whether to use legacy global data format
        split_seed: Unused; kept for API compatibility.

    Returns:
        list of cached_data dicts (one per cell), same format as preload_training_data()
    """
    print("\n" + "="*70)
    print(f"PRE-LOADING DATA FOR {len(cell_patches)} CONTINENT CELLS")
    print("="*70)

    cached_cells = []
    for i, (continent, patch_idx, patch_array) in enumerate(cell_patches):
        lat_vals = patch_array[0]
        lon_vals = patch_array[1]

        print(f"\n  Cell {i+1}/{len(cell_patches)}: {continent} patch {patch_idx} "
              f"(lat {lat_vals.min():.1f}-{lat_vals.max():.1f}, "
              f"lon {lon_vals.min():.1f}-{lon_vals.max():.1f})")

        try:
            (fc, fc_output, obs, lead_time_indices, day_of_year_features, train_times,
             lat_u, lon_u, n_lat, n_lon, n_training_vars, n_output_vars, _) = \
                load_forecasts(data_dir, args, lat_vals, lon_vals, train=True,
                               use_legacy_global_data=use_legacy_global_data)

            # Normalize data
            stats_train = {'mean': fc.mean(0), 'std': fc.std(0) + 1e-8}
            stats_out = {'mean': fc_output.mean(0), 'std': fc_output.std(0) + 1e-8}
            fc_norm = (fc - stats_train['mean']) / stats_train['std']
            fc_output_norm = (fc_output - stats_out['mean']) / stats_out['std']
            obs_norm = (obs - stats_out['mean']) / stats_out['std']

            n_samples = len(fc)
            train_idx, val_idx, holdout_year = _year_holdout_split(train_times)

            cached_cells.append({
                'fc_norm': fc_norm,
                'fc_output_norm': fc_output_norm,
                'obs_norm': obs_norm,
                'lead_time_indices': lead_time_indices,
                'day_of_year_features': day_of_year_features,
                'train_idx': train_idx,
                'val_idx': val_idx,
                'n_lat': n_lat,
                'n_lon': n_lon,
                'n_training_vars': n_training_vars,
                'n_output_vars': n_output_vars,
                'stats_train': stats_train,
                'stats_out': stats_out,
                'train_times': train_times,
                'continent': continent,
                'patch_idx': patch_idx,
            })
            print(f"    Loaded {n_samples} samples ({len(train_idx)} train, {len(val_idx)} val, holdout_year={holdout_year})")

        except Exception as e:
            print(f"    WARNING: Failed to load cell {continent} patch {patch_idx}: {e}")
            continue

    print(f"\n  Successfully loaded {len(cached_cells)}/{len(cell_patches)} cells")
    print("="*70 + "\n")
    return cached_cells


def evaluate_hyperparameters(hyperparams: Dict[str, Any],
                            args: SimpleNamespace,
                            data_dir: str,
                            architecture: str,
                            device: torch.device,
                            cached_data: Dict[str, Any] = None,
                            use_snapshot: bool = False,
                            use_block_ltho: bool = False,
                            use_per_lt: bool = False,
                            split_seed: int = 42,
                            snapshot_objective_runs: int = 3,
                            snapshot_epochs: int = 210) -> Dict[str, Any]:
    """
    Evaluate a set of hyperparameters for a single region/variable configuration.

    Args:
        hyperparams: Dictionary of hyperparameters to evaluate
        args: Configuration with region, variables, lead times, etc.
        data_dir: Path to data directory
        architecture: 'mlp' or 'unet'
        device: torch device
        cached_data: Optional pre-loaded training data to avoid repeated loading
        use_snapshot: Use random-split snapshot ensemble objective
        use_block_ltho: Use leave-three-out block CV objective (overrides use_snapshot)
        use_per_lt: Use per-lead-time snapshot ensemble objective (overrides use_snapshot)

    Returns:
        dict: {'loss': validation_loss, 'status': STATUS_OK, 'epochs_trained': num_epochs}
    """
    if use_block_ltho:
        mode = 'block_ltho'
    elif use_per_lt:
        mode = 'per_lt_snapshot'
    elif use_snapshot:
        mode = 'snapshot'
    else:
        mode = 'early-stopping'
    print(f"\nEvaluating hyperparameters:")
    print(f"  Architecture: {architecture} ({mode})")
    print(f"  Learning rate: {hyperparams['learning_rate']:.6f}")
    print(f"  Hidden dim: {hyperparams['hidden_dim']}")
    print(f"  Batch size: {hyperparams['batch_size']}")

    # --- Block LTHO path ---
    if use_block_ltho:
        if cached_data is None:
            raise ValueError("Block LTHO evaluation requires cached_data (call preload_training_data first).")
        if 'train_times' not in cached_data:
            raise ValueError("cached_data must include 'train_times' for block LTHO evaluation.")
        print(f"  snapshot_T0: {hyperparams.get('snapshot_T0', 10)}")
        print(f"  snapshot_T_mult: {hyperparams.get('snapshot_T_mult', 1)}")
        val_loss = evaluate_block_ltho_hyperparameters(
            hyperparams, args, cached_data, device, snapshot_epochs=snapshot_epochs
        )
        print(f"  Block LTHO CV loss: {val_loss:.6f}")
        return {
            'loss': val_loss,
            'status': STATUS_OK,
            'epochs_trained': snapshot_epochs,
            'hyperparams': hyperparams,
        }

    # --- Per-LT snapshot path ---
    if use_per_lt:
        if cached_data is None:
            raise ValueError("Per-LT evaluation requires cached_data (call preload_training_data first).")
        print(f"  Snapshot T0: {hyperparams.get('snapshot_T0', 30)}")
        print(f"  Snapshot T_mult: {hyperparams.get('snapshot_T_mult', 1)}")
        print(f"  Objective runs per lead time: {snapshot_objective_runs}")
        val_loss = evaluate_per_lt_hyperparameters(
            hyperparams, args, cached_data, device,
            snapshot_epochs=snapshot_epochs,
            snapshot_objective_runs=snapshot_objective_runs
        )
        return {
            'loss': val_loss,
            'status': STATUS_OK,
            'epochs_trained': snapshot_epochs,
            'hyperparams': hyperparams,
        }

    if use_snapshot:
        print(f"  Snapshot T0: {hyperparams.get('snapshot_T0', 30)}")
        print(f"  Snapshot T_mult: {hyperparams.get('snapshot_T_mult', 1)}")
        print(f"  Objective runs: {snapshot_objective_runs}")
    else:
        print(f"  Patience: {hyperparams['patience']}")

    # Use cached data if available, otherwise load from scratch
    if cached_data is not None:
        print(f"  Using cached training data (fast path)")
        fc_norm = cached_data['fc_norm']
        fc_output_norm = cached_data['fc_output_norm']
        obs_norm = cached_data['obs_norm']
        lead_time_indices = cached_data['lead_time_indices']
        day_of_year_features = cached_data['day_of_year_features']
        train_idx = cached_data['train_idx']
        val_idx = cached_data['val_idx']
        n_lat = cached_data['n_lat']
        n_lon = cached_data['n_lon']
        n_training_vars = cached_data['n_training_vars']
        n_output_vars = cached_data['n_output_vars']
        stats_out = cached_data['stats_out']
    else:
        print(f"  Loading training data (slow path - no cache)")
        # Get region grid
        lat_vals, lon_vals = get_region_grid(args)

        # Load training data
        (fc, fc_output, obs, lead_time_indices, day_of_year_features, train_times,
         lat_u, lon_u, n_lat, n_lon, n_training_vars, n_output_vars, _) = \
            load_forecasts(data_dir, args, lat_vals, lon_vals, train=True, use_legacy_global_data=USE_LEGACY_GLOBAL_DATA)

        # Normalize data
        stats_train = {'mean': fc.mean(0), 'std': fc.std(0) + 1e-8}
        stats_out = {'mean': fc_output.mean(0), 'std': fc_output.std(0) + 1e-8}
        fc_norm = (fc - stats_train['mean']) / stats_train['std']
        fc_output_norm = (fc_output - stats_out['mean']) / stats_out['std']
        obs_norm = (obs - stats_out['mean']) / stats_out['std']

        # Split train/validation (80/20)
        n_samples = len(fc)
        rng = np.random.default_rng(split_seed)
        indices = rng.permutation(n_samples)
        split_idx = int(0.8 * n_samples)
        train_idx = indices[:split_idx]
        val_idx = indices[split_idx:]

    # Snapshot objective: match production behavior by training multiple
    # independent snapshot runs (different splits/seeds), then averaging all
    # snapshot predictions on a fixed held-out evaluation set.
    if use_snapshot:
        if getattr(args, 'alternate_loss_fn', None) is not None:
            raise ValueError("Snapshot hyperparameter tuning currently supports only MSE (alternate_loss_fn must be None).")

        input_dim = n_training_vars * n_lat * n_lon
        output_dim = n_output_vars * n_lat * n_lon
        n_lead_times = len(args.lead_time_hours)

        def _build_mlp_model():
            return SimpleMLP(
                input_dim=input_dim,
                hidden_dim=hyperparams['hidden_dim'],
                output_dim=output_dim,
                num_hidden_layers=hyperparams['num_layers'],
                n_lead_times=n_lead_times,
                lead_time_embedding_dim=hyperparams.get('lead_time_embedding_dim', 4),
                dropout_rate=hyperparams['dropout_rate']
            ).to(device)

        # Fixed held-out evaluation set for comparing hyperparameter trials.
        # Use make_eval_dataloader (shuffle=False) so that predictions from different
        # snapshot runs are collected in the same sample order and can be correctly averaged.
        eval_loader = make_eval_dataloader(
            fc_norm[val_idx],
            fc_output_norm[val_idx],
            obs_norm[val_idx],
            lead_time_indices[val_idx],
            day_of_year_features[val_idx],
            batch_size=hyperparams['batch_size'],
        )

        # Train only on the train pool, and create different train/val splits
        # per run (matching production snapshot ensemble behavior).
        train_pool_idx = np.array(train_idx)
        objective_seed = getattr(args, 'random_seed', split_seed)
        all_snapshot_preds = []
        eval_targets = None
        total_snapshots = 0

        for run_i in range(snapshot_objective_runs):
            run_seed = run_i * 17 + 1
            torch.manual_seed(run_seed)
            if device.type == 'cuda':
                torch.cuda.manual_seed_all(run_seed)

            split_rng = np.random.default_rng(run_seed * 13 + 7 + objective_seed)
            shuffled_pool = split_rng.permutation(train_pool_idx)
            split_idx = int(0.8 * len(shuffled_pool))
            run_train_idx = shuffled_pool[:split_idx]
            run_val_idx = shuffled_pool[split_idx:]

            run_train_loader = create_dataloader(
                fc_norm[run_train_idx],
                fc_output_norm[run_train_idx],
                obs_norm[run_train_idx],
                lead_time_indices[run_train_idx],
                day_of_year_features[run_train_idx],
                batch_size=hyperparams['batch_size'],
                device=device
            )
            run_val_loader = create_dataloader(
                fc_norm[run_val_idx],
                fc_output_norm[run_val_idx],
                obs_norm[run_val_idx],
                lead_time_indices[run_val_idx],
                day_of_year_features[run_val_idx],
                batch_size=hyperparams['batch_size'],
                device=device
            )

            model = _build_mlp_model()
            snapshots, _ = train_snapshot_ensemble(
                model, run_train_loader, run_val_loader,
                epochs=snapshot_epochs,
                lr=hyperparams['learning_rate'],
                device=device,
                weight_decay=hyperparams['weight_decay'],
                grad_clip=1.0,
                T_0=hyperparams.get('snapshot_T0', 30),
                T_mult=hyperparams.get('snapshot_T_mult', 1)
            )

            if not snapshots:
                continue

            for snap_weights, _ in snapshots:
                model.load_state_dict(snap_weights)
                model.eval()

                preds_batches = []
                targets_batches = []
                with torch.no_grad():
                    # eval_loader is non-shuffling, so every snapshot sees samples in the
                    # same order — predictions stack correctly into an aligned ensemble tensor.
                    for fc_input_batch, fc_output_batch, y_batch, lead_time_batch, doy_batch in eval_loader:
                        fc_input_batch = fc_input_batch.to(device)
                        fc_output_batch = fc_output_batch.to(device)
                        lead_time_batch = lead_time_batch.to(device)
                        doy_batch = doy_batch.to(device)

                        pred_error = model(fc_input_batch, lead_time_batch, doy_batch)
                        preds = fc_output_batch + pred_error

                        preds_batches.append(preds.cpu())
                        targets_batches.append(y_batch.cpu())

                snapshot_preds = torch.cat(preds_batches, dim=0)

                # Collect targets once (all passes give the same order since shuffle=False)
                if eval_targets is None:
                    eval_targets = torch.cat(targets_batches, dim=0)

                all_snapshot_preds.append(snapshot_preds)
                total_snapshots += 1

        if total_snapshots == 0:
            return {
                'loss': float('inf'),
                'status': STATUS_OK,
                'epochs_trained': 0,
                'hyperparams': hyperparams,
                'n_snapshots': 0,
                'snapshot_objective_runs': snapshot_objective_runs
            }

        ensemble_pred = torch.stack(all_snapshot_preds, dim=0).mean(dim=0)
        val_loss = float(nn.MSELoss()(ensemble_pred, eval_targets).item())

        print(f"  Validation loss: {val_loss:.6f} (across {snapshot_objective_runs} runs, {total_snapshots} snapshots)")
        return {
            'loss': val_loss,
            'status': STATUS_OK,
            'epochs_trained': snapshot_epochs,
            'hyperparams': hyperparams,
            'n_snapshots': total_snapshots,
            'snapshot_objective_runs': snapshot_objective_runs
        }

    # Create data loaders with device-specific optimizations
    train_loader = create_dataloader(
        fc_norm[train_idx],
        fc_output_norm[train_idx],
        obs_norm[train_idx],
        lead_time_indices[train_idx],
        day_of_year_features[train_idx],
        batch_size=hyperparams['batch_size'],
        device=device
    )
    val_loader = create_dataloader(
        fc_norm[val_idx],
        fc_output_norm[val_idx],
        obs_norm[val_idx],
        lead_time_indices[val_idx],
        day_of_year_features[val_idx],
        batch_size=hyperparams['batch_size'],
        device=device
    )

    # Initialize model
    input_dim = n_training_vars * n_lat * n_lon
    output_dim = n_output_vars * n_lat * n_lon
    n_lead_times = len(args.lead_time_hours)

    if architecture == 'mlp':
        model = SimpleMLP(
            input_dim=input_dim,
            hidden_dim=hyperparams['hidden_dim'],
            output_dim=output_dim,
            num_hidden_layers=hyperparams['num_layers'],
            n_lead_times=n_lead_times,
            lead_time_embedding_dim=hyperparams.get('lead_time_embedding_dim', 4),
            dropout_rate=hyperparams['dropout_rate']
        ).to(device)
    elif architecture == 'unet':
        model = UNet(
            input_dim=input_dim,
            hidden_dim=hyperparams['hidden_dim'],
            output_dim=output_dim,
            n_lat=n_lat,
            n_lon=n_lon,
            n_input_vars=n_training_vars,
            n_output_vars=n_output_vars,
            n_lead_times=n_lead_times,
            lead_time_embedding_dim=hyperparams.get('lead_time_embedding_dim', 4),
            dropout_rate=hyperparams['dropout_rate']
        ).to(device)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    alternate_loss_fn = getattr(args, 'alternate_loss_fn', None)
    lead_time_days = [h // 24 for h in args.lead_time_hours] if hasattr(args, 'lead_time_hours') else None
    val_loss, epochs_trained = train_with_early_stopping(
        model, train_loader, val_loader, hyperparams, device,
        alternate_loss_fn=alternate_loss_fn,
        stats_out=stats_out,
        n_output_vars=n_output_vars,
        n_lat=n_lat,
        n_lon=n_lon,
        n_lead_times=n_lead_times,
        lead_time_days=lead_time_days
    )
    print(f"  Validation loss: {val_loss:.6f} (trained {epochs_trained} epochs)")

    return {
        'loss': val_loss,
        'status': STATUS_OK,
        'epochs_trained': epochs_trained,
        'hyperparams': hyperparams
    }


def evaluate_block_ltho_hyperparameters(hyperparams: Dict[str, Any],
                                        args: SimpleNamespace,
                                        cached_data: Dict[str, Any],
                                        device: torch.device,
                                        snapshot_epochs: int = 210) -> float:
    """
    Evaluate hyperparameters using leave-one-year-out mega-ensemble cross-validation.

    For each holdout year, trains one snapshot ensemble per remaining training year,
    then combines ALL those snapshots into a single val-loss-weighted mega-ensemble
    and evaluates it on the holdout year. This directly mirrors production, where all
    blocks' snapshots are combined into one mega-ensemble evaluated on the test set.

    With 4 training years this gives 4 LOO folds, each using 3 training-year blocks
    combined into a mega-ensemble (~63 snapshots with T0=10). The objective is the
    mean mega-ensemble MSE across all 4 holdout years.

    Why this is better than the old per-block approach:
      - Old: evaluated each block's ensemble independently (train 1yr, val 3yrs),
        averaged the 4 block MSEs. This rewarded tiny models (low variance per block)
        even though larger models benefit more from the cross-block mega-ensemble.
      - New: combines all training-year blocks into a mega-ensemble before evaluating,
        which correctly rewards model capacity that benefits from ensemble diversity.

    Args:
        hyperparams: Dict of hyperparameters including hidden_dim, num_layers,
            dropout_rate, learning_rate, weight_decay, batch_size,
            snapshot_T0, snapshot_T_mult
        args: SimpleNamespace with lead_time_hours
        cached_data: From preload_training_data; must include 'train_times'
        device: torch device
        snapshot_epochs: Fixed epochs per block (default 210)

    Returns:
        float: Mean LOO mega-ensemble MSE across all holdout years
    """
    import pandas as pd

    fc_norm = cached_data['fc_norm']
    fc_output_norm = cached_data['fc_output_norm']
    obs_norm = cached_data['obs_norm']
    lead_time_indices = cached_data['lead_time_indices']
    day_of_year_features = cached_data['day_of_year_features']
    train_times = cached_data['train_times']
    n_lat = cached_data['n_lat']
    n_lon = cached_data['n_lon']
    n_training_vars = cached_data['n_training_vars']
    n_output_vars = cached_data['n_output_vars']

    input_dim = n_training_vars * n_lat * n_lon
    output_dim = n_output_vars * n_lat * n_lon
    n_lead_times = len(args.lead_time_hours)

    # lead_time_embedding_dim is fixed at 4 to match the production default in finetune.py.
    # It is not part of the search space so that the hyperopt tunes for the same
    # architecture that production uses when no explicit override is given.
    LEAD_TIME_EMB_DIM = 4

    def _build_model():
        return SimpleMLP(
            input_dim=input_dim,
            hidden_dim=hyperparams['hidden_dim'],
            output_dim=output_dim,
            num_hidden_layers=hyperparams['num_layers'],
            n_lead_times=n_lead_times,
            lead_time_embedding_dim=LEAD_TIME_EMB_DIM,
            dropout_rate=hyperparams['dropout_rate']
        ).to(device)

    sample_years = pd.DatetimeIndex(train_times).year.values
    all_years = sorted(set(sample_years.tolist()))

    T0 = hyperparams.get('snapshot_T0', 10)
    T_mult = hyperparams.get('snapshot_T_mult', 1)
    batch_size = hyperparams['batch_size']
    criterion = nn.MSELoss()

    print(f"  Block LTHO mega-ensemble LOO: {len(all_years)} years, "
          f"T0={T0}, T_mult={T_mult}, epochs={snapshot_epochs}")

    fold_losses = []

    for holdout_year in all_years:
        holdout_mask = sample_years == holdout_year
        holdout_idx = np.where(holdout_mask)[0]

        if len(holdout_idx) == 0:
            continue

        # Non-shuffling eval loader for the holdout year.
        # Must be shuffle=False so all snapshot predictions align sample-by-sample.
        run_eval_loader = make_eval_dataloader(
            fc_norm[holdout_idx], fc_output_norm[holdout_idx],
            obs_norm[holdout_idx], lead_time_indices[holdout_idx],
            day_of_year_features[holdout_idx],
            batch_size=batch_size
        )

        holdout_targets = None
        mega_preds = []
        mega_weights = []

        train_years_for_fold = [y for y in all_years if y != holdout_year]

        for train_year in train_years_for_fold:
            seed_key = holdout_year * 100 + train_year
            torch.manual_seed(seed_key)
            np.random.seed(seed_key * 13)

            t_mask = sample_years == train_year
            # Val set = non-holdout, non-training years (used for snapshot val_loss weighting).
            # Excludes the holdout year so snapshot weights are unbiased w.r.t. final eval.
            v_mask = ~holdout_mask & ~t_mask
            train_block_idx = np.where(t_mask)[0]
            val_block_idx = np.where(v_mask)[0]

            if len(train_block_idx) == 0 or len(val_block_idx) == 0:
                print(f"    Skipping holdout={holdout_year}, train={train_year}: empty split")
                continue

            run_train_loader = create_dataloader(
                fc_norm[train_block_idx], fc_output_norm[train_block_idx],
                obs_norm[train_block_idx], lead_time_indices[train_block_idx],
                day_of_year_features[train_block_idx],
                batch_size=batch_size, device=device
            )
            run_val_loader = create_dataloader(
                fc_norm[val_block_idx], fc_output_norm[val_block_idx],
                obs_norm[val_block_idx], lead_time_indices[val_block_idx],
                day_of_year_features[val_block_idx],
                batch_size=batch_size, device=device
            )

            model = _build_model()
            snapshots, _ = train_snapshot_ensemble(
                model, run_train_loader, run_val_loader,
                epochs=snapshot_epochs,
                lr=hyperparams['learning_rate'],
                device=device,
                weight_decay=hyperparams['weight_decay'],
                grad_clip=1.0,
                T_0=T0,
                T_mult=T_mult
            )

            if not snapshots:
                print(f"    holdout={holdout_year} train={train_year}: no snapshots, skipping")
                continue

            # Collect each snapshot's predictions on the holdout year.
            # All snapshots from all training years are accumulated into the mega-ensemble.
            with torch.no_grad():
                for snap_state, snap_val_loss in snapshots:
                    model.load_state_dict(snap_state)
                    model.eval()
                    batch_preds = []
                    batch_targets = []
                    for fc_in, fc_out, y, lt, doy in run_eval_loader:
                        fc_in = fc_in.to(device)
                        fc_out = fc_out.to(device)
                        lt = lt.to(device)
                        doy = doy.to(device)
                        pred_error = model(fc_in, lt, doy)
                        preds = fc_out + pred_error
                        batch_preds.append(preds.cpu())
                        batch_targets.append(y.cpu())
                    mega_preds.append(torch.cat(batch_preds, dim=0))
                    mega_weights.append(1.0 / max(snap_val_loss, 1e-12))
                    if holdout_targets is None:
                        holdout_targets = torch.cat(batch_targets, dim=0)

            print(f"    holdout={holdout_year} train={train_year}: "
                  f"{len(snapshots)} snaps added (mega total: {len(mega_preds)})")

        if not mega_preds or holdout_targets is None:
            print(f"  WARNING: holdout={holdout_year} — no valid blocks, skipping fold")
            continue

        # Val-loss-weighted mega-ensemble prediction (mirrors production inference).
        # Combines snapshots from ALL training-year blocks into a single prediction.
        w = np.array(mega_weights, dtype=np.float64)
        w = w / w.sum()
        mega_pred = sum(float(wi) * p for wi, p in zip(w, mega_preds))
        fold_mse = float(criterion(mega_pred, holdout_targets).item())
        fold_losses.append(fold_mse)
        print(f"  holdout={holdout_year}: {len(mega_preds)} total snaps → mega-ensemble MSE={fold_mse:.6f}")

    if not fold_losses:
        print("  WARNING: All folds failed — returning inf loss")
        return float('inf')

    mean_loss = float(np.mean(fold_losses))
    print(f"  Mean LOO mega-ensemble MSE across {len(fold_losses)} folds: {mean_loss:.6f}")
    return mean_loss


def evaluate_per_lt_hyperparameters(hyperparams: Dict[str, Any],
                                    args: SimpleNamespace,
                                    cached_data: Dict[str, Any],
                                    device: torch.device,
                                    snapshot_epochs: int = 210,
                                    snapshot_objective_runs: int = 3) -> float:
    """
    Evaluate hyperparameters using the per-lead-time snapshot ensemble objective.

    Mirrors the per-LT production path in run_subregion_experiment(): trains a
    separate snapshot ensemble for each lead time using n_lead_times=1, so that
    each model only sees samples from its own horizon (no gradient competition
    across horizons). Seeds and the 80/20 shuffle-split within each lead time
    match production exactly (seed = run_i * 17 + lt_idx * 1000 + 1).

    Args:
        hyperparams: Dict with hidden_dim, num_layers, dropout_rate, learning_rate,
            weight_decay, batch_size, snapshot_T0, snapshot_T_mult,
            lead_time_embedding_dim.
        args: SimpleNamespace with lead_time_hours.
        cached_data: From preload_training_data; must include lead_time_indices.
        device: torch device.
        snapshot_epochs: Fixed epochs per snapshot run (default 210).
        snapshot_objective_runs: Independent snapshot runs per lead time (default 3).

    Returns:
        float: MSE of the per-LT ensemble on the held-out validation set,
            combining predictions across all lead times.
    """
    fc_norm = cached_data['fc_norm']
    fc_output_norm = cached_data['fc_output_norm']
    obs_norm = cached_data['obs_norm']
    lead_time_indices = cached_data['lead_time_indices']
    day_of_year_features = cached_data['day_of_year_features']
    train_idx = cached_data['train_idx']
    val_idx = cached_data['val_idx']
    n_lat = cached_data['n_lat']
    n_lon = cached_data['n_lon']
    n_training_vars = cached_data['n_training_vars']
    n_output_vars = cached_data['n_output_vars']

    input_dim = n_training_vars * n_lat * n_lon
    output_dim = n_output_vars * n_lat * n_lon
    n_lead_times = len(args.lead_time_hours)

    T0 = hyperparams.get('snapshot_T0', 30)
    T_mult = hyperparams.get('snapshot_T_mult', 1)
    batch_size = hyperparams['batch_size']
    lead_time_emb_dim = hyperparams.get('lead_time_embedding_dim', 4)
    criterion = nn.MSELoss()

    def _build_model():
        """Create a single-lead-time SimpleMLP (n_lead_times=1)."""
        return SimpleMLP(
            input_dim=input_dim,
            hidden_dim=hyperparams['hidden_dim'],
            output_dim=output_dim,
            num_hidden_layers=hyperparams['num_layers'],
            n_lead_times=1,
            lead_time_embedding_dim=lead_time_emb_dim,
            dropout_rate=hyperparams['dropout_rate']
        ).to(device)

    train_pool = np.array(train_idx)
    all_lt_preds = []
    all_lt_targets = []

    for lt_idx in range(n_lead_times):
        lt_hours = args.lead_time_hours[lt_idx]

        # Filter train pool and eval set to this lead time only.
        lt_train_pool = train_pool[lead_time_indices[train_pool] == lt_idx]
        lt_val_mask = lead_time_indices[val_idx] == lt_idx
        lt_val_idx = val_idx[lt_val_mask]

        if len(lt_train_pool) == 0 or len(lt_val_idx) == 0:
            print(f"  Skipping lt_idx={lt_idx} ({lt_hours}h): empty split")
            continue

        # Each per-LT model uses a single lead-time slot (index 0).
        lti_zeros_val = np.zeros(len(lt_val_idx), dtype=np.int64)
        eval_loader = make_eval_dataloader(
            fc_norm[lt_val_idx], fc_output_norm[lt_val_idx],
            obs_norm[lt_val_idx], lti_zeros_val,
            day_of_year_features[lt_val_idx],
            batch_size=batch_size
        )

        lt_snapshot_preds = []
        lt_targets = None

        for run_i in range(snapshot_objective_runs):
            seed = run_i * 17 + lt_idx * 1000 + 1
            torch.manual_seed(seed)
            np.random.seed(seed * 13 + 7)

            run_idx = np.arange(len(lt_train_pool))
            np.random.shuffle(run_idx)
            split = int(0.8 * len(run_idx))
            run_train_idx = lt_train_pool[run_idx[:split]]
            run_val_idx = lt_train_pool[run_idx[split:]]

            run_train_loader = create_dataloader(
                fc_norm[run_train_idx], fc_output_norm[run_train_idx],
                obs_norm[run_train_idx], np.zeros(len(run_train_idx), dtype=np.int64),
                day_of_year_features[run_train_idx],
                batch_size=batch_size, device=device
            )
            run_val_loader = create_dataloader(
                fc_norm[run_val_idx], fc_output_norm[run_val_idx],
                obs_norm[run_val_idx], np.zeros(len(run_val_idx), dtype=np.int64),
                day_of_year_features[run_val_idx],
                batch_size=batch_size, device=device
            )

            model = _build_model()
            snapshots, _ = train_snapshot_ensemble(
                model, run_train_loader, run_val_loader,
                epochs=snapshot_epochs,
                lr=hyperparams['learning_rate'],
                device=device,
                weight_decay=hyperparams['weight_decay'],
                grad_clip=1.0,
                T_0=T0, T_mult=T_mult
            )

            if not snapshots:
                continue

            for snap_weights, _ in snapshots:
                model.load_state_dict(snap_weights)
                model.eval()
                preds_batches = []
                targets_batches = []
                with torch.no_grad():
                    for fc_in, fc_out, y, lt, doy in eval_loader:
                        fc_in = fc_in.to(device)
                        fc_out = fc_out.to(device)
                        lt = lt.to(device)
                        doy = doy.to(device)
                        pred_error = model(fc_in, lt, doy)
                        preds = fc_out + pred_error
                        preds_batches.append(preds.cpu())
                        targets_batches.append(y.cpu())
                lt_snapshot_preds.append(torch.cat(preds_batches, dim=0))
                if lt_targets is None:
                    lt_targets = torch.cat(targets_batches, dim=0)

        if not lt_snapshot_preds or lt_targets is None:
            print(f"  lt={lt_hours}h: no snapshots collected, skipping")
            continue

        lt_ensemble_pred = torch.stack(lt_snapshot_preds, dim=0).mean(dim=0)
        lt_mse = float(criterion(lt_ensemble_pred, lt_targets).item())
        print(f"  lt={lt_hours}h: {len(lt_snapshot_preds)} snapshots → MSE={lt_mse:.6f}")
        all_lt_preds.append(lt_ensemble_pred)
        all_lt_targets.append(lt_targets)

    if not all_lt_preds:
        print("  WARNING: No lead times produced predictions — returning inf")
        return float('inf')

    combined_preds = torch.cat(all_lt_preds, dim=0)
    combined_targets = torch.cat(all_lt_targets, dim=0)
    val_loss = float(criterion(combined_preds, combined_targets).item())
    print(f"  Per-LT objective: {val_loss:.6f} (across {len(all_lt_preds)} lead times, "
          f"{snapshot_objective_runs} runs each)")
    return val_loss


def optimize_hyperparameters(args: SimpleNamespace,
                            data_dir: str,
                            architecture: str,
                            max_evals: int = 100,
                            output_dir: str = None,
                            device: torch.device = None,
                            random_seed: int = 42,
                            resume: bool = False,
                            use_snapshot: bool = False,
                            use_block_ltho: bool = False,
                            use_per_lt: bool = False,
                            snapshot_objective_runs: int = 3,
                            snapshot_epochs: int = 210,
                            multi_cell_data: list = None) -> Dict[str, Any]:
    """
    Optimize hyperparameters for a single region/variable configuration.

    Args:
        args: Configuration with region, variables, lead times, etc.
        data_dir: Path to data directory
        architecture: 'mlp' or 'unet'
        max_evals: Maximum number of evaluations
        output_dir: Directory to save results
        device: torch device (auto-detected if None)
        random_seed: Random seed for reproducibility
        resume: If True, continue from previous trials
        use_snapshot: Optimize for random-split snapshot ensemble
        use_block_ltho: Optimize for block leave-three-out ensemble (overrides use_snapshot)
        use_per_lt: Optimize for per-lead-time snapshot ensemble (overrides use_snapshot)
        multi_cell_data: If provided, list of cached_data dicts from preload_multi_cell_data().
            Each trial evaluates on a different cell (round-robin), making hyperparameters
            robust across diverse geographic regions.

    Returns:
        dict: Best hyperparameters and optimization results
    """
    # Validate inputs
    if architecture not in ['mlp', 'unet']:
        raise ValueError(f"Architecture must be 'mlp' or 'unet', got: {architecture}")
    if (use_snapshot or use_block_ltho or use_per_lt) and architecture != 'mlp':
        raise ValueError("Snapshot/block LTHO/per-LT ensemble tuning is only supported for 'mlp' architecture")
    # Enforce mutual exclusivity: block_ltho > per_lt > snapshot
    if use_block_ltho:
        use_snapshot = False
        use_per_lt = False
    elif use_per_lt:
        use_snapshot = False

    # Set random seeds
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Setup device
    if device is None:
        device = torch.device(
            'cuda' if torch.cuda.is_available() else
            'mps' if torch.backends.mps.is_available() else
            'cpu'
        )
    print(f"Using device: {device}")
    if use_block_ltho:
        mode_str = 'block_ltho'
    elif use_per_lt:
        mode_str = 'per_lt_snapshot'
    elif use_snapshot:
        mode_str = 'snapshot'
    else:
        mode_str = 'early-stopping'

    use_multi_cell = multi_cell_data is not None and len(multi_cell_data) > 0
    if use_multi_cell:
        print(f"Optimizing {architecture.upper()} ({mode_str}) across {len(multi_cell_data)} continent cells, "
              f"variable(s) {args.output_vars}, lead times {args.lead_time_hours}h")
    else:
        print(f"Optimizing {architecture.upper()} ({mode_str}) for region '{args.region}', "
              f"variable(s) {args.output_vars}, lead times {args.lead_time_hours}h")
    if use_snapshot:
        print(f"Snapshot objective settings: runs={snapshot_objective_runs}, epochs={snapshot_epochs}")
    if use_block_ltho:
        print(f"Block LTHO objective: 4-fold year CV, epochs={snapshot_epochs} per block")
    if use_per_lt:
        print(f"Per-LT snapshot objective: runs={snapshot_objective_runs} per lead time, epochs={snapshot_epochs}")

    # Persist seed/objective settings in args for downstream helpers.
    args.random_seed = random_seed
    args.snapshot_epochs = snapshot_epochs
    args.snapshot_objective_runs = snapshot_objective_runs

    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"hyperopt_results_{architecture}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Create search space
    if use_block_ltho:
        search_space = create_block_ltho_search_space()
    elif use_snapshot:
        search_space = create_mlp_snapshot_search_space()
    elif architecture == 'mlp':
        search_space = create_mlp_search_space()
    else:  # unet
        search_space = create_unet_search_space()

    # Pre-load training data once for all trials (major performance optimization)
    if not use_multi_cell:
        cached_data = preload_training_data(
            args,
            data_dir,
            device,
            use_legacy_global_data=USE_LEGACY_GLOBAL_DATA,
            split_seed=random_seed
        )

    # Trial counter for round-robin cell selection in multi-cell mode.
    # CELLS_PER_TRIAL controls how many cells each hyperopt trial evaluates on.
    # Using multiple cells per trial reduces noise from cell-level heterogeneity
    # (some regions are much easier to improve than others), preventing the
    # optimizer from rewarding a bad architecture that happened to land on an
    # easy cell.
    CELLS_PER_TRIAL = 3
    trial_counter = [0]

    # Define objective function
    def objective(hyperparams):
        if use_multi_cell:
            # Evaluate on CELLS_PER_TRIAL different cells and average the loss.
            # Cells are selected round-robin so every cell gets roughly equal use.
            n_cells = len(multi_cell_data)
            cell_losses = []
            cell_infos = []
            cell_epochs = []

            base_idx = (trial_counter[0] * CELLS_PER_TRIAL) % n_cells
            trial_counter[0] += 1

            for k in range(CELLS_PER_TRIAL):
                cell_idx = (base_idx + k) % n_cells
                cell_data = multi_cell_data[cell_idx]
                cell_info = f"{cell_data.get('continent', '?')} patch {cell_data.get('patch_idx', '?')}"
                cell_infos.append(cell_info)

                print(f"\n  Trial {trial_counter[0]}, cell {k+1}/{CELLS_PER_TRIAL}: "
                      f"idx {cell_idx} ({cell_info})")

                result_k = evaluate_hyperparameters(
                    hyperparams, args, data_dir, architecture, device,
                    cached_data=cell_data,
                    use_snapshot=use_snapshot,
                    use_block_ltho=use_block_ltho,
                    use_per_lt=use_per_lt,
                    split_seed=random_seed,
                    snapshot_objective_runs=snapshot_objective_runs,
                    snapshot_epochs=snapshot_epochs
                )
                cell_losses.append(result_k['loss'])
                cell_epochs.append(result_k['epochs_trained'])

            mean_loss = float(np.mean(cell_losses))
            print(f"\n  Trial {trial_counter[0]} mean loss across {CELLS_PER_TRIAL} cells: "
                  f"{mean_loss:.6f}  (individual: {[f'{l:.4f}' for l in cell_losses]})")

            result = {
                'loss': mean_loss,
                'status': STATUS_OK,
                'epochs_trained': int(np.mean(cell_epochs)),
                'hyperparams': hyperparams,
                'cell_losses': cell_losses,
                'cell_infos': cell_infos,
            }
            cell_info_str = "; ".join(cell_infos)
        else:
            result = evaluate_hyperparameters(
                hyperparams, args, data_dir, architecture, device,
                cached_data=cached_data,
                use_snapshot=use_snapshot,
                use_block_ltho=use_block_ltho,
                use_per_lt=use_per_lt,
                split_seed=random_seed,
                snapshot_objective_runs=snapshot_objective_runs,
                snapshot_epochs=snapshot_epochs
            )
            cell_info_str = args.region

        # Save intermediate result
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = os.path.join(output_dir, f'eval_{timestamp}.json')
        save_dict = {
            'hyperparams': hyperparams,
            'loss': result['loss'],
            'epochs_trained': result['epochs_trained'],
            'architecture': architecture,
            'mode': mode_str,
            'snapshot_objective_runs': snapshot_objective_runs if (use_snapshot or use_per_lt) else None,
            'snapshot_epochs': snapshot_epochs,
            'n_snapshots': result.get('n_snapshots'),
            'cell_info': cell_info_str,
        }
        if use_multi_cell:
            save_dict['cell_losses'] = result.get('cell_losses')
            save_dict['cell_infos'] = result.get('cell_infos')
        with open(result_file, 'w') as f:
            json.dump(save_dict, f, indent=2)

        return result

    # Load or initialize trials
    trials_file = os.path.join(output_dir, f'trials_{architecture}.pkl')

    if resume and os.path.exists(trials_file):
        print(f"Resuming from {trials_file}")
        with open(trials_file, 'rb') as f:
            trials = pickle.load(f)
        n_previous = len(trials.trials)
        print(f"Loaded {n_previous} previous trials")
        trial_counter[0] = n_previous  # Continue round-robin from where we left off

        if n_previous >= max_evals:
            print(f"Already completed {n_previous} evaluations (>= {max_evals})")
            # Extract best results
            best_idx = np.argmin([t['result']['loss'] for t in trials.trials])
            best_trial = trials.trials[best_idx]
            best_hyperparams = space_eval(search_space, best_trial['misc']['vals'])

            return {
                'architecture': architecture,
                'best_hyperparams': best_hyperparams,
                'best_loss': best_trial['result']['loss'],
                'n_evaluations': len(trials.trials)
            }
    else:
        trials = Trials()
        if resume:
            print(f"No previous trials found, starting fresh")

    # Run optimization
    print(f"\nStarting Bayesian optimization with {max_evals} evaluations...")
    best = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
    )

    # Get best hyperparameters
    best_hyperparams = space_eval(search_space, best)
    best_idx = np.argmin([t['result']['loss'] for t in trials.trials])
    best_trial = trials.trials[best_idx]

    # Prepare results
    region_info = f"multi_cell_{len(multi_cell_data)}_cells" if use_multi_cell else args.region
    results = {
        'architecture': architecture,
        'mode': mode_str,
        'best_hyperparams': best_hyperparams,
        'best_loss': best_trial['result']['loss'],
        'best_epochs_trained': best_trial['result']['epochs_trained'],
        'snapshot_objective_runs': snapshot_objective_runs if (use_snapshot or use_per_lt) else None,
        'snapshot_epochs': snapshot_epochs if (use_snapshot or use_block_ltho or use_per_lt) else None,
        'best_n_snapshots': best_trial['result'].get('n_snapshots') if use_snapshot else None,
        'n_evaluations': len(trials.trials),
        'region': region_info,
        'variables': args.output_vars,
        'lead_times': args.lead_time_hours
    }

    # Save results
    results_file = os.path.join(output_dir, f'optimization_results_{architecture}.json')
    with open(results_file, 'w') as f:
        json.dump({
            'architecture': architecture,
            'mode': mode_str,
            'best_hyperparams': best_hyperparams,
            'best_loss': results['best_loss'],
            'best_epochs_trained': results['best_epochs_trained'],
            'snapshot_objective_runs': results['snapshot_objective_runs'],
            'snapshot_epochs': results['snapshot_epochs'],
            'best_n_snapshots': results['best_n_snapshots'],
            'n_evaluations': results['n_evaluations'],
            'region': region_info,
            'variables': args.output_vars,
            'lead_times': args.lead_time_hours
        }, f, indent=2)

    # Save trials
    with open(trials_file, 'wb') as f:
        pickle.dump(trials, f)

    print(f"\n{'='*60}")
    print(f"Optimization complete!")
    print(f"Best hyperparameters: {best_hyperparams}")
    print(f"Best validation loss: {results['best_loss']:.6f}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}\n")

    return results


# Example usage
if __name__ == "__main__":

    # ========================================================================
    # LEGACY FLAG: Set to True to use global yearly files (legacy format)
    # TO REMOVE: Remove this flag when legacy data is no longer needed
    # ========================================================================
    USE_LEGACY_GLOBAL_DATA = True # <-- EDIT THIS FLAG
    # ========================================================================

    # ========================================================================
    # CONFIGURE TUNING RUN HERE
    #
    # TUNING_MODE options:
    #   "temperature"            — tune standard MLP for 2m_temperature correction
    #   "wind"                   — tune standard MLP for 10m_wind_speed correction
    #   "joint"                  — tune joint temp+wind MLP (joint_temp_wind_loss)
    #   "block_ltho_temperature" — tune Block LTHO MLP for 2m_temperature
    #   "block_ltho_wind"        — tune Block LTHO MLP for 10m_wind_speed
    #   "per_lt_temperature"     — tune per-lead-time snapshot MLP for 2m_temperature ← RECOMMENDED for per-LT
    #   "per_lt_wind"            — tune per-lead-time snapshot MLP for 10m_wind_speed
    #
    # USE_SNAPSHOT_ENSEMBLE (ignored for block_ltho_* and per_lt_* modes):
    #   True  — tune for random-split snapshot ensemble MLP
    #           Results saved to hyperopt_results_snapshot_{var}_mlp/
    #   False — tune for standard MLP with early stopping
    #           Results saved to hyperopt_results_{var}_mlp/
    #
    # For block_ltho_* modes:
    #   - USE_SNAPSHOT_ENSEMBLE is automatically set to False
    #   - Objective = 4-fold leave-three-out year CV (each fold trains 1 yr, evals 3 yrs)
    #   - Results saved to hyperopt_results_block_ltho_{var}_mlp/
    #   - finetune.py auto-loads these when --block_ensemble is passed
    #
    # For per_lt_* modes:
    #   - USE_SNAPSHOT_ENSEMBLE is automatically set to False
    #   - Objective = per-lead-time snapshot ensemble; trains n_lead_times × SNAPSHOT_OBJECTIVE_RUNS
    #     models per trial (one model per lead time per run, each with n_lead_times=1)
    #   - Results saved to hyperopt_results_per_lt_{var}_mlp/
    #   - finetune.py auto-loads these when --per_lead_time is passed
    #
    # SNAPSHOT_OBJECTIVE_RUNS:
    #   Number of independent snapshot runs per hyperopt trial.
    #   For per_lt_* modes, this applies per lead time (total runs = SNAPSHOT_OBJECTIVE_RUNS × n_lead_times).
    #
    # SNAPSHOT_EPOCHS:
    #   Fixed training epochs per run/block during hyperopt.
    #   For block LTHO, each of the 4 blocks trains for this many epochs.
    #   Runtime estimate (M3 Max): ~0.1 min/run × 3 runs × 3 lead times = ~0.9 min/trial for per_lt
    # ========================================================================
    TUNING_MODE = "per_lt_wind"   # <-- EDIT THIS
    USE_SNAPSHOT_ENSEMBLE = True  # <-- EDIT THIS (ignored for block_ltho_* and per_lt_* modes)
    SNAPSHOT_OBJECTIVE_RUNS = 3   # <-- EDIT THIS
    SNAPSHOT_EPOCHS = 210         # <-- EDIT THIS
    # ========================================================================

    dirs = setup_directories()
    data_dir = dirs['raw']

    # Setup device
    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )
    print(f"Using device: {device}")

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        print("Enabled cudnn benchmarking for faster GPU training")
        print("Using mixed precision training (AMP) for CUDA operations")

    use_block_ltho = TUNING_MODE.startswith("block_ltho_")
    use_per_lt = TUNING_MODE.startswith("per_lt_")
    if use_block_ltho or use_per_lt:
        USE_SNAPSHOT_ENSEMBLE = False
    snapshot_prefix = "snapshot_" if USE_SNAPSHOT_ENSEMBLE else ""

    if TUNING_MODE == "temperature":
        config = SimpleNamespace(
            model_name="pangu",
            training_vars=["2m_temperature"],
            output_vars=["2m_temperature"],
            train_start="2018-01-01",
            train_end="2021-12-31",
            region="india",
            subregion="6x6",
            ground_truth_source="",
            lead_time_hours=[24, 120, 216],
            alternate_loss_fn=None,
            growing_season_only=False
        )
        output_dir = f"hyperopt_results_{snapshot_prefix}temperature_mlp"

    elif TUNING_MODE == "wind":
        config = SimpleNamespace(
            model_name="pangu",
            training_vars=["10m_wind_speed"],
            output_vars=["10m_wind_speed"],
            train_start="2018-01-01",
            train_end="2021-12-31",
            region="india",
            subregion="6x6",
            ground_truth_source="",
            lead_time_hours=[24, 120, 216],
            alternate_loss_fn=None,
            growing_season_only=False
        )
        output_dir = f"hyperopt_results_{snapshot_prefix}wind_mlp"

    elif TUNING_MODE == "joint":
        config = SimpleNamespace(
            model_name="pangu",
            training_vars=["2m_temperature", "10m_wind_speed"],
            output_vars=["2m_temperature", "10m_wind_speed"],
            train_start="2018-01-01",
            train_end="2021-12-31",
            region="usa_south",
            subregion="6x6",
            ground_truth_source="",
            lead_time_hours=[24],
            alternate_loss_fn="joint_temp_wind_loss",
            growing_season_only=False
        )
        output_dir = f"hyperopt_results_{snapshot_prefix}joint_wind_temperature_24h_mlp"

    elif TUNING_MODE == "block_ltho_temperature":
        # Block LTHO: tune across a 25% random sample of continent 6x6 cells.
        # Each trial trains on a different cell (round-robin), so hyperparameters
        # are optimized for diverse geographic conditions rather than a single region.
        config = SimpleNamespace(
            model_name="pangu",
            training_vars=["2m_temperature"],
            output_vars=["2m_temperature"],
            train_start="2018-01-01",
            train_end="2021-12-31",
            region="multi_cell",
            subregion="6x6",
            ground_truth_source="",
            lead_time_hours=[24, 120, 216],
            alternate_loss_fn=None,
            growing_season_only=False
        )
        output_dir = "hyperopt_results_block_ltho_temperature_mlp"

    elif TUNING_MODE == "block_ltho_wind":
        config = SimpleNamespace(
            model_name="pangu",
            training_vars=["10m_wind_speed"],
            output_vars=["10m_wind_speed"],
            train_start="2018-01-01",
            train_end="2021-12-31",
            region="multi_cell",
            subregion="6x6",
            ground_truth_source="",
            lead_time_hours=[24, 120, 216],
            alternate_loss_fn=None,
            growing_season_only=False
        )
        output_dir = "hyperopt_results_block_ltho_wind_mlp"

    elif TUNING_MODE == "per_lt_temperature":
        config = SimpleNamespace(
            model_name="pangu",
            training_vars=["2m_temperature"],
            output_vars=["2m_temperature"],
            train_start="2018-01-01",
            train_end="2021-12-31",
            region="india",
            subregion="6x6",
            ground_truth_source="",
            lead_time_hours=[24, 120, 216],
            alternate_loss_fn=None,
            growing_season_only=False
        )
        output_dir = "hyperopt_results_per_lt_temperature_mlp"

    elif TUNING_MODE == "per_lt_wind":
        config = SimpleNamespace(
            model_name="pangu",
            training_vars=["10m_wind_speed"],
            output_vars=["10m_wind_speed"],
            train_start="2018-01-01",
            train_end="2021-12-31",
            region="india",
            subregion="6x6",
            ground_truth_source="",
            lead_time_hours=[24, 120, 216],
            alternate_loss_fn=None,
            growing_season_only=False
        )
        output_dir = "hyperopt_results_per_lt_wind_mlp"

    else:
        raise ValueError(
            f"Unknown TUNING_MODE: '{TUNING_MODE}'. "
            "Choose 'temperature', 'wind', 'joint', 'block_ltho_temperature', 'block_ltho_wind', "
            "'per_lt_temperature', or 'per_lt_wind'."
        )

    print(f"\nTuning mode: {TUNING_MODE}")
    if use_block_ltho:
        print(f"Block LTHO objective: 4-fold year CV, {SNAPSHOT_EPOCHS} epochs/block")
    elif use_per_lt:
        print(f"Per-LT snapshot objective: {SNAPSHOT_OBJECTIVE_RUNS} runs/lead time, {SNAPSHOT_EPOCHS} epochs/run")
    else:
        print(f"Snapshot ensemble: {USE_SNAPSHOT_ENSEMBLE}")
    print(f"Output dir: {output_dir}\n")

    # For block LTHO modes, pre-load a 25% sample of continent cells
    multi_cell_data = None
    if use_block_ltho:
        cell_patches = sample_continent_patches(
            dirs['processed'], fraction=0.25, seed=42, split='hyperopt'
        )
        print(f"Sampled {len(cell_patches)} continent cells for hyperopt")
        multi_cell_data = preload_multi_cell_data(
            cell_patches, config, data_dir, device,
            use_legacy_global_data=USE_LEGACY_GLOBAL_DATA, split_seed=42
        )

    mlp_results = optimize_hyperparameters(
        args=config,
        data_dir=data_dir,
        architecture="mlp",
        max_evals=100,
        output_dir=output_dir,
        device=device,
        random_seed=42,
        resume=False,  # Set to True to continue from a previous run
        use_snapshot=USE_SNAPSHOT_ENSEMBLE,
        use_block_ltho=use_block_ltho,
        use_per_lt=use_per_lt,
        snapshot_objective_runs=SNAPSHOT_OBJECTIVE_RUNS,
        snapshot_epochs=SNAPSHOT_EPOCHS,
        multi_cell_data=multi_cell_data
    )
    print(f"MLP optimization finished with best loss: {mlp_results['best_loss']:.6f}")

    # # Optionally optimize UNet architecture (early-stopping only)
    # unet_results = optimize_hyperparameters(
    #     args=config,
    #     data_dir=data_dir,
    #     architecture="unet",
    #     max_evals=100,
    #     output_dir=f"hyperopt_results_{snapshot_prefix}{TUNING_MODE}_unet",
    #     device=device,
    #     random_seed=42,
    #     resume=False
    # )
    # print(f"UNet optimization finished with best loss: {unet_results['best_loss']:.6f}")