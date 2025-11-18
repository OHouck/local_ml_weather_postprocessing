#!/usr/bin/env python3
"""
filename: hyperparam_tuning.py
Author: Ozma Houck
Date created: 07/17/2025
Date modified: 11/07/2025

Hyperparameter optimization module for weather forecast fine-tuning using hyperopt.
Uses Bayesian optimization with early stopping for both MLP and UNet architectures.

Updates (11/07/2025):
- Replaced cosine annealing with early stopping
- Simplified to optimize one region/variable at a time
- Removed config_list complexity
- Added early stopping hyperparameters: patience, min_delta
"""

import os
import sys
import numpy as np
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
from finetuning.finetune import SimpleMLP, UNet, load_forecasts, create_dataloader, get_region_grid
from helper_funcs import setup_directories


def create_mlp_search_space():
    """
    Define the hyperparameter search space for MLP architecture with early stopping.

    Search space is centered around optimal values from architecture experiments:
    - mlp_moderate: hidden_dim=1024, num_layers=6, dropout=0.25

    Returns:
        dict: Search space definition for hyperopt
    """
    return {
        # Model architecture - centered on mlp_moderate optimal values
        'hidden_dim': hp.choice('hidden_dim', [512, 768, 1024, 1280, 1536, 2048]),
        'num_layers': hp.choice('num_layers', [4, 5, 6, 7, 8]),

        # Training parameters
        'learning_rate': hp.loguniform('learning_rate', np.log(1e-6), np.log(1e-2)),
        'batch_size': hp.choice('batch_size', [64, 128, 256]),
        'weight_decay': hp.loguniform('weight_decay', np.log(1e-6), np.log(1e-2)),

        # Early stopping parameters
        'patience': hp.choice('patience', [40, 50, 60, 75]),
        'min_delta': hp.loguniform('min_delta', np.log(1e-6), np.log(1e-4)),

        # Embedding and regularization - centered on optimal dropout of 0.25
        'lead_time_embedding_dim': hp.choice('lead_time_embedding_dim', [4, 8, 16]),
        'dropout_rate': hp.uniform('dropout_rate', 0.15, 0.35),
    }


def create_unet_search_space():
    """
    Define the hyperparameter search space for UNet architecture with early stopping.

    Search space is centered around optimal values from architecture experiments:
    - unet_medium: hidden_dim=64, dropout=0.1

    Returns:
        dict: Search space definition for hyperopt
    """
    return {
        # Model architecture - centered on unet_medium optimal values
        'hidden_dim': hp.choice('hidden_dim', [64, 128]),

        # Training parameters
        'learning_rate': hp.loguniform('learning_rate', np.log(1e-6), np.log(1e-2)),
        'batch_size': hp.choice('batch_size', [32, 64, 128, 256]),
        'weight_decay': hp.loguniform('weight_decay', np.log(1e-6), np.log(1e-2)),

        # Early stopping parameters
        'patience': hp.choice('patience', [40, 50, 60, 75]),
        'min_delta': hp.loguniform('min_delta', np.log(1e-6), np.log(1e-4)),

        # Embedding and regularization - centered on optimal dropout of 0.1
        'lead_time_embedding_dim': hp.choice('lead_time_embedding_dim', [4, 8, 16]),
        'dropout_rate': hp.uniform('dropout_rate', 0.05, 0.20),
    }


def train_with_early_stopping(model, train_loader, valid_loader, hyperparams, device):
    """
    Train model with early stopping, mixed precision, and GPU optimizations.

    Args:
        model: The neural network model
        train_loader: Training data loader
        valid_loader: Validation data loader
        hyperparams: Dictionary of hyperparameters
        device: torch device

    Returns:
        tuple: (best_val_loss, num_epochs_trained)
    """
    import time

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=hyperparams['learning_rate'],
        weight_decay=hyperparams['weight_decay']
    )

    # Add ReduceLROnPlateau scheduler for better convergence
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        min_lr=1e-7
    )

    patience = hyperparams['patience']
    min_delta = hyperparams['min_delta']
    max_epochs = 1000  # Maximum epochs before stopping

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
                with torch.cuda.amp.autocast():
                    # Model takes training inputs and predicts error
                    pred_error = model(fc_input_batch, lead_time_batch, doy_batch)
                    # Apply error to output forecast
                    preds = fc_output_batch + pred_error
                    loss = criterion(preds, y_batch)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training for CPU/MPS
                pred_error = model(fc_input_batch, lead_time_batch, doy_batch)
                preds = fc_output_batch + pred_error
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
                    with torch.cuda.amp.autocast():
                        pred_error = model(fc_input_batch, lead_time_batch, doy_batch)
                        preds = fc_output_batch + pred_error
                        loss = criterion(preds, y_batch)
                else:
                    pred_error = model(fc_input_batch, lead_time_batch, doy_batch)
                    preds = fc_output_batch + pred_error
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


def evaluate_hyperparameters(hyperparams: Dict[str, Any],
                            args: SimpleNamespace,
                            data_dir: str,
                            architecture: str,
                            device: torch.device) -> Dict[str, Any]:
    """
    Evaluate a set of hyperparameters for a single region/variable configuration.

    Args:
        hyperparams: Dictionary of hyperparameters to evaluate
        args: Configuration with region, variables, lead times, etc.
        data_dir: Path to data directory
        architecture: 'mlp' or 'unet'
        device: torch device

    Returns:
        dict: {'loss': validation_loss, 'status': STATUS_OK, 'epochs_trained': num_epochs}
    """
    print(f"\nEvaluating hyperparameters:")
    print(f"  Architecture: {architecture}")
    print(f"  Learning rate: {hyperparams['learning_rate']:.6f}")
    print(f"  Hidden dim: {hyperparams['hidden_dim']}")
    print(f"  Batch size: {hyperparams['batch_size']}")
    print(f"  Patience: {hyperparams['patience']}")

    # Get region grid
    lat_vals, lon_vals = get_region_grid(args)

    # Load training data
    (fc, fc_output, obs, lead_time_indices, day_of_year_features, train_times,
     lat_u, lon_u, n_lat, n_lon, n_training_vars, n_output_vars, _) = \
        load_forecasts(data_dir, args, lat_vals, lon_vals, train=True)

    # Normalize data
    stats_train = {'mean': fc.mean(0), 'std': fc.std(0) + 1e-8}
    stats_out = {'mean': fc_output.mean(0), 'std': fc_output.std(0) + 1e-8}
    fc_norm = (fc - stats_train['mean']) / stats_train['std']
    fc_output_norm = (fc_output - stats_out['mean']) / stats_out['std']
    obs_norm = (obs - stats_out['mean']) / stats_out['std']

    # Split train/validation (80/20)
    n_samples = len(fc)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    split_idx = int(0.8 * n_samples)
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

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
            lead_time_embedding_dim=hyperparams['lead_time_embedding_dim'],
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
            lead_time_embedding_dim=hyperparams['lead_time_embedding_dim'],
            dropout_rate=hyperparams['dropout_rate']
        ).to(device)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    # Train model
    val_loss, epochs_trained = train_with_early_stopping(
        model, train_loader, val_loader, hyperparams, device
    )

    print(f"  Validation loss: {val_loss:.6f} (trained {epochs_trained} epochs)")

    return {
        'loss': val_loss,
        'status': STATUS_OK,
        'epochs_trained': epochs_trained,
        'hyperparams': hyperparams
    }


def optimize_hyperparameters(args: SimpleNamespace,
                            data_dir: str,
                            architecture: str,
                            max_evals: int = 100,
                            output_dir: str = None,
                            device: torch.device = None,
                            random_seed: int = 42,
                            resume: bool = False) -> Dict[str, Any]:
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

    Returns:
        dict: Best hyperparameters and optimization results
    """
    # Validate inputs
    if architecture not in ['mlp', 'unet']:
        raise ValueError(f"Architecture must be 'mlp' or 'unet', got: {architecture}")

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
    print(f"Optimizing {architecture.upper()} for region '{args.region}', "
          f"variable(s) {args.output_vars}, lead times {args.lead_time_hours}h")

    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"hyperopt_results_{architecture}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Create search space
    if architecture == 'mlp':
        search_space = create_mlp_search_space()
    else:  # unet
        search_space = create_unet_search_space()

    # Define objective function
    def objective(hyperparams):
        result = evaluate_hyperparameters(
            hyperparams, args, data_dir, architecture, device
        )

        # Save intermediate result
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = os.path.join(output_dir, f'eval_{timestamp}.json')
        with open(result_file, 'w') as f:
            json.dump({
                'hyperparams': hyperparams,
                'loss': result['loss'],
                'epochs_trained': result['epochs_trained'],
                'architecture': architecture
            }, f, indent=2)

        return result

    # Load or initialize trials
    trials_file = os.path.join(output_dir, f'trials_{architecture}.pkl')

    if resume and os.path.exists(trials_file):
        print(f"Resuming from {trials_file}")
        with open(trials_file, 'rb') as f:
            trials = pickle.load(f)
        n_previous = len(trials.trials)
        print(f"Loaded {n_previous} previous trials")

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
    results = {
        'architecture': architecture,
        'best_hyperparams': best_hyperparams,
        'best_loss': best_trial['result']['loss'],
        'best_epochs_trained': best_trial['result']['epochs_trained'],
        'n_evaluations': len(trials.trials),
        'region': args.region,
        'variables': args.output_vars,
        'lead_times': args.lead_time_hours
    }

    # Save results
    results_file = os.path.join(output_dir, f'optimization_results_{architecture}.json')
    with open(results_file, 'w') as f:
        json.dump({
            'architecture': architecture,
            'best_hyperparams': best_hyperparams,
            'best_loss': results['best_loss'],
            'best_epochs_trained': results['best_epochs_trained'],
            'n_evaluations': results['n_evaluations'],
            'region': args.region,
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
    # Setup directories
    dirs = setup_directories()
    data_dir = dirs['raw']

    # Define configuration for optimization
    # Using full variable set based on architecture experiment results
    config = SimpleNamespace(
        model_name="pangu",
        training_vars=[
            "2m_temperature"
        ],
        output_vars=["2m_temperature"],
        train_start="2018-01-01",
        train_end="2021-12-31",
        test_start="2022-01-01",
        test_end="2022-12-31",
        region='india',
        subregion='6x6',
        ground_truth_source='',  # Will default to era5 for pangu
        lead_time_hours=[24, 120, 216],
        growing_season_only=False
    )

    # Setup device
    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )
    print(f"Using device: {device}")

    # Enable cudnn benchmarking for faster training on GPU
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        print("Enabled cudnn benchmarking for faster GPU training")
        print("Using mixed precision training (AMP) for CUDA operations")

    # Optimize MLP architecture
    # mlp_results = optimize_hyperparameters(
    #     args=config,
    #     data_dir=data_dir,
    #     architecture="mlp",
    #     max_evals=100,
    #     output_dir="hyperopt_results_mlp",
    #     device=device,
    #     random_seed=42,
    #     resume=True  # Set to True to continue from previous runs
    # )

    print(f"MLP optimization finished with best loss: {mlp_results['best_loss']:.6f}")

    # Optionally optimize UNet architecture
    unet_results = optimize_hyperparameters(
        args=config,
        data_dir=data_dir,
        architecture="unet",
        max_evals=100,
        output_dir="hyperopt_results_unet",
        device=device,
        random_seed=42,
        resume=False
    )
