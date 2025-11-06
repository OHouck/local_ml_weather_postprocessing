#!/usr/bin/env python3
"""
filename: hyperparam_tuning.py
Author: Ozma Houck
Date created: 07/17/2025
Date modified: 10/28/2025

Enhanced hyperparameter optimization module for weather forecast fine-tuning using hyperopt.
This module provides Bayesian optimization for both MLP and UNet hyperparameters across
multiple regions and lead time groups.

Updates (10/28/2025):
- Replaced early stopping with cosine annealing with warm restarts
- Added hyperparameters for cosine annealing: T_0, T_mult, eta_min, num_epochs
- Replaced month embeddings with day-of-year sin/cos features
- Removed modified model classes, now imports from finetune.py
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import hyperopt
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval
from hyperopt.pyll import scope
import json
from datetime import datetime
import copy
from typing import Dict, List, Tuple, Any, Union
import pickle

# Import model classes and utilities from finetune.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from finetuning.finetune import SimpleMLP, UNet, load_forecasts, create_dataloader, get_region_grid
from helper_funcs import setup_directories


def create_mlp_search_space():
    """
    Define the hyperparameter search space for MLP architecture.

    Returns:
        dict: Search space definition for MLP hyperopt
    """
    search_space = {
        # Model architecture parameters
        'mlp_hidden_dim': hp.choice('mlp_hidden_dim', [64, 128, 256, 512, 1024]),
        'mlp_layers': hp.choice('mlp_layers', [2, 3, 4, 5, 6, 7]),

        # Training parameters
        'learning_rate': hp.loguniform('learning_rate', low=np.log(1e-6), high=np.log(1e-2)),
        'batch_size': hp.choice('batch_size', [16, 32, 64, 128]),
        'num_epochs': hp.choice('num_epochs', [50, 100, 250, 500, 750]),

        # Cosine annealing parameters
        'T_0': hp.choice('T_0', [5, 10, 15, 20, 30]), # number of epochs for the first restart
        'T_mult': hp.choice('T_mult', [1, 2, 3]), # factor to increase T_0 after each restart
        'eta_min': hp.loguniform('eta_min', low=np.log(1e-8), high=np.log(1e-5)), # minimum learning rate

        # Embedding dimensions
        'lead_time_embedding_dim': hp.choice('lead_time_embedding_dim', [4, 8, 16, 32]),

        # Regularization
        'dropout_rate': hp.uniform('dropout_rate', 0.0, 0.5),
        'weight_decay': hp.loguniform('weight_decay', low=np.log(1e-6), high=np.log(1e-2))
    }

    return search_space


def create_unet_search_space():
    """
    Define the hyperparameter search space for UNet architecture.

    Returns:
        dict: Search space definition for UNet hyperopt
    """
    search_space = {
        # UNet architecture parameters
        'unet_hidden_dim': hp.choice('unet_hidden_dim', [16, 32, 64, 128]),

        # Training parameters
        'learning_rate': hp.loguniform('learning_rate', low=np.log(1e-6), high=np.log(1e-2)),
        'batch_size': hp.choice('batch_size', [4, 8, 16, 32]),  # Smaller batches for UNet due to memory
        'num_epochs': hp.choice('num_epochs', [50, 75, 100, 300, 500]),

        # Cosine annealing parameters
        'T_0': hp.choice('T_0', [5, 10, 15, 20, 30]),
        'T_mult': hp.choice('T_mult', [1, 2, 3]),
        'eta_min': hp.loguniform('eta_min', low=np.log(1e-8), high=np.log(1e-5)),

        # Embedding dimensions
        'lead_time_embedding_dim': hp.choice('lead_time_embedding_dim', [4, 8, 16, 32]),

        # Regularization
        'dropout_rate': hp.uniform('dropout_rate', 0.0, 0.3),  # Generally lower for UNet
        'weight_decay': hp.loguniform('weight_decay', low=np.log(1e-6), high=np.log(1e-2))
    }

    return search_space


def train_model_with_hyperparams(model, train_loader, valid_loader, hyperparams, device):
    """
    Train model with specific hyperparameters using cosine annealing.

    Args:
        model: The neural network model
        train_loader: Training data loader
        valid_loader: Validation data loader
        hyperparams: Dictionary of hyperparameters
        device: torch device

    Returns:
        tuple: (trained_model, best_val_loss, training_time_minutes)
    """
    import time
    import torch.optim as optim

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=hyperparams['learning_rate'],
        weight_decay=hyperparams['weight_decay']
    )

    # Add cosine annealing with warm restarts scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=hyperparams['T_0'],
        T_mult=hyperparams['T_mult'],
        eta_min=hyperparams['eta_min']
    )

    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())

    train_start_time = time.time()
    num_epochs = hyperparams['num_epochs']

    for epoch in range(1, num_epochs + 1):
        # Training step
        model.train()
        train_loss = 0.0
        for x_batch, y_batch, lead_time_batch, doy_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            lead_time_batch, doy_batch = lead_time_batch.to(device), doy_batch.to(device)

            optimizer.zero_grad()
            pred_error = model(x_batch, lead_time_batch, doy_batch)
            preds = x_batch + pred_error
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch, lead_time_batch, doy_batch in valid_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                lead_time_batch, doy_batch = lead_time_batch.to(device), doy_batch.to(device)

                pred_error = model(x_batch, lead_time_batch, doy_batch)
                preds = x_batch + pred_error
                loss = criterion(preds, y_batch)
                val_loss += loss.item() * x_batch.size(0)
        val_loss /= len(valid_loader.dataset)

        # Update learning rate
        scheduler.step()

        # Track best model (no early stopping, but still save best)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

    train_end_time = time.time()
    training_time_minutes = (train_end_time - train_start_time) / 60.0

    # Load best weights
    model.load_state_dict(best_model_wts)
    return model, best_val_loss, training_time_minutes


def evaluate_hyperparameters(hyperparams: Dict[str, Any],
                           config_list: List[Dict[str, Any]],
                           device: torch.device,
                           architecture: str = None,
                           verbose: bool = False) -> Dict[str, Any]:
    """
    Evaluate a set of hyperparameters across multiple regions and lead times.

    Args:
        hyperparams: Dictionary of hyperparameters to evaluate
        config_list: List of region configurations
        device: torch device
        architecture: Force specific architecture ('mlp' or 'unet'), overrides hyperparams
        verbose: Whether to print progress

    Returns:
        dict: {'loss': weighted_average_loss, 'status': STATUS_OK, 'region_losses': {...}}
    """
    # Determine architecture
    if architecture is not None:
        arch = architecture
    else:
        arch = hyperparams.get('architecture', 'mlp')

    total_weighted_loss = 0.0
    total_weight = 0.0
    region_losses = {}

    for config in config_list:
        args = config['args']
        data_dir = config['data_dir']
        weight = config.get('weight', 1.0)

        # Get region grid
        if hasattr(args, 'lat_vals') and hasattr(args, 'lon_vals'):
            lat_vals, lon_vals = args.lat_vals, args.lon_vals
        else:
            lat_vals, lon_vals = get_region_grid(args)

        # Load training data (now returns day_of_year_features instead of month_indices)
        (fc, fc_output, obs, lead_time_indices, day_of_year_features, train_times,
            lat_u, lon_u, n_lat, n_lon, n_training_vars, n_output_vars, _) = \
            load_forecasts(data_dir, args, lat_vals, lon_vals, train=True)

        # Normalize data
        stats_train = {'mean': fc.mean(0), 'std': fc.std(0) + 1e-8}
        stats_out = {'mean': fc_output.mean(0), 'std': fc_output.std(0) + 1e-8}
        fc_norm = (fc - stats_train['mean']) / stats_train['std']
        obs_norm = (obs - stats_out['mean']) / stats_out['std']

        # Split train/validation
        n_train = len(fc)
        idx = np.arange(n_train)
        np.random.shuffle(idx)
        split = int(0.8 * n_train)
        t_idx, v_idx = idx[:split], idx[split:]

        # Adjust batch size based on architecture
        batch_size = hyperparams['batch_size']
        if arch == 'unet' and batch_size > 32:
            batch_size = min(batch_size, 16)  # Reduce for UNet due to memory constraints

        # Create data loaders
        train_loader = create_dataloader(
            fc_norm[t_idx], obs_norm[t_idx],
            lead_time_indices[t_idx], day_of_year_features[t_idx],
            batch_size=batch_size
        )
        val_loader = create_dataloader(
            fc_norm[v_idx], obs_norm[v_idx],
            lead_time_indices[v_idx], day_of_year_features[v_idx],
            batch_size=batch_size
        )

        # Initialize model
        input_dim = n_training_vars * n_lat * n_lon
        output_dim = n_output_vars * n_lat * n_lon
        n_lead_times = len(args.lead_time_hours)

        if arch == 'mlp':
            model = SimpleMLP(
                input_dim=input_dim,
                hidden_dim=hyperparams['mlp_hidden_dim'],
                output_dim=output_dim,
                num_hidden_layers=hyperparams['mlp_layers'],
                n_lead_times=n_lead_times,
                lead_time_embedding_dim=hyperparams['lead_time_embedding_dim'],
                dropout_rate=hyperparams['dropout_rate']
            ).to(device)

        elif arch == 'unet':
            model = UNet(
                input_dim=input_dim,
                hidden_dim=hyperparams['unet_hidden_dim'],
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
            raise ValueError(f"Unknown architecture: {arch}")

        # Train model
        model, val_loss, training_time = train_model_with_hyperparams(
            model, train_loader, val_loader, hyperparams, device
        )

        # Store results
        region_key = f"{args.region}_{args.subregion}_{'_'.join(map(str, args.lead_time_hours))}h_{arch}"
        region_losses[region_key] = val_loss

        # Add to weighted average
        total_weighted_loss += val_loss * weight
        total_weight += weight

        if verbose:
            print(f"Region {region_key}: val_loss = {val_loss:.6f}")
                
    # Calculate weighted average loss
    avg_loss = total_weighted_loss / total_weight if total_weight > 0 else 1e10
    
    return {
        'loss': avg_loss,
        'status': STATUS_OK,
        'region_losses': region_losses,
        'hyperparams': hyperparams,
        'architecture': arch
    }


def optimize_hyperparameters(config_list: List[Dict[str, Any]],
                           architecture: str,
                           max_evals: int = 100,
                           output_dir: str = None,
                           device: torch.device = None,
                           random_seed: int = 42,
                           resume: bool = False) -> Dict[str, Any]:
    """
    Optimize hyperparameters for a specific architecture (MLP or UNet).

    Args:
        config_list: List of configurations
        architecture: 'mlp' or 'unet'
        max_evals: Maximum number of evaluations (total, including resumed trials)
        output_dir: Directory to save results
        device: torch device (if None, will auto-detect)
        random_seed: Random seed for reproducibility
        resume: If True, load previous trials from output_dir and continue optimization

    Returns:
        dict: Best hyperparameters and optimization results
    """
    # Validate architecture
    if architecture not in ['mlp', 'unet']:
        raise ValueError(f"Architecture must be 'mlp' or 'unet', got: {architecture}")

    # Set random seeds
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else
                            'mps' if torch.backends.mps.is_available() else
                            'cpu')
    print(f"Using device: {device}")

    # Create output directory
    if output_dir is None:
        output_dir = f"hyperopt_results_{architecture}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    # Define objective function for hyperopt
    def objective(hyperparams):
        """Objective function to minimize."""
        print(f"\nEvaluating {architecture.upper()} hyperparameters: {hyperparams}")
        result = evaluate_hyperparameters(hyperparams, config_list, device,
                                        architecture=architecture, verbose=True)

        # Save intermediate results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = os.path.join(output_dir, f'eval_{timestamp}.json')
        with open(result_file, 'w') as f:
            json.dump({
                'hyperparams': hyperparams,
                'loss': result['loss'],
                'region_losses': result.get('region_losses', {}),
                'architecture': architecture
            }, f, indent=2)

        return result

    # Create architecture-specific search space
    if architecture == 'mlp':
        search_space = create_mlp_search_space()
    elif architecture == 'unet':
        search_space = create_unet_search_space()
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    # Load or initialize trials
    trials_file = os.path.join(output_dir, f'trials_{architecture}.pkl')

    if resume and os.path.exists(trials_file):
        # Load existing trials
        print(f"Resuming optimization from {trials_file}")
        with open(trials_file, 'rb') as f:
            trials = pickle.load(f)
        n_previous = len(trials.trials)
        print(f"Loaded {n_previous} previous trials")
        print(f"Will run {max_evals - n_previous} additional evaluations")

        if n_previous >= max_evals:
            print(f"Already completed {n_previous} evaluations (>= max_evals={max_evals})")
            print("No additional evaluations needed. Loading best results...")

            # Just extract and return the best results
            best_trial_idx = np.argmin([t['result']['loss'] for t in trials.trials])
            best_trial = trials.trials[best_trial_idx]

            # Get best hyperparameters
            best_hyperparams = space_eval(search_space, best_trial['misc']['vals'])

            results = {
                'architecture': architecture,
                'best_hyperparams': best_hyperparams,
                'best_loss': best_trial['result']['loss'],
                'best_region_losses': best_trial['result'].get('region_losses', {}),
                'n_evaluations': len(trials.trials),
                'all_trials': trials.trials
            }

            return results
    else:
        # Start fresh
        trials = Trials()
        if resume:
            print(f"No previous trials found at {trials_file}, starting fresh")

    # minimize objective function using hyperopt
    best = hyperopt.fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
    )
    
    # Get the best hyperparameters with actual values (not indices)
    best_hyperparams = space_eval(search_space, best)
    
    # Find the trial with minimum loss
    best_trial_idx = np.argmin([t['result']['loss'] for t in trials.trials])
    best_trial = trials.trials[best_trial_idx]
    
    # Prepare results
    results = {
        'architecture': architecture,
        'best_hyperparams': best_hyperparams,
        'best_loss': best_trial['result']['loss'],
        'best_region_losses': best_trial['result'].get('region_losses', {}),
        'n_evaluations': len(trials.trials),
        'all_trials': trials.trials
    }
    
    # Save final results
    results_file = os.path.join(output_dir, f'optimization_results_{architecture}.json')
    with open(results_file, 'w') as f:
        json.dump({
            'architecture': architecture,
            'best_hyperparams': best_hyperparams,
            'best_loss': results['best_loss'],
            'best_region_losses': results['best_region_losses'],
            'n_evaluations': results['n_evaluations']
        }, f, indent=2)
    
    # Save trials object for later analysis
    trials_file = os.path.join(output_dir, f'trials_{architecture}.pkl')
    with open(trials_file, 'wb') as f:
        pickle.dump(trials, f)
    
    print(f"\n{architecture.upper()} optimization complete!")
    print(f"Best hyperparameters: {best_hyperparams}")
    print(f"Best loss: {results['best_loss']:.6f}")
    print(f"Results saved to: {output_dir}")
    
    return results

def analyze_optimization_results(results_dir: str):
    """
    Analyze and visualize optimization results.
    
    Args:
        results_dir: Directory containing optimization results
    """
    import matplotlib.pyplot as plt
    
    # Load trials
    trials_files = [f for f in os.listdir(results_dir) if f.startswith('trials_') and f.endswith('.pkl')]
    
    for trials_file in trials_files:
        arch = trials_file.replace('trials_', '').replace('.pkl', '')
        
        with open(os.path.join(results_dir, trials_file), 'rb') as f:
            trials = pickle.load(f)
        
        # Extract losses
        losses = [t['result']['loss'] for t in trials.trials]
        
        # Plot convergence
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.plot(losses)
        plt.title(f'{arch.upper()} - Loss per Trial')
        plt.xlabel('Trial')
        plt.ylabel('Validation Loss')
        
        # Plot best loss over time
        plt.subplot(1, 2, 2)
        best_losses = []
        current_best = float('inf')
        for loss in losses:
            if loss < current_best:
                current_best = loss
            best_losses.append(current_best)
        
        plt.plot(best_losses)
        plt.title(f'{arch.upper()} - Best Loss Over Time')
        plt.xlabel('Trial')
        plt.ylabel('Best Validation Loss')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'optimization_progress_{arch}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Analysis plot saved for {arch.upper()}")


# Example usage
if __name__ == "__main__":
    # Example of how to use the enhanced hyperparameter optimization
    from types import SimpleNamespace
    
    # Base configuration
    pangu_args = SimpleNamespace(
        model_name="pangu",
        training_vars=["2m_temperature"],
        output_vars=["2m_temperature"],
        train_start="2018-01-01",
        train_end="2021-12-31",
        test_start="2022-01-01",
        test_end="2022-12-31",
        region = 'india',
        subregion = '6x6',
        ground_truth_source = 'era5',
        lead_time_hours = [24, 120, 216],
        growing_season_only = False 
    )
    pangu_extreme_heat_args = SimpleNamespace(
        model_name="pangu",
        training_vars=["2m_temperature"],
        output_vars=["2m_temperature"],
        train_start="2018-01-01",
        train_end="2021-12-31",
        test_start="2022-01-01",
        test_end="2022-12-31",
        region = 'india',
        subregion = '6x6',
        ground_truth_source = 'era5',
        lead_time_hours = [24, 120, 216],
        growing_season_only = False,
        alternative_loss_fn = 'extreme_heat_loss'
    )
    aifs_args = SimpleNamespace(
        model_name="aifs",
        training_vars=["total_precipitation"],
        output_vars=["total_precipitation"],
        train_start="2021-01-01",
        train_end="2023-12-31",
        test_start="2024-01-01",
        test_end="2024-12-31",
        region = 'india',
        subregion = '6x6',
        ground_truth_source = 'era5',
        lead_time_hours = [24, 120, 216],
        growing_season_only = True
    )

    dirs = setup_directories()

    # where raw data is stored
    data_dir = dirs['raw']
    config_list = [
    {
        'args': pangu_args,
        'data_dir': data_dir,
        'weight': 1.0  # Optional, defaults to 1.0 if not specified
    },
]

    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else
                          'cpu')

    # Example: Run 35 evaluations
    # To resume and add more evaluations, set resume=True and increase max_evals
    # e.g., first run with max_evals=20, then run again with max_evals=50 and resume=True
    mlp_results = optimize_hyperparameters(
        config_list=config_list,
        architecture="mlp",
        max_evals=100, # total trainings = max_evals * len(config_list)
        output_dir="hyperopt_results_mlp_2m_temp",
        device=device,
        random_seed=42,
        resume=False # Set to True to continue from previous runs
    )
    print(f"MLP best loss: {mlp_results['best_loss']:.6f}")

    # unet_results = optimize_hyperparameters(
    #     config_list=config_list,
    #     architecture="unet",
    #     max_evals=1,
    #     output_dir="hyperopt_results_unet",
    #     device=device,
    #     random_seed=42
    # )
    # print(f"UNet best loss: {unet_results['best_loss']:.6f}")
    