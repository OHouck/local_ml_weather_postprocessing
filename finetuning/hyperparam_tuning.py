#!/usr/bin/env python3
"""
filename: 0.75_hyperparam_tuning.py
Author: Ozma Houck
Date created: 07/17/2025

Hyperparameter optimization module for weather forecast fine-tuning using hyperopt.
This module provides Bayesian optimization for MLP hyperparameters across multiple
regions and lead time groups.
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
from typing import Dict, List, Tuple, Any
import pickle

# Import necessary functions from the main module
# These would need to be imported from your main file
# from finetuning.1_finetune import (
#     SimpleMLP, UNet, load_forecasts, create_dataloader, 
#     train_model, apply_correction, get_region_grid, get_patch_shape
# )


def create_search_space():
    """
    Define the hyperparameter search space for hyperopt.
    
    Returns:
        dict: Search space definition for hyperopt
    """
    search_space = {
        # Model architecture parameters
        'mlp_hidden_dim': hp.choice('mlp_hidden_dim', [64, 128, 256, 512, 1024]),
        'mlp_layers': hp.choice('mlp_layers', [2, 3, 4, 5, 6, 7]),
        
        # Training parameters
        'learning_rate': hp.loguniform('learning_rate', low=np.log(1e-6), high=np.log(1e-2)),
        'batch_size': hp.choice('batch_size', [16, 32, 64, 128]),
        
        # Embedding dimensions
        'lead_time_embedding_dim': hp.choice('lead_time_embedding_dim', [4, 8, 16, 32]),
        'month_embedding_dim': hp.choice('month_embedding_dim', [4, 8, 16, 32]),
        
        # Regularization
        'dropout_rate': hp.uniform('dropout_rate', 0.0, 0.5),
        'weight_decay': hp.loguniform('weight_decay', low=np.log(1e-6), high=np.log(1e-2)),
        
        # Early stopping parameters
        'patience': hp.choice('patience', [20, 30, 50, 70]),
        'min_delta': hp.loguniform('min_delta', low=np.log(1e-6), high=np.log(1e-3))
    }
    
    return search_space


class ModifiedSimpleMLP(nn.Module):
    """
    Modified SimpleMLP with dropout for hyperparameter optimization.
    """
    def __init__(self, input_dim, hidden_dim=128, output_dim=1, num_hidden_layers=3, 
                 n_lead_times=1, lead_time_embedding_dim=16,
                 month_embedding_dim=16, dropout_rate=0.0):
        super(ModifiedSimpleMLP, self).__init__()
        
        # Lead time embedding
        self.n_lead_times = n_lead_times
        self.lead_time_embedding = None
        
        # Month embedding (12 months)
        self.month_embedding = nn.Embedding(12, month_embedding_dim)
        
        # Calculate actual input dimension
        actual_input_dim = input_dim + month_embedding_dim
        
        if n_lead_times > 1:
            self.lead_time_embedding = nn.Embedding(n_lead_times, lead_time_embedding_dim)
            actual_input_dim += lead_time_embedding_dim
            
        # Build network with dropout
        layers = [nn.Linear(actual_input_dim, hidden_dim), nn.ReLU()]
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
            
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
                
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x, lead_time_idx=None, month_idx=None):
        # Always add month embedding
        if month_idx is not None:
            month_emb = self.month_embedding(month_idx)
            x = torch.cat([x, month_emb], dim=-1)
            
        # Add lead time embedding if available
        if self.lead_time_embedding is not None and lead_time_idx is not None:
            lead_time_emb = self.lead_time_embedding(lead_time_idx)
            x = torch.cat([x, lead_time_emb], dim=-1)
            
        return self.net(x)


def train_model_with_hyperparams(model, train_loader, valid_loader, hyperparams, device):
    """
    Train model with specific hyperparameters.
    
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
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    train_start_time = time.time()
    max_epochs = 300  # Maximum epochs to prevent infinite training
    
    for epoch in range(1, max_epochs + 1):
        # Training step
        model.train()
        train_loss = 0.0
        for x_batch, y_batch, lead_time_batch, month_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            lead_time_batch, month_batch = lead_time_batch.to(device), month_batch.to(device)
            
            optimizer.zero_grad()
            preds = model(x_batch, lead_time_batch, month_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)
        train_loss /= len(train_loader.dataset)
        
        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch, lead_time_batch, month_batch in valid_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                lead_time_batch, month_batch = lead_time_batch.to(device), month_batch.to(device)
                
                preds = model(x_batch, lead_time_batch, month_batch)
                loss = criterion(preds, y_batch)
                val_loss += loss.item() * x_batch.size(0)
        val_loss /= len(valid_loader.dataset)
        
        # Early stopping check
        if val_loss + hyperparams['min_delta'] < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= hyperparams['patience']:
                break
    
    train_end_time = time.time()
    training_time_minutes = (train_end_time - train_start_time) / 60.0
    
    # Load best weights
    model.load_state_dict(best_model_wts)
    return model, best_val_loss, training_time_minutes


def evaluate_hyperparameters(hyperparams: Dict[str, Any], 
                           regions_config: List[Dict[str, Any]], 
                           device: torch.device,
                           verbose: bool = False) -> Dict[str, Any]:
    """
    Evaluate a set of hyperparameters across multiple regions and lead times.
    
    Args:
        hyperparams: Dictionary of hyperparameters to evaluate
        regions_config: List of region configurations, each containing:
            - 'args': Namespace with region-specific arguments
            - 'data_dir': Path to data directory
            - 'weight': Weight for this region in the overall loss (optional)
        device: torch device
        verbose: Whether to print progress
        
    Returns:
        dict: {'loss': weighted_average_loss, 'status': STATUS_OK, 'region_losses': {...}}
    """
    # Import from main module 
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from finetune import load_forecasts, create_dataloader, get_region_grid
    
    total_weighted_loss = 0.0
    total_weight = 0.0
    region_losses = {}
    
    for config in regions_config:
        args = config['args']
        data_dir = config['data_dir']
        weight = config.get('weight', 1.0)
        
        try:
            # Get region grid
            if hasattr(args, 'lat_vals') and hasattr(args, 'lon_vals'):
                lat_vals, lon_vals = args.lat_vals, args.lon_vals
            else:
                lat_vals, lon_vals = get_region_grid(args)
            
            # Load training data
            (fc, fc_output, obs, lead_time_indices, month_indices, train_times, 
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
            
            # Create data loaders with hyperparameter batch size
            train_loader = create_dataloader(
                fc_norm[t_idx], obs_norm[t_idx], 
                lead_time_indices[t_idx], month_indices[t_idx], 
                batch_size=hyperparams['batch_size']
            )
            val_loader = create_dataloader(
                fc_norm[v_idx], obs_norm[v_idx], 
                lead_time_indices[v_idx], month_indices[v_idx], 
                batch_size=hyperparams['batch_size']
            )
            
            # Initialize model with hyperparameters
            input_dim = n_training_vars * n_lat * n_lon
            output_dim = n_output_vars * n_lat * n_lon
            n_lead_times = len(args.lead_time_hours)
            
            model = ModifiedSimpleMLP(
                input_dim=input_dim,
                hidden_dim=hyperparams['mlp_hidden_dim'],
                output_dim=output_dim,
                num_hidden_layers=hyperparams['mlp_layers'],
                n_lead_times=n_lead_times,
                lead_time_embedding_dim=hyperparams['lead_time_embedding_dim'],
                month_embedding_dim=hyperparams['month_embedding_dim'],
                dropout_rate=hyperparams['dropout_rate']
            ).to(device)
            
            # Train model
            model, val_loss, training_time = train_model_with_hyperparams(
                model, train_loader, val_loader, hyperparams, device
            )
            
            # Store results
            region_key = f"{args.region}_{args.subregion}_{'_'.join(map(str, args.lead_time_hours))}h"
            region_losses[region_key] = val_loss
            
            # Add to weighted average
            total_weighted_loss += val_loss * weight
            total_weight += weight
            
            if verbose:
                print(f"Region {region_key}: val_loss = {val_loss:.6f}")
                
        except Exception as e:
            print(f"Error evaluating region {args.region}: {str(e)}")
            # Return a high loss for failed evaluations
            return {'loss': 1e10, 'status': STATUS_OK, 'error': str(e)}
    
    # Calculate weighted average loss
    avg_loss = total_weighted_loss / total_weight if total_weight > 0 else 1e10
    
    return {
        'loss': avg_loss,
        'status': STATUS_OK,
        'region_losses': region_losses,
        'hyperparams': hyperparams
    }


def optimize_hyperparameters(regions_config: List[Dict[str, Any]], 
                           max_evals: int = 100,
                           output_dir: str = None,
                           device: torch.device = None,
                           random_seed: int = 42) -> Dict[str, Any]:
    """
    Main function to optimize hyperparameters across multiple regions.
    
    Args:
        regions_config: List of region configurations
        max_evals: Maximum number of evaluations
        output_dir: Directory to save results
        device: torch device (if None, will auto-detect)
        random_seed: Random seed for reproducibility
        
    Returns:
        dict: Best hyperparameters and optimization results
    """
    # Set random seeds
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else
                            'mps' if torch.backends.mps.is_available() else
                            'cpu')
    
    # Create output directory
    if output_dir is None:
        output_dir = f"hyperopt_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define objective function for hyperopt
    def objective(hyperparams):
        """Objective function to minimize."""
        print(f"\nEvaluating hyperparameters: {hyperparams}")
        result = evaluate_hyperparameters(hyperparams, regions_config, device, verbose=True)
        
        # Save intermediate results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = os.path.join(output_dir, f'eval_{timestamp}.json')
        with open(result_file, 'w') as f:
            json.dump({
                'hyperparams': hyperparams,
                'loss': result['loss'],
                'region_losses': result.get('region_losses', {})
            }, f, indent=2)
        
        return result
    
    # Create search space
    search_space = create_search_space()
    
    # Run optimization
    trials = Trials()
    
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
        'best_hyperparams': best_hyperparams,
        'best_loss': best_trial['result']['loss'],
        'best_region_losses': best_trial['result'].get('region_losses', {}),
        'n_evaluations': len(trials.trials),
        'all_trials': trials.trials
    }
    
    # Save final results
    results_file = os.path.join(output_dir, 'optimization_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'best_hyperparams': best_hyperparams,
            'best_loss': results['best_loss'],
            'best_region_losses': results['best_region_losses'],
            'n_evaluations': results['n_evaluations']
        }, f, indent=2)
    
    # Save trials object for later analysis
    trials_file = os.path.join(output_dir, 'trials.pkl')
    with open(trials_file, 'wb') as f:
        pickle.dump(trials, f)
    
    print(f"\nOptimization complete!")
    print(f"Best hyperparameters: {best_hyperparams}")
    print(f"Best loss: {results['best_loss']:.6f}")
    print(f"Results saved to: {output_dir}")
    
    return results


def create_region_config(region_specs: List[Dict[str, Any]], 
                        base_args: Any,
                        data_dir: str) -> List[Dict[str, Any]]:
    """
    Helper function to create region configurations for optimization.
    
    Args:
        region_specs: List of region specifications, each containing:
            - 'region': Region name
            - 'subregion': Subregion specification (e.g., '2x2')
            - 'lead_time_hours': List of lead times
            - 'weight': Optional weight for this region
        base_args: Base arguments namespace to copy
        data_dir: Data directory path
        
    Returns:
        List of region configurations
    """
    import copy
    from types import SimpleNamespace
    
    configs = []
    for spec in region_specs:
        # Create a copy of base args
        args = copy.deepcopy(base_args) if hasattr(base_args, '__dict__') else SimpleNamespace()
        
        # Update with region-specific settings
        args.region = spec['region']
        args.subregion = spec['subregion']
        args.lead_time_hours = spec['lead_time_hours']
        
        # Copy other relevant attributes from base_args if they exist
        for attr in ['training_vars', 'output_vars', 'train_start', 'train_end', 
                     'test_start', 'test_end', 'model_name']:
            if hasattr(base_args, attr):
                setattr(args, attr, getattr(base_args, attr))
        
        config = {
            'args': args,
            'data_dir': data_dir,
            'weight': spec.get('weight', 1.0)
        }
        configs.append(config)
    
    return configs


# Example usage
if __name__ == "__main__":
    # Example of how to use the hyperparameter optimization
    from types import SimpleNamespace
    
    # Base configuration
    base_args = SimpleNamespace(
        model_name="pangu",
        training_vars=["2m_temperature"],
        output_vars=["2m_temperature"],
        train_start="2018-01-01",
        train_end="2021-12-31",
        test_start="2022-01-01",
        test_end="2022-12-31"
    )
    
    # Define regions to optimize across
    region_specs = [
        {
            'region': 'india',
            'subregion': '4x4',
            'lead_time_hours': [120, 144, 168],
            'weight': 1.0
        }
    ]
    
    # Create region configurations
    # Update this path to your actual data directory
    data_dir = "/path/to/your/data"  # Change this to your actual path
    data_dir="/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/data/processed/cleaned_weatherbench_downloads" 
    regions_config = create_region_config(region_specs, base_args, data_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else
                          'cpu')
    
    # Run optimization with fewer evaluations for testing
    results = optimize_hyperparameters(
        regions_config=regions_config,
        max_evals=100,  # Reduced for testing
        output_dir="hyperopt_results",
        device = device,
        random_seed=42
    )
    
    print(f"\nBest hyperparameters found: {results['best_hyperparams']}")
    print(f"Best loss: {results['best_loss']:.6f}")