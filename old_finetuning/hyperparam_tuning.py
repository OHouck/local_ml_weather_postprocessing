#!/usr/bin/env python3
"""
filename: 0.75_hyperparam_tuning_enhanced.py
Author: Ozma Houck
Date created: 07/17/2025 (Enhanced)

Enhanced hyperparameter optimization module for weather forecast fine-tuning using hyperopt.
This module provides Bayesian optimization for both MLP and UNet hyperparameters across 
multiple regions and lead time groups.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import hyperopt
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, space_eval
from hyperopt.pyll import scope
import json
from datetime import datetime
import copy
from typing import Dict, List, Tuple, Any, Union
import pickle

# Import necessary functions from the main module
# These would need to be imported from your main file
# from finetuning.1_finetune import (
#     SimpleMLP, UNet, load_forecasts, create_dataloader, 
#     train_model, apply_correction, get_region_grid, get_patch_shape
# )


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


def create_unet_search_space():
    """
    Define the hyperparameter search space for UNet architecture.
    
    Returns:
        dict: Search space definition for UNet hyperopt
    """
    search_space = {
        # UNet architecture parameters
        'unet_init_features': hp.choice('unet_init_features', [16, 32, 64, 128]),
        'unet_depth': hp.choice('unet_depth', [3, 4, 5]),
        'unet_kernel_size': hp.choice('unet_kernel_size', [3, 5, 7]),
        'unet_batch_norm': hp.choice('unet_batch_norm', [True, False]),
        'unet_upsampling_mode': hp.choice('unet_upsampling_mode', ['bilinear', 'nearest']),
        
        # Training parameters
        'learning_rate': hp.loguniform('learning_rate', low=np.log(1e-6), high=np.log(1e-2)),
        'batch_size': hp.choice('batch_size', [4, 8, 16, 32]),  # Smaller batches for UNet due to memory
        
        # Embedding dimensions
        'lead_time_embedding_dim': hp.choice('lead_time_embedding_dim', [4, 8, 16, 32]),
        'month_embedding_dim': hp.choice('month_embedding_dim', [4, 8, 16, 32]),
        
        # Regularization
        'dropout_rate': hp.uniform('dropout_rate', 0.0, 0.3),  # Generally lower for UNet
        'weight_decay': hp.loguniform('weight_decay', low=np.log(1e-6), high=np.log(1e-2)),
        
        # Early stopping parameters
        'patience': hp.choice('patience', [20, 30, 50, 70]),
        'min_delta': hp.loguniform('min_delta', low=np.log(1e-6), high=np.log(1e-3))
    }
    
    return search_space


def create_combined_search_space():
    """
    Define a combined search space that includes both MLP and UNet parameters.
    Architecture is chosen as part of the search space.
    
    Returns:
        dict: Combined search space definition for hyperopt
    """
    search_space = {
        # Architecture selection
        'architecture': hp.choice('architecture', ['mlp', 'unet']),
        
        # MLP-specific parameters (used when architecture == 'mlp')
        'mlp_hidden_dim': hp.choice('mlp_hidden_dim', [64, 128, 256, 512, 1024]),
        'mlp_layers': hp.choice('mlp_layers', [2, 3, 4, 5, 6, 7]),
        
        # UNet-specific parameters (used when architecture == 'unet')
        'unet_init_features': hp.choice('unet_init_features', [16, 32, 64, 128]),
        'unet_depth': hp.choice('unet_depth', [3, 4, 5]),
        'unet_kernel_size': hp.choice('unet_kernel_size', [3, 5, 7]),
        'unet_batch_norm': hp.choice('unet_batch_norm', [True, False]),
        'unet_upsampling_mode': hp.choice('unet_upsampling_mode', ['bilinear', 'nearest']),
        
        # Common parameters
        'learning_rate': hp.loguniform('learning_rate', low=np.log(1e-6), high=np.log(1e-2)),
        'batch_size': hp.choice('batch_size', [8, 16, 32, 64]),  # Compromise between MLP and UNet
        
        # Embedding dimensions
        'lead_time_embedding_dim': hp.choice('lead_time_embedding_dim', [4, 8, 16, 32]),
        'month_embedding_dim': hp.choice('month_embedding_dim', [4, 8, 16, 32]),
        
        # Regularization
        'dropout_rate': hp.uniform('dropout_rate', 0.0, 0.4),
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


class UNetBlock(nn.Module):
    """Basic UNet building block with optional batch normalization and dropout."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, use_batch_norm=True, dropout_rate=0.0):
        super(UNetBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        if self.use_batch_norm:
            x = self.bn1(x)
        if self.dropout is not None:
            x = self.dropout(x)
            
        x = F.relu(self.conv2(x))
        if self.use_batch_norm:
            x = self.bn2(x)
        if self.dropout is not None:
            x = self.dropout(x)
            
        return x


class ModifiedUNet(nn.Module):
    """
    Modified UNet architecture for weather forecasting with hyperparameter optimization support.
    """
    
    def __init__(self, input_channels, output_channels, init_features=64, depth=4,
                 kernel_size=3, use_batch_norm=True, dropout_rate=0.1,
                 upsampling_mode='bilinear', n_lead_times=1, 
                 lead_time_embedding_dim=16, month_embedding_dim=16):
        super(ModifiedUNet, self).__init__()
        
        self.depth = depth
        self.upsampling_mode = upsampling_mode
        
        # Embedding layers
        self.n_lead_times = n_lead_times
        self.lead_time_embedding = None
        self.month_embedding = nn.Embedding(12, month_embedding_dim)
        
        # Calculate actual input channels including embeddings
        actual_input_channels = input_channels + month_embedding_dim
        
        if n_lead_times > 1:
            self.lead_time_embedding = nn.Embedding(n_lead_times, lead_time_embedding_dim)
            actual_input_channels += lead_time_embedding_dim
        
        # Encoder path
        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        
        features = init_features
        in_channels = actual_input_channels
        
        for i in range(depth):
            self.encoder_blocks.append(
                UNetBlock(in_channels, features, kernel_size, use_batch_norm, dropout_rate)
            )
            in_channels = features
            features *= 2
        
        # Bottleneck
        self.bottleneck = UNetBlock(in_channels, features, kernel_size, use_batch_norm, dropout_rate)
        
        # Decoder path
        self.decoder_blocks = nn.ModuleList()
        self.upconv_blocks = nn.ModuleList()
        
        for i in range(depth):
            self.upconv_blocks.append(
                nn.ConvTranspose2d(features, features // 2, 2, stride=2)
            )
            self.decoder_blocks.append(
                UNetBlock(features, features // 2, kernel_size, use_batch_norm, dropout_rate)
            )
            features //= 2
        
        # Final output layer
        self.final_conv = nn.Conv2d(features, output_channels, 1)
        
    def forward(self, x, lead_time_idx=None, month_idx=None):
        batch_size, channels, height, width = x.shape
        
        # Add embeddings to input
        if month_idx is not None:
            month_emb = self.month_embedding(month_idx)  # [batch_size, month_embedding_dim]
            month_emb = month_emb.unsqueeze(-1).unsqueeze(-1)  # [batch_size, month_embedding_dim, 1, 1]
            month_emb = month_emb.expand(-1, -1, height, width)  # [batch_size, month_embedding_dim, height, width]
            x = torch.cat([x, month_emb], dim=1)
            
        if self.lead_time_embedding is not None and lead_time_idx is not None:
            lead_time_emb = self.lead_time_embedding(lead_time_idx)  # [batch_size, lead_time_embedding_dim]
            lead_time_emb = lead_time_emb.unsqueeze(-1).unsqueeze(-1)  # [batch_size, lead_time_embedding_dim, 1, 1]
            lead_time_emb = lead_time_emb.expand(-1, -1, height, width)  # [batch_size, lead_time_embedding_dim, height, width]
            x = torch.cat([x, lead_time_emb], dim=1)
        
        # Encoder path
        encoder_features = []
        for i, encoder_block in enumerate(self.encoder_blocks):
            x = encoder_block(x)
            encoder_features.append(x)
            if i < len(self.encoder_blocks) - 1:  # Don't pool after last encoder block
                x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path
        for i, (upconv, decoder_block) in enumerate(zip(self.upconv_blocks, self.decoder_blocks)):
            x = upconv(x)
            
            # Get corresponding encoder feature
            encoder_feature = encoder_features[-(i+1)]
            
            # Handle size mismatch due to pooling/upsampling
            if x.shape != encoder_feature.shape:
                x = F.interpolate(x, size=encoder_feature.shape[2:], mode=self.upsampling_mode, align_corners=False)
            
            # Concatenate skip connection
            x = torch.cat([x, encoder_feature], dim=1)
            x = decoder_block(x)
        
        # Final output
        x = self.final_conv(x)
        
        return x


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
                           architecture: str = None,
                           verbose: bool = False) -> Dict[str, Any]:
    """
    Evaluate a set of hyperparameters across multiple regions and lead times.
    
    Args:
        hyperparams: Dictionary of hyperparameters to evaluate
        regions_config: List of region configurations
        device: torch device
        architecture: Force specific architecture ('mlp' or 'unet'), overrides hyperparams
        verbose: Whether to print progress
        
    Returns:
        dict: {'loss': weighted_average_loss, 'status': STATUS_OK, 'region_losses': {...}}
    """
    # Import from main module 
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from finetune import load_forecasts, create_dataloader, get_region_grid
    
    # Determine architecture
    if architecture is not None:
        arch = architecture
    else:
        arch = hyperparams.get('architecture', 'mlp')
    
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
            
            # Adjust batch size based on architecture
            batch_size = hyperparams['batch_size']
            if arch == 'unet' and batch_size > 32:
                batch_size = min(batch_size, 16)  # Reduce for UNet due to memory constraints
            
            # Create data loaders
            if arch == 'mlp':
                # For MLP, flatten spatial dimensions
                train_loader = create_dataloader(
                    fc_norm[t_idx], obs_norm[t_idx], 
                    lead_time_indices[t_idx], month_indices[t_idx], 
                    batch_size=batch_size
                )
                val_loader = create_dataloader(
                    fc_norm[v_idx], obs_norm[v_idx], 
                    lead_time_indices[v_idx], month_indices[v_idx], 
                    batch_size=batch_size
                )
                
                # Initialize MLP model
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
                
            elif arch == 'unet':
                # For UNet, keep spatial structure but need different data loader
                # This would need to be implemented based on your UNet data loader
                # For now, assuming similar interface to MLP loader
                train_loader = create_dataloader(
                    fc_norm[t_idx], obs_norm[t_idx], 
                    lead_time_indices[t_idx], month_indices[t_idx], 
                    batch_size=batch_size
                )
                val_loader = create_dataloader(
                    fc_norm[v_idx], obs_norm[v_idx], 
                    lead_time_indices[v_idx], month_indices[v_idx], 
                    batch_size=batch_size
                )
                
                # Initialize UNet model
                input_channels = n_training_vars
                output_channels = n_output_vars
                n_lead_times = len(args.lead_time_hours)
                
                model = ModifiedUNet(
                    input_channels=input_channels,
                    output_channels=output_channels,
                    init_features=hyperparams['unet_init_features'],
                    depth=hyperparams['unet_depth'],
                    kernel_size=hyperparams['unet_kernel_size'],
                    use_batch_norm=hyperparams['unet_batch_norm'],
                    dropout_rate=hyperparams['dropout_rate'],
                    upsampling_mode=hyperparams['unet_upsampling_mode'],
                    n_lead_times=n_lead_times,
                    lead_time_embedding_dim=hyperparams['lead_time_embedding_dim'],
                    month_embedding_dim=hyperparams['month_embedding_dim']
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
                
        except Exception as e:
            print(f"Error evaluating region {args.region} with {arch}: {str(e)}")
            # Return a high loss for failed evaluations
            return {'loss': 1e10, 'status': STATUS_OK, 'error': str(e)}
    
    # Calculate weighted average loss
    avg_loss = total_weighted_loss / total_weight if total_weight > 0 else 1e10
    
    return {
        'loss': avg_loss,
        'status': STATUS_OK,
        'region_losses': region_losses,
        'hyperparams': hyperparams,
        'architecture': arch
    }


def optimize_hyperparameters(regions_config: List[Dict[str, Any]], 
                           architecture: str,
                           max_evals: int = 100,
                           output_dir: str = None,
                           device: torch.device = None,
                           random_seed: int = 42) -> Dict[str, Any]:
    """
    Optimize hyperparameters for a specific architecture (MLP or UNet).
    
    Args:
        regions_config: List of region configurations
        architecture: 'mlp' or 'unet'
        max_evals: Maximum number of evaluations
        output_dir: Directory to save results
        device: torch device (if None, will auto-detect)
        random_seed: Random seed for reproducibility
        
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
    
    # Create output directory
    if output_dir is None:
        output_dir = f"hyperopt_results_{architecture}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define objective function for hyperopt
    def objective(hyperparams):
        """Objective function to minimize."""
        print(f"\nEvaluating {architecture.upper()} hyperparameters: {hyperparams}")
        result = evaluate_hyperparameters(hyperparams, regions_config, device, 
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
    data_dir = "/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/data/processed/cleaned_weatherbench_downloads" 
    regions_config = create_region_config(region_specs, base_args, data_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else
                          'cpu')
    
    # Example 1: Optimize MLP only
    print("Example 1: MLP optimization")
    mlp_results = optimize_hyperparameters(
        regions_config=regions_config,
        architecture="mlp",
        max_evals=50,
        output_dir="hyperopt_results_mlp",
        device=device,
        random_seed=42
    )
    
    # Example 2: Optimize UNet only
    print("\nExample 2: UNet optimization")
    unet_results = optimize_hyperparameters(
        regions_config=regions_config,
        architecture="unet",
        max_evals=50,
        output_dir="hyperopt_results_unet",
        device=device,
        random_seed=42
    )
    
    print(f"\nOptimizations complete!")
    print(f"MLP best loss: {mlp_results['best_loss']:.6f}")
    print(f"UNet best loss: {unet_results['best_loss']:.6f}")
    
    # Determine which architecture performed better
    if mlp_results['best_loss'] < unet_results['best_loss']:
        print(f"MLP performed better with loss {mlp_results['best_loss']:.6f}")
        print(f"Best MLP hyperparameters: {mlp_results['best_hyperparams']}")
    else:
        print(f"UNet performed better with loss {unet_results['best_loss']:.6f}")
        print(f"Best UNet hyperparameters: {unet_results['best_hyperparams']}")