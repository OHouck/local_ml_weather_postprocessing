
import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from types import SimpleNamespace
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from functools import lru_cache
import socket
from typing import List, Tuple, Dict, Optional, Union

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from helper_funcs import setup_directories
from helper_funcs import generate_output_path

@lru_cache(maxsize=256)
def load_zarr_cached(file_path):
    """Cache zarr dataset loading to avoid redundant file reads."""
    print(f"Loading dataset from: {file_path}")
    return xr.open_zarr(file_path)

def extract_forecast_data(ds, prediction_var, lead_time):
    """Extract forecast data arrays for a specific lead time."""
    var_suffix = f"_lt{lead_time}h"
    
    ground_truth = ds[f"{prediction_var}_ground_truth{var_suffix}"]
    original = ds[f"{prediction_var}_original{var_suffix}"]
    corrected = ds[f"{prediction_var}_corrected{var_suffix}"]
    mean_corrected = ds.get(f"{prediction_var}_mean_corrected{var_suffix}", None)
    
    return ground_truth, original, corrected, mean_corrected


#=========================
# loss functions functions
#=========================
def calculate_rmse(predictions, ground_truth):
    """Calculate RMSE between predictions and ground truth."""
    return float(np.sqrt(((predictions- ground_truth) ** 2).mean()))

def calculate_extreme_heat_rmse(preds, targets):
    """
    up-weight loss fro negative errors for high temperature values
    """
    # convert to C
    targets_c = targets - 273.15
    preds_c = preds - 273.15
    errors = preds_c - targets_c
    weights =np.ones_like(errors)

    # Add penalties for under-prediction at high temps
    weights += ((targets_c > 25) & (targets_c < 30) & (errors < 0)).astype(float) * 2
    weights += ((targets_c >= 30) & (errors < 0)).astype(float) * 10

    squared_errors = errors ** 2

    weights = weights / weights.sum()  # sum to 1 for interpretability
    weighted_mse = (weights * squared_errors).sum()

    return float(np.sqrt(weighted_mse))

def calculate_improvement_percentage(rmse_original, rmse_corrected):
    """Calculate percentage improvement in RMSE."""
    if rmse_original == 0:
        return 0
    return (rmse_original - rmse_corrected) / rmse_original * 100

def calculate_and_save_statistics(
        dirs: Dict[str, str],
        models: List[str],
        variable_configs: List[Dict[str, Union[str, Tuple]]],
        nn_architectures: List[str] = ["mlp"],
        geographic_regions: Optional[List[str]] = None,
        bootstrap_regions: Optional[List[str]] = None,
        subregions: List[str] = ["2x2", "6x6", "10x10"],
        lead_times: Optional[List[int]] = None,
        loss_fns: Optional[List[str]] = None,
        simultaneous: bool = False,
        output_csv_path: Optional[str] = None,
        ground_truth_source: str = ""
) -> pd.DataFrame:
    """
    Enhanced version that handles multiple variable configurations and both geographic and climate regions.
    Now includes mean forecast values and error frequency metrics.
    
    Parameters
    ----------
    dirs : dict
        Dictionary of directories
    models : list
        List of model names (e.g., ['pangu', 'ifs'])
    variable_configs : list of dict
        List of variable configurations, each containing:
        - 'training_vars': tuple/list of training variables
        - 'output_vars': tuple/list of output variables  
        - 'prediction_var': str, variable to predict
    nn_architectures : list
        List of architectures: ["mlp"], ["unet"], or ["mlp", "unet"]
    geographic_regions : list, optional
        List of geographic regions (e.g., ["india", "usa_south"])
    bootstrap_regions: list, optional
        List of regions that need bootrapping(e.g., ["tropical", "arid", "temperate"])
    subregions : list
        List of patch sizes (e.g., ["2x2", "6x6", "10x10"])
    lead_times : list
        List of lead times in hours
    loss_fns : list
        List of loss functions used during training
    simultaneous : bool
        If True, use data from model that trained all lead times simultaneously
    output_csv_path : str
        Path to save the CSV file. If None, auto-generates path
    ground_truth_source : str
        Optional string to specify a different ground truth source
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing all calculated statistics
    """
    
    # Define error cutoffs for each variable
    ERROR_CUTOFFS = {
        '2m_temperature': {'value': 2.5, 'type': 'absolute', 'units': 'K'},
        '10m_wind_speed': {'value': 2.0, 'type': 'absolute', 'units': 'm/s'},
        'total_precipitation': {'value': 0.0, 'type': 'binary', 'units': 'rain/no-rain'}
    }
    
    # Default values
    if geographic_regions is None:
        geographic_regions = []
    if bootstrap_regions is None:
        bootstrap_regions = []
    if lead_times is None:
        lead_times = [24, 120, 216]
    
    # Combine all regions with their types 
    all_regions = []
    for region in geographic_regions:
        all_regions.append({'name': region, 'type': 'geographic', 'bootstrap': False})

    # Classify bootstrap regions by type
    climate_regions = ['arid', 'temperate', 'tropical']
    topographic_regions = ['flat', 'hilly', 'mountainous']
    for region in bootstrap_regions:
        if region in climate_regions:
            region_type = 'climate'
        elif region in topographic_regions:
            region_type = 'topographic'
        else:
            region_type = 'climate'  # Default to climate if not recognized
        all_regions.append({'name': region, 'type': region_type, 'bootstrap': True})
    
    # Storage for all results
    all_results = []

    
    # Process each variable configuration
    for var_config in variable_configs:
        training_vars = var_config['training_vars']
        output_vars = var_config['output_vars']
        prediction_var = var_config['prediction_var']
        
        # Ensure variables are lists
        if not isinstance(training_vars, (list, tuple)):
            training_vars = [training_vars]
        if not isinstance(output_vars, (list, tuple)):
            output_vars = [output_vars]
        
        print(f"\n{'='*60}")
        print(f"Processing variable configuration:")
        print(f"  Training vars: {training_vars}")
        print(f"  Output vars: {output_vars}")
        print(f"  Prediction var: {prediction_var}")
        print(f"{'='*60}")

        # Process each combination
        for growing_season_only in (False, True):
            for loss_fn in (loss_fns if loss_fns else [None]):
                # if loss_fn is extreme_heat_loss and prediction_var is not temperature, skip
                if loss_fn == "extreme_heat_loss" and prediction_var != "2m_temperature":
                    continue
                for subregion in subregions:
                    for region_info in all_regions:
                        region = region_info['name']
                        region_type = region_info['type']
                        bootstrap = region_info['bootstrap']
                        
                        for model in models:
                            for arch in nn_architectures:
                                for lead_time in lead_times:
                                    # Set up args for generate_output_path
                                    if simultaneous:
                                        lead_time_hours = lead_times
                                    else:
                                        lead_time_hours = lead_time
                                    if model == "ifs" or model == "pangu":
                                        train_start="2018-01-01"
                                        train_end="2021-12-31"
                                        test_start="2022-01-01"
                                        test_end="2022-12-31"
                                    elif model == "aifs":
                                        train_start="2021-01-01"
                                        train_end="2023-12-31"
                                        test_start="2024-01-01"
                                        test_end="2024-12-31"
                                    else:
                                        raise ValueError(f"Unknown model: {model}")
                                    if loss_fn == "mse":
                                        alternate_loss_fn = None
                                    elif loss_fn == "extreme_heat_loss":
                                        alternate_loss_fn = "extreme_heat_loss"
                                    else:
                                        alternate_loss_fn = None

                                    args = SimpleNamespace(
                                        model_name=model,
                                        ground_truth_source=ground_truth_source,
                                        region=region,
                                        subregion=subregion,
                                        alternate_loss_fn=alternate_loss_fn,
                                        train_start=train_start,
                                        train_end=train_end,
                                        test_start=test_start,
                                        test_end=test_end,
                                        training_vars=training_vars,
                                        output_vars=output_vars,
                                        lead_time_hours=lead_time_hours,
                                        nn_architecture=arch,
                                        growing_season_only=growing_season_only
                                    )
                                    
                                    # Construct file paths
                                    if bootstrap:
                                        file_pattern = os.path.join(dirs['input'], 
                                            generate_output_path(args).replace('.zarr', '*bs*.zarr'))
                                    else:
                                        file_pattern = os.path.join(dirs['input'], 
                                            generate_output_path(args))
                                    
                                    files = glob.glob(file_pattern)

                                    
                                    if not files:
                                        print(f"No files found for {prediction_var} {model} {arch} {region} {subregion} {lead_time}h")
                                        continue
                                    
                                    if not bootstrap and len(files) > 1:
                                        print(f"Warning: Multiple files found for {prediction_var} {model} {arch} {region} {subregion} {lead_time}h")
                                        files = files[:1]
                                    
                                    # Process each file (multiple for bootstrap)
                                    file_results = []
                                    ground_truth_values = []
                                    
                                    for idx, file_path in enumerate(files):
                                        ds = load_zarr_cached(file_path)
                                        # Extract data
                                        ground_truth, original, corrected, mean_bias_corrected = extract_forecast_data(
                                            ds, prediction_var, lead_time
                                        )

                                        # Flatten arrays for statistics
                                        gt_flat = ground_truth.values.flatten()
                                        orig_flat = original.values.flatten()
                                        corr_flat = corrected.values.flatten()
                                        mean_bias_flat = mean_bias_corrected.values.flatten() if mean_bias_corrected is not None else None

                                        # Calculate RMSE values
                                        rmse_original = calculate_rmse(orig_flat, gt_flat)
                                        rmse_corrected = calculate_rmse(corr_flat, gt_flat)
                                        rmse_pct_improvement = (rmse_original - rmse_corrected) / rmse_original * 100

                                        rmse_og_extreme_heat = calculate_extreme_heat_rmse(orig_flat, gt_flat) 
                                        rmse_corr_extreme_heat = calculate_extreme_heat_rmse(corr_flat, gt_flat) 
                                        rmse_pct_improvement_extreme_heat = (rmse_og_extreme_heat - rmse_corr_extreme_heat) / rmse_og_extreme_heat * 100 if rmse_og_extreme_heat and rmse_corr_extreme_heat else None   

                                        # Remove NaN values - fixed logic here
                                        mask = ~(np.isnan(gt_flat) | np.isnan(orig_flat) | np.isnan(corr_flat))
                                        if mean_bias_flat is not None:
                                            mask = mask & ~np.isnan(mean_bias_flat)
                                        
                                        gt_flat = gt_flat[mask]
                                        orig_flat = orig_flat[mask]
                                        corr_flat = corr_flat[mask]
                                        mean_bias_flat = mean_bias_flat[mask] if mean_bias_flat is not None else None
                                        
                                        # Store ground truth for statistics
                                        ground_truth_values.extend(gt_flat)
                                        
                                        # Calculate mean forecast values
                                        mean_original = np.mean(orig_flat)
                                        mean_corrected = np.mean(corr_flat)
                                        
                                        
                                        # Calculate error frequency metrics
                                        if prediction_var in ERROR_CUTOFFS:
                                            cutoff_info = ERROR_CUTOFFS[prediction_var]
                                            
                                            if cutoff_info['type'] == 'absolute':
                                                # Calculate percentage of errors above cutoff
                                                errors_orig = np.abs(orig_flat - gt_flat)
                                                errors_corr = np.abs(corr_flat - gt_flat)
                                                pct_error_cutoff_original = (errors_orig > cutoff_info['value']).mean() * 100
                                                pct_error_cutoff_corrected = (errors_corr > cutoff_info['value']).mean() * 100
                                                
                                            elif cutoff_info['type'] == 'binary':
                                                # For precipitation: calculate misclassification rate for rain/no-rain
                                                gt_binary = (gt_flat > 0).astype(int)
                                                orig_binary = (orig_flat > 0).astype(int)
                                                corr_binary = (corr_flat > 0).astype(int)
                                                pct_error_cutoff_original = (gt_binary != orig_binary).mean() * 100
                                                pct_error_cutoff_corrected = (gt_binary != corr_binary).mean() * 100
                                        else:
                                            pct_error_cutoff_original = None
                                            pct_error_cutoff_corrected = None
                                        
                                        # Calculate mean bias corrected RMSE if available
                                        rmse_mean_corrected = None
                                        pct_improvement_mean_correction = None  # Fixed variable name
                                        if mean_bias_corrected is not None:
                                            rmse_mean_corrected = np.sqrt(np.mean((mean_bias_flat - gt_flat)**2))
                                            pct_improvement_mean_correction = (rmse_original - rmse_mean_corrected) / rmse_original * 100

                                        file_results.append({
                                            'rmse_original': rmse_original,
                                            'rmse_corrected': rmse_corrected,
                                            'rmse_mean_corrected': rmse_mean_corrected,
                                            'rmse_pct_improvement': rmse_pct_improvement,
                                            'rmse_pct_improvement_extreme_heat': rmse_pct_improvement_extreme_heat,
                                            'pct_improvement_mean_correction': pct_improvement_mean_correction,  # Fixed key name
                                            'mean_original': mean_original,
                                            'mean_corrected': mean_corrected,
                                            'pct_error_cutoff_original': pct_error_cutoff_original,
                                            'pct_error_cutoff_corrected': pct_error_cutoff_corrected,
                                            'bootstrap_idx': idx if bootstrap else None
                                        })
                                    if not file_results:
                                        continue
                                    
                                    # Calculate statistics across bootstrap samples if applicable
                                    if bootstrap:
                                        rmse_orig_values = [r['rmse_original'] for r in file_results]
                                        rmse_corr_values = [r['rmse_corrected'] for r in file_results]
                                        rmse_pct_imp_values = [r['rmse_pct_improvement'] for r in file_results]
                                        rmse_pct_improvement_extreme_heat_values = [r['rmse_pct_improvement_extreme_heat'] for r in file_results if r['rmse_pct_improvement_extreme_heat'] is not None]
                                        mean_orig_values = [r['mean_original'] for r in file_results]
                                        mean_corr_values = [r['mean_corrected'] for r in file_results]
                                        pct_error_orig_values = [r['pct_error_cutoff_original'] for r in file_results if r['pct_error_cutoff_original'] is not None]
                                        pct_error_corr_values = [r['pct_error_cutoff_corrected'] for r in file_results if r['pct_error_cutoff_corrected'] is not None]
                                        
                                        n = len(file_results)
                                        
                                        # Calculate means
                                        rmse_orig_mean = np.mean(rmse_orig_values)
                                        rmse_corr_mean = np.mean(rmse_corr_values)
                                        rmse_pct_imp_mean = np.mean(rmse_pct_imp_values)
                                        rmse_pct_imp_extreme_heat_mean = np.mean(rmse_pct_improvement_extreme_heat_values) if rmse_pct_improvement_extreme_heat_values else None
                                        mean_orig_forecast = np.mean(mean_orig_values)
                                        mean_corr_forecast = np.mean(mean_corr_values)
                                        pct_error_orig_mean = np.mean(pct_error_orig_values) if pct_error_orig_values else None
                                        pct_error_corr_mean = np.mean(pct_error_corr_values) if pct_error_corr_values else None
                                        
                                        # Calculate standard errors and confidence intervals
                                        rmse_orig_se = np.std(rmse_orig_values, ddof=1) / np.sqrt(n)
                                        rmse_corr_se = np.std(rmse_corr_values, ddof=1) / np.sqrt(n)
                                        rmse_pct_imp_se = np.std(rmse_pct_imp_values, ddof=1) / np.sqrt(n)  # Fixed variable name
                                        
                                        # 95% CI using t-distribution
                                        alpha_ci = 0.05
                                        t_crit = stats.t.ppf(1 - alpha_ci/2, df=n-1)
                                        
                                        rmse_orig_ci_lower = rmse_orig_mean - (t_crit * rmse_orig_se)
                                        rmse_orig_ci_upper = rmse_orig_mean + (t_crit * rmse_orig_se)
                                        rmse_corr_ci_lower = rmse_corr_mean - (t_crit * rmse_corr_se)
                                        rmse_corr_ci_upper = rmse_corr_mean + (t_crit * rmse_corr_se)
                                        rmse_pct_imp_ci_lower = rmse_pct_imp_mean - (t_crit * rmse_pct_imp_se)  # Fixed variable name
                                        rmse_pct_imp_ci_upper = rmse_pct_imp_mean + (t_crit * rmse_pct_imp_se)  # Fixed variable name
                                        
                                        # Handle mean corrected if available
                                        rmse_mc_mean = None
                                        pct_imp_mc_mean = None
                                        if file_results[0]['rmse_mean_corrected'] is not None:
                                            rmse_mc_values = [r['rmse_mean_corrected'] for r in file_results]
                                            pct_imp_mc_values = [r['pct_improvement_mean_correction'] for r in file_results]  # Fixed key name
                                            rmse_mc_mean = np.mean(rmse_mc_values)
                                            pct_imp_mc_mean = np.mean(pct_imp_mc_values)
                                    else:
                                        # Single file case
                                        result = file_results[0]
                                        rmse_orig_mean = result['rmse_original']
                                        rmse_corr_mean = result['rmse_corrected']
                                        rmse_pct_imp_mean = result['rmse_pct_improvement']
                                        rmse_pct_imp_extreme_heat_mean = result['rmse_pct_improvement_extreme_heat']
                                        mean_orig_forecast = result['mean_original']
                                        mean_corr_forecast = result['mean_corrected']
                                        pct_error_orig_mean = result['pct_error_cutoff_original']
                                        pct_error_corr_mean = result['pct_error_cutoff_corrected']
                                        rmse_mc_mean = result['rmse_mean_corrected']
                                        pct_imp_mc_mean = result['pct_improvement_mean_correction']  # Fixed key name
                                        
                                        # No confidence intervals for single file
                                        rmse_orig_ci_lower = rmse_orig_ci_upper = None
                                        rmse_corr_ci_lower = rmse_corr_ci_upper = None
                                        rmse_pct_imp_ci_lower = rmse_pct_imp_ci_upper = None  # Fixed variable names
                                        n = 1
                                    
                                    # Calculate ground truth statistics
                                    if ground_truth_values:
                                        gt_mean = np.mean(ground_truth_values)
                                        gt_std = np.std(ground_truth_values)
                                    else:
                                        gt_mean = gt_std = None
                                    
                                    # Prepare metadata string (only for first 3 rows of each variable)
                                    metadata_str = None
                                    if prediction_var in ERROR_CUTOFFS:
                                        cutoff_info = ERROR_CUTOFFS[prediction_var]
                                        if cutoff_info['type'] == 'absolute':
                                            metadata_str = f"Error cutoff: >{cutoff_info['value']} {cutoff_info['units']}"
                                        elif cutoff_info['type'] == 'binary':
                                            metadata_str = f"Error type: {cutoff_info['units']} misclassification"
                                    else:
                                        metadata_str = "No error cutoff defined"
                                        print(f"Warning: No error cutoff defined for variable: {prediction_var}")
                                    
                                    # Create row for results
                                    row = {
                                        'variable': prediction_var,
                                        'model': model,
                                        'architecture': arch,
                                        'region': region,
                                        'region_type': region_type,
                                        'subregion': subregion,
                                        'lead_time': lead_time,
                                        'training_vars': "_".join(training_vars),
                                        'output_vars': "_".join(output_vars),
                                        'train_period': f"{train_start}_{train_end}",
                                        'test_period': f"{test_start}_{test_end}",
                                        'growing_season_only': growing_season_only,
                                        'loss_fn': loss_fn,
                                        'rmse_original': rmse_orig_mean,
                                        'rmse_corrected': rmse_corr_mean,
                                        'rmse_mean_corrected': rmse_mc_mean,
                                        'rmse_pct_improvement': rmse_pct_imp_mean,
                                        'rmse_pct_improvement_extreme_heat': rmse_pct_imp_extreme_heat_mean,
                                        'pct_improvement_mean_corrected': pct_imp_mc_mean,
                                        'mean_original_forecast': mean_orig_forecast,
                                        'mean_corrected_forecast': mean_corr_forecast,
                                        'pct_error_cutoff_original': pct_error_orig_mean,
                                        'pct_error_cutoff_corrected': pct_error_corr_mean,
                                        'metadata': metadata_str,
                                        'ground_truth_mean': gt_mean,
                                        'ground_truth_std': gt_std,
                                        'bootstrap': bootstrap,
                                        'n_samples': n
                                    }
                                    # Add confidence intervals if bootstrap
                                    if bootstrap:
                                        row.update({
                                            'rmse_original_ci_lower': rmse_orig_ci_lower,
                                            'rmse_original_ci_upper': rmse_orig_ci_upper,
                                            'rmse_corrected_ci_lower': rmse_corr_ci_lower,
                                            'rmse_corrected_ci_upper': rmse_corr_ci_upper,
                                            'rmse_pct_improvement_ci_lower': rmse_pct_imp_ci_lower,  # Fixed key name
                                            'rmse_pct_improvement_ci_upper': rmse_pct_imp_ci_upper   # Fixed key name
                                        })
                                    
                                    all_results.append(row)
                                    
                                    # Optional progress message
                                    if pct_error_orig_mean is not None:
                                        print(f"Processed: {prediction_var} {model} {arch} {region} {subregion} {lead_time}h - "
                                            f"Improvement: {rmse_pct_imp_mean:.1f}%, "
                                            f"Error rate orig: {pct_error_orig_mean:.1f}% -> corr: {pct_error_corr_mean:.1f}%")
                                    else:
                                        print(f"Processed: {prediction_var} {model} {arch} {region} {subregion} {lead_time}h - "
                                            f"Improvement: {rmse_pct_imp_mean:.1f}%")
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Save to CSV
    if output_csv_path is None:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_csv_path = os.path.join(dirs['processed'], 
                                      f'forecast_statistics_all_vars_{timestamp}.csv')
    
    df.to_csv(output_csv_path, index=False)
    print(f"\nStatistics saved to: {output_csv_path}")
    
    return df

def main():
    
    # create data outputs:
    dirs = setup_directories()

    variable_configs = [
    {
        'training_vars': ['2m_temperature'],
        'output_vars': ['2m_temperature'],
        'prediction_var': '2m_temperature'
    },
    {
        'training_vars': ['10m_wind_speed'],
        'output_vars': ['10m_wind_speed'],
        'prediction_var': '10m_wind_speed'
    },
    {
        'training_vars': ['total_precipitation'],
        'output_vars': ['total_precipitation'],
        'prediction_var': 'total_precipitation'
    }
]

    loss_function_list = ['mse', 'extreme_heat_loss']

    geographic_regions=["india", "ethiopia", "amazon", "corn_belt", "usa_south"]
    # regions that require bootstrapping
    bootstrap_regions=["arid", "tropical", "temperate", "flat", "hilly", "mountainous"]
    df = calculate_and_save_statistics(
        dirs=dirs,
        variable_configs=variable_configs,
        geographic_regions=geographic_regions,
        bootstrap_regions=bootstrap_regions,
        models=["pangu", "ifs", "aifs"],  
        nn_architectures=["mlp", "unet"],  # Can also include "unet"
        subregions=["2x2", "6x6", "10x10"],  # All subregions
        lead_times=[24, 120, 216],
        loss_fns = loss_function_list,
        simultaneous=True,
        output_csv_path=f"{dirs['processed']}/forecast_improvement_stats.csv"
    )

if __name__ == "__main__":
    main() 