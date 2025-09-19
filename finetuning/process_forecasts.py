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

def setup_directories():
    """Set up directory structure based on environment."""
    nodename = socket.gethostname()
    if nodename == "oMac.local":
        root = os.path.expanduser("~/OneDrive - The University of Chicago/ai_weather_ag/data")
    else:
        raise Exception(f"Unknown environment, Please specify the root directory. "
                        f"Nodename found: {nodename}")

    dirs = {
        'root': root,
        'raw': os.path.join(root, "raw"),
        'globus': os.path.expanduser("~/globus/forecast_data/processed"),
        'processed': os.path.join(root, "processed"),
        'fig': os.path.join(root, "../figures/finetuning"),
        'input': os.path.join(root, "fine_tuning_output")
    }

    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


@lru_cache(maxsize=256)
def load_zarr_cached(file_path):
    """Cache zarr dataset loading to avoid redundant file reads."""
    return xr.open_zarr(file_path)

def extract_forecast_data(ds, prediction_var, lead_time):
    """Extract forecast data arrays for a specific lead time."""
    var_suffix = f"_lt{lead_time}h"
    
    ground_truth = ds[f"{prediction_var}_ground_truth{var_suffix}"]
    original = ds[f"{prediction_var}_original{var_suffix}"]
    corrected = ds[f"{prediction_var}_corrected{var_suffix}"]
    mean_corrected = ds.get(f"{prediction_var}_mean_corrected{var_suffix}", None)
    
    return ground_truth, original, corrected, mean_corrected


def calculate_rmse(predictions, ground_truth):
    """Calculate RMSE between predictions and ground truth."""
    return float(np.sqrt(((predictions - ground_truth) ** 2).mean().values))


def calculate_improvement_percentage(rmse_original, rmse_corrected):
    """Calculate percentage improvement in RMSE."""
    if rmse_original == 0:
        return 0
    return (rmse_original - rmse_corrected) / rmse_original * 100

def generate_output_path(args):
    region_str = f"{args.region}"
    subregion_str = f"{args.subregion}"
    dates_str = f"train{args.train_start}-{args.train_end}_test{args.test_start}-{args.test_end}"
    training_vars_str = "_".join(args.training_vars)
    output_vars_str = "_".join(args.output_vars)

    if args.nn_architecture == "UNet":
        model_str = "unet"
    else: 
        model_str = "mlp"

    output_path = f"{args.model_name}/{args.ground_truth_source}{region_str}/train_{training_vars_str}_test_{output_vars_str}_dim{subregion_str}_leadtime_{args.lead_time_hours}h_{dates_str}_{model_str}.zarr"
    return output_path 

def calculate_and_save_statistics(
        dirs: Dict[str, str],
        train_start: str, 
        train_end: str,
        test_start: str, 
        test_end: str,
        models: List[str],
        variable_configs: List[Dict[str, Union[str, Tuple]]],
        nn_architectures: List[str] = ["mlp"],
        geographic_regions: Optional[List[str]] = None,
        climate_regions: Optional[List[str]] = None,
        subregions: List[str] = ["2x2", "6x6", "10x10"],
        lead_times: Optional[List[int]] = None,
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
    train_start, train_end, test_start, test_end : str
        Date strings for train/test periods
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
    climate_regions : list, optional
        List of climate regions (e.g., ["tropical", "arid", "temperate"])
    subregions : list
        List of patch sizes (e.g., ["2x2", "6x6", "10x10"])
    lead_times : list
        List of lead times in hours
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
        '2m_temperature': {'value': 5.0, 'type': 'absolute', 'units': 'K'},
        '10m_wind_speed': {'value': 2.0, 'type': 'absolute', 'units': 'm/s'},
        'total_precipitation': {'value': 0.0, 'type': 'binary', 'units': 'rain/no-rain'}
    }
    
    # Default values
    if geographic_regions is None:
        geographic_regions = []
    if climate_regions is None:
        climate_regions = []
    if lead_times is None:
        lead_times = [24, 120, 240]
    
    # Combine all regions with their types
    all_regions = []
    for region in geographic_regions:
        all_regions.append({'name': region, 'type': 'geographic', 'bootstrap': False})
    for region in climate_regions:
        all_regions.append({'name': region, 'type': 'climate', 'bootstrap': True})
    
    # Storage for all results
    all_results = []
    
    # Track metadata row indices for each variable
    metadata_row_count = {}
    
    # Process each variable configuration
    for var_config in variable_configs:
        training_vars = var_config['training_vars']
        output_vars = var_config['output_vars']
        prediction_var = var_config['prediction_var']
        
        # Initialize metadata counter for this variable
        metadata_row_count[prediction_var] = 0
        
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
                                lead_time_hours = "_".join(str(lt) for lt in lead_times)
                            else:
                                lead_time_hours = lead_time
                            
                            args = SimpleNamespace(
                                model_name=model,
                                ground_truth_source=ground_truth_source,
                                region=region,
                                subregion=subregion,
                                train_start=train_start,
                                train_end=train_end,
                                test_start=test_start,
                                test_end=test_end,
                                training_vars=training_vars,
                                output_vars=output_vars,
                                lead_time_hours=lead_time_hours,
                                nn_architecture=arch
                            )
                            
                            # Construct file paths
                            if bootstrap:
                                file_pattern = os.path.join(dirs['input'], 
                                                           generate_output_path(args).replace('.zarr', '*bs*.zarr'))
                            else:
                                file_pattern = os.path.join(dirs['input'], generate_output_path(args))
                            
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
                                try:
                                    ds = load_zarr_cached(file_path)
                                    print(file_path)
                                    
                                    # Extract data
                                    ground_truth, original, corrected, mean_bias_corrected = extract_forecast_data(
                                        ds, prediction_var, lead_time
                                    )
                                    
                                    # Flatten arrays for statistics
                                    gt_flat = ground_truth.values.flatten()
                                    orig_flat = original.values.flatten()
                                    corr_flat = corrected.values.flatten()
                                    mean_bias_flat = mean_bias_corrected.values.flatten() if mean_bias_corrected is not None else None
                                    
                                    # Remove NaN values
                                    mask = ~(np.isnan(gt_flat) | np.isnan(orig_flat) | np.isnan(corr_flat)) | (mean_bias_flat is not None and np.isnan(mean_bias_flat))
                                    gt_flat = gt_flat[mask]
                                    orig_flat = orig_flat[mask]
                                    corr_flat = corr_flat[mask]
                                    mean_bias_flat = mean_bias_flat[mask] if mean_bias_flat is not None else None
                                    
                                    # Store ground truth for statistics
                                    ground_truth_values.extend(gt_flat)
                                    
                                    # Calculate mean forecast values
                                    mean_original = np.mean(orig_flat)
                                    mean_corrected = np.mean(corr_flat)
                                    
                                    # Calculate RMSE values
                                    rmse_original = np.sqrt(np.mean((orig_flat - gt_flat)**2))
                                    rmse_corrected = np.sqrt(np.mean((corr_flat - gt_flat)**2))
                                    
                                    # Calculate percent improvement
                                    pct_improvement = (rmse_original - rmse_corrected) / rmse_original * 100
                                    
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
                                    pct_improvement_mean = None
                                    if mean_bias_corrected is not None:
                                        rmse_mean_bias_corrected = np.sqrt(np.mean((mean_bias_flat - gt_flat)**2))
                                        pct_improvement_mean = (rmse_original - rmse_mean_bias_corrected) / rmse_original * 100

                                    file_results.append({
                                        'rmse_original': rmse_original,
                                        'rmse_corrected': rmse_corrected,
                                        'rmse_mean_corrected': rmse_mean_corrected,
                                        'pct_improvement': pct_improvement,
                                        'pct_improvement_mean': pct_improvement_mean,
                                        'mean_original': mean_original,
                                        'mean_corrected': mean_corrected,
                                        'pct_error_cutoff_original': pct_error_cutoff_original,
                                        'pct_error_cutoff_corrected': pct_error_cutoff_corrected,
                                        'bootstrap_idx': idx if bootstrap else None
                                    })
                                    
                                except Exception as e:
                                    print(f"Error processing {file_path}: {e}")
                                    exit()
                                    continue
                            
                            if not file_results:
                                continue
                            
                            # Calculate statistics across bootstrap samples if applicable
                            if bootstrap:
                                rmse_orig_values = [r['rmse_original'] for r in file_results]
                                rmse_corr_values = [r['rmse_corrected'] for r in file_results]
                                pct_imp_values = [r['pct_improvement'] for r in file_results]
                                mean_orig_values = [r['mean_original'] for r in file_results]
                                mean_corr_values = [r['mean_corrected'] for r in file_results]
                                pct_error_orig_values = [r['pct_error_cutoff_original'] for r in file_results if r['pct_error_cutoff_original'] is not None]
                                pct_error_corr_values = [r['pct_error_cutoff_corrected'] for r in file_results if r['pct_error_cutoff_corrected'] is not None]
                                
                                n = len(file_results)
                                
                                # Calculate means
                                rmse_orig_mean = np.mean(rmse_orig_values)
                                rmse_corr_mean = np.mean(rmse_corr_values)
                                pct_imp_mean = np.mean(pct_imp_values)
                                mean_orig_forecast = np.mean(mean_orig_values)
                                mean_corr_forecast = np.mean(mean_corr_values)
                                pct_error_orig_mean = np.mean(pct_error_orig_values) if pct_error_orig_values else None
                                pct_error_corr_mean = np.mean(pct_error_corr_values) if pct_error_corr_values else None
                                
                                # Calculate standard errors and confidence intervals
                                rmse_orig_se = np.std(rmse_orig_values, ddof=1) / np.sqrt(n)
                                rmse_corr_se = np.std(rmse_corr_values, ddof=1) / np.sqrt(n)
                                pct_imp_se = np.std(pct_imp_values, ddof=1) / np.sqrt(n)
                                
                                # 95% CI using t-distribution
                                alpha_ci = 0.05
                                t_crit = stats.t.ppf(1 - alpha_ci/2, df=n-1)
                                
                                rmse_orig_ci_lower = rmse_orig_mean - (t_crit * rmse_orig_se)
                                rmse_orig_ci_upper = rmse_orig_mean + (t_crit * rmse_orig_se)
                                rmse_corr_ci_lower = rmse_corr_mean - (t_crit * rmse_corr_se)
                                rmse_corr_ci_upper = rmse_corr_mean + (t_crit * rmse_corr_se)
                                pct_imp_ci_lower = pct_imp_mean - (t_crit * pct_imp_se)
                                pct_imp_ci_upper = pct_imp_mean + (t_crit * pct_imp_se)
                                
                                # Handle mean corrected if available
                                rmse_mc_mean = None
                                pct_imp_mc_mean = None
                                if file_results[0]['rmse_mean_corrected'] is not None:
                                    rmse_mc_values = [r['rmse_mean_corrected'] for r in file_results]
                                    pct_imp_mc_values = [r['pct_improvement_mean'] for r in file_results]
                                    rmse_mc_mean = np.mean(rmse_mc_values)
                                    pct_imp_mc_mean = np.mean(pct_imp_mc_values)
                            else:
                                # Single file case
                                result = file_results[0]
                                rmse_orig_mean = result['rmse_original']
                                rmse_corr_mean = result['rmse_corrected']
                                pct_imp_mean = result['pct_improvement']
                                mean_orig_forecast = result['mean_original']
                                mean_corr_forecast = result['mean_corrected']
                                pct_error_orig_mean = result['pct_error_cutoff_original']
                                pct_error_corr_mean = result['pct_error_cutoff_corrected']
                                rmse_mc_mean = result['rmse_mean_corrected']
                                pct_imp_mc_mean = result['pct_improvement_mean']
                                
                                # No confidence intervals for single file
                                rmse_orig_ci_lower = rmse_orig_ci_upper = None
                                rmse_corr_ci_lower = rmse_corr_ci_upper = None
                                pct_imp_ci_lower = pct_imp_ci_upper = None
                                n = 1
                            
                            # Calculate ground truth statistics
                            if ground_truth_values:
                                gt_mean = np.mean(ground_truth_values)
                                gt_std = np.std(ground_truth_values)
                            else:
                                gt_mean = gt_std = None
                            
                            # Prepare metadata string (only for first 3 rows of each variable)
                            metadata_str = None
                            if prediction_var in ERROR_CUTOFFS and metadata_row_count[prediction_var] < 3:
                                cutoff_info = ERROR_CUTOFFS[prediction_var]
                                if cutoff_info['type'] == 'absolute':
                                    metadata_str = f"Error cutoff: >{cutoff_info['value']} {cutoff_info['units']}"
                                elif cutoff_info['type'] == 'binary':
                                    metadata_str = f"Error type: {cutoff_info['units']} misclassification"
                                metadata_row_count[prediction_var] += 1
                            
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
                                'rmse_original': rmse_orig_mean,
                                'rmse_corrected': rmse_corr_mean,
                                'rmse_mean_corrected': rmse_mc_mean,
                                'pct_improvement': pct_imp_mean,
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
                                    'pct_improvement_ci_lower': pct_imp_ci_lower,
                                    'pct_improvement_ci_upper': pct_imp_ci_upper
                                })
                            
                            all_results.append(row)
                            print(f"Processed: {prediction_var} {model} {arch} {region} {subregion} {lead_time}h - "
                                  f"Improvement: {pct_imp_mean:.1f}%, "
                                  f"Error rate orig: {pct_error_orig_mean:.1f}% -> corr: {pct_error_corr_mean:.1f}%" 
                                  if pct_error_orig_mean is not None else f"Improvement: {pct_imp_mean:.1f}%")
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    print(df.columns)
    
    # Save to CSV
    if output_csv_path is None:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_csv_path = os.path.join(dirs['globus'], 
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
]

    df = calculate_and_save_statistics(
        dirs=dirs,
        variable_configs=variable_configs,
        geographic_regions=["india", "ethiopia", "amazon", "british_columbia", "usa_south"],
        climate_regions=["tropical", "arid", "temperate"],  # Automatically bootstrapped
        train_start="2018-01-01",
        train_end="2021-12-31",
        test_start="2022-01-01",
        test_end="2022-12-31",
        models=["pangu", "ifs"],  # Both models
        nn_architectures=["mlp"],  # Can also include "unet"
        subregions=["2x2", "6x6", "10x10"],  # All subregions
        lead_times=[24, 120, 216],
        simultaneous=True,
        output_csv_path=f"{dirs['globus']}/forecast_improvement_stats.csv"
    )

if __name__ == "__main__":
    main()
