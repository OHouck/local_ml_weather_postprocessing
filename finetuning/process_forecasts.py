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
    
    # Format lead times
    lead_times_str = "leadtime_" + "_".join([str(lt) for lt in args.lead_time_hours]) + "h"

    output_path = f"{args.model_name}/{args.ground_truth_source}{region_str}/train_{training_vars_str}_test_{output_vars_str}_dim{subregion_str}_{lead_times_str}_{dates_str}_{model_str}.zarr"
    return output_path 


def calculate_and_save_statistics(
        dirs,
        train_start, train_end,
        test_start, test_end,
        models,  # Now accepts list of models
        training_output_vars,
        prediction_var,
        nn_architectures=["mlp"],
        regions=None,
        subregions=["2x2", "4x4", "10x10"],  # Now accepts list
        bootstrap=False,
        lead_times=None,
        simultaneous=False,
        output_csv_path=None
):
    """
    Calculate statistics for all forecast types and save to CSV.
    
    Parameters
    ----------
    dirs : dict
        Dictionary of directories
    train_start, train_end, test_start, test_end : str
        Date strings for train/test periods
    models : list
        List of model names (e.g., ['pangu', 'ifs'])
    training_output_vars : tuple
        Tuple of (training_vars, output_vars)
    prediction_var : str
        Variable to predict (e.g., '2m_temperature', '10m_wind_speed')
    nn_architectures : list
        List of architectures: ["mlp"], ["unet"], or ["mlp", "unet"]
    regions : list
        List of regions to analyze
    subregions : list
        List of patch sizes (e.g., ["2x2", "4x4", "10x10"])
    bootstrap : bool
        If True, uses bootstrap samples
    lead_times : list
        List of lead times in hours
    simultaneous : bool
        If True, use data from model that trained all lead times simultaneously
    output_csv_path : str
        Path to save the CSV file. If None, auto-generates path
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing all calculated statistics
    """
    
    # Parse training and output variables
    training_vars, output_vars = training_output_vars
    if not isinstance(training_vars, (list, tuple)):
        training_vars = [training_vars]
    if not isinstance(output_vars, (list, tuple)):
        output_vars = [output_vars]
    
    # Default values
    if regions is None:
        regions = ["india", "usa_south", "british_columbia", "amazon", "ethiopia"]
    if lead_times is None:
        lead_times = [24, 120, 216]
    
    # Storage for all results
    all_results = []
    
    # Process each combination
    for subregion in subregions:
        for region in regions:
            # Determine if it's a climate zone
            is_climate_zone = region in ["tropical", "arid", "temperate", "cold", "polar"]
            
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
                            print(f"No files found for {model} {arch} {region} {subregion} {lead_time}h")
                            continue
                        
                        if not bootstrap and len(files) > 1:
                            print(f"Warning: Multiple files found for {model} {arch} {region} {subregion} {lead_time}h")
                            files = files[:1]  # Use first file
                        
                        # Process each file (multiple for bootstrap)
                        file_results = []
                        ground_truth_values = []
                        
                        for idx, file_path in enumerate(files):
                            try:
                                ds = load_zarr_cached(file_path)
                                
                                # Extract data
                                ground_truth, original, corrected, mean_corrected = extract_forecast_data(
                                    ds, prediction_var, lead_time
                                )
                                
                                # Flatten arrays for statistics
                                gt_flat = ground_truth.values.flatten()
                                orig_flat = original.values.flatten()
                                corr_flat = corrected.values.flatten()
                                
                                # Remove NaN values
                                mask = ~(np.isnan(gt_flat) | np.isnan(orig_flat) | np.isnan(corr_flat))
                                gt_flat = gt_flat[mask]
                                orig_flat = orig_flat[mask]
                                corr_flat = corr_flat[mask]
                                
                                # Store ground truth for statistics
                                ground_truth_values.extend(gt_flat)
                                
                                # Calculate RMSE values
                                rmse_original = np.sqrt(np.mean((orig_flat - gt_flat)**2))
                                rmse_corrected = np.sqrt(np.mean((corr_flat - gt_flat)**2))
                                
                                # Calculate percent improvement
                                pct_improvement = (rmse_original - rmse_corrected) / rmse_original * 100
                                
                                # Calculate mean corrected RMSE if available
                                rmse_mean_corrected = None
                                pct_improvement_mean = None
                                if mean_corrected is not None:
                                    mc_flat = mean_corrected.values.flatten()[mask]
                                    rmse_mean_corrected = np.sqrt(np.mean((mc_flat - gt_flat)**2))
                                    pct_improvement_mean = (rmse_original - rmse_mean_corrected) / rmse_original * 100
                                
                                file_results.append({
                                    'rmse_original': rmse_original,
                                    'rmse_corrected': rmse_corrected,
                                    'rmse_mean_corrected': rmse_mean_corrected,
                                    'pct_improvement': pct_improvement,
                                    'pct_improvement_mean': pct_improvement_mean,
                                    'bootstrap_idx': idx if bootstrap else None
                                })
                                
                            except Exception as e:
                                print(f"Error processing {file_path}: {e}")
                                continue
                        
                        if not file_results:
                            continue
                        
                        # Calculate statistics across bootstrap samples if applicable
                        if bootstrap:
                            rmse_orig_values = [r['rmse_original'] for r in file_results]
                            rmse_corr_values = [r['rmse_corrected'] for r in file_results]
                            pct_imp_values = [r['pct_improvement'] for r in file_results]
                            
                            n = len(file_results)
                            
                            # Calculate means
                            rmse_orig_mean = np.mean(rmse_orig_values)
                            rmse_corr_mean = np.mean(rmse_corr_values)
                            pct_imp_mean = np.mean(pct_imp_values)
                            
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
                        
                        # Create row for results
                        row = {
                            'variable': prediction_var,
                            'model': model,
                            'architecture': arch,
                            'region': region,
                            'region_type': 'climate' if is_climate_zone else 'geographic',
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
                        print(f"Processed: {model} {arch} {region} {subregion} {lead_time}h - "
                              f"Improvement: {pct_imp_mean:.1f}%")
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Save to CSV
    if output_csv_path is None:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        output_csv_path = os.path.join(dirs['processed'], 
                                      f'forecast_statistics_{prediction_var}_{timestamp}.csv')
    
    df.to_csv(output_csv_path, index=False)
    print(f"\nStatistics saved to: {output_csv_path}")
    
    return df

def main():
    
    # create data outputs:
    dirs = setup_directories()

    geographic_stats_df = calculate_and_save_statistics(
        dirs=dirs,
        train_start="2018-01-01",
        train_end="2021-12-31",
        test_start="2022-01-01",
        test_end="2022-12-31",
        models=["pangu", "ifs"],  # Both models
        training_output_vars=("2m_temperature", "2m_temperature"),
        prediction_var="2m_temperature",
        nn_architectures=["mlp"],  # Can also include "unet"
        regions=["india", "ethiopia", "amazon", "british_columbia", "usa_south"],
        subregions=["2x2", "6x6", "10x10"],  # All subregions
        bootstrap=False,
        lead_times=[24, 120, 240],
        simultaneous=True,
        output_csv_path=f"{dirs['processed']}/geographic_regions_stats.csv"
    )

if __name__ == "__main__":
    main()
