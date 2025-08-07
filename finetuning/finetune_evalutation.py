import os
import re
import glob
import socket
import calendar
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from types import SimpleNamespace

from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats

import time


#######################
# Utility Functions
#######################
def setup_directories():
    # Determine root directory based on environment.
    nodename = socket.gethostname()
    if nodename == "oMac.local":
        root = os.path.expanduser("~/OneDrive - The University of Chicago/ai_weather_ag/data")
    else:
        raise Exception(f"Unknown environment, Please specify the root directory. \
                        Nodename found: {nodename}")

    dirs = {
        'root': root,
        'raw': os.path.join(root, "raw"),
        'processed': os.path.join(root, "processed"),
        'fig': os.path.join(root, "../figures/finetuning"),
        'input': os.path.join(root, "fine_tuning_output")  # adjusted input directory path
    }

    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs

def generate_output_path(args):
    region_str = f"{args.region}"
    subregion_str = f"{args.subregion}"
    dates_str = f"train{args.train_start}-{args.train_end}_test{args.test_start}-{args.test_end}"
    training_vars_str = "_".join(args.training_vars)
    output_vars_str = "_".join(args.output_vars)
    
    # Handle different nn architectures
    if args.nn_architecture == 'mlp':
        nn_str = "mlp"
    elif args.nn_architecture == 'unet':
        nn_str = "unet"
    else:
        raise ValueError(f"Unknown nn_architecture: {args.nn_architecture}")
    
    lead_time = f"leadtime_{args.lead_time_hours}"

    output_path = f"{args.model_name}/{region_str}/train_{training_vars_str}_test_{output_vars_str}_dim{subregion_str}_{lead_time}h_{dates_str}_{nn_str}.zarr"
    return output_path


def generate_lead_time_plots(
        dirs,
        train_start, train_end,
        test_start, test_end,
        model,
        training_output_vars,
        prediction_var,
        nn_architecture=["mlp"],  
        regions=None,
        subregion="4x4",
        bootstrap=False,
        plot_type="all",
        simultaneous=False
):
    """
    Generates a single plot showing percent improvement in RMSE by lead time for neural network
    and anomaly correction methods across all specified regions.
    
    Parameters
    ----------
    nn_architecture : list
        List of architectures to plot: ["mlp"], ["unet"], or ["mlp", "unet"]
    simultaneous : bool
        If True, use data from model that trained all lead times simultaneously.
    """
    
    # Parse training and output variables
    training_vars, output_vars = training_output_vars
    if not isinstance(training_vars, (list, tuple)):
        training_vars = [training_vars]
    if not isinstance(output_vars, (list, tuple)):
        output_vars = [output_vars]

    # Create string representations for file naming
    training_vars_str = "_".join(training_vars)
    output_vars_str = "_".join(output_vars)
    time_str = f"train{train_start}-{train_end}_test{test_start}-{test_end}"

    lead_times = [24, 120, 240]

    # Define colors for different regions
    region_colors = {
        'india': '#E69F00',          # Orange 
        'usa_south': '#56B4E9',      # Sky Blue
        'british_columbia': '#009E73',  # Teal Green
        'amazon': '#CC79A7',         # Pink/Magenta
        'ethiopia': '#D55E00',       # Reddish-Orange
    }

    climate_region_colors = {
        'tropical': '#228b22', # Green
        'arid': '#FFFF00',  # Yellow
        'temperate': '#90EE90' # Light Green
    }

    # Define markers for different models
    model_markers = {
        'pangu': 'o',  # Circle
        'ifs': '^'     # Triangle
    }
    
    # Define fill styles for different architectures
    architecture_fillstyles = {
        'mlp': 'full',   # Filled markers
        'unet': 'none'   # Hollow markers
    }
    
    correction_styles = {
        'nn': '-',     # Solid line
        'ano': '--'    # Dashed line for mean debiased
    }
    
    # Store all improvements for all regions
    all_improvements = {}
    
    # Process each region
    for region_idx, region in enumerate(regions):
        # Initialize storage for improvement percentages for this region
        improvements = {}
        for arch in nn_architecture:
            improvements[f'pangu_{arch}_nn'] = {}
            improvements[f'pangu_{arch}_ano'] = {}
            improvements[f'ifs_{arch}_nn'] = {}
            improvements[f'ifs_{arch}_ano'] = {}
        
        # Process each lead time
        for lt in lead_times:
            # Process each architecture
            for arch in nn_architecture:
                # Set up args for generate_output_path

                if simultaneous: # determine whether to pass list or single value
                    # convert to string for file naming
                    lead_time_hours = ""
                    for lead_time in lead_times:
                        lead_time_hours += f"{lead_time}"
                        if lead_time != lead_times[-1]:
                            lead_time_hours += "_"
                else:
                    lead_time_hours = lt

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
                    pangu_pattern = os.path.join(dirs['input'], generate_output_path(args).replace('.zarr', '*bs*.zarr'))
                    args.model_name = 'ifs'
                    ifs_pattern = os.path.join(dirs['input'], generate_output_path(args).replace('.zarr', '*bs*.zarr'))
                else:
                    pangu_pattern = os.path.join(dirs['input'], generate_output_path(args))
                    args.model_name = 'ifs'
                    ifs_pattern = os.path.join(dirs['input'], generate_output_path(args))
                
                # Process Pangu model files
                pangu_files = glob.glob(pangu_pattern)
                if not bootstrap and len(pangu_files) > 1:
                    raise ValueError(f"Multiple files found for lead time {lt}h: {pangu_files}")
                

                for idx, file_path in enumerate(pangu_files):
                    try:
                        ds = xr.open_zarr(file_path)
                        
                        # Extract data
                        ground_truth = ds[f"{prediction_var}_ground_truth_lt{lt}h"]
                        original = ds[f"{prediction_var}_original_lt{lt}h"]
                        nn_corrected = ds[f"{prediction_var}_corrected_lt{lt}h"]
                        ano_corrected = ds[f"{prediction_var}_mean_corrected_lt{lt}h"]

                        # Calculate RMSE
                        rmse_original = float(np.sqrt(((original - ground_truth) ** 2).mean().values))
                        rmse_nn = float(np.sqrt(((nn_corrected - ground_truth) ** 2).mean().values))
                        
                        # Calculate percent improvements
                        pct_improvement_nn = ((rmse_original - rmse_nn) / rmse_original * 100 
                                              if rmse_original != 0 else 0)
                        improvements[f'pangu_{arch}_nn'][(lt, idx)] = pct_improvement_nn
                        
                        if ano_corrected is not None and plot_type == "all":
                            rmse_ano = float(np.sqrt(((ano_corrected - ground_truth) ** 2).mean().values))
                            pct_improvement_ano = ((rmse_original - rmse_ano) / rmse_original * 100 
                                                   if rmse_original != 0 else 0)
                            improvements[f'pangu_{arch}_ano'][(lt, idx)] = pct_improvement_ano

                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
                        continue

                # Process IFS model files only if needed
                if plot_type in ["pangu_ifs_nn", "all"]:
                    ifs_files = glob.glob(ifs_pattern)
                    if ifs_files:  # Evaluates to True if the list is not empty
                        assert len(ifs_files) == len(pangu_files)
                        
                    for idx, file_path in enumerate(ifs_files):  
                        try:
                            ds = xr.open_zarr(file_path)
                            
                            # Extract data
                            ground_truth = ds[f"{prediction_var}_ground_truth_lt{lt}h"]
                            original = ds[f"{prediction_var}_original_lt{lt}h"]
                            nn_corrected = ds[f"{prediction_var}_corrected_lt{lt}h"]
                            ano_corrected = ds.get(f"{prediction_var}_mean_corrected_lt{lt}h", None)

                            # Calculate RMSE
                            rmse_original = float(np.sqrt(((original - ground_truth) ** 2).mean().values))
                            rmse_nn = float(np.sqrt(((nn_corrected - ground_truth) ** 2).mean().values))
                            
                            # Calculate percent improvements
                            pct_improvement_nn = ((rmse_original - rmse_nn) / rmse_original * 100 
                                                  if rmse_original != 0 else 0)
                            improvements[f'ifs_{arch}_nn'][(lt, idx)] = pct_improvement_nn
                            
                            if ano_corrected is not None and plot_type == "all":
                                rmse_ano = float(np.sqrt(((ano_corrected - ground_truth) ** 2).mean().values))
                                pct_improvement_ano = ((rmse_original - rmse_ano) / rmse_original * 100 
                                                       if rmse_original != 0 else 0)
                                improvements[f'ifs_{arch}_ano'][(lt, idx)] = pct_improvement_ano

                        except Exception as e:
                            print(f"Error processing IFS file {file_path}: {e}")
                            continue
        
        # Store improvements for this region
        all_improvements[region] = improvements

    # Create a single plot for all regions
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Track which lead times actually have data across all regions
    all_valid_lead_times = set()
    
    # Plot each region's data
    for region_idx, (region, improvements) in enumerate(all_improvements.items()):
        # Get color for this region
        if "temperate" in region or "tropical" in region or "arid" in region:
            color = climate_region_colors.get(region, '#1f77b4')
        else:
            color = region_colors[region]

        # Determine which improvement types to plot based on plot_type
        improvement_types_to_plot = []
        for arch in nn_architecture:
            if plot_type == "pangu_nn":
                improvement_types_to_plot.append(f'pangu_{arch}_nn')
            elif plot_type == "pangu_ifs_nn":
                improvement_types_to_plot.extend([f'pangu_{arch}_nn', f'ifs_{arch}_nn'])
            elif plot_type == "all":
                improvement_types_to_plot.extend([f'pangu_{arch}_nn', f'pangu_{arch}_ano', 
                                                  f'ifs_{arch}_nn', f'ifs_{arch}_ano'])
        
        # Process each improvement type for this region
        for imp_type in improvement_types_to_plot:
            imp_data = improvements.get(imp_type, {})
            if not imp_data:
                continue
            
            # Determine model, architecture, correction type, and get appropriate styles
            model_name = 'ifs' if 'ifs' in imp_type else 'pangu'
            architecture = 'unet' if 'unet' in imp_type else 'mlp'
            correction_type = 'nn' if 'nn' in imp_type else 'ano'
            
            marker = model_markers[model_name]
            fillstyle = architecture_fillstyles[architecture]
            linestyle = correction_styles[correction_type]
            line_alpha = 0.75  # Consistent alpha for all lines
            
            if bootstrap:
                # Aggregate bootstrap results
                aggregated = {}
                for lt in lead_times:
                    # Extract all values for this lead time
                    lt_values = [imp_data[(lead_time, idx)] 
                                 for (lead_time, idx) in imp_data.keys() 
                                 if lead_time == lt]
                    
                    if lt_values:
                        n = len(lt_values)
                        mean = np.mean(lt_values)
                        std = np.std(lt_values, ddof=1)
                        se = std / np.sqrt(n)
                        
                        # Calculate 95% confidence interval using t-distribution
                        alpha_ci = 0.05
                        t_crit = stats.t.ppf(1 - alpha_ci/2, df=n-1)
                        
                        aggregated[lt] = {
                            'mean': mean,
                            'ci_lower': mean - (t_crit * se),
                            'ci_upper': mean + (t_crit * se),
                            'count': n
                        }
                
                # Extract data for plotting
                valid_lead_times = sorted(aggregated.keys())
                if not valid_lead_times:
                    continue
                
                all_valid_lead_times.update(valid_lead_times)
                means = [aggregated[lt]['mean'] for lt in valid_lead_times]
                ci_lower = [aggregated[lt]['ci_lower'] for lt in valid_lead_times]
                ci_upper = [aggregated[lt]['ci_upper'] for lt in valid_lead_times]
                
                # Plot line with markers
                x_pos = [lead_times.index(lt) for lt in valid_lead_times]
                ax.plot(x_pos, means, 
                       marker=marker, 
                       fillstyle=fillstyle,
                       linestyle=linestyle, 
                       color=color, 
                       linewidth=2.5, 
                       markersize=15,
                       alpha=line_alpha,
                       zorder=3)
                
                # Add confidence interval bands
                ax.fill_between(x_pos, ci_lower, ci_upper,
                               color=color, 
                               alpha=0.1 * line_alpha, 
                               zorder=1)
                
            else:
                # Non-bootstrap case
                valid_data = [(lt, imp_data[(lt, 0)]) for lt in lead_times 
                             if (lt, 0) in imp_data]
                if not valid_data:
                    continue
                    
                valid_lead_times, values = zip(*valid_data)
                all_valid_lead_times.update(valid_lead_times)
                
                # Plot line
                x_pos = [lead_times.index(lt) for lt in valid_lead_times]
                ax.plot(x_pos, values, 
                       marker=marker, 
                       fillstyle=fillstyle,
                       linestyle=linestyle,
                       color=color, 
                       linewidth=2.5, 
                       markersize=15,
                       alpha=line_alpha,
                       zorder=3)
    
    # Set consistent axes limits for all plot types
    ax.set_ylim(-15, 45)  # Adjust these values based on your typical data range
    
    # Set x-axis to show all lead times but only label those with data
    ax.set_xticks(range(len(lead_times)))
    ax.set_xticklabels([f"{lt}h" if lt in all_valid_lead_times else "" 
                       for lt in lead_times])
    
    # Customize plot appearance
    ax.set_xlabel("Forecast Lead Time", fontsize=15)
    ax.set_ylabel("RMSE Improvement (%)", fontsize=15)
    
    # Add horizontal line at y=0 for reference
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Title
    regions_str = ", ".join(regions)
    arch_str = "/".join([a.upper() for a in nn_architecture])
    title_parts = [f"RMSE Improvement for {prediction_var.replace('_', ' ').title()} ({arch_str})"]
    title_parts.append(f"Regions: {regions_str}, Patch Size: {subregion}")
    if bootstrap:
        title_parts[0] += " (with 95% CI)"
    ax.set_title('\n'.join(title_parts), fontsize=14, pad=15)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Create custom legend with four groups
    from matplotlib.lines import Line2D
    
    # Region legend entries
    region_handles = []
    for region in regions:
        if "temperate" in region or "tropical" in region or "arid" in region:
            color = climate_region_colors.get(region, '#1f77b4')
        else:
            color = region_colors.get(region, '#1f77b4')
        region_handles.append(Line2D([0], [0], color=color, linewidth=3, 
                                   label=region.replace('_', ' ').title()))
    
    # Model legend entries
    model_handles = []
    if plot_type in ["pangu_nn", "all"]:
        model_handles.append(Line2D([0], [0], color='black', marker='o', linestyle='none',
                                   markersize=15, label='Pangu'))
    if plot_type in ["pangu_ifs_nn", "all"]:
        model_handles.append(Line2D([0], [0], color='black', marker='^', linestyle='none',
                                   markersize=15, label='IFS'))
    
    # Architecture legend entries
    arch_handles = []
    for arch in nn_architecture:
        fillstyle = architecture_fillstyles[arch]
        arch_handles.append(Line2D([0], [0], color='black', marker='o', fillstyle=fillstyle,
                                  markersize=15, linestyle='none', label=arch.upper()))
    
    # Correction type legend entries
    correction_handles = []
    correction_handles.append(Line2D([0], [0], color='black', linestyle='-', linewidth=2,
                                    label='Neural Network'))
    if plot_type == "all":
        correction_handles.append(Line2D([0], [0], color='black', linestyle='--', linewidth=2,
                                        label='Mean Debiased'))
    
    # Position legends
    legend1 = ax.legend(handles=region_handles, title="Region", 
                        loc='lower right', bbox_to_anchor=(1, 0), fontsize=12)
    legend2 = ax.legend(handles=model_handles, title="Model", 
                        loc='lower right', bbox_to_anchor=(1, 0.45), fontsize=12)
    legend3 = ax.legend(handles=arch_handles, title="Architecture", 
                        loc='lower right', bbox_to_anchor=(1, 0.3), fontsize=12)
    legend4 = ax.legend(handles=correction_handles, title="Correction", 
                        loc='lower right', bbox_to_anchor=(1, 0.2), fontsize=12)
    
    # Add all legends to the plot
    ax.add_artist(legend1)
    ax.add_artist(legend2)
    ax.add_artist(legend3)
    
    # Style all legends consistently
    for legend in [legend1, legend2, legend3, legend4]:
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.95)
        legend.get_frame().set_edgecolor('gray')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    out_folder = os.path.join(dirs["fig"], model, "lead_time", "multi_region", subregion)
    os.makedirs(out_folder, exist_ok=True)
    
    bootstrap_suffix = "_bootstrap" if bootstrap else ""
    arch_suffix = "_".join(nn_architecture)
    if "arid" in regions or "tropical" in regions or "temperate" in regions:
        region_type = "climate_zones"
    else:
        region_type = "geographic"
    fname = (f"leadtime_improvement_{prediction_var}_trainedwith_{training_vars_str}_"
            f"{region_type}_{plot_type}_{arch_suffix}{bootstrap_suffix}.png")
    save_path = os.path.join(out_folder, fname)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Multi-region lead-time improvement plot ({plot_type}) saved to: {save_path}")

def diagnose_zarr_performance(path, lead_times=[24, 120, 240]):
    """
    Diagnostic function to understand why different lead times have different performance.
    """
    import time
    
    print(f"Diagnosing Zarr performance for: {path}")
    
    try:
        with xr.open_zarr(path) as ds:
            print(f"Dataset shape: {ds.dims}")
            print(f"Available variables: {list(ds.data_vars.keys())}")
            
            for lead_time in lead_times:
                var_name = f"2m_temperature_original_lt{lead_time}h"
                if var_name in ds:
                    var = ds[var_name]
                    print(f"\n=== Lead Time {lead_time}h ===")
                    print(f"Variable shape: {var.shape}")
                    print(f"Variable chunks: {var.chunks}")
                    print(f"Chunk sizes: {var.chunksizes}")
                    
                    # Get number of chunks correctly
                    if hasattr(var.data, 'nchunks'):
                        print(f"Number of chunks: {var.data.nchunks}")
                    elif hasattr(var.data, 'chunks'):
                        import numpy as np
                        nchunks = np.prod([len(c) for c in var.data.chunks])
                        print(f"Number of chunks: {nchunks}")
                    
                    # Test loading speed
                    start_time = time.time()
                    _ = var.load()  # Force load into memory
                    load_time = time.time() - start_time
                    print(f"Load time: {load_time:.3f} seconds")
                    
                    # Memory usage
                    memory_usage = var.nbytes / (1024**2)  # MB
                    print(f"Memory usage: {memory_usage:.2f} MB")
                    
                    # Dask array info
                    if hasattr(var.data, 'chunks'):
                        print(f"Dask chunks: {var.data.chunks}")
                        print(f"Chunk dtype: {var.data.dtype}")
                        
                        # Calculate chunk efficiency
                        total_elements = var.size
                        chunks_per_dim = [len(c) for c in var.data.chunks]
                        total_chunks = np.prod(chunks_per_dim)
                        avg_elements_per_chunk = total_elements / total_chunks
                        print(f"Average elements per chunk: {avg_elements_per_chunk:.1f}")
                        print(f"Chunks per dimension: {chunks_per_dim}")
                else:
                    print(f"Variable {var_name} not found")
                    
    except Exception as e:
        print(f"Error diagnosing {path}: {e}")
        import traceback
        traceback.print_exc()

def generate_subregion_comparison_plots(dirs, train_start, train_end, test_start,
                                        test_end, model, training_output_vars,
                                        prediction_var, nn_architecture=["mlp"],
                                        lead_time=None, simultaneous=False):
    """
    Creates plot showing how RMSE changes when trained on different sizes of subregions.
    """
    input_folder = dirs['input']
    training_vars, output_vars = training_output_vars
    training_vars = training_vars if isinstance(training_vars, (list,tuple)) else [training_vars]
    output_vars   = output_vars   if isinstance(output_vars,   (list,tuple)) else [output_vars]

    valid_lead_times = [24, 120, 240]
    if lead_time not in valid_lead_times:
        raise ValueError(f"Invalid lead time: {lead_time}. Must be one of {valid_lead_times}.")
    
    regions   = ["usa_south", "british_columbia", "ethiopia", "amazon", "india"]
    subregions = ["2x2","6x6","10x10"]
    degrees    = [int(s.split('x')[0]) for s in subregions]

    # Store improvements for both models and architectures
    improvements = {}
    for arch in nn_architecture:
        improvements[arch] = {
            'pangu': {r: [] for r in regions},
            'ifs': {r: [] for r in regions}
        }

    # Set leadtime arg for file naming
    if simultaneous:
        lead_time_hours = "_".join(str(lt) for lt in valid_lead_times)
    else:
        lead_time_hours = lead_time

    for arch in nn_architecture:
        for region in regions:
            print(f"Processing region: {region}, architecture: {arch}")
            
            # Cache central bounds - compute once per region
            central_bounds = None
            
            for model_name in ['pangu', 'ifs']:
                print(f"  Loading {model_name} data for {region}...")
                
                for sub in subregions:
                    args = SimpleNamespace(
                        model_name=model_name, region=region, subregion=sub,
                        train_start=train_start, train_end=train_end,
                        test_start=test_start, test_end=test_end,
                        training_vars=training_vars, output_vars=output_vars,
                        lead_time_hours=lead_time_hours,
                        nn_architecture=arch
                    )
                    
                    path = os.path.join(input_folder, generate_output_path(args))
                    
                    try:
                        with xr.open_zarr(path, chunks=None) as ds:
                            # Extract central bounds on first successful load
                            if central_bounds is None:
                                if sub == "2x2":
                                    central_bounds = {
                                        'lat_min': float(ds.latitude.min()),
                                        'lat_max': float(ds.latitude.max()),
                                        'lon_min': float(ds.longitude.min()),
                                        'lon_max': float(ds.longitude.max())
                                    }
                                else:
                                    # For larger subregions, extract central 2x2
                                    lat_center = float((ds.latitude.min() + ds.latitude.max()) / 2)
                                    lon_center = float((ds.longitude.min() + ds.longitude.max()) / 2)
                                    central_bounds = {
                                        'lat_min': lat_center - 1.0,
                                        'lat_max': lat_center + 1.0,
                                        'lon_min': lon_center - 1.0,
                                        'lon_max': lon_center + 1.0
                                    }
                            
                            # OPTIMIZATION 3: Load all required variables at once
                            required_vars = [
                                f"{prediction_var}_ground_truth_lt{lead_time}h",
                                f"{prediction_var}_original_lt{lead_time}h",
                                f"{prediction_var}_corrected_lt{lead_time}h"
                            ]
                            
                            # Check if all required variables exist
                            missing_vars = [var for var in required_vars if var not in ds]
                            if missing_vars:
                                print(f"    Missing variables for {model_name}, {sub}: {missing_vars}")
                                continue
                            
                            # OPTIMIZATION 4: Single spatial slice operation
                            ds_subset = ds[required_vars].sel(
                                latitude=slice(central_bounds['lat_min'], central_bounds['lat_max']),
                                longitude=slice(central_bounds['lon_min'], central_bounds['lon_max'])
                            )
                            
                            print(f"    Loading {sub} data...")
                            start_time = time.time()
                            data_loaded = ds_subset.load()
                            load_time = time.time() - start_time
                            print(f"    Load time for {sub}: {load_time:.2f}s")
                            
                            try:
                                gt_n = data_loaded[f"{prediction_var}_ground_truth_lt{lead_time}h"]
                                orig_n = data_loaded[f"{prediction_var}_original_lt{lead_time}h"]
                                corr_n = data_loaded[f"{prediction_var}_corrected_lt{lead_time}h"]

                                # Fast numpy operations on loaded arrays
                                rmse_orig = float(np.sqrt(((orig_n.values - gt_n.values)**2).mean()))
                                rmse_corr = float(np.sqrt(((corr_n.values - gt_n.values)**2).mean()))
                                pct_improvement = (rmse_orig - rmse_corr) / rmse_orig * 100
                                
                                size = int(sub.split('x')[0])
                                improvements[arch][model_name][region].append((size, pct_improvement))
                                print(f"    {sub}: {pct_improvement:.2f}% improvement")
                                
                            except Exception as e:
                                print(f"    Error computing metrics for {model_name}, {region}, {sub}: {e}")
                        
                    except Exception as e:
                        print(f"    {model_name} data not found for {region}, {sub}: {e}")

    print("Improvements collected:", improvements)
    
    # Plotting code remains the same...
    region_colors = plt.get_cmap('Set1')
    region_color_map = {region: region_colors(i) for i, region in enumerate(regions)}
    ls_map = {24: 'solid', 120: 'solid', 240: 'solid'} # change if I want to plot multiple lead times
    
    model_markers = {'pangu': 'o', 'ifs': '^'}
    arch_fillstyles = {'mlp': 'full', 'unet': 'none'}
    
    plt.figure(figsize=(12, 7))
    
    for arch in nn_architecture:
        for model_name in ['pangu', 'ifs']:
            for region in regions:
                data = sorted(improvements[arch][model_name][region], key=lambda x: x[0])
                if not data:
                    continue
                    
                sizes, imps = zip(*data)
                
                region_label = region.replace('_', ' ').title()
                if len(nn_architecture) > 1:
                    label = f"{region_label} {model_name.upper()} {arch.upper()} ({lead_time}h)"
                else:
                    label = f"{region_label} {model_name.upper()} ({lead_time}h)"
                
                plt.ylim(-25, 30)
                plt.plot(sizes, imps, 
                        marker=model_markers[model_name],
                        fillstyle=arch_fillstyles[arch],
                        color=region_color_map[region], 
                        linestyle=ls_map[lead_time],
                        label=label,
                        linewidth=2, markersize=15)
    
    plt.xticks(degrees, subregions)
    plt.xlabel("Patch size (degrees)", fontsize=15)
    plt.ylabel("RMSE % improvement\n(original − corrected)", fontsize=15)
    
    arch_str = "/".join([a.upper() for a in nn_architecture])
    title = f"RMSE % Improvement by Patch Size ({arch_str}) - {lead_time}h Lead Time"
    
    plt.title(title, fontsize=15)
    plt.grid(True, alpha=0.3)
    
    # Legend creation remains the same...
    from matplotlib.lines import Line2D
    
    region_handles = [Line2D([0], [0], color=region_color_map[region], linewidth=3, 
                            label=region.replace('_', ' ').title()) for region in regions]
    
    model_handles = [
        Line2D([0], [0], color='black', marker='o', linestyle='none', markersize=12, label='Pangu'),
        Line2D([0], [0], color='black', marker='^', linestyle='none', markersize=12, label='IFS')
    ]
    
    if len(nn_architecture) > 1:
        arch_handles = [Line2D([0], [0], color='black', marker='o', fillstyle=arch_fillstyles[arch],
                              markersize=12, linestyle='none', label=arch.upper()) 
                       for arch in nn_architecture]
    
    legend1 = plt.legend(handles=region_handles, title="Region", 
                        loc='lower right', bbox_to_anchor=(1, 0))
    legend2 = plt.legend(handles=model_handles, title="Model", 
                        loc='lower right', bbox_to_anchor=(1, 0.35))
    
    if len(nn_architecture) > 1:
        legend3 = plt.legend(handles=arch_handles, title="Architecture", 
                            loc='lower right', bbox_to_anchor=(1, 0.55))
        plt.gca().add_artist(legend3)
    
    plt.gca().add_artist(legend1)
    plt.gca().add_artist(legend2)
    
    plt.tight_layout()

    # Save figure
    out_folder = os.path.join(dirs["fig"], model, "subregion")
    os.makedirs(out_folder, exist_ok=True)
    
    lead_times_suffix = f"_{lead_time}h"
    arch_suffix = "_".join(nn_architecture)
    fname = f"subregion_rmse_improvement_lt{lead_times_suffix}_{'_'.join(training_vars)}_{prediction_var}_{arch_suffix}.png"
    plt.savefig(os.path.join(out_folder, fname), dpi=150, bbox_inches='tight')
    plt.close()


def generate_map_plots(
        dirs,
        train_start, train_end,
        test_start, test_end,
        model,
        training_output_vars,
        prediction_var,
        nn_architecture="mlp",  
        region="usa_south",
        subregion="2x2",
        lead_time=24,
        simultaneous=False
):
    """
    Generates a figure with 2 maps: original forecast RMSE and percent improvement in RMSE.
    Map extent is always 10x10 degrees, with the subregion determining how much is filled.
    
    Parameters
    ----------
    nn_architecture : str
        Architecture to use: "mlp" or "unet"
    """
    
    # Parse training and output variables
    training_vars, output_vars = training_output_vars
    if not isinstance(training_vars, (list, tuple)):
        training_vars = [training_vars]
    if not isinstance(output_vars, (list, tuple)):
        output_vars = [output_vars]

    # Create string representations for file naming
    training_vars_str = "_".join(training_vars)
    output_vars_str = "_".join(output_vars)
    time_str = f"train{train_start}-{train_end}_test{test_start}-{test_end}"

    valid_lead_times = [24, 120, 240]
    # Set leadtime arg for file naming
    if simultaneous:
        lead_time_hours = "_".join(str(lt) for lt in valid_lead_times)
    else:
        lead_time_hours = lead_time

    # Set up args for generate_output_path
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
        nn_architecture=nn_architecture
    )
    

    # Construct file path
    file_path = os.path.join(dirs['input'], generate_output_path(args))

    # Skip if region is "pixel" (no maps for pixel)
    if region == "pixel":
        print(f"Skipping map generation for region 'pixel'")
        return

    try:
        # Load the forecast data
        ds = xr.open_zarr(file_path)
        
        # First, get the 10x10 degree extent for this region
        # We need to load the 10x10 file to get the full extent
        args_10x10 = SimpleNamespace(**vars(args))
        args_10x10.subregion = "10x10"
        path_10x10 = os.path.join(dirs['input'], generate_output_path(args_10x10))
        
        try:
            with xr.open_zarr(path_10x10) as ds_10x10:
                lat_min_10x10 = float(ds_10x10.latitude.min())
                lat_max_10x10 = float(ds_10x10.latitude.max())
                lon_min_10x10 = float(ds_10x10.longitude.min())
                lon_max_10x10 = float(ds_10x10.longitude.max())
        except Exception as e:
            print(f"Warning: Could not load 10x10 extent, using current extent: {e}")
            lat_min_10x10 = float(ds.latitude.min())
            lat_max_10x10 = float(ds.latitude.max())
            lon_min_10x10 = float(ds.longitude.min())
            lon_max_10x10 = float(ds.longitude.max())
        
        # Extract data arrays
        ground_truth = ds[f"{prediction_var}_ground_truth_lt{lead_time}h"]
        fc_original = ds[f"{prediction_var}_original_lt{lead_time}h"]
        fc_corrected = ds[f"{prediction_var}_corrected_lt{lead_time}h"]
        
        # Calculate RMSE for original and corrected forecasts
        mse_spatial_orig = ((fc_original - ground_truth) ** 2).mean(dim="time")
        mse_spatial_corr = ((fc_corrected - ground_truth) ** 2).mean(dim="time")
        
        # Convert MSE to RMSE
        rmse_spatial_orig = np.sqrt(mse_spatial_orig)
        rmse_spatial_corr = np.sqrt(mse_spatial_corr)
        
        # Calculate percent improvement
        pct_improvement = ((rmse_spatial_orig - rmse_spatial_corr) / rmse_spatial_orig * 100)
        
        # Create figure with 2 subplots - optimized for 16:9 slides
        fig = plt.figure(figsize=(14, 4.5))
        
        # Use GridSpec for better control over subplot spacing
        gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.1, hspace=0.5)
        
        # First subplot: Original forecast RMSE
        ax1 = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
        
        # Set the 10x10 degree extent
        ax1.set_extent([lon_min_10x10, lon_max_10x10, lat_min_10x10, lat_max_10x10], 
                       crs=ccrs.PlateCarree())
        
        vmin_orig = float(rmse_spatial_orig.min().values)
        vmax_orig = float(rmse_spatial_orig.max().values)
        
        im1 = rmse_spatial_orig.plot(
            ax=ax1, 
            cmap='viridis', 
            add_colorbar=False,
            vmin=vmin_orig, 
            vmax=vmax_orig
        )
        
        # Add colorbar for first subplot with better positioning
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes("right", size="4%", pad=0.05, axes_class=plt.Axes)
        cbar1 = plt.colorbar(im1, cax=cax1)
        cbar1.set_label('RMSE', fontsize=10)
        cbar1.ax.tick_params(labelsize=9)
        
        ax1.set_title(f"Original {model.upper()} Forecast RMSE\n{prediction_var.replace('_', ' ').title()}", 
                      fontsize=13, pad=2)
        ax1.coastlines(resolution='50m', linewidth=0.5)
        ax1.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
        ax1.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.8, edgecolor='black')
        
        # Add state/province borders based on region
        if region in ['usa_south', 'british_columbia']:
            ax1.add_feature(cfeature.STATES, linestyle=':', linewidth=0.5, edgecolor='gray')
        elif region == 'india':
            ax1.add_feature(cfeature.STATES, linestyle=':', linewidth=0.5, edgecolor='gray')
        elif region == 'amazon':
            ax1.add_feature(cfeature.STATES, linestyle=':', linewidth=0.5, edgecolor='gray')
        
        # Customize gridlines to prevent overlap
        gl1 = ax1.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, 
                           linewidth=0.5, alpha=0.5)
        gl1.right_labels = False
        gl1.top_labels = False
        gl1.xlabel_style = {'size': 9}
        gl1.ylabel_style = {'size': 9}
        
        # Second subplot: Percent improvement
        ax2 = fig.add_subplot(gs[1], projection=ccrs.PlateCarree())
        
        # Set the same 10x10 degree extent
        ax2.set_extent([lon_min_10x10, lon_max_10x10, lat_min_10x10, lat_max_10x10], 
                       crs=ccrs.PlateCarree())
        
        # Use diverging colormap centered at 0
        vmin_pct = float(pct_improvement.min().values)
        vmax_pct = float(pct_improvement.max().values)
        
        # Ensure colormap is centered at 0
        vmax_abs = max(abs(vmin_pct), abs(vmax_pct))
        norm = TwoSlopeNorm(vmin=-vmax_abs, vcenter=0, vmax=vmax_abs)
        
        im2 = pct_improvement.plot(
            ax=ax2, 
            cmap='RdBu', 
            add_colorbar=False,
            norm=norm
        )
        
        # Add colorbar for second subplot with better positioning
        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="4%", pad=0.05, axes_class=plt.Axes)
        cbar2 = plt.colorbar(im2, cax=cax2)
        cbar2.set_label('Improvement (%)', fontsize=10)
        cbar2.ax.tick_params(labelsize=9)
        
        ax2.set_title(f"RMSE Percent Improvement ({nn_architecture.upper()} Corrected)\n{prediction_var.replace('_', ' ').title()}", 
                      fontsize=13, pad=2)
        ax2.coastlines(resolution='50m', linewidth=0.5)
        ax2.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
        ax2.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.8, edgecolor='black')
        
        # Add state/province borders based on region
        if region in ['usa_south', 'british_columbia']:
            ax2.add_feature(cfeature.STATES, linestyle=':', linewidth=0.5, edgecolor='gray')
        elif region == 'india':
            ax2.add_feature(cfeature.STATES, linestyle=':', linewidth=0.5, edgecolor='gray')
        elif region == 'amazon':
            ax2.add_feature(cfeature.STATES, linestyle=':', linewidth=0.5, edgecolor='gray')
        
        # Customize gridlines to prevent overlap
        gl2 = ax2.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                           linewidth=0.5, alpha=0.5)
        gl2.right_labels = False
        gl2.top_labels = False
        gl2.left_labels = False
        gl2.xlabel_style = {'size': 9}
        gl2.ylabel_style = {'size': 9}
        
        # Add overall title - reduce vertical spacing
        fig.suptitle(f"{region.replace('_', ' ').title()} - {lead_time}h Lead Time - Patch Size: {subregion}", 
                     fontsize=13, y=1.00)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        out_folder = os.path.join(dirs["fig"], model, "maps", region, subregion)
        os.makedirs(out_folder, exist_ok=True)
        
        fname = f"rmse_maps_{prediction_var}_trainedwith_{training_vars_str}_{lead_time}h_{nn_architecture}.png"
        save_path = os.path.join(out_folder, fname)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Map plots saved to: {save_path}")
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")


def generate_time_series_plots(
        dirs,
        train_start, train_end,
        test_start, test_end,
        model,
        training_output_vars,
        prediction_var,
        nn_architecture="mlp",  
        region="usa_south",
        subregion="2x2",
        lead_time=24,
        simultaneous=False
):
    """
    Generates a single bar plot showing monthly RMSE for original and corrected forecasts
    for both the main model (e.g., pangu) and IFS.
    
    Parameters
    ----------
    nn_architecture : str
        Architecture to use: "mlp" or "unet"
    """
    
    # Parse training and output variables
    training_vars, output_vars = training_output_vars
    if not isinstance(training_vars, (list, tuple)):
        training_vars = [training_vars]
    if not isinstance(output_vars, (list, tuple)):
        output_vars = [output_vars]

    # Create string representations for file naming
    training_vars_str = "_".join(training_vars)
    output_vars_str = "_".join(output_vars)
    time_str = f"train{train_start}-{train_end}_test{test_start}-{test_end}"

    valid_lead_times = [24, 120, 240]
    if simultaneous:
        lead_time_hours = "_".join(str(lt) for lt in valid_lead_times)
    else:
        lead_time_hours = lead_time

    # Set up args for generate_output_path
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
        nn_architecture=nn_architecture
    )

    # Construct file paths for main model and IFS
    model_file_path = os.path.join(dirs['input'], generate_output_path(args))
    
    args.model_name = 'ifs'
    ifs_file_path = os.path.join(dirs['input'], generate_output_path(args))

    # Initialize storage for monthly RMSE values
    months = []
    model_rmse_orig = []
    model_rmse_corr = []
    ifs_rmse_orig = []
    ifs_rmse_corr = []
    
    try:
        # Load main model data
        ds_model = xr.open_zarr(model_file_path)
        
        # Extract data arrays
        ground_truth = ds_model[f"{prediction_var}_ground_truth_lt{lead_time}h"]
        fc_original = ds_model[f"{prediction_var}_original_lt{lead_time}h"]
        fc_corrected = ds_model[f"{prediction_var}_corrected_lt{lead_time}h"]
        
        # Calculate monthly RMSE for main model
        # First compute MSE, then take mean over spatial dimensions, then group by month
        mse_orig = ((fc_original - ground_truth) ** 2)
        mse_corr = ((fc_corrected - ground_truth) ** 2)
        
        # Average over spatial dimensions first
        mse_orig_spatial_mean = mse_orig.mean(dim=['latitude', 'longitude'])
        mse_corr_spatial_mean = mse_corr.mean(dim=['latitude', 'longitude'])
        
        # Then group by month and average over time
        mse_orig_monthly = mse_orig_spatial_mean.groupby("time.month").mean(dim="time")
        mse_corr_monthly = mse_corr_spatial_mean.groupby("time.month").mean(dim="time")
        
        # Convert to RMSE
        rmse_orig_monthly = np.sqrt(mse_orig_monthly)
        rmse_corr_monthly = np.sqrt(mse_corr_monthly)
        
        # Get month names and values
        months = [calendar.month_name[i] for i in rmse_orig_monthly['month'].values]
        model_rmse_orig = float(rmse_orig_monthly.values) if rmse_orig_monthly.values.ndim == 0 else rmse_orig_monthly.values.flatten()
        model_rmse_corr = float(rmse_corr_monthly.values) if rmse_corr_monthly.values.ndim == 0 else rmse_corr_monthly.values.flatten()
        
        # Ensure we have the correct number of values
        if len(model_rmse_orig) != 12 or len(model_rmse_corr) != 12:
            print(f"Warning: Expected 12 monthly values, but got {len(model_rmse_orig)} for original and {len(model_rmse_corr)} for corrected")
            print(f"Shape of rmse_orig_monthly: {rmse_orig_monthly.shape}")
            print(f"Dimensions: {rmse_orig_monthly.dims}")
        
    except Exception as e:
        print(f"Error loading main model data from {model_file_path}: {e}")
        return
    
    # Try to load IFS data
    has_ifs_data = False
    try:
        ds_ifs = xr.open_zarr(ifs_file_path)
        
        # Extract IFS data arrays
        ifs_ground_truth = ds_ifs[f"{prediction_var}_ground_truth_lt{lead_time}h"]
        ifs_fc_original = ds_ifs[f"{prediction_var}_original_lt{lead_time}h"]
        ifs_fc_corrected = ds_ifs[f"{prediction_var}_corrected_lt{lead_time}h"]
        
        # Calculate monthly RMSE for IFS
        # First compute MSE, then take mean over spatial dimensions, then group by month
        ifs_mse_orig = ((ifs_fc_original - ifs_ground_truth) ** 2)
        ifs_mse_corr = ((ifs_fc_corrected - ifs_ground_truth) ** 2)
        
        # Average over spatial dimensions first
        ifs_mse_orig_spatial_mean = ifs_mse_orig.mean(dim=['latitude', 'longitude'])
        ifs_mse_corr_spatial_mean = ifs_mse_corr.mean(dim=['latitude', 'longitude'])
        
        # Then group by month and average over time
        ifs_mse_orig_monthly = ifs_mse_orig_spatial_mean.groupby("time.month").mean(dim="time")
        ifs_mse_corr_monthly = ifs_mse_corr_spatial_mean.groupby("time.month").mean(dim="time")
        
        # Convert to RMSE
        ifs_rmse_orig_monthly = np.sqrt(ifs_mse_orig_monthly)
        ifs_rmse_corr_monthly = np.sqrt(ifs_mse_corr_monthly)
        
        # Get values
        ifs_rmse_orig = float(ifs_rmse_orig_monthly.values) if ifs_rmse_orig_monthly.values.ndim == 0 else ifs_rmse_orig_monthly.values.flatten()
        ifs_rmse_corr = float(ifs_rmse_corr_monthly.values) if ifs_rmse_corr_monthly.values.ndim == 0 else ifs_rmse_corr_monthly.values.flatten()
        has_ifs_data = True
        
    except Exception as e:
        print(f"IFS data not available for {region}: {e}")
    
    # Create the bar plot - optimized for 16:9 slides
    fig, ax = plt.subplots(figsize=(14, 4.5))
    
    # Set up bar positions
    x = np.arange(len(months))
    bar_width = 0.35
    
    # Plot main model bars (overlapping with transparency)
    ax.bar(x - bar_width/2, model_rmse_orig, bar_width, 
           color='blue', alpha=0.5, label=f'{model.upper()} Original')
    ax.bar(x - bar_width/2, model_rmse_corr, bar_width, 
           color='red', alpha=0.5, label=f'{model.upper()} Corrected ({nn_architecture.upper()})')
    
    # Plot IFS bars if available (offset to the right)
    if has_ifs_data:
        ifs_bar_width = bar_width * 0.5  # Make IFS bars narrower
        offset = bar_width/2 + 0.05  # Small gap between model and IFS bars
        
        ax.bar(x + offset, ifs_rmse_orig, ifs_bar_width, 
               color='darkblue', alpha=0.75, label='IFS Baseline')
        ax.bar(x + offset + ifs_bar_width, ifs_rmse_corr, ifs_bar_width, 
               color='#ADD8E6', alpha=0.75, label=f'IFS Corrected ({nn_architecture.upper()})')
    
    # Customize plot
    ax.set_xlabel('Month', fontsize=15)
    ax.set_ylabel('RMSE', fontsize=15)
    ax.set_title(f'Monthly RMSE Comparison - {region.replace("_", " ").title()}\n'
                 f'{prediction_var.replace("_", " ").title()} - {lead_time}h Lead Time - Patch Size: {subregion}',
                 fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    out_folder = os.path.join(dirs["fig"], model, "time_series", region, subregion)
    os.makedirs(out_folder, exist_ok=True)
    
    fname = f"rmse_monthly_{prediction_var}_trainedwith_{training_vars_str}_{lead_time}h_{nn_architecture}.png"
    save_path = os.path.join(out_folder, fname)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Time series plot saved to: {save_path}")

import os
import numpy as np
import pandas as pd
import xarray as xr
from types import SimpleNamespace


def generate_summary_stat_table(
        dirs,
        train_start, train_end,
        test_start, test_end,
        model,
        training_output_vars,
        prediction_var,
        nn_architecture="mlp",
        regions=None,
        subregion="2x2",
        bootstrap=False,
        lead_times=None,
        simultaneous=False
):
    """
    Generates a single comprehensive summary statistics table for a specific variable.
    
    The table includes:
    - Region and lead time columns
    - Mean ground truth with standard deviation in parentheses
    - RMSE of original and corrected forecasts
    - Percent improvement
    
    Parameters
    ----------
    dirs : dict
        Dictionary of directories
    train_start, train_end, test_start, test_end : str
        Date strings for train/test periods
    model : str
        Model name (e.g., 'pangu')
    variables_config : tuple
        Tuple of (training_vars, output_vars, prediction_var)
    nn_architecture : str
        Architecture type: "mlp" or "unet"
    regions : list
        List of regions to analyze. If None, uses default regions
    subregion : str
        Patch size (e.g., "2x2", "10x10")
    bootstrap : bool
        If True, uses bootstrap samples; otherwise, uses full data
    lead_times : list of int
        Lead times in hours. If None, uses [24, 72, 168]
    simultaneous : bool
        If True, processes all lead times simultaneously; otherwise, processes sequentially
    
    Returns
    -------
    pandas.DataFrame
        DataFrame containing all summary statistics
    """
    
    # Extract variables from config
    training_vars, output_vars = training_output_vars
    
    # Ensure variables are lists
    if not isinstance(training_vars, (list, tuple)):
        training_vars = [training_vars]
    if not isinstance(output_vars, (list, tuple)):
        output_vars = [output_vars]
    
    # Default regions if not specified
    if regions is None:
        regions = ["amazon", "india", "usa_south", "british_columbia", "ethiopia"]
    
    # Default lead times if not specified
    if lead_times is None:
        lead_times = [24, 120, 240]
    
    # Storage for all results
    all_rows = []
    region_ground_truth_stats = {}
    
    # Process each region
    for region in regions:
        region_ground_truth_values = []
        
        for lead_time in lead_times:
            # Set up args for generate_output_path

            if simultaneous: # determine whether to pass list or single value
                # convert to string for file naming
                lead_time_hours = ""
                for lt in lead_times:
                    lead_time_hours += f"{lt}"
                    if lt != lead_times[-1]:
                        lead_time_hours += "_"
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
                nn_architecture=nn_architecture
            )
            
            # Construct file path
            # Construct file paths
            if bootstrap:
                file_pattern = os.path.join(dirs['input'], generate_output_path(args).replace('.zarr', '*bs*.zarr'))
            else:
                file_pattern = os.path.join(dirs['input'], generate_output_path(args))
            
            files = glob.glob(file_pattern)
            if not bootstrap and len(files) > 1:
                raise ValueError(f"Error: Multiple files found for {region} with lead time {lead_time}h. Using first file.")

            # create empty dict to store values
            values = {'rmse_orig': np.nan,
                      'rmse_change': np.nan,
                      'rmse_imp_pct': np.nan}

            for idx, file_path in enumerate(files):
                # Load the data
                ds = xr.open_zarr(file_path)
                
                # Extract data arrays for the specific lead time
                ground_truth = ds[f"{prediction_var}_ground_truth_lt{lead_time}h"]
                fc_original = ds[f"{prediction_var}_original_lt{lead_time}h"]
                fc_corrected = ds[f"{prediction_var}_corrected_lt{lead_time}h"]
                
                # Flatten arrays for statistics
                gt_flat = ground_truth.values.flatten()
                orig_flat = fc_original.values.flatten()
                corr_flat = fc_corrected.values.flatten()
                
                # Remove NaN values
                mask = ~(np.isnan(gt_flat) | np.isnan(orig_flat) | np.isnan(corr_flat))
                gt_flat = gt_flat[mask]
                orig_flat = orig_flat[mask]
                corr_flat = corr_flat[mask]
                
                # Collect ground truth values for region-wide statistics
                region_ground_truth_values.extend(gt_flat)
                
                # RMSE calculations
                error_orig = orig_flat - gt_flat
                error_corr = corr_flat - gt_flat
                rmse_orig = np.sqrt(np.mean(error_orig**2))
                rmse_corr = np.sqrt(np.mean(error_corr**2))
                
                # Calculate RMSE change (negative means improvement)
                rmse_change = rmse_corr - rmse_orig
                
                # Percent improvement
                pct_improvement = (rmse_orig - rmse_corr) / rmse_orig * 100

                # save values to dict
                values['rmse_orig'] = rmse_orig
                values['rmse_change'] = rmse_change
                values['rmse_imp_pct'] = pct_improvement

            # take average if multiple bootstrap samples
            rmse_orig = values['rmse_orig'].mean() if isinstance(values['rmse_orig'], (list, np.ndarray)) else values['rmse_orig']
            rmse_change = values['rmse_change'].mean() if isinstance(values['rmse_change'], (list, np.ndarray)) else values['rmse_change']
            pct_improvement = values['rmse_imp_pct'].mean() if isinstance(values['rmse_imp_pct'], (list, np.ndarray)) else values['rmse_imp_pct']

            # Create row data
            row_data = {
                'Region': region.replace('_', ' ').title(),
                'Lead Time': f"{lead_time}h",
                'RMSE (Orig)': rmse_orig,
                'RMSE Change': rmse_change,
                'RMSE Improvement (%)': pct_improvement
            }

            all_rows.append(row_data)
            print(f"Processed {region} - {lead_time}h: RMSE improvement = {pct_improvement:.1f}%")
                    
        
        # Calculate region-wide ground truth statistics
        if region_ground_truth_values:
            region_mean = np.mean(region_ground_truth_values)
            region_std = np.std(region_ground_truth_values)
            region_ground_truth_stats[region.replace('_', ' ').title()] = f"{region_mean:.2f} ({region_std:.2f})"
    
    # Create DataFrame
    if not all_rows:
        print("No data processed successfully.")
        return None
    
    df = pd.DataFrame(all_rows)
    
    # Create LaTeX table
    _create_latex_table(df, prediction_var, nn_architecture, subregion, dirs, model, region_ground_truth_stats)
    
    return df


def _create_latex_table(df, prediction_var, nn_architecture, subregion, dirs, model, region_ground_truth_stats):
    """
    Creates a LaTeX table with proper formatting where region names appear only once
    and ground truth statistics appear in the second row of each region.
    """
    # Prepare output folder
    out_folder = os.path.join(dirs["fig"], model, "summary_stats")
    os.makedirs(out_folder, exist_ok=True)
    
    # Start building the LaTeX table
    latex_lines = []
    
    # Table setup
    latex_lines.append("\\begin{tabular}{llrr}")
    latex_lines.append("\\toprule")
    latex_lines.append("Region & Lead Time & RMSE & Improvement (\\%) \\\\")
    latex_lines.append("\\midrule")
    
    # Group by region to format the table
    current_region = None
    region_rows = []
    
    for _, row in df.iterrows():
        region = row['Region']
        
        # When we encounter a new region, process the previous region's rows
        if region != current_region and current_region is not None:
            # Add the collected rows for the previous region
            for i, region_row in enumerate(region_rows):
                if i == 0:
                    # First row: show region name
                    latex_lines.append(region_row)
                elif i == 1:
                    # Second row: show ground truth stats
                    gt_stats = region_ground_truth_stats.get(current_region, "N/A")
                    latex_lines.append(f"\\textit{{Ground Truth: {gt_stats}}} & {region_row}")
                else:
                    # Subsequent rows: empty first column
                    latex_lines.append(f" & {region_row}")
            region_rows = []
        
        # Format RMSE with change in parentheses
        rmse_display = f"{row['RMSE (Orig)']:.3f} ({row['RMSE Change']:+.3f})"
        
        # Build the data portion of the row (without region name)
        data_portion = f"{row['Lead Time']} & {rmse_display} & {row['RMSE Improvement (%)']:.1f} \\\\"
        
        if region != current_region:
            # First row of new region
            region_rows.append(f"{region} & {data_portion}")
            current_region = region
        else:
            # Additional rows for same region
            region_rows.append(data_portion)
    
    # Don't forget to process the last region
    if region_rows:
        for i, region_row in enumerate(region_rows):
            if i == 0:
                latex_lines.append(region_row)
            elif i == 1:
                gt_stats = region_ground_truth_stats.get(current_region, "N/A")
                latex_lines.append(f"\\textit{{Ground Truth: {gt_stats}}} & {region_row}")
            else:
                latex_lines.append(f" & {region_row}")
    
    # Close the table
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    
    # Join all lines
    latex_str = "\n".join(latex_lines)
    
    # Save the LaTeX file
    variable_name = prediction_var.replace('_', ' ').title()
    filename = f"summary_stats_{prediction_var}_{nn_architecture}_{subregion}.tex"
    filepath = os.path.join(out_folder, filename)
    
    with open(filepath, 'w') as f:
        f.write(latex_str)
    
    print(f"\nSaved LaTeX table to: {filepath}")
    print(f"Table title: Summary Statistics for {variable_name}")



def main():
    dirs = setup_directories()

    # Two options for training and output variable combinations, uncomment the one you want to use
    # training_vars = ["2m_temperature"]
    # output_vars = ["2m_temperature"]
    # prediction_var = "2m_temperature"

    training_vars = ["10m_wind_speed"]
    output_vars = ["10m_wind_speed"]
    prediction_var = "10m_wind_speed"

    #============================================
    # Summary Stat Tables
    #============================================

    generate_summary_stat_table(
        dirs=dirs,
        train_start="2018-01-01",
        train_end="2021-12-31",
        test_start="2022-01-01",
        test_end="2022-12-31",
        model="pangu",
        training_output_vars=(training_vars, output_vars),
        prediction_var=prediction_var,
        nn_architecture="mlp",
        regions = ["india", "amazon", "ethiopia"],
        subregion="2x2",
        lead_times=[24, 120, 240],  # Multiple lead times
        simultaneous=True
    )

    # climate zones (this takes a long time to run)
    generate_summary_stat_table(
        dirs=dirs,
        train_start="2018-01-01",
        train_end="2021-12-31",
        test_start="2022-01-01",
        test_end="2022-12-31",
        model="pangu",
        training_output_vars=(training_vars, output_vars),
        prediction_var=prediction_var,
        nn_architecture="mlp",
        regions = ["tropical", "arid", "temperate"],
        subregion="2x2",
        bootstrap=True,
        lead_times=[24, 120, 240],  
        simultaneous=False
    )

    exit()

    #============================================
    # Lead Time Plots
    #============================================

    # plot_types = ["pangu_nn", "pangu_ifs_nn", "all"] # possible plot types
    subregions = ["2x2", "6x6", "10x10"]
    for subregion in subregions:
        generate_lead_time_plots(
            dirs = dirs,
            train_start="2018-01-01",
            train_end="2021-12-31",
            test_start="2022-01-01",
            test_end="2022-12-31",
            model="pangu",
            training_output_vars=(training_vars, output_vars),
            prediction_var=prediction_var,
            nn_architecture=["mlp"],  # mlp 
            regions = ["india", "ethiopia", "amazon", "british_columbia", "usa_south"],
            subregion=subregion,
            bootstrap=False,
            plot_type="pangu_ifs_nn",
            simultaneous=False
        )
        generate_lead_time_plots(
            dirs = dirs,
            train_start="2018-01-01",
            train_end="2021-12-31",
            test_start="2022-01-01",
            test_end="2022-12-31",
            model="pangu",
            training_output_vars=(training_vars, output_vars),
            prediction_var=prediction_var,
            nn_architecture=["mlp"],  # mlp 
            regions = ["arid", "temperate", "tropical"], 
            subregion=subregion,
            bootstrap=True,
            plot_type="pangu_ifs_nn",
            simultaneous=False
        )


    #=============================================
    # Subregion Comparison Plots
    #=============================================
    
    generate_subregion_comparison_plots(
        dirs = dirs,
        train_start="2018-01-01",
        train_end="2021-12-31",
        test_start="2022-01-01",
        test_end="2022-12-31",
        model="pangu",
        training_output_vars=(training_vars, output_vars),
        prediction_var=prediction_var,
        nn_architecture=["mlp"],
        lead_time=240,
        simultaneous=False
    )

    #==============================================
    # Generate Maps
    #============================================== 

    regions = ["usa_south", "amazon", "india", "british_columbia", "ethiopia"]
    for region in regions:
        # MLP maps
        generate_map_plots(
            dirs=dirs,
            train_start="2018-01-01",
            train_end="2021-12-31",
            test_start="2022-01-01",
            test_end="2022-12-31",
            model="pangu",
            training_output_vars=(training_vars, output_vars),
            prediction_var=prediction_var,
            nn_architecture="mlp",
            region=region,
            subregion="10x10",
            lead_time=24,
            simultaneous=False
        )

        #===========================================
        # Generate time series plots
        #===========================================

        generate_time_series_plots(
            dirs=dirs,
            train_start="2018-01-01",
            train_end="2021-12-31",
            test_start="2022-01-01",
            test_end="2022-12-31",
            model="pangu",
            training_output_vars=(training_vars, output_vars),
            prediction_var=prediction_var,
            nn_architecture="mlp",
            region=region,
            subregion="10x10",
            lead_time=24,
            simultaneous=False
        )
        

if __name__ == "__main__":
    main()