import os
import re
import glob
import socket
import calendar
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from types import SimpleNamespace

from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats


#######################
# Utility Functions
#######################

def generate_output_path(args):
    region_str = f"{args.region}"
    subregion_str = f"{args.subregion}"
    dates_str = f"train{args.train_start}-{args.train_end}_test{args.test_start}-{args.test_end}"
    training_vars_str = "_".join(args.training_vars)
    output_vars_str = "_".join(args.output_vars)
    mlp_str = f"mlp{args.mlp_hidden_dim}x{args.mlp_layers}"
    lead_time = f"leadtime_{args.lead_time_hours}"

    output_path = f"{args.model_name}/{region_str}/train_{training_vars_str}_test_{output_vars_str}_dim{subregion_str}_{lead_time}h_{dates_str}_{mlp_str}.zarr"
    return output_path 

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

#######################
# Plotting Functions
#######################

def generate_lead_time_plots(
        dirs,
        train_start, train_end,
        test_start, test_end,
        model,
        training_output_vars,
        prediction_var,
        mlp_params,
        regions,
        subregion,
        bootstrap=None,
):
    """
    Generates a single plot showing percent improvement in RMSE by lead time for neural network
    and anomaly correction methods across all specified regions.
    
    Parameters
    ----------
    dirs : dict
        Dictionary containing paths for input and figure directories
    train_start, train_end : str
        Training period boundaries
    test_start, test_end : str
        Testing period boundaries
    model : str
        Model name (e.g., 'pangu', 'ifs')
    training_output_vars : tuple
        Tuple of (training_vars, output_vars)
    prediction_var : str
        Variable to predict
    mlp_params : tuple
        MLP architecture parameters
    regions : list
        List of regions to process
    subregion : str
        Subregion identifier
    bootstrap : bool, optional
        Whether to use bootstrap samples for confidence intervals
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
    mlp_str = f"mlp{mlp_params[0]}x{mlp_params[1]}"
    time_str = f"train{train_start}-{train_end}_test{test_start}-{test_end}"

    lead_times = [24, 48, 72, 96, 120, 144, 168]  # Possible lead times in hours

    # Define colors for different regions (can handle up to 10 regions)
    region_colors = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
        '#bcbd22',  # Olive
        '#17becf'   # Cyan
    ]
    
    # Define line styles and markers for different methods
    method_styles = {
        'nn': {'linestyle': '-', 'marker': 'o', 'markersize': 8},
        'ano': {'linestyle': '--', 'marker': 's', 'markersize': 7}
    }
    
    # Store all improvements for all regions
    all_improvements = {}
    
    # Process each region
    for region_idx, region in enumerate(regions):
        # Initialize storage for improvement percentages for this region
        improvements = {
            'pangu_nn': {},
            'pangu_ano': {},
            'ifs_nn': {},
            'ifs_ano': {}
        }
        
        # Process each lead time
        for lt in lead_times:
            # Construct file paths
            if bootstrap:
                pangu_pattern = os.path.join(
                    dirs['input'], 
                    f"{model}/{region}/train_{training_vars_str}_test_{output_vars_str}_"
                    f"dim{subregion}_leadtime_{lt}h_{time_str}_{mlp_str}*bs*.zarr"
                )
                ifs_pattern = os.path.join(
                    dirs['input'],
                    f"ifs/{region}/train_{training_vars_str}_test_{output_vars_str}_"
                    f"dim{subregion}_leadtime_{lt}h_{time_str}_{mlp_str}*bs*.zarr"
                )
            else:
                pangu_pattern = os.path.join(
                    dirs['input'],
                    f"{model}/{region}/train_{training_vars_str}_test_{output_vars_str}_"
                    f"dim{subregion}_leadtime_{lt}h_{time_str}_{mlp_str}.zarr"
                )
                ifs_pattern = os.path.join(
                    dirs['input'],
                    f"ifs/{region}/train_{training_vars_str}_test_{output_vars_str}_"
                    f"dim{subregion}_leadtime_{lt}h_{time_str}_{mlp_str}.zarr"
                )

            # Process Pangu model files
            pangu_files = glob.glob(pangu_pattern)
            if not bootstrap and len(pangu_files) > 1:
                raise ValueError(f"Multiple files found for lead time {lt}h: {pangu_files}")

            for idx, file_path in enumerate(pangu_files):
                try:
                    ds = xr.open_zarr(file_path)
                    
                    # Extract data
                    ground_truth = ds[f"{prediction_var}_ground_truth"]
                    original = ds[f"{prediction_var}_original"]
                    nn_corrected = ds[f"{prediction_var}_corrected"]
                    ano_corrected = ds.get(f"{prediction_var}_mean_corrected", None)

                    # Calculate RMSE
                    rmse_original = float(np.sqrt(((original - ground_truth) ** 2).mean().values))
                    rmse_nn = float(np.sqrt(((nn_corrected - ground_truth) ** 2).mean().values))
                    
                    # Calculate percent improvements
                    pct_improvement_nn = ((rmse_original - rmse_nn) / rmse_original * 100 
                                          if rmse_original != 0 else 0)
                    improvements['pangu_nn'][(lt, idx)] = pct_improvement_nn
                    
                    if ano_corrected is not None:
                        rmse_ano = float(np.sqrt(((ano_corrected - ground_truth) ** 2).mean().values))
                        pct_improvement_ano = ((rmse_original - rmse_ano) / rmse_original * 100 
                                               if rmse_original != 0 else 0)
                        improvements['pangu_ano'][(lt, idx)] = pct_improvement_ano

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    continue

            # Process IFS model files
            ifs_files = glob.glob(ifs_pattern)
            if ifs_files:  # Evaluates to True if the list is empty
                assert len(ifs_files) == len(pangu_files)
                
            for idx, file_path in enumerate(ifs_files):  
                try:
                    ds = xr.open_zarr(file_path)
                    
                    # Extract data
                    ground_truth = ds[f"{prediction_var}_ground_truth"]
                    original = ds[f"{prediction_var}_original"]
                    nn_corrected = ds[f"{prediction_var}_corrected"]
                    ano_corrected = ds.get(f"{prediction_var}_mean_corrected", None)

                    # Calculate RMSE
                    rmse_original = float(np.sqrt(((original - ground_truth) ** 2).mean().values))
                    rmse_nn = float(np.sqrt(((nn_corrected - ground_truth) ** 2).mean().values))
                    
                    # Calculate percent improvements
                    pct_improvement_nn = ((rmse_original - rmse_nn) / rmse_original * 100 
                                          if rmse_original != 0 else 0)
                    improvements['ifs_nn'][(lt, idx)] = pct_improvement_nn
                    
                    if ano_corrected is not None:
                        rmse_ano = float(np.sqrt(((ano_corrected - ground_truth) ** 2).mean().values))
                        pct_improvement_ano = ((rmse_original - rmse_ano) / rmse_original * 100 
                                               if rmse_original != 0 else 0)
                        improvements['ifs_ano'][(lt, idx)] = pct_improvement_ano

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
        color = region_colors[region_idx % len(region_colors)]

        # Process each improvement type for this region
        for imp_type, imp_data in improvements.items():
            if not imp_data:
                continue
            
            # Determine if this is NN or anomaly correction
            method_type = 'nn' if 'nn' in imp_type else 'ano'
            style = method_styles[method_type]
            
            # Determine if this is Pangu or IFS (for alpha transparency)
            is_ifs = 'ifs' in imp_type
            line_alpha = 0.6 if is_ifs else 1.0
            
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
                
                # Create label
                label = f"{region} - {imp_type.replace('_', ' ').title()}"
                
                # Plot line with markers
                x_pos = [lead_times.index(lt) for lt in valid_lead_times]
                ax.plot(x_pos, means, 
                       marker=style['marker'], 
                       linestyle=style['linestyle'], 
                       color=color, 
                       linewidth=2.5, 
                       markersize=style['markersize'],
                       label=label,
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
                
                # Create label
                label = f"{region} - {imp_type.replace('_', ' ').title()}"
                
                # Plot line
                x_pos = [lead_times.index(lt) for lt in valid_lead_times]
                ax.plot(x_pos, values, 
                       marker=style['marker'], 
                       linestyle=style['linestyle'],
                       color=color, 
                       linewidth=2.5, 
                       markersize=style['markersize'],
                       label=label,
                       alpha=line_alpha,
                       zorder=3)
    
    # Set x-axis to show all lead times but only label those with data
    ax.set_xticks(range(len(lead_times)))
    ax.set_xticklabels([f"{lt}h" if lt in all_valid_lead_times else "" 
                       for lt in lead_times])
    
    # Customize plot appearance
    ax.set_xlabel("Forecast Lead Time", fontsize=13)
    ax.set_ylabel("RMSE Improvement (%)", fontsize=13)
    
    # Add horizontal line at y=0 for reference
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Title
    regions_str = ", ".join(regions)
    title_parts = [f"RMSE Improvement for {prediction_var.replace('_', ' ').title()}"]
    title_parts.append(f"Regions: {regions_str}, Patch Size: {subregion}")
    if bootstrap:
        title_parts[0] += " (with 95% CI)"
    ax.set_title('\n'.join(title_parts), fontsize=14, pad=15)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Legend - position it outside the plot area if many lines
    n_lines = len([h for h in ax.get_legend_handles_labels()[0]])
    if n_lines > 6:
        legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                          frameon=True, fancybox=True,
                          shadow=True, fontsize=10, ncol=1)
    else:
        legend = ax.legend(loc='best', frameon=True, fancybox=True,
                          shadow=True, fontsize=10, ncol=1)
    
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
    if "arid" in regions or "tropical" in regions or "temperate" in regions:
        region_type = "climate_zones"
    else:
        region_type = "geographic"
    fname = (f"leadtime_improvement_{prediction_var}_trainedwith_{training_vars_str}_"
            f"{region_type}{bootstrap_suffix}.png")
    save_path = os.path.join(out_folder, fname)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Multi-region lead-time improvement plot saved to: {save_path}")
    
    if bootstrap:
        # Print summary statistics for each region
        for region in regions:
            print(f"\nRegion {region}:")
            for imp_type in ['pangu_nn', 'pangu_ano', 'ifs_nn', 'ifs_ano']:
                if all_improvements[region][imp_type]:
                    # Get sample count from first lead time
                    first_lt = next(iter([lt for (lt, idx) in all_improvements[region][imp_type].keys()]))
                    n_samples = len([v for (lt, idx), v in all_improvements[region][imp_type].items() 
                                   if lt == first_lt])
                    if n_samples > 0:
                        print(f"  {imp_type}: {n_samples} bootstrap samples")

def generate_subregion_comparison_plots(dirs, train_start, train_end, test_start,
                                        test_end, model, training_output_vars,
                                        prediction_var, mlp_params):
    input_folder = dirs['input']
    training_vars, output_vars = training_output_vars
    training_vars = training_vars if isinstance(training_vars, (list,tuple)) else [training_vars]
    output_vars   = output_vars   if isinstance(output_vars,   (list,tuple)) else [output_vars]
    mlp_str = f"mlp{mlp_params[0]}x{mlp_params[1]}"

    regions   = ["amazon", "india", "usa_south", "british_columbia"]
    subregions = ["2x2","4x4","6x6","8x8","10x10"]
    lead_times = [24,72,168]
    degrees    = [int(s.split('x')[0]) for s in subregions]

    improvement = {r:{lt:[] for lt in lead_times} for r in regions}

    for region in regions:
        # ---- extract central 2×2 bounds and normalization once ----
        central_path = os.path.join(
            input_folder,
            generate_output_path(SimpleNamespace(
                model_name=model, region=region, subregion="2x2",
                train_start=train_start, train_end=train_end,
                test_start=test_start,  test_end=test_end,
                training_vars=training_vars, output_vars=output_vars,
                mlp_hidden_dim=mlp_params[0], mlp_layers=mlp_params[1],
                lead_time_hours=lead_times[0]  # dummy
            ))
        )
        with xr.open_zarr(central_path) as ds2:
            lat_min, lat_max = ds2.latitude.min().item(), ds2.latitude.max().item()
            lon_min, lon_max = ds2.longitude.min().item(), ds2.longitude.max().item()

            gt2 = ds2[f"{prediction_var}_ground_truth"]
            mu, sigma = float(gt2.mean()), float(gt2.std())

        # ---- now loop subregions + lead times, always slicing that 2×2 box ----
        for sub in subregions:
            for lt in lead_times:
                path = os.path.join(
                    input_folder,
                    generate_output_path(SimpleNamespace(
                        model_name=model, region=region, subregion=sub,
                        train_start=train_start, train_end=train_end,
                        test_start=test_start,  test_end=test_end,
                        training_vars=training_vars, output_vars=output_vars,
                        mlp_hidden_dim=mlp_params[0], mlp_layers=mlp_params[1],
                        lead_time_hours=lt
                    ))
                )
                # open and slice to 2x2 grid in the center of the region
                with xr.open_zarr(path) as ds:
                    ds = ds.sel(latitude=slice(lat_min,lat_max),
                                longitude=slice(lon_min,lon_max))
                    
                    # normalize using mu, sigma
                    gt_n   = ds[f"{prediction_var}_ground_truth"] 
                    orig_n = ds[f"{prediction_var}_original"]     
                    corr_n = ds[f"{prediction_var}_corrected"]    

                    rmse_orig = float(np.sqrt(((orig_n - gt_n)**2).mean()))
                    rmse_corr = float(np.sqrt(((corr_n - gt_n)**2).mean()))
                    pct_improvement = (rmse_orig - rmse_corr) / rmse_orig * 100
                    size = int(sub.split('x')[0])
                    improvement[region][lt].append((size, pct_improvement))

    # ---- plotting ----
    # Create color map for regions and line style map for lead times
    region_colors = plt.get_cmap('Set1')
    region_color_map = {region: region_colors(i) for i, region in enumerate(regions)}
    ls_map = {24: 'solid', 72: '--', 168: ':'}
    
    # Create single figure
    plt.figure(figsize=(10, 6))
    
    # Plot all combinations of region and lead time
    for region in regions:
        for lt in lead_times:
            data = sorted(improvement[region][lt], key=lambda x: x[0])
            sizes, imps = zip(*data)
            
            # Format region name for legend
            region_label = region.replace('_', ' ').title()
            
            plt.plot(sizes, imps, marker='o',
                     color=region_color_map[region], 
                     linestyle=ls_map[lt],
                     label=f"{region_label} ({lt}h)",
                     linewidth=2, markersize=6)
    
    # Configure plot
    plt.xticks(degrees, subregions)
    plt.xlabel("Patch size (degrees)", fontsize=12)
    plt.ylabel("RMSE % improvement\n(original − corrected)", fontsize=12)
    plt.title(f"RMSE % Improvement by Patch Size Across Regions and Lead Times", fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Create custom legend
    from matplotlib.lines import Line2D
    
    # Region legend entries
    region_handles = [Line2D([0], [0], color=region_color_map[region], linewidth=3, 
                            label=region.replace('_', ' ').title()) for region in regions]
    
    # Lead time legend entries  
    lt_handles = [Line2D([0], [0], color='black', linestyle=ls_map[lt], linewidth=2,
                        label=f"{lt}h lead") for lt in lead_times]
    
    # Create two-part legend
    legend1 = plt.legend(handles=region_handles, title="Region", 
                        loc='upper left', bbox_to_anchor=(0, 1))
    legend2 = plt.legend(handles=lt_handles, title="Lead Time", 
                        loc='upper left', bbox_to_anchor=(0, 0.7))
    plt.gca().add_artist(legend1)  # Keep both legends
    
    plt.tight_layout()

    # Save figure
    out_folder = os.path.join(dirs["fig"], model, "subregion")
    os.makedirs(out_folder, exist_ok=True)
    fname = f"subregion_rmse_improvement_combined_{'_'.join(training_vars)}_{prediction_var}_{mlp_str}.png"
    plt.savefig(os.path.join(out_folder, fname), dpi=150, bbox_inches='tight')
    plt.close()


def generate_rmse_comparison_plot(dirs, model, training_output_vars, prediction_var, mlp_params):
    """
    Scans the input folder for forecast files matching the given model,
    training/output variables, and MLP parameters, and creates a single bar plot
    that organizes the overall (scalar) RMSE by lead time (first level) and region (second level).
    
    For each (lead time, region) combination, the original and corrected RMSE are plotted
    as overlapping bars with transparency. The x-axis tick labels are generated dynamically
    (with two lines: lead time on the first line and region on the second) so that the plot
    remains legible regardless of whether there are 3 or 5 regions.
    
    Parameters:
      model: Model name.
      training_output_vars: Tuple/list of (training_vars, output_vars) [each as list or str].
      mlp_params: Tuple (num_units, num_layers).
    """
    dirs = setup_directories()
    input_folder = dirs['input']
    training_vars, output_vars = training_output_vars
    if not isinstance(training_vars, (list, tuple)):
        training_vars = [training_vars]
    if not isinstance(output_vars, (list, tuple)):
        output_vars = [output_vars]
    training_vars_str = "_".join(training_vars)
    output_vars_str = "_".join(output_vars)
    mlp_str = f"mlp{mlp_params[0]}x{mlp_params[1]}"
    time_str = "train2018-01-01-2021-12-31_test2022-01-01-2022-12-31"

    # Define the lead times and regions to consider.
    lead_times = [24, 72, 168] # possible lead times 
    regions = ["amazon", "usa_south", "india", "british_columbia"]  # adjust or extend as needed
    subregion ="10x10"

    # Dictionary to store results keyed by (lead_time, region)
    # Each value is a tuple: (avg_rmse_orig, avg_rmse_corr)
    results = {}
    ifs_results = {}

    # Loop over each combination to get original and forecast
    for lt in lead_times:
        for region in regions:

            file_path = os.path.join(input_folder, f"{model}/{region}/train_{training_vars_str}_test_{output_vars_str}_dim{subregion}_leadtime_{lt}h_{time_str}_{mlp_str}.zarr")
            ifs_file_path = os.path.join(input_folder, f"ifs/{region}/train_{training_vars_str}_test_{output_vars_str}_dim{subregion}_leadtime_{lt}h_{time_str}_{mlp_str}.zarr")

            try:
                ds = xr.open_zarr(file_path)
            except Exception as e:
                print(f"Error opening {file_path}: {e}")
                continue

            ground_truth = ds[f"{prediction_var}_ground_truth"]
            orig = ds[f"{prediction_var}_original"]
            corr = ds[f"{prediction_var}_corrected"]

            # normalize by test‐set truth
            mean = ground_truth.mean().values
            std  = ground_truth.std().values
            gt_n   = (ground_truth - mean) / std
            orig_n = (orig            - mean) / std
            corr_n = (corr           - mean) / std

            rmse_total_orig = float(np.sqrt(((orig_n - gt_n) ** 2).mean().values))
            rmse_total_corr = float(np.sqrt(((corr_n - gt_n) ** 2).mean().values))
            results[(lt, region)] = (rmse_total_orig, rmse_total_corr)

            # repeat for ifs
            try:
                ifs_ds = xr.open_zarr(ifs_file_path)
            except Exception as e:
                print(f"Error opening {ifs_file_path}: {e}")
                continue
            
            ifs_ground_truth = ifs_ds[f"{prediction_var}_ground_truth"]
            ifs_fc_original = ifs_ds[f"{prediction_var}_original"]
            ifs_fc_corrected = ifs_ds[f"{prediction_var}_corrected"]

            # Normalize by ifs groundtruth
            ifs_mean = ifs_ground_truth.mean().values
            ifs_std  = ifs_ground_truth.std().values
            ifs_gt_n   = (ifs_ground_truth - ifs_mean) / ifs_std
            ifs_orig_n = (ifs_fc_original - ifs_mean) / ifs_std
            ifs_corr_n = (ifs_fc_corrected - ifs_mean) / ifs_std


            ifs_rmse_total_orig = float(np.sqrt(((ifs_orig_n - ifs_gt_n) ** 2).mean().values))
            ifs_rmse_total_corr = float(np.sqrt(((ifs_corr_n - ifs_gt_n) ** 2).mean().values))
            ifs_results[(lt, region)] = (ifs_rmse_total_orig, ifs_rmse_total_corr)

    # Prepare data for the single grouped bar plot.
    x_positions = []
    x_labels = []
    rmse_orig_vals = []
    rmse_corr_vals = []
    ifs_rmse_orig_vals = []
    ifs_rmse_corr_vals = []
    pos = 0
    group_gap = 1  # extra gap between different lead time groups
    for region in regions:
        # Collect regions that have results for this lead time.
        for lt in sorted(lead_times):
            regions_with_data = [r for r in regions if (lt, r) in results]
            regions_with_ifs_data = [r for r in regions if (lt, r) in ifs_results]

            # check if region is available in both results and ifs_results
            if region not in regions_with_data or region not in regions_with_ifs_data:
                print(f"Skipping region {region} for lead time {lt} as no data is available")
                continue

            x_positions.append(pos)
            # Create a two-line label: first line is lead time, second line is region.
            label = f"{lt}h\n{region.replace('_', ' ').title()}"
            x_labels.append(label)
            rmse_orig, rmse_corr = results[(lt, region)]
            ifs_rmse_orig, ifs_rmse_corr = ifs_results[(lt, region)]

            rmse_orig_vals.append(rmse_orig)
            rmse_corr_vals.append(rmse_corr)
            ifs_rmse_orig_vals.append(ifs_rmse_orig)
            ifs_rmse_corr_vals.append(ifs_rmse_corr)
            pos += 1
        pos += group_gap  # add gap between groups

    x_positions_offset = np.array(x_positions) + 0.3  # Offset for IFS bars

    # # Create the grouped bar plot.
    fig, ax = plt.subplots(figsize=(max(8, len(x_positions)*0.8), 6))
    # Overlap the two bars at the same positions with transparency.
    ax.bar(x_positions, rmse_orig_vals, color='blue', width=0.8, alpha=0.5, label='Original RMSE')
    ax.bar(x_positions, rmse_corr_vals, color='red', width=0.8, alpha=0.5, label='Corrected RMSE')

    ifs_bar_width = 0.18
    # first IFS bar (baseline)
    ax.bar(
        x_positions_offset,
        ifs_rmse_orig_vals,
        width=ifs_bar_width,
        alpha=0.75,
        label='IFS Baseline RMSE'
    )
    # second IFS bar (corrected), shifted over by one bar‐width
    ax.bar(
        x_positions_offset + ifs_bar_width,
        ifs_rmse_corr_vals,
        width=ifs_bar_width,
        alpha=0.75,
        color='#ADD8E6',        # light‐blue
        label='IFS Corrected RMSE'
    )

    # rest stays the same
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_ylabel("Normalized RMSE")
    ax.set_title(f"RMSE Comparison for {model}\nPredicting {prediction_var}")
    ax.legend()
    plt.tight_layout()

    save_path = os.path.join(dirs["fig"], model, "comparison", f"rmse_comparison_{model}_trained_with_{training_vars_str}_output{prediction_var}_{mlp_str}.png")
    plt.savefig(save_path, dpi=150)
    print(f"RMSE comparison bar chart saved to {save_path}")
    plt.close()

def generate_map_plots(
        dirs,
        train_start, train_end,
        test_start, test_end,
        model,
        training_output_vars,
        prediction_var,
        mlp_params,
        region,
        subregion,
        lead_time,
):
    """
    Generates a figure with 2 maps: original forecast RMSE and percent improvement in RMSE.
    
    Parameters
    ----------
    dirs : dict
        Dictionary containing paths for input and figure directories
    train_start, train_end : str
        Training period boundaries
    test_start, test_end : str
        Testing period boundaries
    model : str
        Model name (e.g., 'pangu', 'ifs')
    training_output_vars : tuple
        Tuple of (training_vars, output_vars)
    prediction_var : str
        Variable to predict
    mlp_params : tuple
        MLP architecture parameters (hidden_dim, layers)
    region : str
        Region identifier
    subregion : str
        Subregion identifier
    lead_time : int
        Forecast lead time in hours
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
    mlp_str = f"mlp{mlp_params[0]}x{mlp_params[1]}"
    time_str = f"train{train_start}-{train_end}_test{test_start}-{test_end}"

    # Construct file path
    file_path = os.path.join(
        dirs['input'],
        f"{model}/{region}/train_{training_vars_str}_test_{output_vars_str}_"
        f"dim{subregion}_leadtime_{lead_time}h_{time_str}_{mlp_str}.zarr"
    )

    # Skip if region is "pixel" (no maps for pixel)
    if region == "pixel":
        print(f"Skipping map generation for region 'pixel'")
        return

    try:
        # Load the forecast data
        ds = xr.open_zarr(file_path)
        
        # Extract data arrays
        ground_truth = ds[f"{prediction_var}_ground_truth"]
        fc_original = ds[f"{prediction_var}_original"]
        fc_corrected = ds[f"{prediction_var}_corrected"]
        
        # Calculate RMSE for original and corrected forecasts
        mse_spatial_orig = ((fc_original - ground_truth) ** 2).mean(dim="time")
        mse_spatial_corr = ((fc_corrected - ground_truth) ** 2).mean(dim="time")
        
        # Convert MSE to RMSE
        rmse_spatial_orig = np.sqrt(mse_spatial_orig)
        rmse_spatial_corr = np.sqrt(mse_spatial_corr)
        
        # Calculate percent improvement
        pct_improvement = ((rmse_spatial_orig - rmse_spatial_corr) / rmse_spatial_orig * 100)
        
        # Create figure with 2 subplots - reduce spacing between plots
        fig = plt.figure(figsize=(15, 6))
        
        # Use GridSpec for better control over subplot spacing
        gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.15, hspace=0.1)
        
        # First subplot: Original forecast RMSE
        ax1 = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
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
                      fontsize=12, pad=10)
        ax1.coastlines(resolution='50m', linewidth=0.5)
        ax1.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
        ax1.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.8, edgecolor='black')
        
        # Add state/province borders based on region
        if region in ['usa_south', 'british_columbia']:
            # Add states/provinces for North America
            ax1.add_feature(cfeature.STATES, linestyle=':', linewidth=0.5, edgecolor='gray')
        elif region == 'india':
            # For India, the STATES feature includes Indian states
            ax1.add_feature(cfeature.STATES, linestyle=':', linewidth=0.5, edgecolor='gray')
        elif region == 'amazon':
            # For Brazil, states are also included in the STATES feature
            ax1.add_feature(cfeature.STATES, linestyle=':', linewidth=0.5, edgecolor='gray')
        
        # Customize gridlines to prevent overlap
        gl1 = ax1.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, 
                           linewidth=0.5, alpha=0.5)
        gl1.right_labels = False  # Turn off right labels to prevent overlap with colorbar
        gl1.top_labels = False
        gl1.xlabel_style = {'size': 9}
        gl1.ylabel_style = {'size': 9}
        
        # Second subplot: Percent improvement
        ax2 = fig.add_subplot(gs[1], projection=ccrs.PlateCarree())
        
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
        
        ax2.set_title(f"RMSE Percent Improvement (MLP Corrected)\n{prediction_var.replace('_', ' ').title()}", 
                      fontsize=12, pad=10)
        ax2.coastlines(resolution='50m', linewidth=0.5)
        ax2.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
        ax2.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.9, edgecolor='black')
        
        # Add state/province borders based on region
        if region in ['usa_south', 'british_columbia']:
            # Add states/provinces for North America
            ax2.add_feature(cfeature.STATES, linestyle=':', linewidth=0.7, edgecolor='gray')
        elif region == 'india':
            # For India, the STATES feature includes Indian states
            ax2.add_feature(cfeature.STATES, linestyle=':', linewidth=0.7, edgecolor='gray')
        elif region == 'amazon':
            # For Brazil, states are also included in the STATES feature
            ax2.add_feature(cfeature.STATES, linestyle=':', linewidth=0.7, edgecolor='gray')
        
        # Customize gridlines to prevent overlap
        gl2 = ax2.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                           linewidth=0.5, alpha=0.5)
        gl2.right_labels = False  # Turn off right labels to prevent overlap with colorbar
        gl2.top_labels = False
        gl2.left_labels = False  # Turn off left labels on second plot since they're close to first plot
        gl2.xlabel_style = {'size': 9}
        gl2.ylabel_style = {'size': 9}
        
        # Add overall title
        fig.suptitle(f"{region.replace('_', ' ').title()} - {lead_time}h Lead Time - Patch Size: {subregion}", 
                     fontsize=14, y=1.02)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        out_folder = os.path.join(dirs["fig"], model, "maps", region, subregion)
        os.makedirs(out_folder, exist_ok=True)
        
        fname = f"rmse_maps_{prediction_var}_trainedwith_{training_vars_str}_{lead_time}h.png"
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
        mlp_params,
        region,
        subregion,
        lead_time,
):
    """
    Generates a single bar plot showing monthly RMSE for original and corrected forecasts
    for both the main model (e.g., pangu) and IFS.
    
    Parameters
    ----------
    dirs : dict
        Dictionary containing paths for input and figure directories
    train_start, train_end : str
        Training period boundaries
    test_start, test_end : str
        Testing period boundaries
    model : str
        Model name (e.g., 'pangu', 'ifs')
    training_output_vars : tuple
        Tuple of (training_vars, output_vars)
    prediction_var : str
        Variable to predict
    mlp_params : tuple
        MLP architecture parameters (hidden_dim, layers)
    region : str
        Region identifier
    subregion : str
        Subregion identifier
    lead_time : int
        Forecast lead time in hours
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
    mlp_str = f"mlp{mlp_params[0]}x{mlp_params[1]}"
    time_str = f"train{train_start}-{train_end}_test{test_start}-{test_end}"

    # Construct file paths for main model and IFS
    model_file_path = os.path.join(
        dirs['input'],
        f"{model}/{region}/train_{training_vars_str}_test_{output_vars_str}_"
        f"dim{subregion}_leadtime_{lead_time}h_{time_str}_{mlp_str}.zarr"
    )
    
    ifs_file_path = os.path.join(
        dirs['input'],
        f"ifs/{region}/train_{training_vars_str}_test_{output_vars_str}_"
        f"dim{subregion}_leadtime_{lead_time}h_{time_str}_{mlp_str}.zarr"
    )

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
        ground_truth = ds_model[f"{prediction_var}_ground_truth"]
        fc_original = ds_model[f"{prediction_var}_original"]
        fc_corrected = ds_model[f"{prediction_var}_corrected"]
        
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
        ifs_ground_truth = ds_ifs[f"{prediction_var}_ground_truth"]
        ifs_fc_original = ds_ifs[f"{prediction_var}_original"]
        ifs_fc_corrected = ds_ifs[f"{prediction_var}_corrected"]
        
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
    
    # Create the bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set up bar positions
    x = np.arange(len(months))
    bar_width = 0.35
    
    # Plot main model bars (overlapping with transparency)
    ax.bar(x - bar_width/2, model_rmse_orig, bar_width, 
           color='blue', alpha=0.5, label=f'{model.upper()} Original')
    ax.bar(x - bar_width/2, model_rmse_corr, bar_width, 
           color='red', alpha=0.5, label=f'{model.upper()} Corrected')
    
    # Plot IFS bars if available (offset to the right)
    if has_ifs_data:
        ifs_bar_width = bar_width * 0.5  # Make IFS bars narrower
        offset = bar_width/2 + 0.05  # Small gap between model and IFS bars
        
        ax.bar(x + offset, ifs_rmse_orig, ifs_bar_width, 
               color='darkblue', alpha=0.75, label='IFS Baseline')
        ax.bar(x + offset + ifs_bar_width, ifs_rmse_corr, ifs_bar_width, 
               color='#ADD8E6', alpha=0.75, label='IFS Corrected')
    
    # Customize plot
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title(f'Monthly RMSE Comparison - {region.replace("_", " ").title()}\n'
                 f'{prediction_var.replace("_", " ").title()} - {lead_time}h Lead Time - Patch Size: {subregion}',
                 fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    out_folder = os.path.join(dirs["fig"], model, "time_series", region, subregion)
    os.makedirs(out_folder, exist_ok=True)
    
    fname = f"rmse_monthly_{prediction_var}_trainedwith_{training_vars_str}_{lead_time}h.png"
    save_path = os.path.join(out_folder, fname)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Time series plot saved to: {save_path}")

def generate_map_plots(
        dirs,
        train_start, train_end,
        test_start, test_end,
        model,
        training_output_vars,
        prediction_var,
        mlp_params,
        region,
        subregion,
        lead_time,
):
    """
    Generates a figure with 2 maps: original forecast RMSE and percent improvement in RMSE.
    
    Parameters
    ----------
    dirs : dict
        Dictionary containing paths for input and figure directories
    train_start, train_end : str
        Training period boundaries
    test_start, test_end : str
        Testing period boundaries
    model : str
        Model name (e.g., 'pangu', 'ifs')
    training_output_vars : tuple
        Tuple of (training_vars, output_vars)
    prediction_var : str
        Variable to predict
    mlp_params : tuple
        MLP architecture parameters (hidden_dim, layers)
    region : str
        Region identifier
    subregion : str
        Subregion identifier
    lead_time : int
        Forecast lead time in hours
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
    mlp_str = f"mlp{mlp_params[0]}x{mlp_params[1]}"
    time_str = f"train{train_start}-{train_end}_test{test_start}-{test_end}"

    # Construct file path
    file_path = os.path.join(
        dirs['input'],
        f"{model}/{region}/train_{training_vars_str}_test_{output_vars_str}_"
        f"dim{subregion}_leadtime_{lead_time}h_{time_str}_{mlp_str}.zarr"
    )

    # Skip if region is "pixel" (no maps for pixel)
    if region == "pixel":
        print(f"Skipping map generation for region 'pixel'")
        return

    try:
        # Load the forecast data
        ds = xr.open_zarr(file_path)
        
        # Extract data arrays
        ground_truth = ds[f"{prediction_var}_ground_truth"]
        fc_original = ds[f"{prediction_var}_original"]
        fc_corrected = ds[f"{prediction_var}_corrected"]
        
        # Calculate RMSE for original and corrected forecasts
        mse_spatial_orig = ((fc_original - ground_truth) ** 2).mean(dim="time")
        mse_spatial_corr = ((fc_corrected - ground_truth) ** 2).mean(dim="time")
        
        # Convert MSE to RMSE
        rmse_spatial_orig = np.sqrt(mse_spatial_orig)
        rmse_spatial_corr = np.sqrt(mse_spatial_corr)
        
        # Calculate percent improvement
        pct_improvement = ((rmse_spatial_orig - rmse_spatial_corr) / rmse_spatial_orig * 100)
        
        # Create figure with 2 subplots - reduce spacing between plots
        fig = plt.figure(figsize=(15, 6))
        
        # Use GridSpec for better control over subplot spacing
        gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.15, hspace=0.1)
        
        # First subplot: Original forecast RMSE
        ax1 = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
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
                      fontsize=12, pad=10)
        ax1.coastlines(resolution='50m', linewidth=0.5)
        ax1.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
        ax1.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.8, edgecolor='black')
        
        # Add state/province borders based on region
        if region in ['usa_south', 'british_columbia']:
            # Add states/provinces for North America
            ax1.add_feature(cfeature.STATES, linestyle=':', linewidth=0.5, edgecolor='gray')
        elif region == 'india':
            # For India, the STATES feature includes Indian states
            ax1.add_feature(cfeature.STATES, linestyle=':', linewidth=0.5, edgecolor='gray')
        elif region == 'amazon':
            # For Brazil, states are also included in the STATES feature
            ax1.add_feature(cfeature.STATES, linestyle=':', linewidth=0.5, edgecolor='gray')
        
        # Customize gridlines to prevent overlap
        gl1 = ax1.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, 
                           linewidth=0.5, alpha=0.5)
        gl1.right_labels = False  # Turn off right labels to prevent overlap with colorbar
        gl1.top_labels = False
        gl1.xlabel_style = {'size': 9}
        gl1.ylabel_style = {'size': 9}
        
        # Second subplot: Percent improvement
        ax2 = fig.add_subplot(gs[1], projection=ccrs.PlateCarree())
        
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
        
        ax2.set_title(f"RMSE Percent Improvement (MLP Corrected)\n{prediction_var.replace('_', ' ').title()}", 
                      fontsize=12, pad=10)
        ax2.coastlines(resolution='50m', linewidth=0.5)
        ax2.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
        ax2.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.8, edgecolor='black')
        
        # Add state/province borders based on region
        if region in ['usa_south', 'british_columbia']:
            # Add states/provinces for North America
            ax2.add_feature(cfeature.STATES, linestyle=':', linewidth=0.5, edgecolor='gray')
        elif region == 'india':
            # For India, the STATES feature includes Indian states
            ax2.add_feature(cfeature.STATES, linestyle=':', linewidth=0.5, edgecolor='gray')
        elif region == 'amazon':
            # For Brazil, states are also included in the STATES feature
            ax2.add_feature(cfeature.STATES, linestyle=':', linewidth=0.5, edgecolor='gray')
        
        # Customize gridlines to prevent overlap
        gl2 = ax2.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                           linewidth=0.5, alpha=0.5)
        gl2.right_labels = False  # Turn off right labels to prevent overlap with colorbar
        gl2.top_labels = False
        gl2.left_labels = False  # Turn off left labels on second plot since they're close to first plot
        gl2.xlabel_style = {'size': 9}
        gl2.ylabel_style = {'size': 9}
        
        # Add overall title
        fig.suptitle(f"{region.replace('_', ' ').title()} - {lead_time}h Lead Time - Patch Size: {subregion}", 
                     fontsize=14, y=1.02)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        out_folder = os.path.join(dirs["fig"], model, "maps", region, subregion)
        os.makedirs(out_folder, exist_ok=True)
        
        fname = f"rmse_maps_{prediction_var}_trainedwith_{training_vars_str}_{lead_time}h.png"
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
        mlp_params,
        region,
        subregion,
        lead_time,
):
    """
    Generates a single bar plot showing monthly RMSE for original and corrected forecasts
    for both the main model (e.g., pangu) and IFS.
    
    Parameters
    ----------
    dirs : dict
        Dictionary containing paths for input and figure directories
    train_start, train_end : str
        Training period boundaries
    test_start, test_end : str
        Testing period boundaries
    model : str
        Model name (e.g., 'pangu', 'ifs')
    training_output_vars : tuple
        Tuple of (training_vars, output_vars)
    prediction_var : str
        Variable to predict
    mlp_params : tuple
        MLP architecture parameters (hidden_dim, layers)
    region : str
        Region identifier
    subregion : str
        Subregion identifier
    lead_time : int
        Forecast lead time in hours
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
    mlp_str = f"mlp{mlp_params[0]}x{mlp_params[1]}"
    time_str = f"train{train_start}-{train_end}_test{test_start}-{test_end}"

    # Construct file paths for main model and IFS
    model_file_path = os.path.join(
        dirs['input'],
        f"{model}/{region}/train_{training_vars_str}_test_{output_vars_str}_"
        f"dim{subregion}_leadtime_{lead_time}h_{time_str}_{mlp_str}.zarr"
    )
    
    ifs_file_path = os.path.join(
        dirs['input'],
        f"ifs/{region}/train_{training_vars_str}_test_{output_vars_str}_"
        f"dim{subregion}_leadtime_{lead_time}h_{time_str}_{mlp_str}.zarr"
    )

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
        ground_truth = ds_model[f"{prediction_var}_ground_truth"]
        fc_original = ds_model[f"{prediction_var}_original"]
        fc_corrected = ds_model[f"{prediction_var}_corrected"]
        
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
        ifs_ground_truth = ds_ifs[f"{prediction_var}_ground_truth"]
        ifs_fc_original = ds_ifs[f"{prediction_var}_original"]
        ifs_fc_corrected = ds_ifs[f"{prediction_var}_corrected"]
        
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
    
    # Create the bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set up bar positions
    x = np.arange(len(months))
    bar_width = 0.35
    
    # Plot main model bars (overlapping with transparency)
    ax.bar(x - bar_width/2, model_rmse_orig, bar_width, 
           color='blue', alpha=0.5, label=f'{model.upper()} Original')
    ax.bar(x - bar_width/2, model_rmse_corr, bar_width, 
           color='red', alpha=0.5, label=f'{model.upper()} Corrected')
    
    # Plot IFS bars if available (offset to the right)
    if has_ifs_data:
        ifs_bar_width = bar_width * 0.5  # Make IFS bars narrower
        offset = bar_width/2 + 0.05  # Small gap between model and IFS bars
        
        ax.bar(x + offset, ifs_rmse_orig, ifs_bar_width, 
               color='darkblue', alpha=0.75, label='IFS Baseline')
        ax.bar(x + offset + ifs_bar_width, ifs_rmse_corr, ifs_bar_width, 
               color='#ADD8E6', alpha=0.75, label='IFS Corrected')
    
    # Customize plot
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title(f'Monthly RMSE Comparison - {region.replace("_", " ").title()}\n'
                 f'{prediction_var.replace("_", " ").title()} - {lead_time}h Lead Time - Patch Size: {subregion}',
                 fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    out_folder = os.path.join(dirs["fig"], model, "time_series", region, subregion)
    os.makedirs(out_folder, exist_ok=True)
    
    fname = f"rmse_monthly_{prediction_var}_trainedwith_{training_vars_str}_{lead_time}h.png"
    save_path = os.path.join(out_folder, fname)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Time series plot saved to: {save_path}")

def generate_time_series_plots(
        dirs,
        train_start, train_end,
        test_start, test_end,
        model,
        training_output_vars,
        prediction_var,
        mlp_params,
        region,
        subregion,
        lead_time,
):
    """
    Generates a single bar plot showing monthly RMSE for original and corrected forecasts
    for both the main model (e.g., pangu) and IFS.
    
    Parameters
    ----------
    dirs : dict
        Dictionary containing paths for input and figure directories
    train_start, train_end : str
        Training period boundaries
    test_start, test_end : str
        Testing period boundaries
    model : str
        Model name (e.g., 'pangu', 'ifs')
    training_output_vars : tuple
        Tuple of (training_vars, output_vars)
    prediction_var : str
        Variable to predict
    mlp_params : tuple
        MLP architecture parameters (hidden_dim, layers)
    region : str
        Region identifier
    subregion : str
        Subregion identifier
    lead_time : int
        Forecast lead time in hours
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
    mlp_str = f"mlp{mlp_params[0]}x{mlp_params[1]}"
    time_str = f"train{train_start}-{train_end}_test{test_start}-{test_end}"

    # Construct file paths for main model and IFS
    model_file_path = os.path.join(
        dirs['input'],
        f"{model}/{region}/train_{training_vars_str}_test_{output_vars_str}_"
        f"dim{subregion}_leadtime_{lead_time}h_{time_str}_{mlp_str}.zarr"
    )
    
    ifs_file_path = os.path.join(
        dirs['input'],
        f"ifs/{region}/train_{training_vars_str}_test_{output_vars_str}_"
        f"dim{subregion}_leadtime_{lead_time}h_{time_str}_{mlp_str}.zarr"
    )

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
        ground_truth = ds_model[f"{prediction_var}_ground_truth"]
        fc_original = ds_model[f"{prediction_var}_original"]
        fc_corrected = ds_model[f"{prediction_var}_corrected"]
        
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
        ifs_ground_truth = ds_ifs[f"{prediction_var}_ground_truth"]
        ifs_fc_original = ds_ifs[f"{prediction_var}_original"]
        ifs_fc_corrected = ds_ifs[f"{prediction_var}_corrected"]
        
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
    
    # Create the bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set up bar positions
    x = np.arange(len(months))
    bar_width = 0.35
    
    # Plot main model bars (overlapping with transparency)
    ax.bar(x - bar_width/2, model_rmse_orig, bar_width, 
           color='blue', alpha=0.5, label=f'{model.upper()} Original')
    ax.bar(x - bar_width/2, model_rmse_corr, bar_width, 
           color='red', alpha=0.5, label=f'{model.upper()} Corrected')
    
    # Plot IFS bars if available (offset to the right)
    if has_ifs_data:
        ifs_bar_width = bar_width * 0.5  # Make IFS bars narrower
        offset = bar_width/2 + 0.05  # Small gap between model and IFS bars
        
        ax.bar(x + offset, ifs_rmse_orig, ifs_bar_width, 
               color='darkblue', alpha=0.75, label='IFS Baseline')
        ax.bar(x + offset + ifs_bar_width, ifs_rmse_corr, ifs_bar_width, 
               color='#ADD8E6', alpha=0.75, label='IFS Corrected')
    
    # Customize plot
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title(f'Monthly RMSE Comparison - {region.replace("_", " ").title()}\n'
                 f'{prediction_var.replace("_", " ").title()} - {lead_time}h Lead Time - Patch Size: {subregion}',
                 fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    out_folder = os.path.join(dirs["fig"], model, "time_series", region, subregion)
    os.makedirs(out_folder, exist_ok=True)
    
    fname = f"rmse_monthly_{prediction_var}_trainedwith_{training_vars_str}_{lead_time}h.png"
    save_path = os.path.join(out_folder, fname)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Time series plot saved to: {save_path}")


def main():

    dirs = setup_directories()

    # three options for training and output variable combinations, uncomment the one you want to use

    training_vars = ["2m_temperature"]
    output_vars = ["2m_temperature"]
    prediction_var = "2m_temperature"

    # training_vars = ["10m_wind_speed"]
    # output_vars = ["10m_wind_speed"]
    # prediction_var = "10m_wind_speed"

    # Compare multiple runs across lead times and regions in a single plot.
    generate_rmse_comparison_plot(
        dirs=dirs,
        model="pangu",
        training_output_vars=(training_vars, output_vars),
        prediction_var=prediction_var,
        mlp_params=(512, 5)
    )

    regions = ["india", "amazon", "british_columbia", "usa_south"]
    generate_lead_time_plots(
        dirs = dirs,
        train_start="2018-01-01",
        train_end="2021-12-31",
        test_start="2022-01-01",
        test_end="2022-12-31",
        model="pangu",
        training_output_vars=(training_vars, output_vars),
        prediction_var=prediction_var,
        mlp_params=(512, 5), 
        regions = regions,
        subregion="10x10",
        bootstrap=False
    )
    climate_regions = ["tropical", "arid", "temperate"]
    generate_lead_time_plots(
        dirs = dirs,
        train_start="2018-01-01",
        train_end="2021-12-31",
        test_start="2022-01-01",
        test_end="2022-12-31",
        model="pangu",
        training_output_vars=(training_vars, output_vars),
        prediction_var=prediction_var,
        mlp_params=(512, 5), 
        regions = climate_regions,
        subregion="2x2",
        bootstrap=True
    )

    generate_subregion_comparison_plots(
        dirs = dirs,
        train_start="2018-01-01",
        train_end="2021-12-31",
        test_start="2022-01-01",
        test_end="2022-12-31",
        model="pangu",
        training_output_vars=(training_vars, output_vars),
        prediction_var=prediction_var,
        mlp_params=(512, 5)
    )

    regions = ["usa_south", "amazon", "india", "british_columbia"]
    for region in regions:
        generate_map_plots(
            dirs=dirs,
            train_start="2018-01-01",
            train_end="2021-12-31",
            test_start="2022-01-01",
            test_end="2022-12-31",
            model="pangu",
            training_output_vars=(training_vars, output_vars),
            prediction_var=prediction_var,
            mlp_params=(512, 5),
            region=region,
            subregion="10x10",
            lead_time=24
        )

        # Generate time series plots
        generate_time_series_plots(
            dirs=dirs,
            train_start="2018-01-01",
            train_end="2021-12-31",
            test_start="2022-01-01",
            test_end="2022-12-31",
            model="pangu",
            training_output_vars=(training_vars, output_vars),
            prediction_var=prediction_var,
            mlp_params=(512, 5),
            region=region,
            subregion="10x10",
            lead_time=24
        )

if __name__ == "__main__":
    main()