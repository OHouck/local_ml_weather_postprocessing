# functions shared across files

import socket
import os
import numpy as np

def setup_directories():
    """Set up directory structure based on environment."""
    nodename = socket.gethostname()
    if nodename == "oMac.local":
        root = os.path.expanduser(f"/Users/ohouck/globus/forecast_data")

    elif "midway3" in nodename:
        root = os.path.expanduser("/project/jfranke/ozma/forecast_data")

    else:
        raise Exception(f"Unknown environment, Please specify the root directory. "
                        f"Nodename found: {nodename}")

    dirs = {
        'root': root,
        'raw': os.path.join(root, "raw"),
        'processed': os.path.join(root, "processed"),
        'fig': os.path.join(root, "figures"),
        'input': os.path.join(root, "processed", "finetuning_output")
    }

    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs



def generate_output_path(args):
    """Generate standardized output path for forecast files."""
    region_str = f"{args.region}"
    subregion_str = f"{args.subregion}"
    dates_str = f"train{args.train_start}-{args.train_end}_test{args.test_start}-{args.test_end}"
    training_vars_str = "_".join(args.training_vars)
    output_vars_str = "_".join(args.output_vars)
    
    # Handle different nn architectures
    if args.nn_architecture == 'mlp':
        model_str = "mlp"
    elif args.nn_architecture == 'unet':
        model_str = "unet"
    elif args.nn_architecture == 'gated_mlp':
        model_str = "gated_mlp"
    elif args.nn_architecture == 'pooled_film':
        model_str = "pooled_film"
    else:
        raise ValueError(f"Unknown nn_architecture: {args.nn_architecture}")
    if args.alternate_loss_fn is not None:
        model_str += f"_{args.alternate_loss_fn}"

    # Append PCA suffix if PCA dimensionality reduction is used
    pca_components = getattr(args, 'pca_components', 0)
    if pca_components > 0:
        model_str += f"_pca{pca_components}"

    # Append snapshot/ensemble/swa/block suffix so runs don't collide in the output directory
    n_snapshot = getattr(args, 'snapshot_ensemble', None)
    n_ensemble = getattr(args, 'ensemble', None)
    n_swa = getattr(args, 'swa_ensemble', None)
    use_block = getattr(args, 'block_ensemble', False)
    block_holdout = getattr(args, 'block_holdout', 1)
    if use_block:
        holdout_suffix = f"k{block_holdout}" if block_holdout != 1 else ""
        model_str += f"_block{holdout_suffix}"
        if n_snapshot:
            model_str += f"_snapshot{n_snapshot}"
    elif n_snapshot:
        model_str += f"_snapshot{n_snapshot}"
    elif n_swa:
        model_str += f"_swa{n_swa}"
    elif n_ensemble:
        model_str += f"_ensemble{n_ensemble}"

    # Append lead-time-weighted loss suffix
    lt_weights = getattr(args, 'lead_time_loss_weights', None)
    if lt_weights is not None:
        model_str += "_ltw"

    # Append C-Mixup suffix
    cmixup_alpha = getattr(args, 'cmixup_alpha', 0.0)
    if cmixup_alpha > 0:
        model_str += f"_cmix{cmixup_alpha:.1f}".replace('.', 'p')

    # Append per-lead-time suffix
    if getattr(args, 'per_lead_time', False):
        model_str += "_perlt"

    # Append small output init suffix
    if getattr(args, 'small_output_init', False):
        model_str += "_soi"

    if args.growing_season_only:
        grow_str = "_growing_season"
    else:
        grow_str = ""

     # Format lead times
    lead_times_str = "leadtime_" + "_".join([str(lt) for lt in args.lead_time_hours]) + "h"
    output_path = f"{args.model_name}/{args.ground_truth_source}{region_str}/train_{training_vars_str}_test_{output_vars_str}_dim{subregion_str}_{lead_times_str}{grow_str}_{dates_str}_{model_str}.zarr"

    return output_path


# ============================================================================
# Continent patch sampling for hyperparameter tuning and evaluation
# ============================================================================

CONTINENTS = ['africa', 'asia', 'europe', 'north_america', 'south_america', 'oceania']


def load_all_continent_patches(processed_dir):
    """
    Load all 6x6 degree patches from all continents.

    Returns:
        list of (continent_name, patch_index, patch_array) tuples
        where patch_array is shape (2, 24) with [0]=lats, [1]=lons
    """
    all_patches = []
    for continent in CONTINENTS:
        patches_path = os.path.join(processed_dir, f"{continent}_patches.npy")
        if not os.path.exists(patches_path):
            print(f"Warning: {patches_path} not found, skipping {continent}")
            continue
        patches = np.load(patches_path, allow_pickle=True)
        for i, patch in enumerate(patches):
            all_patches.append((continent, i + 1, patch))
    return all_patches


def sample_continent_patches(processed_dir, fraction=0.1, seed=42, split='hyperopt'):
    """
    Create a deterministic random subset of continent 6x6 patches.

    Uses different seeds for hyperopt vs eval splits to ensure non-overlapping samples.

    Args:
        processed_dir: Path to processed data directory containing *_patches.npy files
        fraction: Fraction of total patches to sample (default: 0.1 = 10%)
        seed: Base random seed for reproducibility
        split: 'hyperopt' or 'eval' — determines which non-overlapping subset is returned

    Returns:
        list of (continent_name, patch_index, patch_array) tuples
    """
    all_patches = load_all_continent_patches(processed_dir)
    n_total = len(all_patches)
    n_sample = max(1, int(n_total * fraction))

    # Shuffle all patches with the base seed, then take the first 2*n_sample
    # and split into two non-overlapping groups
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_total)

    if split == 'hyperopt':
        selected_indices = sorted(indices[:n_sample])
    elif split == 'eval':
        selected_indices = sorted(indices[n_sample:2 * n_sample])
    else:
        raise ValueError(f"split must be 'hyperopt' or 'eval', got: {split}")

    return [all_patches[i] for i in selected_indices]
