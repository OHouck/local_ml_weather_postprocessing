# functions shared across files

import socket
import os

def setup_directories():
    """Set up directory structure based on environment."""
    nodename = socket.gethostname()
    if nodename == "oMac.local":
        root = os.path.expanduser(f"/Users/ohouck/globus/forecast_data")
    elif nodename == "midway3-amd2.rcc.local":
        root = os.path.expanduser("/project/jfranke/ozma/data")

    else:
        raise Exception(f"Unknown environment, Please specify the root directory. "
                        f"Nodename found: {nodename}")

    dirs = {
        'root': root,
        'raw': os.path.join(root, "raw"),
        'processed': os.path.join(root, "processed"),
        'fig': os.path.join(root, "figures/finetuning"),
        'input': os.path.join(root, "fine_tuning_output")
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
    else:
        raise ValueError(f"Unknown nn_architecture: {args.nn_architecture}")
    if args.alternate_loss_fn is not None:
        model_str += f"_{args.alternate_loss_fn}"

    if args.growing_season_only:
        grow_str = "_growing_season"
    else:
        grow_str = ""
    
     # Format lead times
    lead_times_str = "leadtime_" + "_".join([str(lt) for lt in args.lead_time_hours]) + "h"
    output_path = f"{args.model_name}/{args.ground_truth_source}{region_str}/train_{training_vars_str}_test_{output_vars_str}_dim{subregion_str}_{lead_times_str}{grow_str}_{dates_str}_{model_str}.zarr"

    return output_path