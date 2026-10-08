"""
Settings shared by every stage of the pipeline.

This file holds the few things more than one stage needs to agree on:
    - where the data lives on each machine (setup_directories)
    - the forecast models, variables, lead times and train/test years
    - the gridded model variants, and how their output files are named, so
      the training scripts and the figure scripts read and write the same files
    - the list of 6x6 degree land patches the gridded models are trained on
"""

import os
import socket
import warnings

import numpy as np

# Harmless zarr warnings, silenced for every script that imports this one:
# some local archives (rewritten in zarr v3 format) have no consolidated
# metadata, so opening them falls back to listing their directory, where
# macOS's Finder leaves .DS_Store files that zarr does not recognise; and
# writing a zarr v3 store notes that consolidated metadata is not yet part of
# the v3 specification
warnings.filterwarnings("ignore", message="Failed to open Zarr store with consolidated metadata")
warnings.filterwarnings("ignore", message="Object at .DS_Store is not recognized")
warnings.filterwarnings("ignore", message="Consolidated metadata is currently not part")

# ---------------------------------------------------------------------------
# Study design shared by the gridded and the station analyses
# ---------------------------------------------------------------------------
FORECAST_MODELS = ["pangu", "ifs"]
# The corrected variables, in the order of the variable axis of the station arrays
VARIABLES = ["2m_temperature", "10m_wind_speed"]
# The archive variables they are built from (wind speed comes from u and v)
DOWNLOADED_VARIABLES = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"]
LEAD_TIMES_HOURS = [24, 120, 216]
TRAINING_YEARS = [2018, 2019, 2020, 2021]
TEST_YEAR = 2022
ALL_YEARS = TRAINING_YEARS + [TEST_YEAR]
TRAIN_START, TRAIN_END = f"{TRAINING_YEARS[0]}-01-01", f"{TRAINING_YEARS[-1]}-12-31"
TEST_START, TEST_END = f"{TEST_YEAR}-01-01", f"{TEST_YEAR}-12-31"

# Each forecast model is post-processed against its own gridded "truth": Pangu
# was trained on ERA5, and IFS is verified against its own analysis (HRES t0),
# which is the WeatherBench2 convention.
GRIDDED_TRUTH_SOURCE = {"pangu": "era5", "ifs": "hres_t0"}

# The continents the land patches are grouped into. Each one is a
# sub-directory of the gridded output directory.
CONTINENTS = ["africa", "asia", "europe", "north_america", "south_america", "oceania"]

# Every gridded model is trained on a 6x6 degree patch (24 x 24 grid cells).
PATCH_SIZE = "6x6"

# The gridded model variants, named as in their output files. A block ensemble
# trains one network per training year and validates it on the other years.
MODEL_VARIANTS = {
    "mlp": dict(architecture="mlp", snapshot_count=None, block_ensemble=False,
                per_lead_time=False),
    "unet": dict(architecture="unet", snapshot_count=None, block_ensemble=False,
                 per_lead_time=False),
    "mlp_snapshot3": dict(architecture="mlp", snapshot_count=3, block_ensemble=False,
                          per_lead_time=False),
    "mlp_snapshot3_perlt": dict(architecture="mlp", snapshot_count=3, block_ensemble=False,
                                per_lead_time=True),
    "mlp_blockk3_snapshot1": dict(architecture="mlp", snapshot_count=1, block_ensemble=True,
                                  per_lead_time=False),
}
# The paper's production model: an MLP snapshot ensemble of 3 runs
PRODUCTION_MODEL_SUFFIX = "mlp_snapshot3"

# Seed of the shuffle the land patch samples are drawn from, and the share of
# patches used to evaluate the architecture comparison. They have to match
# between the experiment that trains on them and the figure that plots them.
PATCH_SAMPLE_SEED = 42
EVALUATION_PATCH_FRACTION = 0.05

# Station analysis: stations reporting a 00 UTC temperature on fewer than this
# share of training days are not used, and the forecast neighbourhood around a
# station reaches this many grid cells either side (a 5x5 window)
MINIMUM_TRAINING_COVERAGE = 0.70
NEIGHBOURHOOD_HALF_WIDTH = 2

FIGURE_DIRECTORY_NAME = "erc_figures"


def setup_directories():
    """
    Return the data directories for the machine this code is running on.

    Inputs:
        None. The machine is recognised from its hostname.

    Returns:
        dict of str -> str, with keys
            'root'            the data root
            'raw'             downloaded forecasts, reanalyses and station files
            'processed'       derived data (land patch lists, model outputs)
            'gridded_output'  gridded post-processing outputs (zarr files)
            'station_output'  station datasets and station results
            'figures'         the paper figures
        Every directory is created if it does not exist.
    """
    hostname = socket.gethostname()
    if hostname == "oMac.local":
        root = "/Users/ohouck/globus/forecast_data"
    elif "midway3" in hostname:
        root = "/project/jfranke/ozma/forecast_data"
    else:
        raise RuntimeError(f"Unknown machine '{hostname}'. Add its data root "
                           f"to common.setup_directories().")

    directories = {
        "root": root,
        "raw": os.path.join(root, "raw"),
        "processed": os.path.join(root, "processed"),
        "gridded_output": os.path.join(root, "processed", "finetuning_output"),
        "station_output": os.path.join(root, "processed", "station_finetuning"),
        "figures": os.path.join(root, FIGURE_DIRECTORY_NAME),
    }
    for path in directories.values():
        os.makedirs(path, exist_ok=True)
    return directories


def gridded_output_path(directories, forecast_model, continent, patch_number, variable,
                        model_suffix):
    """
    Path of the zarr file holding one patch's gridded post-processing output.

    The name records the training variable, patch size, lead times, dates and
    model variant, so runs with different settings never overwrite each other.

    Inputs:
        directories (dict): output of setup_directories().
        forecast_model (str): 'pangu' or 'ifs'.
        continent (str): continent the patch belongs to.
        patch_number (int): 1-based index of the patch within its continent.
        variable (str): the variable trained on and corrected.
        model_suffix (str): a key of MODEL_VARIANTS, e.g. 'mlp_snapshot3'.

    Returns:
        str, the full path of the zarr file.
    """
    lead_times = "_".join(str(lead_time) for lead_time in LEAD_TIMES_HOURS)
    filename = (f"train_{variable}_test_{variable}_dim{PATCH_SIZE}_leadtime_{lead_times}h"
                f"_train{TRAIN_START}-{TRAIN_END}_test{TEST_START}-{TEST_END}"
                f"_{model_suffix}_{continent}_bs{patch_number}.zarr")
    return os.path.join(directories["gridded_output"], forecast_model, continent, filename)


def load_continent_patches(directories):
    """
    Load every 6x6 degree land patch, grouped by continent.

    The patch lists are written by data_preparation/create_land_patches.py.

    Inputs:
        directories (dict): output of setup_directories().

    Returns:
        list of (continent, patch_number, patch_coordinates) tuples, where
        patch_number is 1-based within its continent and patch_coordinates is
        an array of shape (2, 24): row 0 the latitudes, row 1 the longitudes.
    """
    all_patches = []
    for continent in CONTINENTS:
        patch_file = os.path.join(directories["processed"], f"{continent}_patches.npy")
        continent_patches = np.load(patch_file, allow_pickle=True)
        for patch_index, patch_coordinates in enumerate(continent_patches):
            all_patches.append((continent, patch_index + 1, patch_coordinates))
    return all_patches


def sample_continent_patches(directories, fraction, seed, split):
    """
    Draw a reproducible random subset of the land patches.

    The patches are shuffled once with the seed. The 'hyperopt' split takes
    the first n patches of the shuffle and the 'eval' split the next n, where
    n = fraction x number of patches. Note that n is computed from the
    fraction passed in, so the two splits only stay disjoint when they are
    drawn with the same fraction.

    Inputs:
        directories (dict): output of setup_directories().
        fraction (float): fraction of all patches to draw.
        seed (int): seed of the shuffle.
        split (str): 'hyperopt' or 'eval'.

    Returns:
        list of (continent, patch_number, patch_coordinates) tuples, in the
        same order as load_continent_patches().
    """
    all_patches = load_continent_patches(directories)
    sample_size = max(1, int(len(all_patches) * fraction))
    shuffled_indices = np.random.default_rng(seed).permutation(len(all_patches))

    if split == "hyperopt":
        selected_indices = sorted(shuffled_indices[:sample_size])
    elif split == "eval":
        selected_indices = sorted(shuffled_indices[sample_size:2 * sample_size])
    else:
        raise ValueError(f"split must be 'hyperopt' or 'eval', not {split!r}")
    return [all_patches[index] for index in selected_indices]
