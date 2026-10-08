"""
Section 4 (part 1): hyperparameter search for the gridded post-processing models.

For each model variant and variable, a Bayesian search (hyperopt's tree of
Parzen estimators, 100 trials) looks for the hyperparameters that minimise
the error of the corrected forecast in a held-out development year. Each
trial trains the variant on 2018-2020 exactly as the production run trains on
2018-2021 (same code, weekly validation blocks), corrects the 2021 forecasts,
and is scored by their mean squared error in normalised units, over every
grid cell and lead time.

Pangu forecasts are used throughout. Most variants are searched on one 6x6
degree patch over India. The block leave-time-out ensemble is searched over a
reproducible 25% sample of the land patches instead, scoring each trial on 3
of them in rotation. Its networks train on one year each and validate on the
other search years, as in production.

The best hyperparameters of each search replace that variant's entry in
tuned_hyperparameters.json, which the training scripts read. Every trial is
also kept in hyperparameter_searches/{variant}_{variable}.pkl, so an
interrupted search resumes where it stopped.

Run on its own with:
    uv run python gridded_post_processing/hyperparameter_search.py
"""

import json
import os
import pickle
import sys
import warnings

import numpy as np

# hyperopt imports the deprecated pkg_resources, which prints a harmless warning
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
from hyperopt import STATUS_OK, Trials, fmin, hp, space_eval, tpe  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (MODEL_VARIANTS, PATCH_SAMPLE_SEED, VARIABLES,  # noqa: E402
                    sample_continent_patches, setup_directories)
from gridded_post_processing.training import (HYPERPARAMETER_FILE,  # noqa: E402
                                              post_process_patch, select_device)
from gridded_post_processing.training_data import load_patch_data  # noqa: E402

NUMBER_OF_TRIALS = 100
SEARCH_SEED = 42
SEARCH_TRAINING_YEARS = ("2018-01-01", "2020-12-31")
DEVELOPMENT_YEAR = ("2021-01-01", "2021-12-31")
SEARCH_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "hyperparameter_searches")

# The single search patch: 6x6 degrees centred on 22N, 77E (India)
INDIA_LATITUDES = np.arange(19.0, 25.0, 0.25)
INDIA_LONGITUDES = np.arange(74.0, 80.0, 0.25)

# The block leave-time-out search draws from this share of the land patches,
# scoring each trial on this many of them
BLOCK_SEARCH_PATCH_FRACTION = 0.25
BLOCK_SEARCH_PATCHES_PER_TRIAL = 3

# Search space of each model variant
EARLY_STOPPING_SPACE = {
    "hidden_dim": hp.choice("hidden_dim", [64, 128, 256, 512, 1024]),
    "num_layers": hp.choice("num_layers", [2, 3, 4, 5, 6, 8, 10]),
    "learning_rate": hp.loguniform("learning_rate", np.log(1e-6), np.log(1e-2)),
    "batch_size": hp.choice("batch_size", [64, 128, 256]),
    "weight_decay": hp.loguniform("weight_decay", np.log(1e-6), np.log(1e-3)),
    "patience": hp.choice("patience", [15, 20, 25, 30, 50, 100]),
    "min_delta": hp.loguniform("min_delta", np.log(1e-5), np.log(1e-3)),
    "lead_time_embedding_dim": hp.choice("lead_time_embedding_dim", [4, 8, 16]),
    "dropout_rate": hp.uniform("dropout_rate", 0.1, 0.3),
}
SNAPSHOT_SPACE = {
    "hidden_dim": hp.choice("hidden_dim", [32, 64, 128, 256, 512, 1024]),
    "num_layers": hp.choice("num_layers", [2, 3, 4, 5, 6, 8, 10]),
    "lead_time_embedding_dim": hp.choice("lead_time_embedding_dim", [4, 8, 16]),
    "learning_rate": hp.loguniform("learning_rate", np.log(1e-4), np.log(1e-2)),
    "batch_size": hp.choice("batch_size", [64, 128, 256]),
    "weight_decay": hp.loguniform("weight_decay", np.log(1e-6), np.log(1e-3)),
    "snapshot_T0": hp.choice("snapshot_T0", [20, 30, 42]),
    "snapshot_T_mult": hp.choice("snapshot_T_mult", [1, 2]),
    "dropout_rate": hp.uniform("dropout_rate", 0.1, 0.3),
}
SEARCH_SPACES = {
    "mlp": EARLY_STOPPING_SPACE,
    "mlp_snapshot3": SNAPSHOT_SPACE,
    "mlp_snapshot3_perlt": SNAPSHOT_SPACE,
    "mlp_blockk3_snapshot1": {
        "hidden_dim": hp.choice("hidden_dim", [64, 128, 256, 512, 1024]),
        "num_layers": hp.choice("num_layers", [2, 3, 4, 5, 6]),
        "learning_rate": hp.loguniform("learning_rate", np.log(5e-5), np.log(5e-3)),
        "batch_size": hp.choice("batch_size", [64, 128, 256]),
        "weight_decay": hp.loguniform("weight_decay", np.log(1e-7), np.log(1e-3)),
        "snapshot_T0": hp.choice("snapshot_T0", [5, 7, 10, 15, 21]),
        "snapshot_T_mult": hp.choice("snapshot_T_mult", [1, 2]),
        "dropout_rate": hp.uniform("dropout_rate", 0.1, 0.4),
    },
    "unet": {
        "hidden_dim": hp.choice("hidden_dim", [64, 128]),
        "learning_rate": hp.loguniform("learning_rate", np.log(1e-4), np.log(1e-2)),
        "batch_size": hp.choice("batch_size", [64, 128, 256]),
        "weight_decay": hp.loguniform("weight_decay", np.log(1e-6), np.log(1e-3)),
        "patience": hp.choice("patience", [15, 20, 25, 30]),
        "min_delta": hp.loguniform("min_delta", np.log(1e-5), np.log(1e-3)),
        "lead_time_embedding_dim": hp.choice("lead_time_embedding_dim", [4, 8, 16]),
        "dropout_rate": hp.uniform("dropout_rate", 0.05, 0.20),
    },
}


def search_hyperparameters(directories, model_suffix, variable, device):
    """
    Run the hyperparameter search of one model variant and variable, and save the best.

    Inputs:
        directories (dict): output of common.setup_directories().
        model_suffix (str): a key of SEARCH_SPACES.
        variable (str): '2m_temperature' or '10m_wind_speed'.
        device (torch.device): output of training.select_device().

    Returns:
        dict, the best hyperparameters found.
    """
    # Resume from the trials saved so far
    os.makedirs(SEARCH_DIRECTORY, exist_ok=True)
    trials_path = os.path.join(SEARCH_DIRECTORY, f"{model_suffix}_{variable}.pkl")
    trials = Trials()
    if os.path.exists(trials_path):
        with open(trials_path, "rb") as trials_file:
            trials = pickle.load(trials_file)
    trial_count = len(trials.trials)

    # Load the search patches once: (search-year data, development-year data).
    # A finished search needs no data.
    search_patches = []
    if trial_count < NUMBER_OF_TRIALS:
        if MODEL_VARIANTS[model_suffix]["block_ensemble"]:
            patch_coordinates = [(coordinates[0], coordinates[1]) for _, _, coordinates
                                 in sample_continent_patches(directories,
                                                             BLOCK_SEARCH_PATCH_FRACTION,
                                                             PATCH_SAMPLE_SEED, split="hyperopt")]
        else:
            patch_coordinates = [(INDIA_LATITUDES, INDIA_LONGITUDES)]
        search_patches = [(load_patch_data(directories, "pangu", variable, latitudes, longitudes,
                                           *SEARCH_TRAINING_YEARS),
                           load_patch_data(directories, "pangu", variable, latitudes, longitudes,
                                           *DEVELOPMENT_YEAR))
                          for latitudes, longitudes in patch_coordinates]
    patches_per_trial = min(BLOCK_SEARCH_PATCHES_PER_TRIAL, max(len(search_patches), 1))

    def development_year_error(hyperparameters):
        """
        Train with one set of hyperparameters and score it on the development year.

        Inputs:
            hyperparameters (dict): one point of the search space.

        Returns:
            dict with the hyperopt 'loss' (normalised mean squared error,
            averaged over this trial's patches) and 'status'.
        """
        # Rotate through the search patches, a few per trial
        nonlocal trial_count
        first_patch = trial_count * patches_per_trial
        trial_count += 1
        patch_errors = []
        for offset in range(patches_per_trial):
            training_data, development_data = search_patches[(first_patch + offset)
                                                             % len(search_patches)]
            corrected, _ = post_process_patch(training_data, development_data, model_suffix,
                                              hyperparameters, device)
            forecast_deviation = training_data["forecast"].std(0) + 1e-8
            normalised_error = (corrected - development_data["truth"]) / forecast_deviation
            patch_errors.append(float(np.mean(normalised_error ** 2)))
        return {"loss": float(np.mean(patch_errors)), "status": STATUS_OK}

    best_point = fmin(development_year_error, SEARCH_SPACES[model_suffix], algo=tpe.suggest,
                      max_evals=NUMBER_OF_TRIALS, trials=trials,
                      rstate=np.random.default_rng(SEARCH_SEED), trials_save_file=trials_path)
    best_hyperparameters = space_eval(SEARCH_SPACES[model_suffix], best_point)

    # Record the best hyperparameters for the training scripts
    with open(HYPERPARAMETER_FILE) as hyperparameter_file:
        tuned = json.load(hyperparameter_file)
    tuned[model_suffix][variable] = best_hyperparameters
    with open(HYPERPARAMETER_FILE, "w") as hyperparameter_file:
        json.dump(tuned, hyperparameter_file, indent=2)
    print(f"  {model_suffix} {variable}: best loss {min(trials.losses()):.5f}, "
          f"{best_hyperparameters}", flush=True)
    return best_hyperparameters


def run():
    """
    Search the hyperparameters of every model variant for both variables.

    Comment out variants in the loop below to skip them.

    Inputs:
        None.

    Returns:
        None. Updates tuned_hyperparameters.json.
    """
    directories = setup_directories()
    device = select_device()
    for model_suffix in SEARCH_SPACES:
        for variable in VARIABLES:
            search_hyperparameters(directories, model_suffix, variable, device)


if __name__ == "__main__":
    run()
