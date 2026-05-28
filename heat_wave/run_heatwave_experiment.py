"""
heat_wave/run_heatwave_experiment.py

Train a lightweight MLP to post-process the AIFS heat-wave 2m_temperature forecasts,
using the refactored importable API in finetuning/post_process.py
(PostProcessConfig + post_process_forecasts) rather than the old argparse/CLI path.

Expects the cleaned forecast and matching ERA5 ground-truth datasets produced by
heat_wave/preprocess_temp_forecast.py. Builds a PostProcessConfig, reuses the tuned
snapshot-MLP hyperparameters, writes the corrected-forecast zarr, and prints the
test-set metrics (RMSE original/corrected, % improvement).

Run with: uv run python heat_wave/run_heatwave_experiment.py
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import xarray as xr

sys.path.insert(0, str(Path(__file__).parent.parent))
from helper_funcs import setup_directories, generate_output_path
from finetuning.post_process import (
    PostProcessConfig,
    post_process_forecasts,
    load_optimal_hyperparameters,
)
from heat_wave.preprocess_temp_forecast import cache_paths

# Experiment settings
REGION = "pnw"
SUBREGION = "6x6"
MODEL_NAME = "aifs"
TRAINING_VARS = ["2m_temperature"]
OUTPUT_VARS = ["2m_temperature"]
LEAD_TIME_HOURS = [24]
TRAIN_START, TRAIN_END = "2000-01-01", "2020-12-31"
TEST_START, TEST_END = "2021-01-01", "2021-12-31"


def build_config():
    """Build the PostProcessConfig for the heat-wave run with tuned hyperparameters.

    Mirrors the hyperparameter merge in post_process.main(): loads the tuned snapshot
    MLP hyperparameters for 2m_temperature and folds them into the config.

    Inputs: none (uses module-level experiment settings).

    Returns:
        PostProcessConfig: fully populated run configuration.
    """
    config = PostProcessConfig(
        training_vars=TRAINING_VARS,
        output_vars=OUTPUT_VARS,
        lead_time_hours=LEAD_TIME_HOURS,
        train_start=TRAIN_START,
        train_end=TRAIN_END,
        test_start=TEST_START,
        test_end=TEST_END,
        nn_architecture="mlp",
        alternate_loss_fn=None,
        snapshot_ensemble=3,
        snapshot_epochs=210,
        snapshot_T0=30,
        model_name=MODEL_NAME,
    )

    hp = load_optimal_hyperparameters(
        "mlp", TRAINING_VARS, OUTPUT_VARS, None, use_snapshot=True
    )
    if hp:
        config.mlp_hidden_dim = hp.get("hidden_dim", config.mlp_hidden_dim)
        config.mlp_num_layers = hp.get("num_layers", config.mlp_num_layers)
        config.mlp_dropout = hp.get("dropout_rate", config.mlp_dropout)
        config.optimal_lr = hp.get("learning_rate")
        config.optimal_batch_size = hp.get("batch_size")
        config.optimal_weight_decay = hp.get("weight_decay")
        config.optimal_patience = hp.get("patience")
        config.optimal_min_delta = hp.get("min_delta")
        config.optimal_lead_time_embedding_dim = hp.get("lead_time_embedding_dim")
        config.optimal_snapshot_T0 = hp.get("snapshot_T0")
        config.optimal_snapshot_T_mult = hp.get("snapshot_T_mult")
        print("Using tuned snapshot MLP hyperparameters.")
    else:
        print("No tuned hyperparameters found; using PostProcessConfig defaults.")

    return config


def build_output_path(config):
    """Build the standardized corrected-forecast zarr path for this run.

    Inputs:
        config (PostProcessConfig): run configuration.

    Returns:
        str: absolute path under <processed>/finetuning_output/.
    """
    dirs = setup_directories()
    output_dir = dirs["input"]  # <processed>/finetuning_output
    # generate_output_path reads region/subregion/dates/etc. off an args-like object.
    args = SimpleNamespace(
        region=REGION,
        subregion=SUBREGION,
        train_start=config.train_start,
        train_end=config.train_end,
        test_start=config.test_start,
        test_end=config.test_end,
        training_vars=config.training_vars,
        output_vars=config.output_vars,
        nn_architecture=config.nn_architecture,
        alternate_loss_fn=config.alternate_loss_fn,
        snapshot_ensemble=config.snapshot_ensemble,
        block_ensemble=config.block_ensemble,
        block_holdout=config.block_holdout,
        per_lead_time=config.per_lead_time,
        growing_season_only=config.growing_season_only,
        lead_time_hours=config.lead_time_hours,
        model_name=config.model_name,
        ground_truth_source=config.ground_truth_source,
    )
    return os.path.join(output_dir, generate_output_path(args))


def main():
    """Load preprocessed data, train the post-processing model, and report metrics."""
    forecast_path, era5_path = cache_paths()
    if not (os.path.exists(forecast_path) and os.path.exists(era5_path)):
        raise FileNotFoundError(
            "Preprocessed data not found. Run "
            "`uv run python heat_wave/preprocess_temp_forecast.py` first."
        )

    forecast_ds = xr.open_dataset(forecast_path)
    ground_truth_ds = xr.open_dataset(era5_path)

    config = build_config()
    output_path = build_output_path(config)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Output zarr: {output_path}")

    result = post_process_forecasts(
        forecast_ds, ground_truth_ds, config, output_path=output_path
    )

    print(f"\nTraining time: {result.training_time_minutes:.1f} min")
    print("Test metrics:")
    print(result.metrics.to_string(index=False))


if __name__ == "__main__":
    main()
