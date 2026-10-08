"""
Section 5: run the gridded (ERA5 / HRES analysis) post-processing.

Trains the production model, an MLP snapshot ensemble of 3 runs, separately
on every 6x6 degree land patch, for Pangu (truth: ERA5) and IFS (truth: its
own HRES t=0 analysis), for 2m temperature and 10m wind speed. Each network is
trained on 2018-2021 and corrects the 2022 forecasts. Every patch's output is
written to
    processed/finetuning_output/{model}/{continent}/train_{variable}_..._mlp_snapshot3_{continent}_bs{N}.zarr

The run can be stopped and restarted: a patch whose output was already written
by this pipeline is skipped. Outputs written before the switch to weekly
validation blocks carry no 'validation_split' attribute and are redone.

Run on its own with:
    uv run python gridded_post_processing/run_post_processing.py
"""

import os
import shutil
import sys
import time

import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (FORECAST_MODELS, PRODUCTION_MODEL_SUFFIX, TEST_END, TEST_START,  # noqa: E402
                    TRAIN_END, TRAIN_START, VARIABLES, gridded_output_path,
                    load_continent_patches, setup_directories)
from gridded_post_processing.training import (build_output_dataset,  # noqa: E402
                                              load_tuned_hyperparameters, post_process_patch,
                                              select_device)
from gridded_post_processing.training_data import load_patch_data  # noqa: E402


def post_process_and_save(directories, forecast_model, variable, continent, patch_number,
                          patch_coordinates, model_suffix, device):
    """
    Train one model variant on one patch and write its corrected test forecasts.

    Skips the patch if its output already exists and was written by this
    pipeline (it carries a 'validation_split' attribute).

    Inputs:
        directories (dict): output of common.setup_directories().
        forecast_model (str): 'pangu' or 'ifs'.
        variable (str): '2m_temperature' or '10m_wind_speed'.
        continent (str), patch_number (int): which patch, as in
            common.load_continent_patches().
        patch_coordinates (numpy.ndarray): (2, 24) patch latitudes and longitudes.
        model_suffix (str): model variant, a key of common.MODEL_VARIANTS.
        device (torch.device): output of training.select_device().

    Returns:
        None. Writes the patch's zarr file.
    """
    output_path = gridded_output_path(directories, forecast_model, continent, patch_number,
                                      variable, model_suffix)
    if os.path.exists(output_path) and "validation_split" in xr.open_zarr(output_path).attrs:
        return

    start_time = time.time()
    latitudes, longitudes = patch_coordinates[0], patch_coordinates[1]
    training_data = load_patch_data(directories, forecast_model, variable, latitudes, longitudes,
                                    TRAIN_START, TRAIN_END)
    test_data = load_patch_data(directories, forecast_model, variable, latitudes, longitudes,
                                TEST_START, TEST_END)
    hyperparameters = load_tuned_hyperparameters(model_suffix, variable)
    corrected, training_minutes = post_process_patch(training_data, test_data, model_suffix,
                                                     hyperparameters, device)

    output = build_output_dataset(training_data, test_data, corrected, variable, forecast_model,
                                  model_suffix, training_minutes)

    # Write beside the target and swap it in when done, so an interrupted
    # write never leaves a partial file that looks finished
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    partial_path = f"{output_path}.partial"
    shutil.rmtree(partial_path, ignore_errors=True)
    output.to_zarr(partial_path, mode="w")
    shutil.rmtree(output_path, ignore_errors=True)
    os.rename(partial_path, output_path)
    print(f"  {forecast_model} {variable} {continent} patch {patch_number} {model_suffix}: "
          f"{(time.time() - start_time) / 60:.1f} min", flush=True)


def run():
    """
    Post-process every land patch with the production model, for both
    forecast models and both variables.

    Inputs:
        None.

    Returns:
        None.
    """
    directories = setup_directories()
    device = select_device()
    all_patches = load_continent_patches(directories)
    print(f"Gridded post-processing of {len(all_patches)} patches on {device}")
    for forecast_model in FORECAST_MODELS:
        for variable in VARIABLES:
            for continent, patch_number, patch_coordinates in all_patches:
                post_process_and_save(directories, forecast_model, variable, continent,
                                      patch_number, patch_coordinates, PRODUCTION_MODEL_SUFFIX,
                                      device)


if __name__ == "__main__":
    run()
