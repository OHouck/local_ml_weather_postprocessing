#!/bin/bash
#SBATCH --job-name=run_heatwave_aifs_mlp
#SBATCH --account=pi-jfranke
#SBATCH --output=run_heatwave_aifs_mlp%J.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#
# Post-process the AIFS heat-wave 2m_temperature forecasts. All run configuration
# lives in the Python driver via PostProcessConfig (see run_heatwave_experiment.py);
# this wrapper just runs the two steps.

# 1. Build the cleaned forecast + matching ERA5 ground truth (cached after first run).
uv run python heat_wave/preprocess_temp_forecast.py

# 2. Train the post-processing model and write the corrected-forecast zarr.
uv run python heat_wave/run_heatwave_experiment.py
