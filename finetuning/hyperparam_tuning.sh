#!/bin/bash
#SBATCH --job-name=hyperparam_tuning
#SBATCH --account=pi-jfranke
#SBATCH --output=hyperparam_tuning-%J.txt
#SBATCH --error=hyperparam_tuning-%J.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G

# OPTIMIZATION NOTES:
# - Data is cached once and reused across all trials (60-70% speedup)
# - Snapshot ensemble trials are ~0.2 min each on GPU (~3.5 hrs for 100 evals)
# - Early-stopping trials are ~5-10 min each; use fewer evals (50-75) on cluster
# - Edit TUNING_MODE and USE_SNAPSHOT_ENSEMBLE in hyperparam_tuning.py before submitting
#
# TUNING_MODE options (set in hyperparam_tuning.py __main__):
#   "temperature"  → tunes for 2m_temperature, saves to hyperopt_results_[snapshot_]temperature_mlp/
#   "wind"         → tunes for 10m_wind_speed,  saves to hyperopt_results_[snapshot_]wind_mlp/
#   "joint"        → tunes joint temp+wind loss, saves to hyperopt_results_[snapshot_]joint_..._mlp/
#
# USE_SNAPSHOT_ENSEMBLE options (set in hyperparam_tuning.py __main__):
#   True  → snapshot ensemble search space (lr, hidden_dim, T0, etc.)
#   False → early-stopping search space (adds patience, min_delta)

module load python/3.11.9
source .venv/bin/activate

# Enable better GPU utilization
export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=8
uv run finetuning/hyperparam_tuning.py
