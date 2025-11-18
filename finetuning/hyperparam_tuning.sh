#!/bin/bash
#SBATCH --exclusive
#SBATCH --job-name=hyperparam_tune
#SBATCH --account=pi-jfranke
#SBATCH --output=logs/hyperparam_tuning-%J.txt
#SBATCH --error=logs/hyperparam_tuning-%J.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

# Hyperparameter Tuning Script
# ============================
# This script runs Bayesian hyperparameter optimization using hyperopt.
#
# Search spaces are centered around optimal architectures found in experiments:
# - MLP: hidden_dim=1024, num_layers=6, dropout=0.25 (mlp_moderate)
# - UNet: hidden_dim=64, dropout=0.1 (unet_medium)
#
# The tuning searches for optimal:
# - Learning rate, batch size, weight decay
# - Early stopping parameters (patience, min_delta)
# - Lead time embedding dimension
# - Regularization (dropout)
#
# Results are saved to:
# - hyperopt_results_mlp/ (for MLP optimization)
# - hyperopt_results_unet/ (for UNet optimization)
#
# To resume from previous run: The script automatically resumes if trials file exists
# To start fresh: Delete the hyperopt_results_*/ directory

# Create logs directory if it doesn't exist
mkdir -p logs

echo "=========================================="
echo "HYPERPARAMETER TUNING"
echo "=========================================="
echo "Start time: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
echo ""

module load python/3.11.9
source .venv/bin/activate

echo "Running hyperparameter optimization..."
echo "See configuration in finetuning/hyperparam_tuning.py"
echo ""

uv run finetuning/hyperparam_tuning.py

echo ""
echo "=========================================="
echo "COMPLETE"
echo "=========================================="
echo "End time: $(date)"
echo ""
echo "Results saved to:"
echo "  - hyperopt_results_mlp/"
echo "  - hyperopt_results_unet/ (if enabled)"
echo ""
echo "To view best parameters:"
echo "  cat hyperopt_results_mlp/best_params.json"

