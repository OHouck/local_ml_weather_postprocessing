#!/bin/bash
#SBATCH --job-name=run_arch_experiments_pangu
#SBATCH --account=pi-jfranke
#SBATCH --output=run_arch_experiments_pangu%J.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
# lat must be between -90 and 90
# lon must be between 0 and 360 (0 is at prime meridian)
#
source .venv/bin/activate

# ============================================================================
# ARCHITECTURE EXPERIMENTS — EVAL ON 10% CONTINENT CELL SAMPLE
# ============================================================================
# This script evaluates the Block LTHO ensemble method on a 10% random sample
# of continent 6x6 cells (the "eval" split, disjoint from the hyperopt split).
# Results are averaged across cells in plot_arch_experiment_results.py.
#
# The Python driver handles cell sampling, lat/lon extraction, and calling
# finetune.py's run_subregion_experiment directly for each cell.
# ============================================================================

python3 finetuning/run_arch_experiments_eval.py
