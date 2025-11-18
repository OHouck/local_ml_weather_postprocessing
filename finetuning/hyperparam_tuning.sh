#!/bin/bash
#SBATCH --exclusive
#SBATCH --job-name=mlp_finetune
#SBATCH --account=pi-jfranke
#SBATCH --output=hyperparam_mlp_temp-%J.txt
#SBATCH --error=hyperparam_mlp_temp-%J.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1 
#SBATCH --mem=32G
 
module load python/3.11.9
source .venv/bin/activate

uv run finetuning/hyperparam_tuning.py 
 
