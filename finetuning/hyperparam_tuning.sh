#!/bin/bash
#SBATCH --exclusive
#SBATCH --job-name=mlp_finetune
#SBATCH --account=pi-jfranke
#SBATCH --output=hyperparam_2m_temp-%J.txt
#SBATCH --error=hyperparam_2m_temp-%J.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4 
#SBATCH --mem=64G
 
module load python/3.11.9
source .venv/bin/activate

uv run finetuning/hyperparam_tuning.py 
 
