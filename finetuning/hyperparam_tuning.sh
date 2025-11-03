#!/bin/bash
#SBATCH --exclusive
#SBATCH --job-name=mlp_finetune
#SBATCH --account=pi-jfranke
#SBATCH --output=output-%J.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1 
 
module load python/3.11.9
source .venv/bin/activate

uv run finetuning/run_experiments.sh 
 
