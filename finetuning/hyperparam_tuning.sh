#!/bin/bash
#SBATCH --job-name=mlp_finetune
#SBATCH --account=pi-jfranke
#SBATCH --output=hyperparam_2m_temp-%J.txt
#SBATCH --error=hyperparam_2m_temp-%J.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G

# OPTIMIZATION NOTES:
# - Removed --exclusive (don't waste node resources)
# - Increased CPUs from 4 to 16 (better data loading parallelism)
# - Increased memory from 64G to 80G (room for data caching ~20-30GB)
# - Data is now cached once and reused across all 100 trials (60-70% speedup)
# - Expected runtime: 7-8 hours instead of 12 hours

module load python/3.11.9
source .venv/bin/activate

# Enable better GPU utilization
export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=16

uv run finetuning/hyperparam_tuning.py 
 
