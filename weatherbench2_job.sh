#!/bin/sh -l
# FILENAME: test_job.sbatch

#SBATCH --account=atm170020-gpu # Allocation name
#SBATCH -p gpu # GPU partition
#SBATCH --time=03:00:00
#SBATCH --mem=128G #24G total
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1 # Number of GPUs per node

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

#SBATCH --job-name weatherbench2 
#SBATCH -e job.e%j
#SBATCH -o job.o%j
#SBATCH --mail-user=ohouck@uchicago.edu
#SBATCH --mail-type=all # send email to above address at start and end of job

# load module and python enviroment

module load anaconda/2024.02-py311
conda activate /home/x-ohouck/aiw_env

# move to code directory
cd /anvil/projects/x-atm170020/ohouck/ai_weather_ag

# Set environment variables for Apache Beam
export BEAM_TEMP=$SCRATCH

set -x
srun -u  \
    bash -c "
    TORCH_USE_CUDA_DSA=1 python weatherbench2_eval.py
    "
