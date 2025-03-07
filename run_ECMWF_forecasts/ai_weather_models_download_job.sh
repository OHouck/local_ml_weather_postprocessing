#!/bin/sh -l
# FILENAME: ai_weather_models_download.job.sh

#SBATCH --account=atm170020-gpu # Allocation name
#SBATCH -p gpu # GPU partition
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=6G #32total
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1 # Number of GPUs per node

#SBATCH --ntasks-per-node=1 # total number of nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

#SBATCH --job-name download_forecasts_ecmwf
#SBATCH -e forecast_download.job.e%j
#SBATCH -o forecast_download.job.o%j
#SBATCH --mail-user=ohouck@uchicago.edu
#SBATCH --mail-type=all # send email to above address at start and end of job

# load module and python enviroment

module load anaconda/2024.02-py311
conda activate /home/x-ohouck/aiw_env

# move to asset directory since my updated path wasn't working?
cd /anvil/projects/x-atm170020/ohouck/ai_weather_assets

set -x
srun -u --mpi=pmi2 \
    bash -c "
    TORCH_USE_CUDA_DSA=1 python ../ai_weather_ag/download_forecasts.py
    "

