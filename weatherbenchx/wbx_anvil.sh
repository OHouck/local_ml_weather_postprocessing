#!/bin/sh -l
# FILENAME: wbx_anvil.sh

#SBATCH --account=atm170020-gpu # Allocation name
#SBATCH -p gpu # GPU partition
#SBATCH --time=12:00:00
#SBATCH --mem=256G 
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1 # Number of GPUs per node

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

#SBATCH --job-name weatherbenchx
#SBATCH -e job.e%j
#SBATCH -o job.o%j
#SBATCH --mail-user=ohouck@uchicago.edu
#SBATCH --mail-type=all # send email to above address at start and end of job

module load anaconda/2024.02-py311
conda activate /home/x-ohouck/aiw_env

# move to code directory
cd /anvil/projects/x-atm170020/ohouck/ai_weather_ag

python3 weatherbenchx/custom_weatherbench_download.py