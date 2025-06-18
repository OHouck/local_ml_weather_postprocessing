#!/bin/sh -l
# FILENAME: test_job.sbatch 

#SBATCH --account=atm170020-gpu # Allocation name
#SBATCH -p gpu # GPU partition
#SBATCH --time=12:00:00
#SBATCH --mem=256G 
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1 # Number of GPUs per node

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

#SBATCH --job-name weatherbench2 
#SBATCH -e job.e%j
#SBATCH -o job.o%j
#SBATCH --mail-user=ohouck@uchicago.edu
#SBATCH --mail-type=all # send email to above address at start and end of job