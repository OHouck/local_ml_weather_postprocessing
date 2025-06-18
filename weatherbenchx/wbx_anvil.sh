#!/bin/sh -l
# FILENAME: wbx_anvil_optimized.sh

#SBATCH --account=atm170020-gpu # Allocation name
#SBATCH -p gpu 
#SBATCH --time=6:00:00
#SBATCH --mem=128G 
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1 # Number of GPUs per node

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16 
#SBATCH --job-name weatherbenchx_download

#SBATCH -e logs/wbx_download_%j.err
#SBATCH -o logs/wbx_download_%j.out

#SBATCH --mail-user=ohouck@uchicago.edu
#SBATCH --mail-type=BEGIN,END,FAIL # More selective email notifications

# Create logs directory if it doesn't exist
mkdir -p /anvil/projects/x-atm170020/ohouck/ai_weather_ag/logs

# Load modules and activate environment
module load anaconda/2024.02-py311
conda activate /home/x-ohouck/aiw_env

# Move to code directory
cd /anvil/projects/x-atm170020/ohouck/ai_weather_ag

# Set up logging environment variables
export PYTHONUNBUFFERED=1  # Ensure Python output is not buffered
export DASK_LOGGING__DISTRIBUTED=info
export BEAM_LOG_LEVEL=INFO

# Print job information for debugging
echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: $SLURM_MEM_PER_NODE MB"
echo "Working directory: $(pwd)"
echo "Python version: $(python --version)"
echo "NumPy version: $(python -c 'import numpy; print(numpy.__version__)')"
echo "=================================="

# Run the script with enhanced logging
# Using 'script' command to capture all output with timestamps
script -f -c "python3 -u weatherbenchx/custom_weatherbench_download.py" \
  /anvil/projects/x-atm170020/ohouck/ai_weather_ag/logs/wbx_detailed_${SLURM_JOB_ID}.log

# Capture exit code
EXIT_CODE=$?

echo "=================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "Log files:"
echo "  SLURM output: logs/wbx_download_${SLURM_JOB_ID}.out"
echo "  SLURM error:  logs/wbx_download_${SLURM_JOB_ID}.err"
echo "  Detailed log: logs/wbx_detailed_${SLURM_JOB_ID}.log"

# Exit with the same code as the Python script
exit $EXIT_CODE