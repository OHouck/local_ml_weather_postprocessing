#!/bin/sh -l
# FILENAME: submit_combine_and_subset_simple.sh
#SBATCH --account=atm170020-gpu
#SBATCH -p gpu 
#SBATCH --time=24:00:00  # Increased to 24 hours for full dataset
#SBATCH --mem=128GB
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name weatherbenchx_download
#SBATCH -e logs/combine_subset_%j.err
#SBATCH -o logs/combine_subset_%j.out
#SBATCH --mail-user=ohouck@uchicago.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Create logs directory
mkdir -p /anvil/projects/x-atm170020/ohouck/ai_weather_ag/logs

# Load modules and activate environment
module load anaconda/2024.02-py311
conda activate /home/x-ohouck/aiw_env

# Move to code directory
cd /anvil/projects/x-atm170020/ohouck/ai_weather_ag

# Set essential environment variables
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

# Simple monitoring function
monitor_progress() {
    while true; do
        echo "[$(date)] Progress check:"
        find /anvil/projects/x-atm170020/ohouck/data/processed/ -name "*.nc" -exec ls -lh {} \; 2>/dev/null
        echo "---"
        sleep 900  # Check every 15 minutes
    done
}

# Start monitoring in background
monitor_progress &
MONITOR_PID=$!

# Run the script
echo "=================================="
echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "=================================="

python3 -u weatherbenchx/combine_and_subset.py --use-incremental

# Capture exit code and cleanup
EXIT_CODE=$?
kill $MONITOR_PID 2>/dev/null

echo "=================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "Final output files:"
find /anvil/projects/x-atm170020/ohouck/data/processed -name "*.nc" -exec ls -lh {} \; 2>/dev/null
echo "=================================="

exit $EXIT_CODE