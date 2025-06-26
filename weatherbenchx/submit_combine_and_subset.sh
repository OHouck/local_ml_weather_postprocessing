#!/bin/sh -l
# FILENAME: submit_combine_and_subset.sh
#SBATCH --account=atm170020-gpu
#SBATCH -p gpu 
#SBATCH --time=6:00:00  
#SBATCH --mem=128GB
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name combine_and_subset
#SBATCH -e logs/combine_and_subset_%j.err
#SBATCH -o logs/combine_and_subset_%j.out
#SBATCH --mail-user=ohouck@uchicago.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Create logs directory if it doesn't exist
mkdir -p /anvil/projects/x-atm170020/ohouck/ai_weather_ag/logs

# Load modules and activate environment
module load anaconda/2024.02-py311
conda activate /home/x-ohouck/aiw_env

# Move to code directory
cd /anvil/projects/x-atm170020/ohouck/ai_weather_ag

# Function to monitor disk usage
monitor_disk_usage() {
    while true; do
        echo "[$(date)] Disk usage in data directory:"
        df -h /anvil/projects/x-atm170020/ohouck/data
        echo "[$(date)] Number of files downloaded:"
        find /anvil/projects/x-atm170020/ohouck/data/processed/ -name "*.nc" | wc -l
        echo "---"
        sleep 300  # Check every 5 minutes
    done
}

# Start disk monitoring in background
monitor_disk_usage &
MONITOR_PID=$!

# Set up logging environment variables
export PYTHONUNBUFFERED=1
export DASK_LOGGING__DISTRIBUTED=info
export BEAM_LOG_LEVEL=INFO

# Print job information
echo "=================================="
echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: $SLURM_MEM_PER_NODE MB"
echo "Working directory: $(pwd)"
echo "Python version: $(python --version)"

# Function to handle interruption
cleanup() {
    echo "Job interrupted - checkpoint saved for resumption"
    kill $MONITOR_PID 2>/dev/null
    exit 0
}

# Set up trap for graceful shutdown
trap cleanup SIGTERM SIGINT

python3 -u weatherbenchx/combine_and_subset.py \
    2>&1 | tee /anvil/projects/x-atm170020/ohouck/ai_weather_ag/logs/combined_and_subset_detailed_${SLURM_JOB_ID}.log

# Capture exit code
EXIT_CODE=$?

# Stop monitoring
kill $MONITOR_PID 2>/dev/null

echo "=================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"

echo "Final disk usage:"
df -h /anvil/projects/x-atm170020/ohouck/data
echo "Total files downloaded:"
find /anvil/projects/x-atm170020/ohouck/data/processed -name "*.nc" | wc -l


echo "Log files:"
echo "  SLURM output: logs/combine_and_subset${SLURM_JOB_ID}.out"
echo "  SLURM error:  logs/combine_and_subset${SLURM_JOB_ID}.err"
echo "  Detailed log: logs/combine_and_subset${SLURM_JOB_ID}.log"

exit $EXIT_CODE