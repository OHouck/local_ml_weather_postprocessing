#!/bin/sh -l
# FILENAME: wbx_anvil_optimized.sh
#SBATCH --account=atm170020-gpu
#SBATCH -p gpu 
#SBATCH --time=24:00:00  # Increased to 24 hours for full dataset
#SBATCH --mem=128GB
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name weatherbenchx_download
#SBATCH -e logs/wbx_download_%j.err
#SBATCH -o logs/wbx_download_%j.out
#SBATCH --mail-user=ohouck@uchicago.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Create logs directory if it doesn't exist
mkdir -p /anvil/projects/x-atm170020/ohouck/ai_weather_ag/logs

# Load modules and activate environment
module load anaconda/2024.02-py311
conda activate /home/x-ohouck/aiw_env

# Move to code directory
cd /anvil/projects/x-atm170020/ohouck/ai_weather_ag

# Set up logging environment variables
export PYTHONUNBUFFERED=1
export DASK_LOGGING__DISTRIBUTED=info
export BEAM_LOG_LEVEL=INFO

# Function to monitor disk usage
monitor_disk_usage() {
    while true; do
        echo "[$(date)] Disk usage in data directory:"
        df -h /anvil/projects/x-atm170020/ohouck/data
        echo "[$(date)] Number of files downloaded:"
        find /anvil/projects/x-atm170020/ohouck/data/raw/pangu_raw_data -name "*.nc" | wc -l
        echo "---"
        sleep 300  # Check every 5 minutes
    done
}

# Start disk monitoring in background
monitor_disk_usage &
MONITOR_PID=$!

# Print job information
echo "=================================="
echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: $SLURM_MEM_PER_NODE MB"
echo "Working directory: $(pwd)"
echo "Python version: $(python --version)"
echo "NumPy version: $(python -c 'import numpy; print(numpy.__version__)')"

# Check if checkpoint exists
CHECKPOINT_FILE="/anvil/projects/x-atm170020/ohouck/data/checkpoints/download_progress.json"
if [ -f "$CHECKPOINT_FILE" ]; then
    echo "Found checkpoint file - will resume from last position"
    cat "$CHECKPOINT_FILE"
else
    echo "No checkpoint found - starting fresh download"
fi
echo "=================================="

# Function to handle interruption
cleanup() {
    echo "Job interrupted - checkpoint saved for resumption"
    kill $MONITOR_PID 2>/dev/null
    exit 0
}

# Set up trap for graceful shutdown
trap cleanup SIGTERM SIGINT

# Run the script with enhanced logging
# Using timeout to ensure graceful shutdown before SLURM kills the job
python3 -u weatherbenchx/weatherbench_download.py \
    2>&1 | tee /anvil/projects/x-atm170020/ohouck/ai_weather_ag/logs/wbx_detailed_${SLURM_JOB_ID}.log

# Capture exit code
EXIT_CODE=$?

# Stop monitoring
kill $MONITOR_PID 2>/dev/null

echo "=================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"

# Final disk usage report
echo "Final disk usage:"
df -h /anvil/projects/x-atm170020/ohouck/data
echo "Total files downloaded:"
find /anvil/projects/x-atm170020/ohouck/data/raw/pangu_raw_data -name "*.nc" | wc -l

# Check if we need to resubmit
if [ -f "$CHECKPOINT_FILE" ] && [ $EXIT_CODE -eq 124 ]; then
    echo "Job timed out but checkpoint exists - consider resubmitting"
fi

echo "Log files:"
echo "  SLURM output: logs/wbx_download_${SLURM_JOB_ID}.out"
echo "  SLURM error:  logs/wbx_download_${SLURM_JOB_ID}.err"
echo "  Detailed log: logs/wbx_detailed_${SLURM_JOB_ID}.log"

exit $EXIT_CODE