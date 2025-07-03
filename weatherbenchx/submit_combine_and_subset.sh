#!/bin/sh -l
# FILENAME: submit_combine_and_subset_simple.sh
#SBATCH --account=atm170020-gpu
#SBATCH -p gpu
#SBATCH --time=24:00:00  # Increased to 24 hours for full dataset
#SBATCH --mem=64GB
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
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
export OMP_NUM_THREADS=1  # Prevent threading conflicts

monitor_progress() {
    while true; do
        echo "[$(date)] System status:"
        echo "Memory usage:"
        free -h
        echo ""
        echo "Disk usage:"
        df -h /anvil/projects/x-atm170020/ohouck/data/
        echo ""
        echo "Processing progress:"
        find /anvil/projects/x-atm170020/ohouck/data/processed/ -name "*.nc" -exec ls -lh {} \; 2>/dev/null | tail -n 10
        echo ""
        echo "Temporary files:"
        find /anvil/projects/x-atm170020/ohouck/data/temp/ -name "*.nc" 2>/dev/null | wc -l
        echo "---"
        sleep 600  # Check every 10 minutes
    done
}


# Start monitoring in background
monitor_progress &
MONITOR_PID=$!

# Trap to ensure cleanup on exit
cleanup() {
    echo "Cleaning up..."
    kill $MONITOR_PID 2>/dev/null
    # Clean up any leftover temp files
    find /anvil/projects/x-atm170020/ohouck/data/temp/ -name "*.nc" -mtime +1 -delete 2>/dev/null
}
trap cleanup EXIT

# Run the script
echo "=================================="
echo "Job started at: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE"
echo "=================================="

python3 -u weatherbenchx/combine_and_subset.py 

# Capture exit code and cleanup
EXIT_CODE=$?
kill $MONITOR_PID 2>/dev/null

echo "=================================="
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "Final output files:"
find /anvil/projects/x-atm170020/ohouck/data/processed -name "*.nc" -exec ls -lh {} \; 2>/dev/null
echo ""
echo "Disk usage after completion:"
du -sh /anvil/projects/x-atm170020/ohouck/data/processed/
echo "=================================="

exit $EXIT_CODE