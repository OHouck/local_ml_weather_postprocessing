#!/bin/bash -l
#SBATCH -p general
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -o aifs_processing_%j.out
#SBATCH -J aifs_processing

# Configuration
REMOTE_DIR="/net/monsoon/marchakitus/model_data/AIFS/output_daily_march_15_october"
SCRATCH_DIR="/home/ohouck/temp_data"
LOCAL_DEST="ohouck@oMac.local:/Users/ohouck/globus/forecast_data"
LEAD_DAYS="1,5,9"
ERROR_LOG="failed_files_$(date +%Y%m%d_%H%M%S).txt"

# Load conda environment
conda activate /home/ohouck/conda_env # Change to your environment name

echo "Starting AIFS processing at $(date)"
echo "Python: $(which python)"
echo "Working directory: $SCRATCH_DIR"

# Create scratch directory
mkdir -p "$SCRATCH_DIR"
cd "$SCRATCH_DIR"

# Copy Python script to scratch directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cp "${SCRIPT_DIR}/aifs_cleaning.py" .

# Get list of files
FILES=$(ls ${REMOTE_DIR}/*.nc 2>/dev/null)

if [ -z "$FILES" ]; then
    echo "No .nc files found in ${REMOTE_DIR}"
    exit 1
fi

# Convert to array
files_array=($FILES)
NUM_FILES=${#files_array[@]}
echo "Found ${NUM_FILES} files to process"

# Initialize counters
SUCCESS_COUNT=0
FAIL_COUNT=0

# Process files sequentially
for filepath in "${files_array[@]}"; do
    filename=$(basename "$filepath")
    echo "Processing $filename..."
    
    if python process_aifs_combined.py \
           "$filepath" \
           "$SCRATCH_DIR" \
           "${LEAD_DAYS}"; then
        ((SUCCESS_COUNT++))
        echo "✓ Successfully processed $filename"
    else
        ((FAIL_COUNT++))
        echo "✗ Failed to process $filename"
        echo "$filename" >> "$ERROR_LOG"
    fi
done

echo "========================================="
echo "Processing complete!"
echo "Successful: $SUCCESS_COUNT"
echo "Failed: $FAIL_COUNT"
echo "========================================="

# Combine into yearly files and transfer to local machine
echo "Creating yearly files..."
python process_aifs_combined.py \
    --combine-only \
    "$SCRATCH_DIR" \
    "/home/ohouck/aifs_yearly"

# Copy error log to home directory if it exists
if [ -f "$ERROR_LOG" ] && [ -s "$ERROR_LOG" ]; then
    cp "$ERROR_LOG" "$HOME/"
    echo "Error log saved to $HOME/$ERROR_LOG"
fi

# Cleanup scratch directory
echo "Cleaning up scratch directory..."
rm -rf "$SCRATCH_DIR"

echo "Processing finished at $(date)"