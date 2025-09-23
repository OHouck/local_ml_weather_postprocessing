#!/bin/bash -l
#SBATCH -p general
#SBATCH --time=00:10:00
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -o aifs_test.out
#SBATCH -J aifs_test

# Test configuration - process only 2 files
REMOTE_DIR="/net/monsoon/marchakitus/model_data/AIFS/output_daily_march_15_october"
SCRATCH_DIR="/home/ohouck/temp_data"
LOCAL_DEST="ohouck@oMac.local:/Users/ohouck/globus/forecast_data/test"
LEAD_DAYS="1,5,9"
ERROR_LOG="test_failed_files.txt"

# Load conda environment
conda activate /home/ohouck/conda_env # Change to your environment name

echo "========================================="
echo "AIFS TEST MODE - Processing 2 files only"
echo "========================================="
echo "Starting at $(date)"
echo "Python: $(which python)"
echo "Scratch directory: $SCRATCH_DIR"
echo "Output destination: $LOCAL_DEST"

# Create scratch directory
mkdir -p "$SCRATCH_DIR"
cd "$SCRATCH_DIR"

# Get first 2 files only
FILES=$(ls ${REMOTE_DIR}/*.nc 2>/dev/null | head -2)

if [ -z "$FILES" ]; then
    echo "No .nc files found in ${REMOTE_DIR}"
    exit 1
fi

# Convert to array
files_array=($FILES)
NUM_FILES=${#files_array[@]}
echo "Testing with ${NUM_FILES} files:"
for f in "${files_array[@]}"; do
    echo "  - $(basename $f)"
done
echo ""

# Initialize counters
SUCCESS_COUNT=0
FAIL_COUNT=0

# Process files
for filepath in "${files_array[@]}"; do
    filename=$(basename "$filepath")
    echo "========================================="
    echo "Processing $filename..."
    echo "========================================="
    
    if python /home/ohouck/process_aifs_combined.py \
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
    echo ""
done

echo "========================================="
echo "Test processing complete!"
echo "Successful: $SUCCESS_COUNT / $NUM_FILES"
echo "Failed: $FAIL_COUNT / $NUM_FILES"
echo "========================================="

# Create test directory on local machine
echo "Creating test directory on local machine..."
ssh ohouck@oMac.local "mkdir -p /Users/ohouck/globus/forecast_data/test" 2>/dev/null || true

# Transfer processed files
echo "Transferring test files to local machine..."
for zarr_file in "$SCRATCH_DIR"/processed_*.zarr; do
    if [ -d "$zarr_file" ]; then
        file_name=$(basename "$zarr_file")
        echo "Transferring $file_name..."
        
        rsync -avz --progress \
            "$zarr_file" \
            "$LOCAL_DEST/"
        
        if [ $? -eq 0 ]; then
            echo "✓ Successfully transferred $file_name"
        else
            echo "✗ Failed to transfer $file_name"
        fi
    fi
done

# List what was created
echo ""
echo "Files created in scratch:"
ls -lh "$SCRATCH_DIR"/*.zarr 2>/dev/null || echo "No zarr files created"

# Copy error log to home if exists
if [ -f "$ERROR_LOG" ] && [ -s "$ERROR_LOG" ]; then
    cp "$ERROR_LOG" "$HOME/"
    echo "Error log saved to $HOME/$ERROR_LOG"
fi

# Optional: Ask whether to clean up
echo ""
echo "Test complete at $(date)"
echo "Scratch directory: $SCRATCH_DIR"
echo "Note: Scratch directory will NOT be automatically deleted in test mode"
echo "To manually clean up, run: rm -rf $SCRATCH_DIR"
echo ""
echo "To view transferred files on local machine:"
echo "  ssh ohouck@oMac.local 'ls -lh /Users/ohouck/globus/forecast_data/test/'"