#!/bin/bash

# Configuration
CLUSTER_HOST="dsi"  # Using SSH config alias
REMOTE_DIR="/net/monsoon/marchakitus/model_data/AIFS/output_daily_march_15_october"
LOCAL_DIR="/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/data/raw/aifs"
LOCAL_TEMP="/Users/ohouck/tmp/aifs_temp_$$"  # Local temp directory for raw files
LEAD_DAYS="1,5,10"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # Directory where this script is located
PYTHON_SCRIPT="${SCRIPT_DIR}/aifs_cleaning.py"

# Create local directories
mkdir -p "$LOCAL_DIR"
mkdir -p "$LOCAL_TEMP"

# Function to process single file
process_file() {
    local filename=$1
    local remote_file="${REMOTE_DIR}/${filename}"
    local temp_local_file="${LOCAL_TEMP}/${filename}"
    # Change extension from .nc to .zarr
    local processed_name="processed_${filename%.nc}.zarr"
    local final_output="${LOCAL_DIR}/${processed_name}"
    
    echo "========================================="
    echo "Processing: ${filename}"
    echo "========================================="
    
    # Step 1: Download raw file from cluster to temp location
    echo "Step 1/3: Downloading ${filename} from cluster..."
    rsync -avz --progress \
          "${CLUSTER_HOST}:${remote_file}" \
          "${temp_local_file}"
    
    if [ $? -ne 0 ]; then
        echo "✗ Failed to download ${filename}"
        return 1
    fi
    
    # Get file size for info
    local file_size=$(du -h "${temp_local_file}" | cut -f1)
    echo "Downloaded ${filename} (${file_size})"
    
    # Step 2: Process locally using existing aifs_cleaning.py
    echo "Step 2/3: Processing ${filename} locally..."
    python "${PYTHON_SCRIPT}" \
           "${temp_local_file}" \
           "${final_output}" \
           "${LEAD_DAYS}" 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully processed ${filename}"
        
        # Step 3: Delete the temporary raw file (only the local copy!)
        echo "Step 3/3: Cleaning up temporary file..."
        rm -f "${temp_local_file}"
        
        if [ $? -eq 0 ]; then
            echo "✓ Deleted temporary file: ${temp_local_file}"
        else
            echo "⚠ Warning: Could not delete temporary file: ${temp_local_file}"
        fi
        
        # Check processed file exists
        if [ -d "${final_output}" ]; then
            local processed_size=$(du -sh "${final_output}" | cut -f1)
            echo "✓ Saved processed file: ${processed_name} (${processed_size})"
        fi
        
        echo "✓ Completed processing ${filename}"
        return 0
    else
        echo "✗ Failed to process ${filename}"
        # Clean up temp file even if processing failed
        rm -f "${temp_local_file}"
        return 1
    fi
}

# Cleanup function
cleanup() {
    echo "Cleaning up local temporary directory..."
    rm -rf "${LOCAL_TEMP}"
}

# Set trap to cleanup on exit
trap cleanup EXIT

# Main execution
main() {
    # Check if aifs_cleaning.py exists
    if [ ! -f "${PYTHON_SCRIPT}" ]; then
        echo "Error: aifs_cleaning.py not found at ${PYTHON_SCRIPT}"
        echo "Please ensure aifs_cleaning.py is in the same directory as this script"
        exit 1
    fi
    
    # Check SSH key authentication
    ssh -o BatchMode=yes -o ConnectTimeout=5 ${CLUSTER_HOST} "echo 'SSH connection test'" &>/dev/null
    if [ $? -ne 0 ]; then
        echo "Error: SSH key authentication not configured for ${CLUSTER_HOST}"
        echo "Please set up SSH keys first"
        exit 1
    fi
    
    # Get list of files from cluster
    echo "Fetching file list from cluster..."
    FILES=$(ssh ${CLUSTER_HOST} "ls ${REMOTE_DIR}/*.nc 2>/dev/null")
    
    if [ -z "$FILES" ]; then
        echo "No .nc files found in ${REMOTE_DIR}"
        exit 1
    fi
    
    # Count files
    NUM_FILES=$(echo "$FILES" | wc -w | tr -d ' ')
    echo "Found ${NUM_FILES} files to process"
    echo ""
    
    # Process files based on arguments
    if [ $# -ge 1 ]; then
        if [ "$1" == "--test" ]; then
            # Test mode: process only first file
            echo "TEST MODE: Processing first file only"
            FIRST_FILE=$(echo "$FILES" | head -n1 | xargs basename)
            process_file "$FIRST_FILE"
        else
            # Process specific file
            process_file "$1"
        fi
    else
        # Process all files sequentially, one at a time
        COUNTER=0
        SUCCESS_COUNT=0
        FAIL_COUNT=0
        
        for filepath in ${FILES}; do
            filename=$(basename "$filepath")
            ((COUNTER++))
            
            echo ""
            echo "File ${COUNTER}/${NUM_FILES}"
            
            if process_file "$filename"; then
                ((SUCCESS_COUNT++))
            else
                ((FAIL_COUNT++))
                echo "⚠ Failed to process ${filename}, continuing with next file..."
            fi
            
            # Show progress summary
            echo "Progress: ${SUCCESS_COUNT} succeeded, ${FAIL_COUNT} failed, $((NUM_FILES - COUNTER)) remaining"
            echo ""
            
            # Optional: Add a small delay between files to avoid overwhelming the system
            sleep 1
        done
        
        echo "========================================="
        echo "Processing complete!"
        echo "Total files: ${NUM_FILES}"
        echo "Successful: ${SUCCESS_COUNT}"
        echo "Failed: ${FAIL_COUNT}"
        echo "========================================="
    fi
    
    echo "Processed files saved in: ${LOCAL_DIR}"
}

# Show usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] [FILENAME]

Download and process AIFS forecast data from cluster
Downloads files one at a time, processes locally, then deletes temp files

Options:
    --test            Process only the first file (for testing)
    FILENAME          Process specific file
    (no args)         Process all files sequentially

Examples:
    $0                              # Process all files one by one
    $0 init_2024071100.nc          # Process single file
    $0 --test                      # Test with first file only

Configuration:
    Edit script variables to change:
    - CLUSTER_HOST: SSH alias for cluster
    - REMOTE_DIR: Path to data on cluster
    - LOCAL_DIR: Local destination for processed files
    - LEAD_DAYS: Lead time days to extract

Process flow for each file:
    1. Download raw .nc file from cluster to local temp directory
    2. Process locally using aifs_cleaning.py and save as .zarr in final directory
    3. Delete local temp copy (original on cluster is never touched)

Output:
    Processed files are saved as Zarr stores (directories) with naming:
    processed_<original_filename>.zarr
EOF
}

# Check for help flag
if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    usage
    exit 0
fi

# Check available disk space before starting
echo "Checking available disk space..."
AVAILABLE_SPACE=$(df -h /tmp | tail -1 | awk '{print $4}')
echo "Available space in /tmp: ${AVAILABLE_SPACE}"
AVAILABLE_LOCAL=$(df -h "${LOCAL_DIR}" | tail -1 | awk '{print $4}')
echo "Available space in output directory: ${AVAILABLE_LOCAL}"
echo ""

read -p "Continue with processing? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted by user"
    exit 0
fi

# Run main function
main "$@"