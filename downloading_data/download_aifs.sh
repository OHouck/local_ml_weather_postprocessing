# download_aifs.sh
#!/bin/bash

# Configuration
CLUSTER_HOST="dsi"
REMOTE_DIR="/net/monsoon/marchakitus/model_data/AIFS/output_daily_march_15_october"
LOCAL_DIR="/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/data/raw/aifs"
LOCAL_TEMP="/Users/ohouck/tmp/aifs_temp_$"
ERROR_LOG_DIR="${LOCAL_DIR}/error_logs"
LEAD_DAYS="1,5,9"

# More robust script directory detection
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Script is being executed directly
    SCRIPT_DIR="$(cd "$(dirname "${0}")" && pwd)"
else
    # Script is being sourced
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

PYTHON_SCRIPT="${SCRIPT_DIR}/aifs_cleaning.py"

# Debug path information
echo "=== DEBUG PATH INFORMATION ==="
echo "Script directory: ${SCRIPT_DIR}"
echo "Python script path: ${PYTHON_SCRIPT}"
echo "Current working directory: $(pwd)"
echo "Script exists: $([ -f "${PYTHON_SCRIPT}" ] && echo "YES" || echo "NO")"
if [ -f "${PYTHON_SCRIPT}" ]; then
    echo "Python script permissions: $(ls -la "${PYTHON_SCRIPT}")"
fi
echo "============================="

# Parallel processing settings
MAX_PARALLEL_JOBS=3  # Adjust based on your system and network capacity

# Create local directories
mkdir -p "$LOCAL_DIR"
mkdir -p "$LOCAL_TEMP"
mkdir -p "$ERROR_LOG_DIR"

# Function to process single file
process_file() {
    local filename=$1
    local remote_file="${REMOTE_DIR}/${filename}"
    local temp_local_file="${LOCAL_TEMP}/${filename}"
    local processed_name="processed_${filename%.nc}.zarr"
    local final_output="${LOCAL_DIR}/${processed_name}"
    local error_log="${ERROR_LOG_DIR}/${filename%.nc}_error.log"
    
    # Skip file if already processed
    if [ -d "${final_output}" ]; then
        echo "✓ Skipping ${filename}: already processed (${processed_name})"
        return 2
    fi

    echo "========================================="
    echo "Processing: ${filename} (PID: $$)"
    echo "========================================="

    echo "Step 1/3: Preparing ${filename}..."

    # If file already exists locally, don’t redownload
    if [ -f "${temp_local_file}" ]; then
        echo "✓ Found existing local copy of ${filename}, skipping download"
    else
        echo "Downloading ${filename} from cluster..."
        rsync -avz --partial --progress --compress-level=1 \
              --timeout=300 --contimeout=60 \
              "${CLUSTER_HOST}:${remote_file}" \
              "${temp_local_file}"

        local rsync_exit_code=$?
        if [ $rsync_exit_code -ne 0 ]; then
            echo "✗ Failed to download ${filename} (rsync exit code: $rsync_exit_code)"
            echo "$(date): Failed to download ${filename} - rsync exit code: $rsync_exit_code" >> "$error_log"
            return 1
        fi
    fi

    local file_size=$(du -h "${temp_local_file}" | cut -f1)
    echo "Ready to process ${filename} (${file_size})"

    echo "Step 2/3: Processing ${filename} locally..."
    
    if python3 "${PYTHON_SCRIPT}" \
           "${temp_local_file}" \
           "${final_output}" \
           "${LEAD_DAYS}" 2>&1 | tee -a "$error_log"; then
        
        echo "✓ Successfully processed ${filename}"

        # Delete temporary file only in normal/parallel mode, not in test mode
        if [ "$TEST_MODE" != "true" ]; then
            echo "Step 3/3: Cleaning up temporary file..."
            rm -f "${temp_local_file}"
        else
            echo "TEST MODE: Keeping local file for debugging: ${temp_local_file}"
        fi

        if [ -d "${final_output}" ]; then
            local processed_size=$(du -sh "${final_output}" | cut -f1)
            echo "✓ Saved processed file: ${processed_name} (${processed_size})"
        fi

        rm -f "$error_log"  # remove error log if successful
        return 0
    else
        echo "✗ Failed to process ${filename}"
        echo "$(date): Python processing failed for ${filename}" >> "$error_log"
        echo "✓ Keeping downloaded file for debugging: ${temp_local_file}"
        echo "✓ Error details saved to: ${error_log}"
        return 1
    fi
}

# Function to process files in parallel
process_files_parallel() {
    local files=("$@")
    local pids=()
    local results=()
    local job_count=0
    
    for filename in "${files[@]}"; do
        # Wait if we've reached max parallel jobs
        while [ ${#pids[@]} -ge $MAX_PARALLEL_JOBS ]; do
            # Check for completed jobs
            for i in "${!pids[@]}"; do
                if ! kill -0 "${pids[$i]}" 2>/dev/null; then
                    wait "${pids[$i]}"
                    local exit_code=$?
                    results[${pids[$i]}]=$exit_code
                    unset pids[$i]
                fi
            done
            pids=("${pids[@]}")  # Reindex array
            sleep 0.1
        done
        
        # Start new job
        echo "Starting parallel job for: $filename"
        process_file "$filename" &
        local pid=$!
        pids+=($pid)
        ((job_count++))
        
        echo "Started job $job_count/${#files[@]} (PID: $pid)"
    done
    
    # Wait for remaining jobs to complete
    for pid in "${pids[@]}"; do
        wait "$pid"
        results[$pid]=$?
    done
    
    # Count results
    local success_count=0
    local fail_count=0
    local skip_count=0
    
    for exit_code in "${results[@]}"; do
        case $exit_code in
            0) ((success_count++)) ;;
            2) ((skip_count++)) ;;
            *) ((fail_count++)) ;;
        esac
    done
    
    echo "========================================="
    echo "Parallel processing complete!"
    echo "Total files: ${#files[@]}"
    echo "Successful: $success_count"
    echo "Skipped (already processed): $skip_count"
    echo "Failed: $fail_count"
    echo "========================================="
}

# Cleanup function
cleanup() {
    echo "Cleaning up background jobs..."
    jobs -p | xargs -r kill 2>/dev/null
    echo "Temp directory cleanup will be manual due to potential debugging files"
    echo "Check ${LOCAL_TEMP} for any files that need inspection"
}

# Set trap to cleanup on exit
trap cleanup EXIT

# Main execution
main() {
    # Enhanced Python script checking
    echo "Checking for Python script..."
    echo "Looking for: ${PYTHON_SCRIPT}"
    
    if [ ! -f "${PYTHON_SCRIPT}" ]; then
        echo "Error: aifs_cleaning.py not found at ${PYTHON_SCRIPT}"
        echo ""
        echo "Debugging information:"
        echo "- Script directory: ${SCRIPT_DIR}"
        echo "- Files in script directory:"
        ls -la "${SCRIPT_DIR}/"
        echo ""
        echo "Possible solutions:"
        echo "1. Ensure both scripts are in the same directory"
        echo "2. Check if the Python script is named exactly 'aifs_cleaning.py'"
        echo "3. Try running from the directory containing both scripts"
        echo ""
        
        # Try to find the Python script in common locations
        echo "Searching for aifs_cleaning.py in nearby directories..."
        find "$(dirname "$SCRIPT_DIR")" -name "aifs_cleaning.py" -type f 2>/dev/null | head -5
        
        exit 1
    fi
    
    # Check if Python script is executable (not required but good practice)
    if [ ! -x "${PYTHON_SCRIPT}" ]; then
        echo "Making Python script executable..."
        chmod +x "${PYTHON_SCRIPT}"
    fi
    
    # Test Python script can be found by Python
    echo "Testing Python script syntax..."
    if ! python3 -m py_compile "${PYTHON_SCRIPT}"; then
        echo "Error: Python script has syntax errors"
        exit 1
    fi
    echo "Python script found and syntax is valid."
    echo ""
    
    # Check SSH key authentication with faster timeout
    ssh -o BatchMode=yes -o ConnectTimeout=5 -o ServerAliveInterval=60 \
        ${CLUSTER_HOST} "echo 'SSH connection test'" &>/dev/null
    if [ $? -ne 0 ]; then
        echo "Error: SSH key authentication not configured for ${CLUSTER_HOST}"
        echo "Please set up SSH keys first"
        exit 1
    fi
    
    # Get list of files from cluster
    echo "Fetching file list from cluster..."
    FILES=$(ssh ${CLUSTER_HOST} "ls ${REMOTE_DIR}/*.nc 2>/dev/null" | xargs -n1 basename)
    
    if [ -z "$FILES" ]; then
        echo "No .nc files found in ${REMOTE_DIR}"
        exit 1
    fi
    
    # Convert to array
    files_array=($FILES)
    NUM_FILES=${#files_array[@]}
    echo "Found ${NUM_FILES} files to process"
    echo "Will process with up to ${MAX_PARALLEL_JOBS} parallel jobs"
    echo ""
    
    # Process files based on arguments
    if [ $# -ge 1 ]; then
        if [ "$1" == "--test" ]; then
            echo "TEST MODE: Processing first file only"
            TEST_MODE=true
            process_file "${files_array[0]}"
        elif [ "$1" == "--parallel" ]; then
            # Parallel mode for all files
            echo "PARALLEL MODE: Processing all files with ${MAX_PARALLEL_JOBS} parallel jobs"
            process_files_parallel "${files_array[@]}"
        else
            # Process specific file
            process_file "$1"
        fi
    else
        # Ask user for processing mode
        echo "Choose processing mode:"
        echo "1) Sequential (one at a time) - safer, slower"
        echo "2) Parallel (${MAX_PARALLEL_JOBS} at a time) - faster, uses more resources"
        read -p "Enter choice [1-2]: " -n 1 -r
        echo
        
        case $REPLY in
            2)
                echo "Using parallel processing..."
                process_files_parallel "${files_array[@]}"
                ;;
            *)
                echo "Using sequential processing..."
                # Sequential processing (original logic)
                COUNTER=0
                SUCCESS_COUNT=0
                FAIL_COUNT=0
                SKIPPED_COUNT=0
                
                for filename in "${files_array[@]}"; do
                    ((COUNTER++))
                    echo ""
                    echo "File ${COUNTER}/${NUM_FILES}"

                    if process_file "$filename"; then
                        ((SUCCESS_COUNT++))
                    else
                        case $? in
                            2) ((SKIPPED_COUNT++)) ;;
                            *) ((FAIL_COUNT++))
                               echo "⚠ Failed to process ${filename}, continuing with next file..." ;;
                        esac
                    fi

                    echo "Progress: ${SUCCESS_COUNT} succeeded, ${FAIL_COUNT} failed, ${SKIPPED_COUNT} skipped, $((NUM_FILES - COUNTER)) remaining"
                    echo ""
                    sleep 0.1
                done
                
                echo "========================================="
                echo "Sequential processing complete!"
                echo "Total files: ${NUM_FILES}"
                echo "Successful: ${SUCCESS_COUNT}"
                echo "Skipped (already processed): ${SKIPPED_COUNT}"
                echo "Failed: ${FAIL_COUNT}"
                echo "========================================="
                ;;
        esac
    fi
    
    echo "Processed files saved in: ${LOCAL_DIR}"
    if [ -d "$ERROR_LOG_DIR" ] && [ "$(ls -A $ERROR_LOG_DIR 2>/dev/null)" ]; then
        echo "Error logs saved in: ${ERROR_LOG_DIR}"
    fi
}

# Show usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] [FILENAME]

Download and process AIFS forecast data from cluster
Downloads files one at a time or in parallel, processes locally, saves error logs

Options:
    --test            Process only the first file (for testing)
    --parallel        Process all files in parallel (${MAX_PARALLEL_JOBS} jobs)
    FILENAME          Process specific file
    (no args)         Interactive mode to choose sequential or parallel

Examples:
    $0                              # Interactive mode
    $0 --parallel                   # Process all files in parallel
    $0 init_2024071100.nc          # Process single file
    $0 --test                      # Test with first file only

Performance improvements:
    - Optimized rsync with compression and partial transfers
    - Parallel processing option for faster throughput
    - Better error handling and logging
    - Failed downloads preserve temp files for debugging

Output:
    - Processed files: ${LOCAL_DIR}/processed_*.zarr
    - Error logs: ${ERROR_LOG_DIR}/*_error.log
    - Failed temp files: ${LOCAL_TEMP}/ (for debugging)
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