#!/bin/bash
# submit_weatherbench_download.sh
# Helper script to submit or resubmit the WeatherBench download job

# Check if we're resuming or starting fresh
CHECKPOINT_FILE="/anvil/projects/x-atm170020/ohouck/data/checkpoints/download_progress.json"

if [ -f "$CHECKPOINT_FILE" ]; then
    echo "Found existing checkpoint - will resume download"
    echo "Current checkpoint status:"
    python3 -c "
import json
with open('$CHECKPOINT_FILE', 'r') as f:
    cp = json.load(f)
    print(f\"  Last year: {cp.get('last_year', 'N/A')}\")
    print(f\"  Last month: {cp.get('last_month', 'N/A')}\")
    print(f\"  Completed chunks: {len(cp.get('completed_chunks', []))}\")
"
    echo ""
    read -p "Continue with download? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
else
    echo "No checkpoint found - will start fresh download"
    echo "This will download ~5 years of data (2018-2022)"
    read -p "Start download? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# Submit the job
echo "Submitting job..."
sbatch wbx_anvil.sh

# Show queue status
echo ""
echo "Current job queue:"
squeue -u $USER
