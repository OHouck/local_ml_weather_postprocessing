#!/usr/bin/env python3
"""
Monitor the progress of the WeatherBench download
"""
import os
import json
from datetime import datetime
from pathlib import Path

def check_download_progress(data_dir: str, checkpoint_dir: str):
    """Check and report download progress"""
    
    # Check checkpoint
    checkpoint_file = os.path.join(checkpoint_dir, "download_progress.json")
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        
        print(f"Checkpoint Status:")
        print(f"  Last processed: {checkpoint.get('last_year', 'N/A')}-{checkpoint.get('last_month', 'N/A'):02d}")
        print(f"  Completed chunks: {len(checkpoint.get('completed_chunks', []))}")
        print()
    else:
        print("No checkpoint file found")
        print()
    
    # Check downloaded files
    raw_data_dir = os.path.join(data_dir, "raw", "pangu_raw_data")
    if os.path.exists(raw_data_dir):
        pred_files = list(Path(raw_data_dir).glob("predictions_*.nc"))
        target_files = list(Path(raw_data_dir).glob("targets_*.nc"))
        
        print(f"Downloaded Files:")
        print(f"  Prediction files: {len(pred_files)}")
        print(f"  Target files: {len(target_files)}")
        print()
        
        # Calculate total size
        total_size = 0
        for f in pred_files + target_files:
            total_size += f.stat().st_size
        
        print(f"Total disk usage: {total_size / (1024**3):.2f} GB")
        print()
        
        # Show date range
        if pred_files:
            dates = []
            for f in pred_files:
                # Extract date from filename
                fname = f.stem
                if "predictions_" in fname:
                    date_part = fname.replace("predictions_", "")
                    dates.append(date_part.split("_")[0])
            
            dates = sorted(dates)
            print(f"Date range: {dates[0]} to {dates[-1]}")
            
            # Check for gaps
            expected_files = 365 * 5 / 7  # Approximately, since we chunk by 7 days
            completion_pct = len(pred_files) / expected_files * 100
            print(f"Estimated completion: {completion_pct:.1f}%")
    else:
        print(f"Data directory not found: {raw_data_dir}")

def main():
    # Adjust these paths based on your environment
    import socket
    nodename = socket.gethostname()
    
    if "anvil" in nodename.lower():
        data_dir = "/anvil/projects/x-atm170020/ohouck/data"
    else:
        data_dir = os.path.expanduser("~/ai_weather_ag/data")
    
    checkpoint_dir = os.path.join(data_dir, "checkpoints")
    
    print(f"WeatherBench Download Progress Monitor")
    print(f"Time: {datetime.now()}")
    print(f"Data directory: {data_dir}")
    print("=" * 50)
    print()
    
    check_download_progress(data_dir, checkpoint_dir)

if __name__ == "__main__":
    main()
