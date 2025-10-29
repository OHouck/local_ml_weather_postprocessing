"""
Author: Ozma Houck
File name: download_aurora_2022.py
Date created: 10/29/2025
Purpose: Download aurora data for all days in 2022
"""
from pathlib import Path
import pandas as pd
import sys
import os
sys.path.insert(0, str(Path(__file__).parent.parent))
from helper_funcs import setup_directories

# Import the download function from the original script
from prepare_aurora_data import download_aurora_data

def main():
    # Setup directories
    dirs = setup_directories()
    download_path = os.path.join(dirs["raw"], "aurora")
    os.makedirs(download_path, exist_ok=True)
    
    # Generate all dates in 2022
    dates = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
    
    print(f"Total days to process: {len(dates)}")
    print("=" * 60)
    
    # Track progress
    skipped = 0
    downloaded = 0
    
    for date in dates:
        day_str = date.strftime('%Y-%m-%d')
        
        # Check if both files already exist
        surface_file = Path(download_path) / f"{day_str}-surface-level.nc"
        atmos_file = Path(download_path) / f"{day_str}-atmospheric.nc"
        
        if surface_file.exists() and atmos_file.exists():
            print(f"{day_str}: Both files exist, skipping.")
            skipped += 1
            continue
        
        print(f"\n{'=' * 60}")
        print(f"Downloading data for {day_str}...")
        print(f"{'=' * 60}")
        
        try:
            download_aurora_data(day_str)
            downloaded += 1
        except Exception as e:
            print(f"ERROR downloading {day_str}: {e}")
            continue
    
    print(f"\n{'=' * 60}")
    print("Download Summary:")
    print(f"  Days downloaded: {downloaded}")
    print(f"  Days skipped (already exist): {skipped}")
    print(f"  Total days: {len(dates)}")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
