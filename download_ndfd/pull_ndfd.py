import subprocess
import os
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from helper_funcs import setup_directories

def check_data_availability(element='maxt', start_year=2020, end_year=2025):
    """Check which months have data for each year"""
    print(f"\n=== Data Availability for {element} ===\n")
    
    for year in range(start_year, end_year + 1):
        path = f"s3://noaa-ndfd-pds/wmo/{element}/{year}/"
        cmd = ["aws", "s3", "ls", "--no-sign-request", path]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and result.stdout.strip():
            months = [line.split()[1].replace('/', '').replace('PRE', '').strip() 
                     for line in result.stdout.split('\n') if 'PRE' in line]
            months = [m for m in months if m.isdigit()]
            
            if months:
                print(f"{year}: Months {min(months)}-{max(months)} ({len(months)} months)")
            else:
                print(f"{year}: No data or irregular structure")
        else:
            print(f"{year}: No data")

def download_ndfd_month(element, year, month, base_dir):
    """Download entire month of NDFD data"""
    output_dir = os.path.join(base_dir, element, str(year), f"{month:02d}")
    os.makedirs(output_dir, exist_ok=True)
    
    s3_path = f"s3://noaa-ndfd-pds/wmo/{element}/{year}/{month:02d}/"
    
    cmd = [
        "aws", "s3", "sync",
        "--no-sign-request",
        s3_path,
        output_dir
    ]
    
    print(f"Downloading {element} for {year}-{month:02d}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        # Count files downloaded
        file_count = sum(len(files) for _, _, files in os.walk(output_dir))
        print(f"✓ Downloaded {element} {year}-{month:02d}: {file_count} files")
        return True
    else:
        print(f"✗ Error: {result.stderr}")
        return False

def download_year_data(year, elements, base_dir):
    """Download data for an entire year"""
    current_month = datetime.now().month if year == datetime.now().year else 12
    
    for element in elements:
        print(f"\n=== Downloading {element} for {year} ===")
        for month in range(1, current_month + 1):
            download_ndfd_month(element, year, month, base_dir)


def main():

    dirs = setup_directories()

    # Run the check
    # check_data_availability('maxt', 2020, 2025)

    year = 2025

    # elements = [
    #     'maxt',      # Maximum temperature
    #     'mint',      # Minimum temperature  
    #     'temp',      # Temperature
    #     'wspd',      # Wind speed
    #     'wdir',      # Wind direction
    #     'pop12',     # 12-hour probability of precipitation
    #     'qpf',       # Quantitative precipitation forecast
    #     'rhm',       # Relative humidity
    # ]

    elements = [
        'maxt',      # Maximum temperature
    ]

    output_dir = os.path.join(dirs['raw'], 'ndfd_data')

    print(f"Starting download for {year}")
    print(f"Output directory: {output_dir}")
    print(f"Elements: {', '.join(elements)}")

    # Confirm before starting
    response = input("\nProceed with download? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Download cancelled.")
        return

    download_year_data(year, elements, output_dir)

if __name__ == "__main__":
    main()