"""
estimate_memory.py - Estimate memory requirements for combine_and_subset.py
"""

import os
import glob
import xarray as xr
import numpy as np
from combine_and_subset import setup_directories, get_region_bounds, preprocess_and_subset

def estimate_memory_requirements():
    """Estimate memory requirements for processing"""
    print("\n" + "="*60)
    print("MEMORY REQUIREMENT ESTIMATION")
    print("="*60)
    
    dirs = setup_directories()
    
    # Get sample files
    pangu_files = sorted(glob.glob(os.path.join(dirs["raw"], "pangu_raw_data", "predictions*.nc")))
    era5_files = sorted(glob.glob(os.path.join(dirs["raw"], "pangu_raw_data", "targets*.nc")))
    
    if not pangu_files or not era5_files:
        print("ERROR: No files found for analysis")
        return
    
    # Analyze a sample file
    print("\n1. Analyzing sample files...")
    sample_pangu = pangu_files[0]
    sample_era5 = era5_files[0]
    
    regions = ["india", "usa_south", "amazon", "british_columbia"]
    
    for region in regions:
        print(f"\n{region.upper()}:")
        
        # Analyze Pangu data
        with xr.open_dataset(sample_pangu, engine='netcdf4') as ds:
            ds_subset = preprocess_and_subset(ds, region)
            
            # Calculate memory per file
            n_vars = len(ds_subset.data_vars)
            dims = dict(ds_subset.sizes)
            
            # Assuming float32 (4 bytes per value)
            values_per_file = n_vars
            for dim, size in dims.items():
                values_per_file *= size
            
            memory_per_file = values_per_file * 4 / 1e6  # MB
            
            print(f"  Pangu predictions:")
            print(f"    Dimensions after subset: {dims}")
            print(f"    Variables: {n_vars}")
            print(f"    Memory per file: {memory_per_file:.1f} MB")
            print(f"    Files per year: ~{365/7:.0f}")
            print(f"    Memory per year: ~{memory_per_file * 365/7:.1f} MB")
            print(f"    Memory for 5 years: ~{memory_per_file * 365/7 * 5:.1f} MB")
            
            pangu_yearly_mb = memory_per_file * 365/7
            pangu_total_mb = pangu_yearly_mb * 5
        
        # Analyze ERA5 data
        with xr.open_dataset(sample_era5, engine='netcdf4') as ds:
            ds_subset = preprocess_and_subset(ds, region)
            
            # Calculate memory per file
            n_vars = len(ds_subset.data_vars)
            dims = dict(ds_subset.sizes)
            
            values_per_file = n_vars
            for dim, size in dims.items():
                values_per_file *= size
            
            memory_per_file = values_per_file * 4 / 1e6  # MB
            
            print(f"  ERA5 targets:")
            print(f"    Dimensions after subset: {dims}")
            print(f"    Variables: {n_vars}")
            print(f"    Memory per file: {memory_per_file:.1f} MB")
            print(f"    Files per year: ~{365/7:.0f}")
            print(f"    Memory per year: ~{memory_per_file * 365/7:.1f} MB")
            print(f"    Memory for 5 years: ~{memory_per_file * 365/7 * 5:.1f} MB")
            
            era5_yearly_mb = memory_per_file * 365/7
            era5_total_mb = era5_yearly_mb * 5
        
        # Total estimates
        print(f"\n  TOTAL ESTIMATES for {region}:")
        print(f"    Peak memory during yearly processing: ~{max(pangu_yearly_mb, era5_yearly_mb) * 2:.0f} MB")
        print(f"    Peak memory during final combination: ~{(pangu_total_mb + era5_total_mb) * 1.5:.0f} MB")
        print(f"    Recommended memory allocation: {max((pangu_total_mb + era5_total_mb) * 2, 16000):.0f} MB")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS:")
    print("="*60)
    print("1. For India region: 64GB should be sufficient with efficient processing")
    print("2. For larger regions: Consider 128GB or use incremental processing")
    print("3. The OOM error occurred during final combination of 5 years")
    print("4. Solutions:")
    print("   - Use the improved batch combination approach")
    print("   - Use the incremental year-by-year approach")
    print("   - Increase memory to 128GB for safety margin")

if __name__ == "__main__":
    estimate_memory_requirements()
