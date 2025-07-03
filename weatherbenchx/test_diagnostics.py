"""
Test script to verify the processing approach on a small subset of data
Run this before the full job to ensure everything works correctly
"""

import os
import sys
import xarray as xr
import numpy as np
import glob
from datetime import datetime

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from combine_and_subset import (
    setup_directories, get_region_bounds, preprocess_and_subset,
    validate_netcdf_file
)

def test_single_file_processing():
    """Test processing a single file"""
    print("\n" + "="*60)
    print("TEST 1: Single File Processing")
    print("="*60)
    
    dirs = setup_directories()
    test_files = sorted(glob.glob(os.path.join(dirs["raw"], "pangu_raw_data", "predictions*.nc")))[:1]
    
    if not test_files:
        print("ERROR: No test files found!")
        return False
    
    test_file = test_files[0]
    print(f"Testing with file: {os.path.basename(test_file)}")
    
    try:
        # Test file validation
        print("\n1. Testing file validation...")
        is_valid = validate_netcdf_file(test_file)
        print(f"   File valid: {is_valid}")
        
        if not is_valid:
            return False
        
        # Test loading and preprocessing
        print("\n2. Testing file loading and preprocessing...")
        with xr.open_dataset(test_file, engine='netcdf4') as ds:
            print(f"   Original dimensions: {dict(ds.sizes)}")
            print(f"   Original size: {sum(ds[var].nbytes for var in ds.data_vars) / 1e9:.3f} GB")
            
            # Test subsetting for India
            print("\n3. Testing regional subsetting (India)...")
            ds_india = preprocess_and_subset(ds, 'india')
            print(f"   Subset dimensions: {dict(ds_india.sizes)}")
            print(f"   Subset size: {sum(ds_india[var].nbytes for var in ds_india.data_vars) / 1e9:.3f} GB")
            
            # Calculate size reduction
            original_size = ds.latitude.size * ds.longitude.size
            subset_size = ds_india.latitude.size * ds_india.longitude.size
            reduction = original_size / subset_size
            print(f"   Size reduction: {reduction:.1f}x")
            
            # Test loading into memory
            print("\n4. Testing memory loading...")
            start_time = datetime.now()
            ds_loaded = ds_india.load()
            load_time = (datetime.now() - start_time).total_seconds()
            print(f"   Loaded in {load_time:.2f} seconds")
            
            # Test saving
            print("\n5. Testing file saving...")
            test_output = os.path.join(dirs['processed'], 'test_output.nc')
            encoding = {var: {'zlib': True, 'complevel': 4} for var in ds_loaded.data_vars}
            
            start_time = datetime.now()
            ds_loaded.to_netcdf(test_output, encoding=encoding, engine='netcdf4')
            save_time = (datetime.now() - start_time).total_seconds()
            
            output_size = os.path.getsize(test_output) / 1e6
            print(f"   Saved in {save_time:.2f} seconds")
            print(f"   Output file size: {output_size:.2f} MB")
            
            # Clean up
            os.remove(test_output)
            
        print("\n✓ Single file processing test PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ Single file processing test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_file_combining():
    """Test combining multiple files"""
    print("\n" + "="*60)
    print("TEST 2: Multiple File Combining")
    print("="*60)
    
    dirs = setup_directories()
    test_files = sorted(glob.glob(os.path.join(dirs["raw"], "pangu_raw_data", "predictions*.nc")))[:3]
    
    if len(test_files) < 3:
        print(f"WARNING: Only {len(test_files)} files found, need at least 3 for this test")
        return False
    
    print(f"Testing with {len(test_files)} files")
    
    try:
        # Process files individually
        print("\n1. Processing files individually...")
        processed_datasets = []
        
        for i, file_path in enumerate(test_files):
            with xr.open_dataset(file_path, engine='netcdf4') as ds:
                ds_subset = preprocess_and_subset(ds, 'india')
                ds_loaded = ds_subset.load()
                processed_datasets.append(ds_loaded)
                print(f"   Processed file {i+1}/{len(test_files)}")
        
        # Test combining
        print("\n2. Testing concatenation...")
        combined = xr.concat(processed_datasets, dim='time')
        print(f"   Combined dimensions: {dict(combined.sizes)}")
        print(f"   Time steps: {combined.sizes['time']}")
        
        # Verify time ordering
        time_diff = np.diff(combined.time.values)
        is_monotonic = np.all(time_diff > np.timedelta64(0, 's'))
        print(f"   Time ordering correct: {is_monotonic}")
        
        # Test saving combined dataset
        print("\n3. Testing combined save...")
        test_output = os.path.join(dirs['processed'], 'test_combined.nc')
        encoding = {var: {'zlib': True, 'complevel': 4} for var in combined.data_vars}
        
        start_time = datetime.now()
        combined.to_netcdf(test_output, encoding=encoding, engine='netcdf4')
        save_time = (datetime.now() - start_time).total_seconds()
        
        output_size = os.path.getsize(test_output) / 1e6
        print(f"   Saved in {save_time:.2f} seconds")
        print(f"   Output file size: {output_size:.2f} MB")
        
        # Verify saved file
        print("\n4. Verifying saved file...")
        with xr.open_dataset(test_output) as ds_verify:
            print(f"   Verified dimensions: {dict(ds_verify.sizes)}")
            print(f"   Variables present: {list(ds_verify.data_vars)}")
        
        # Clean up
        os.remove(test_output)
        
        print("\n✓ Multiple file combining test PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ Multiple file combining test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_usage():
    """Test memory usage patterns"""
    print("\n" + "="*60)
    print("TEST 3: Memory Usage Analysis")
    print("="*60)
    
    import psutil
    process = psutil.Process()
    
    dirs = setup_directories()
    test_file = sorted(glob.glob(os.path.join(dirs["raw"], "pangu_raw_data", "predictions*.nc")))[0]
    
    # Initial memory
    mem_start = process.memory_info().rss / 1e9
    print(f"Initial memory: {mem_start:.2f} GB")
    
    # After loading original
    with xr.open_dataset(test_file, engine='netcdf4') as ds:
        mem_opened = process.memory_info().rss / 1e9
        print(f"After opening file: {mem_opened:.2f} GB (+{mem_opened - mem_start:.2f} GB)")
        
        # After subsetting
        ds_subset = preprocess_and_subset(ds, 'india')
        mem_subset = process.memory_info().rss / 1e9
        print(f"After subsetting: {mem_subset:.2f} GB (+{mem_subset - mem_opened:.2f} GB)")
        
        # After loading
        ds_loaded = ds_subset.load()
        mem_loaded = process.memory_info().rss / 1e9
        print(f"After loading to memory: {mem_loaded:.2f} GB (+{mem_loaded - mem_subset:.2f} GB)")
    
    # After closing
    import gc
    gc.collect()
    mem_final = process.memory_info().rss / 1e9
    print(f"After cleanup: {mem_final:.2f} GB")
    
    print("\n✓ Memory usage test completed")
    return True

def main():
    """Run all diagnostic tests"""
    print("WeatherBench Data Processing Diagnostics")
    print("========================================")
    
    dirs = setup_directories()
    
    # Run tests
    tests = [
        test_single_file_processing,
        test_multiple_file_combining,
        test_memory_usage
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"\nTest {test_func.__name__} encountered error: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed! Safe to run full processing.")
    else:
        print("\n✗ Some tests failed. Please fix issues before running full processing.")
    
    return passed == total

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
