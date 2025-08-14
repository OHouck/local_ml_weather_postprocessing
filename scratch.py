import xarray as xr
import time
import os

def diagnose_zarr_performance(path):
    print(f"\n{'='*60}")
    print(f"Diagnosing: {os.path.basename(path)}")
    print(f"{'='*60}")
    
    # Open dataset
    start = time.time()
    ds = xr.open_zarr(path)
    open_time = time.time() - start
    print(f"Dataset open time: {open_time:.3f}s")
    
    # Print basic info
    print(f"Dimensions: {ds.dims}")
    print(f"Variables: {list(ds.data_vars)}")
    
    # Check each variable
    for var in ds.data_vars:
        print(f"\n{var}:")
        print(f"  Shape: {ds[var].shape}")
        print(f"  Chunks: {ds[var].chunks}")
        print(f"  Dtype: {ds[var].dtype}")
        print(f"  Size: {ds[var].nbytes / 1e6:.2f} MB")
        
        # Time loading
        start = time.time()
        data = ds[var].load()
        load_time = time.time() - start
        print(f"  Load time: {load_time:.3f}s")
    
    # Get actual file size
    if os.path.isdir(path):
        total_size = sum(
            os.path.getsize(os.path.join(root, file))
            for root, dirs, files in os.walk(path)
            for file in files
        ) / 1e6
        print(f"\nTotal file size on disk: {total_size:.2f} MB")
    
    return ds

# Test both wind and temperature
wind_path = "/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/ai_weather_ag/data/fine_tuning_output/pangu/amazon/train_10m_wind_speed_test_10m_wind_speed_dim10x10_leadtime_240h_train2018-01-01-2019-12-31_test2022-01-01-2022-12-31_mlp.zarr"
temp_path = wind_path.replace("10m_wind_speed", "2m_temperature")

if os.path.exists(wind_path):
    ds_wind = diagnose_zarr_performance(wind_path)

if os.path.exists(temp_path):
    ds_temp = diagnose_zarr_performance(temp_path)