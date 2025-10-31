import time
import sys
from datetime import datetime
import psutil
import os
import gc
import warnings
import pandas as pd
import xarray as xr
import numpy as np
import dask
from dask.distributed import Client, as_completed
from dask.diagnostics import ProgressBar
from pathlib import Path
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))
from helper_funcs import setup_directories


def print_time_and_memory(step_name, start_time):
    """Print elapsed time and current memory usage"""
    elapsed = time.time() - start_time
    memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024  # GB
    print(f"\n✓ {step_name}: {elapsed:.2f} seconds | Memory: {memory:.2f} GB")
    return time.time()


def save_to_zarr_with_fallback(subset_rechunked, output_path, available_vars):
    """Try different methods to save data to Zarr"""
    total_size_gb = sum(subset_rechunked[var].nbytes for var in available_vars) / 1024**3
    save_successful = False
    
    # Method 1: Save without explicit encoding
    try:
        print("\n  Trying save without encoding...")
        with ProgressBar():
            save_start = time.time()
            subset_rechunked.to_zarr(
                output_path,
                mode='w',
                consolidated=True,
                zarr_version=2
            )
            save_time = time.time() - save_start
            print(f"\n  Save completed in {save_time:.2f} seconds")
            print(f"  Average speed: {total_size_gb / save_time:.2f} GB/s")
            save_successful = True
            
    except Exception as e:
        print(f"  Method 1 failed: {e}")
        
        # Method 2: Save to NetCDF first, then convert
        try:
            print("\n  Trying NetCDF intermediate save...")
            temp_nc = 'temp.nc'
            
            with ProgressBar():
                subset_rechunked.compute().to_netcdf(temp_nc)
            
            ds_temp = xr.open_dataset(temp_nc)
            ds_temp.to_zarr(output_path, mode='w', consolidated=True)
            ds_temp.close()
            
            os.remove(temp_nc)
            save_successful = True
            print("  NetCDF conversion method succeeded!")
            
        except Exception as e2:
            print(f"  Method 2 failed: {e2}")
            
            # Method 3: Save variable by variable
            try:
                print("\n  Trying variable-by-variable save...")
                import zarr
                
                store = zarr.DirectoryStore(output_path)
                root = zarr.open(store, mode='w')
                
                for var in available_vars:
                    print(f"    Saving {var}...")
                    data = subset_rechunked[var].compute()
                    
                    ds_single = xr.Dataset({var: data})
                    
                    temp_path = f'temp_{var}.zarr'
                    ds_single.to_zarr(temp_path, mode='w')
                    
                    temp_store = zarr.open(temp_path)
                    zarr.copy(temp_store[var], root, name=var)
                    
                    import shutil
                    shutil.rmtree(temp_path)
                
                subset_rechunked.coords.to_zarr(output_path, mode='a')
                
                save_successful = True
                print("  Variable-by-variable method succeeded!")
                
            except Exception as e3:
                print(f"  Method 3 failed: {e3}")
                
                # Method 4: Use legacy zarr format
                try:
                    print("\n  Trying legacy zarr format...")
                    
                    computed_data = subset_rechunked.compute()
                    
                    computed_data.to_zarr(
                        output_path,
                        mode='w',
                        consolidated=True,
                        zarr_version=2
                    )
                    save_successful = True
                    print("  Legacy format method succeeded!")
                    
                except Exception as e4:
                    print(f"  Method 4 failed: {e4}")
                    raise Exception("All save methods failed!")
    
    return save_successful


def download_data(data_name, year, dirs):
    """Main download function"""
    # Start timing
    script_start = time.time()
    start_time = time.time()
    
    print("=== Download Script Started ===")
    print(f"Start time: {datetime.now()}")
    
    # 1. Set up Dask client with 16GB RAM
    print("\n1. Setting up Dask client...")
    
    client = Client(
        n_workers=2,
        threads_per_worker=4,
        processes=False,
        memory_limit='8GB',
        silence_logs=30
    )
    
    print(f"Dask client: {client}")
    print(f"Dashboard: {client.dashboard_link}")
    start_time = print_time_and_memory("Dask setup", start_time)
    
    # 2. Define variables and parameters
    print("\n2. Defining download parameters...")
    
    atmos_vars = None  # Initialize to None
    
    if data_name == 'era5':
        surface_vars = [
            '2m_temperature',
            '10m_u_component_of_wind',
            '10m_v_component_of_wind',
            'total_precipitation_6hr'
        ]
    elif data_name == 'hres_t0':

        # for some reason precip was empty when last downloaded in summer 2025
        surface_vars = [
            '2m_temperature',
            '10m_u_component_of_wind',
            '10m_v_component_of_wind',
            'mean_sea_level_pressure',
        ]
        atmos_vars = [
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
            "specific_humidity",
            "geopotential",
        ]
    
    # test XX
    time_range = [f'{year}-01-01', f'{year}-12-31']
    
    # Output paths XX
    output_path_surface = os.path.join(dirs['raw'], f"{data_name}_{year}.zarr")
    output_path_atmos = os.path.join(dirs["raw"], f"{data_name}_{year}_atmospheric.zarr")
    
    start_time = print_time_and_memory("Parameter setup", start_time)
    
    # 3. Open the dataset
    print("\n3. Opening dataset...")
    try:
        if data_name == 'era5':
            print("  Using ARCO-ERA5 dataset")
            ds = xr.open_zarr(
                'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3',
                consolidated=True,
                storage_options={"token": "anon"}
            )
            ds = ds.rename({'total_precipitation': 'total_precipitation_6hr'})

        elif data_name == 'hres_t0':
            print("  Using Weatherbench2-HRES-T0 dataset")
            ds = xr.open_zarr(
                "gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr",
                consolidated=True,
                storage_options={"token": "anon"}
            )
        else:
            raise ValueError(f"Unknown dataset name: {data_name}")
        
        vars = list(ds.data_vars)
        print(f"Available variables in dataset: {vars}")
        print(f"Dataset opened successfully")
        print(f"Dataset dimensions: {ds.dims}")
        
        # Check which requested surface variables are available
        available_surface_vars = []
        for var_name in surface_vars:
            if var_name in ds.data_vars:
                available_surface_vars.append(var_name)
                print(f"  ✓ {var_name} found")
            else:
                similar = [v for v in ds.data_vars if var_name.split('_')[-1] in v]
                if similar:
                    print(f"  ✗ {var_name} not found. Similar: {similar[:3]}")
                else:
                    print(f"  ✗ {var_name} not found")
        
        # Check which requested atmospheric variables are available
        available_atmos_vars = []
        if atmos_vars is not None:
            for var_name in atmos_vars:
                if var_name in ds.data_vars:
                    available_atmos_vars.append(var_name)
                    print(f"  ✓ {var_name} found")
                else:
                    similar = [v for v in ds.data_vars if var_name.split('_')[-1] in v]
                    if similar:
                        print(f"  ✗ {var_name} not found. Similar: {similar[:3]}")
                    else:
                        print(f"  ✗ {var_name} not found")
    
    except Exception as e:
        print(f"Error opening dataset: {e}")
        client.close()
        raise

    start_time = print_time_and_memory("Dataset opened", start_time)
    
    # ==================== PROCESS SURFACE VARIABLES ====================
    print("\n" + "="*70)
    print("PROCESSING SURFACE VARIABLES")
    print("="*70)
    
    # 4. Select surface subset
    print("\n4. Selecting surface data subset...")
    print(f"  Time range: {time_range[0]} to {time_range[1]}")
    print(f"  Variables: {available_surface_vars}")

    subset = ds[available_surface_vars].sel(
        time=slice(time_range[0], time_range[1])
    )

    # Calculate cumulative precipitation for the entire day
    if 'total_precipitation_6hr' in available_surface_vars:
        six_hour_precip = subset['total_precipitation_6hr']

        # Shift time coordinate forward by 6 hours
        six_hour_precip = six_hour_precip.assign_coords(
            time=six_hour_precip.time + pd.Timedelta(hours=6)
        )

        # Create daily precipitation sums
        daily_precip = six_hour_precip.resample(time='1D').sum()
        daily_precip = daily_precip.rename('total_precipitation')

        # Broadcast back to 6-hourly resolution
        total_precipitation = daily_precip.resample(time='6H').ffill()

        # Merge back
        subset = subset.drop_vars('total_precipitation_6hr')
        subset["total_precipitation"] = total_precipitation

    # Filter for only hours 0, 6, 12. 0 and 12 for evalutating accuracy but 6 is needed to initalize aurora model
    subset = subset.sel(time=subset.time.dt.hour.isin([0, 6, 12]))

    # Update available_vars to reflect changes
    available_surface_vars = list(subset.data_vars)
    
    # Print subset info
    print(f"\nSurface subset shape:")
    for var in available_surface_vars:
        var_shape = subset[var].shape
        var_size_gb = subset[var].nbytes / 1024**3
        print(f"  {var}: {var_shape} (~{var_size_gb:.2f} GB)")
    
    total_size_gb = sum(subset[var].nbytes for var in available_surface_vars) / 1024**3
    print(f"\nTotal surface size: ~{total_size_gb:.2f} GB")
    
    print(f"\nOriginal chunk structure for {available_surface_vars[0]}:")
    print(f"  {subset[available_surface_vars[0]].chunks}")
    
    start_time = print_time_and_memory("Surface subset selected", start_time)
    
    # 5. Rechunk surface data for optimal performance
    print("\n5. Rechunking surface data for optimal download...")
    
    n_times = len(subset.time)
    time_chunk = min(240, n_times // 10)
    
    chunk_dict = {
        'time': time_chunk,
        'latitude': len(subset.latitude),
        'longitude': len(subset.longitude)
    }
    
    subset_rechunked = subset.chunk(chunk_dict)
    print(f"  New chunks: {subset_rechunked[available_surface_vars[0]].chunks}")
    
    start_time = print_time_and_memory("Surface rechunking complete", start_time)
    
    # 6. Save surface data to Zarr
    print("\n6. Saving surface data to Zarr format...")
    print(f"  Output path: {output_path_surface}")
    
    save_successful = save_to_zarr_with_fallback(
        subset_rechunked, 
        output_path_surface, 
        available_surface_vars
    )
    
    if not save_successful:
        client.close()
        raise Exception("Failed to save surface data")
    
    start_time = print_time_and_memory("Surface data saved", start_time)
    
    # 7. Verify the saved surface data
    print("\n7. Verifying saved surface data...")
    ds_saved = xr.open_zarr(output_path_surface)
    print(f"  Saved dataset shape: {ds_saved.dims}")
    print(f"  Variables saved: {list(ds_saved.data_vars)}")
    print(f"  Time range: {ds_saved.time.values[0]} to {ds_saved.time.values[-1]}")
    
    if os.path.exists(output_path_surface):
        import glob
        total_size = sum(
            os.path.getsize(f) 
            for f in glob.glob(f"{output_path_surface}/**/*", recursive=True)
            if os.path.isfile(f)
        ) / 1024**3
        print(f"  Total file size on disk: {total_size:.2f} GB")
    
    start_time = print_time_and_memory("Surface verification complete", start_time)
    
    # ==================== PROCESS ATMOSPHERIC VARIABLES ====================
    if available_atmos_vars:
        print("\n" + "="*70)
        print("PROCESSING ATMOSPHERIC VARIABLES")
        print("="*70)
        
        # 8. Select atmospheric subset
        print("\n8. Selecting atmospheric data subset...")
        print(f"  Time range: {time_range[0]} to {time_range[1]}")
        print(f"  Variables: {available_atmos_vars}")

        atmos_subset = ds[available_atmos_vars].sel(
            time=slice(time_range[0], time_range[1])
        )

        # Filter for only hours 0 and 12 (same as surface)
        atmos_subset = atmos_subset.sel(time=atmos_subset.time.dt.hour.isin([0, 12]))
        
        # Print subset info
        print(f"\nAtmospheric subset shape:")
        for var in available_atmos_vars:
            var_shape = atmos_subset[var].shape
            var_size_gb = atmos_subset[var].nbytes / 1024**3
            print(f"  {var}: {var_shape} (~{var_size_gb:.2f} GB)")
        
        total_atmos_size_gb = sum(atmos_subset[var].nbytes for var in available_atmos_vars) / 1024**3
        print(f"\nTotal atmospheric size: ~{total_atmos_size_gb:.2f} GB")
        
        print(f"\nOriginal chunk structure for {available_atmos_vars[0]}:")
        print(f"  {atmos_subset[available_atmos_vars[0]].chunks}")
        
        start_time = print_time_and_memory("Atmospheric subset selected", start_time)
        
        # 9. Rechunk atmospheric data
        print("\n9. Rechunking atmospheric data for optimal download...")
        
        chunk_dict_atmos = {
            'time': time_chunk,
            'level': len(atmos_subset.level) if 'level' in atmos_subset.dims else None,
            'latitude': len(atmos_subset.latitude),
            'longitude': len(atmos_subset.longitude)
        }
        # Remove None values
        chunk_dict_atmos = {k: v for k, v in chunk_dict_atmos.items() if v is not None}
        
        atmos_subset_rechunked = atmos_subset.chunk(chunk_dict_atmos)
        print(f"  New chunks: {atmos_subset_rechunked[available_atmos_vars[0]].chunks}")
        
        start_time = print_time_and_memory("Atmospheric rechunking complete", start_time)
        
        # 10. Save atmospheric data to Zarr
        print("\n10. Saving atmospheric data to Zarr format...")
        print(f"  Output path: {output_path_atmos}")
        
        save_successful_atmos = save_to_zarr_with_fallback(
            atmos_subset_rechunked, 
            output_path_atmos, 
            available_atmos_vars
        )
        
        if not save_successful_atmos:
            client.close()
            raise Exception("Failed to save atmospheric data")
        
        start_time = print_time_and_memory("Atmospheric data saved", start_time)
        
        # 11. Verify the saved atmospheric data
        print("\n11. Verifying saved atmospheric data...")
        ds_saved_atmos = xr.open_zarr(output_path_atmos)
        print(f"  Saved dataset shape: {ds_saved_atmos.dims}")
        print(f"  Variables saved: {list(ds_saved_atmos.data_vars)}")
        print(f"  Time range: {ds_saved_atmos.time.values[0]} to {ds_saved_atmos.time.values[-1]}")
        
        if os.path.exists(output_path_atmos):
            import glob
            total_size_atmos = sum(
                os.path.getsize(f) 
                for f in glob.glob(f"{output_path_atmos}/**/*", recursive=True)
                if os.path.isfile(f)
            ) / 1024**3
            print(f"  Total file size on disk: {total_size_atmos:.2f} GB")
        
        start_time = print_time_and_memory("Atmospheric verification complete", start_time)
    
    # ==================== FINAL SUMMARY ====================
    print("\n" + "="*70)
    print("DOWNLOAD SUMMARY")
    print("="*70)
    total_time = time.time() - script_start
    print(f"Total execution time: {total_time/60:.2f} minutes")
    print(f"\nSurface data:")
    print(f"  Output: {output_path_surface}")
    print(f"  Size: ~{total_size_gb:.2f} GB")
    
    if available_atmos_vars:
        print(f"\nAtmospheric data:")
        print(f"  Output: {output_path_atmos}")
        print(f"  Size: ~{total_atmos_size_gb:.2f} GB")
    
    print(f"\nEnd time: {datetime.now()}")
    
    # Close Dask client
    client.close()
    print("\nDask client closed.")
    
    return output_path_surface, output_path_atmos if available_atmos_vars else None


if __name__ == '__main__':
    # Check package versions
    import zarr
    import numcodecs
    print(f"Package versions:")
    print(f"  xarray: {xr.__version__}")
    print(f"  zarr: {zarr.__version__}")
    print(f"  numcodecs: {numcodecs.__version__}")
    print(f"  dask: {dask.__version__}")

    # years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
    years = [2019, 2023]
    data_source = 'era5'  # hres_t0 or era5
    dirs = setup_directories()
    
    # Try the download
    for year in years:
        output_surface, output_atmos = download_data(data_source, year, dirs)
        print(f"\nSuccess!")
        print(f"  Surface data saved to: {output_surface}")
        if output_atmos:
            print(f"  Atmospheric data saved to: {output_atmos}")