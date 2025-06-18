"""
Simplified approach using environment variables to force anonymous access
"""

import os
import warnings

# CRITICAL: Set these environment variables before importing any Google/GCS libraries
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/dev/null'  # Force no credentials
os.environ['GCSFS_TOKEN'] = 'anon'  # Force anonymous token
os.environ['GCSFS_ACCESS'] = 'read_only'  # Force read-only access

# Disable authentication entirely
os.environ['GOOGLE_AUTH_DISABLE'] = 'true'
os.environ['GCLOUD_PROJECT'] = ''
os.environ['GOOGLE_CLOUD_PROJECT'] = ''

# Other environment settings
os.environ['NUMEXPR_MAX_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# Now import packages
import sys
import socket
from typing import List, Tuple
import time
from datetime import datetime
import numpy as np
import logging

try:
    import apache_beam as beam
    from apache_beam.options.pipeline_options import PipelineOptions
    import xarray as xr
    import weatherbenchX
    from weatherbenchX.data_loaders import xarray_loaders
    from weatherbenchX import time_chunks
    import dask
    
    # Configure dask
    dask.config.set({'distributed.comm.timeouts.tcp': '60s'})
    
    # Set xarray to use anonymous access by default
    xr.set_options(warn_for_unclosed_files=False)
    
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='google')

def patch_gcsfs_for_anonymous():
    """
    Monkey patch gcsfs to use anonymous access by default
    """
    try:
        import gcsfs
        
        # Store original init
        original_init = gcsfs.GCSFileSystem.__init__
        
        def anonymous_init(self, *args, **kwargs):
            # Force anonymous parameters
            kwargs['token'] = 'anon'
            kwargs['access'] = 'read_only'
            kwargs['project'] = None
            # Call original init with forced params
            original_init(self, *args, **kwargs)
        
        # Replace the init method
        gcsfs.GCSFileSystem.__init__ = anonymous_init
        
        logger.info("Successfully patched gcsfs for anonymous access")
        return True
        
    except Exception as e:
        logger.error(f"Failed to patch gcsfs: {e}")
        return False

def setup_directories():
    """Setup directory structure"""
    nodename = socket.gethostname()
    if "anvil" in nodename.lower():
        root = "/anvil/projects/x-atm170020/ohouck/data"
    else:
        root = os.path.expanduser("~/ai_weather_ag/data")

    dirs = {
        "root": root,
        "raw": os.path.join(root, "raw"),
        "processed": os.path.join(root, "processed")
    }
    
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    
    return dirs

def test_simple_xarray_access(prediction_path, target_path):
    """Test if we can access the data with simple xarray calls"""
    try:
        logger.info("Testing direct xarray access...")
        
        # Try to open with xarray directly
        pred_ds = xr.open_zarr(prediction_path, chunks={'time': 1})
        logger.info(f"Successfully opened prediction dataset: {list(pred_ds.dims.keys())}")
        
        target_ds = xr.open_zarr(target_path, chunks={'time': 1})
        logger.info(f"Successfully opened target dataset: {list(target_ds.dims.keys())}")
        
        return True
        
    except Exception as e:
        logger.error(f"Direct xarray access failed: {e}")
        return False

def test_weatherbenchx_loaders(prediction_path, target_path, variables):
    """Test WeatherBenchX data loaders"""
    try:
        logger.info("Testing WeatherBenchX loaders...")
        
        # Create loaders
        pred_loader = xarray_loaders.PredictionsFromXarray(
            path=prediction_path,
            variables=variables[:1]  # Just one variable for testing
        )
        
        target_loader = xarray_loaders.TargetsFromXarray(
            path=target_path,
            variables=variables[:1]
        )
        
        # Test loading a small chunk
        test_init_time = np.array(['2018-01-01T12:00:00'], dtype='datetime64[ns]')
        test_lead_time = np.array([24 * 3600 * 10**9], dtype='timedelta64[ns]')
        
        logger.info("Loading test prediction chunk...")
        pred_data = pred_loader.load_chunk(test_init_time, test_lead_time)
        
        logger.info("Loading test target chunk...")
        target_time = test_init_time + test_lead_time
        target_data = target_loader.load_chunk(target_time)
        
        logger.info("WeatherBenchX loaders test successful!")
        return True, pred_loader, target_loader
        
    except Exception as e:
        logger.error(f"WeatherBenchX loaders test failed: {e}")
        return False, None, None

def download_sample_data(output_dir):
    """Download a small sample using the working approach"""
    
    # Configuration
    prediction_path = 'gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr'
    target_path = 'gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr'
    variables = ["2m_temperature"]  # Start with just one variable
    
    os.makedirs(output_dir, exist_ok=True)
    
    # First test direct xarray access
    if not test_simple_xarray_access(prediction_path, target_path):
        logger.error("Cannot access data with xarray. Check network connectivity.")
        return False
    
    # Test WeatherBenchX loaders
    success, pred_loader, target_loader = test_weatherbenchx_loaders(
        prediction_path, target_path, variables
    )
    
    if not success:
        logger.error("WeatherBenchX loaders failed. Try updating packages.")
        return False
    
    # If we get here, everything is working
    logger.info("All tests passed! WeatherBenchX should work with your environment.")
    
    # Download a small sample
    try:
        test_init_times = np.array(['2018-01-01T12:00:00', '2018-01-02T12:00:00'], dtype='datetime64[ns]')
        test_lead_times = np.array([24 * 3600 * 10**9], dtype='timedelta64[ns]')
        
        logger.info("Downloading sample data...")
        pred_data = pred_loader.load_chunk(test_init_times, test_lead_times)
        
        target_times = test_init_times + test_lead_times[0]
        target_data = target_loader.load_chunk(target_times)
        
        # Save sample data
        pred_path = os.path.join(output_dir, 'sample_predictions.nc')
        target_path_out = os.path.join(output_dir, 'sample_targets.nc')
        
        pred_data.to_netcdf(pred_path)
        target_data.to_netcdf(target_path_out)
        
        logger.info(f"Sample data saved to {output_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Sample download failed: {e}")
        return False

def main():
    """Main function to test and download data"""
    
    # Setup environment for anonymous access
    logger.info("Setting up anonymous access...")
    patch_gcsfs_for_anonymous()
    
    # Setup directories
    dirs = setup_directories()
    output_dir = os.path.join(dirs['raw'], 'weatherbench_test')
    
    # Run test and sample download
    if download_sample_data(output_dir):
        logger.info("SUCCESS! Your environment is properly configured.")
        logger.info("You can now run your full WeatherBenchX download script.")
        logger.info("Make sure to add the environment variable setup at the top of your script.")
    else:
        logger.error("FAILED! Check the error messages above.")
        logger.info("Try updating packages: pip install --upgrade xarray dask gcsfs")

if __name__ == "__main__":
    main()