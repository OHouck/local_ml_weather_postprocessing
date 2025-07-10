#!/usr/bin/env python3
"""
Name: weatherbench_download.py
Author: Ozma Houck
Date Created: 6/24/25

Purpose: WeatherBench download script with checkpointing,
error handling, and improved performance.
"""
import os
import warnings
import json
from pathlib import Path

# CRITICAL: Set these environment variables BEFORE importing any packages
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/dev/null'
os.environ['GCSFS_TOKEN'] = 'anon'
os.environ['GCSFS_ACCESS'] = 'read_only'
os.environ['GOOGLE_AUTH_DISABLE'] = 'true'
os.environ['GCLOUD_PROJECT'] = ''
os.environ['GOOGLE_CLOUD_PROJECT'] = ''

# Performance settings
os.environ['NUMEXPR_MAX_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP'] = '60s'
os.environ['DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT'] = '60s'
os.environ['GCSFS_REQUEST_TIMEOUT'] = '60'

import sys
import socket
from typing import List, Tuple, Dict
import time
from datetime import datetime, timedelta
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
    dask.config.set({'distributed.comm.timeouts.tcp': '60s'})
except ImportError as e:
    print(f"Import error: {e}")
    print("\nPlease run the following to fix dependencies:")
    print("pip install --force-reinstall 'numpy<2.0'")
    print("pip install --force-reinstall pandas==2.0.3 pyarrow==12.0.1")
    print("pip install --force-reinstall apache-beam==2.52.0")
    sys.exit(1)


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='weatherbenchX.data_loaders.xarray_loaders')
warnings.filterwarnings('ignore', category=UserWarning, module='google')

def patch_gcsfs_for_anonymous():
    """Monkey patch gcsfs to use anonymous access by default"""
    try:
        import gcsfs
        original_init = gcsfs.GCSFileSystem.__init__
        
        def anonymous_init(self, *args, **kwargs):
            kwargs['token'] = 'anon'
            kwargs['access'] = 'read_only'
            kwargs['project'] = None
            original_init(self, *args, **kwargs)
        
        gcsfs.GCSFileSystem.__init__ = anonymous_init
        logger.info("Successfully patched gcsfs for anonymous access")
        return True
    except Exception as e:
        logger.error(f"Failed to patch gcsfs: {e}")
        return False

def setup_directories():
    """Set up directory structure based on environment"""
    nodename = socket.gethostname()
    if nodename == "oMac.local":
        root = os.path.expanduser("~/OneDrive - The University of Chicago/ai_weather_ag/data")
    elif "anvil" in nodename.lower():
        root = os.path.expanduser("/anvil/projects/x-atm170020/ohouck/data")
    else:
        root = os.path.expanduser("~/ai_weather_ag/data")
        logger.warning(f"Unknown environment '{nodename}', using default root: {root}")

    dirs = {
        "root": root,
        "raw": os.path.join(root, "raw"),
        "processed": os.path.join(root, "processed"),
        "fig": os.path.join(root, "../figures/finetuning"),
        "checkpoint": os.path.join(root, "checkpoints")
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs

class CheckpointManager:
    """Manage checkpoints for resuming downloads"""
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(checkpoint_dir, "download_progress.json")
        
    def load_checkpoint(self) -> Dict:
        """Load checkpoint data"""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return {
            "completed_chunks": [],
            "partial_chunks": [],  # Track chunks with only predictions
            "last_year": None,
            "last_month": None
        }
    
    def save_checkpoint(self, data: Dict):
        """Save checkpoint data"""
        with open(self.checkpoint_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def is_chunk_completed(self, chunk_id: str) -> bool:
        """Check if a chunk has been completed"""
        checkpoint = self.load_checkpoint()
        return chunk_id in checkpoint["completed_chunks"]
    
    def mark_chunk_completed(self, chunk_id: str):
        """Mark a chunk as completed"""
        checkpoint = self.load_checkpoint()
        if chunk_id not in checkpoint["completed_chunks"]:
            checkpoint["completed_chunks"].append(chunk_id)
            # Remove from partial if it was there
            if chunk_id in checkpoint.get("partial_chunks", []):
                checkpoint["partial_chunks"].remove(chunk_id)
            self.save_checkpoint(checkpoint)
    
    def mark_chunk_partial(self, chunk_id: str):
        """Mark a chunk as partially completed (predictions only)"""
        checkpoint = self.load_checkpoint()
        if chunk_id not in checkpoint.get("partial_chunks", []):
            if "partial_chunks" not in checkpoint:
                checkpoint["partial_chunks"] = []
            checkpoint["partial_chunks"].append(chunk_id)
            self.save_checkpoint(checkpoint)

class LoadAndSaveChunk(beam.DoFn):
    """Custom DoFn to load data chunks and save them to disk with checkpointing"""
    def __init__(self, prediction_loader, target_loader, output_path, checkpoint_manager):
        self.prediction_loader = prediction_loader
        self.target_loader = target_loader
        self.output_path = output_path
        self.checkpoint_manager = checkpoint_manager
        if self.output_path.endswith('.nc'):
            self.output_path = self.output_path[:-3]
    
    def setup(self):
        """Setup method called once per worker"""
        os.makedirs(self.output_path, exist_ok=True)
        patch_gcsfs_for_anonymous()
    
    def process(self, element):
        """Process a single time chunk"""
        init_times, lead_times = element
        
        # Create chunk ID
        chunk_id = f"{str(init_times[0])[:10]}_{str(init_times[-1])[:10]}".replace(":", "-")
        
        # Skip if already completed
        if self.checkpoint_manager.is_chunk_completed(chunk_id):
            logger.info(f"Skipping already completed chunk: {chunk_id}")
            yield f"Skipped completed chunk {chunk_id}"
            return
        
        init_times = np.array(init_times, dtype='datetime64[ns]')
        lead_times = np.array(lead_times, dtype='timedelta64[ns]')
        
        prediction_saved = False
        target_saved = False
        
        try:
            # Load prediction data
            logger.info(f"Loading predictions for chunk {chunk_id}")
            prediction_chunk = self.prediction_loader.load_chunk(init_times, lead_times)
            logger.info(f"Loaded prediction data with shape: {dict(prediction_chunk.sizes)}")
            
            # Save prediction data immediately after loading
            pred_filename = f"predictions_{chunk_id}.nc"
            pred_output_path = os.path.join(self.output_path, pred_filename)
            
            encoding = {var: {'zlib': True, 'complevel': 4} for var in prediction_chunk.data_vars}
            prediction_chunk.to_netcdf(pred_output_path, encoding=encoding)
            logger.info(f"Saved predictions to: {pred_output_path}")
            prediction_saved = True
            
            # Calculate valid times for targets
            valid_times = []
            for init_time in init_times:
                for lead_time in lead_times:
                    valid_times.append(init_time + lead_time)
            valid_times = np.unique(np.array(valid_times, dtype='datetime64[ns]'))
            logger.info(f"Calculated {len(valid_times)} valid times for targets")
            
            # Try to load and save target data
            try:
                logger.info(f"Loading targets for chunk {chunk_id} (valid times: {valid_times[0]} to {valid_times[-1]})")
                target_chunk = self.target_loader.load_chunk(valid_times)
                logger.info(f"Loaded target data with shape: {dict(target_chunk.sizes)}")
                
                target_filename = f"targets_{chunk_id}.nc"
                target_output_path = os.path.join(self.output_path, target_filename)
                
                encoding = {var: {'zlib': True, 'complevel': 4} for var in target_chunk.data_vars}
                target_chunk.to_netcdf(target_output_path, encoding=encoding)
                logger.info(f"Saved targets to: {target_output_path}")
                target_saved = True
                
            except Exception as target_error:
                logger.error(f"Failed to load/save targets for chunk {chunk_id}: {str(target_error)}")
                # Continue even if targets fail - at least we have predictions
            
            # Only mark as completed if both were saved successfully
            if prediction_saved and target_saved:
                self.checkpoint_manager.mark_chunk_completed(chunk_id)
                logger.info(f"Successfully saved both predictions and targets for chunk {chunk_id}")
                yield f"Successfully saved chunk {chunk_id}"
            elif prediction_saved:
                self.checkpoint_manager.mark_chunk_partial(chunk_id)
                logger.warning(f"Only predictions saved for chunk {chunk_id} - targets failed")
                yield f"Partially saved chunk {chunk_id} (predictions only)"
            else:
                yield f"Failed to save chunk {chunk_id}"
            
        except Exception as e:
            error_msg = f"Error processing chunk {chunk_id}: {str(e)}"
            logger.error(error_msg)
            if prediction_saved:
                yield f"Partial success for chunk {chunk_id} - predictions saved, error: {str(e)}"
            else:
                yield error_msg

def create_monthly_chunks(year: int, month: int, lead_times: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create chunks for a specific month"""
    start_date = datetime(year, month, 1)
    if month == 12:
        end_date = datetime(year + 1, 1, 1)
    else:
        end_date = datetime(year, month + 1, 1)
    
    # Create daily init times at 12:00
    dates = []
    current_date = start_date
    while current_date < end_date:
        dates.append(np.datetime64(current_date) + np.timedelta64(12, 'h'))
        current_date += timedelta(days=1)
    
    init_times = np.array(dates, dtype='datetime64[ns]')
    
    # Create chunks of 7 days each
    chunks = []
    for i in range(0, len(init_times), 7):
        chunk_init_times = init_times[i:i+7]
        chunks.append((chunk_init_times, lead_times))
    
    return chunks

def run_download_by_month(
    prediction_path: str,
    target_path: str,
    variables: List[str],
    output_path: str,
    start_year: int,
    end_year: int,
    checkpoint_manager: CheckpointManager
):
    """Download data month by month with checkpointing"""
    
    # Load checkpoint to see where we left off
    checkpoint = checkpoint_manager.load_checkpoint()
    last_completed_year = checkpoint.get("last_year")
    last_completed_month = checkpoint.get("last_month")
    
    # Determine where to start 
    if last_completed_year and last_completed_month:
        # Start from the next month after the last completed one
        if last_completed_month == 12:
            start_from_year = last_completed_year + 1
            start_from_month = 1
        else:
            start_from_year = last_completed_year
            start_from_month = last_completed_month + 1
            
        logger.info(f"Resuming from {start_from_year}-{start_from_month:02d}")
    else:
        # No checkpoint, start from the beginning
        start_from_year = start_year
        start_from_month = 1
        logger.info(f"Starting fresh from {start_from_year}-{start_from_month:02d}")
    
    # Lead times (in hours converted to nanoseconds)
    lead_times = np.array([24, 48, 72, 96, 120, 144, 168], dtype='timedelta64[h]').astype('timedelta64[ns]')
    
    # Create data loaders
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning)
        
        target_loader = xarray_loaders.TargetsFromXarray(
            path=target_path,
            variables=variables,
        )
        
        prediction_loader = xarray_loaders.PredictionsFromXarray(
            path=prediction_path,
            variables=variables,
        )
    
    # Process each month
    for year in range(start_from_year, end_year + 1):
        start_month = 1 if year > start_from_year else start_from_month
        
        for month in range(start_month, 13):
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing {year}-{month:02d}")
            logger.info(f"{'='*50}")
            
            # Create chunks for this month
            month_chunks = create_monthly_chunks(year, month, lead_times)
            logger.info(f"Created {len(month_chunks)} chunks for {year}-{month:02d}")
            
            # Check if all chunks for this month are already completed
            all_completed = True
            for init_times, _ in month_chunks:
                chunk_id = f"{str(init_times[0])[:10]}_{str(init_times[-1])[:10]}".replace(":", "-")
                if not checkpoint_manager.is_chunk_completed(chunk_id):
                    all_completed = False
                    break
            
            if all_completed:
                logger.info(f"All chunks for {year}-{month:02d} already completed, skipping to next month")
                # Update checkpoint to mark this month as completed
                checkpoint["last_year"] = year
                checkpoint["last_month"] = month
                checkpoint_manager.save_checkpoint(checkpoint)
                continue
            
            # Process chunks using Beam
            pipeline_options = PipelineOptions(
                runner='DirectRunner',
                direct_num_workers=1,
                direct_running_mode='multi_threading'
            )
            
            with beam.Pipeline(options=pipeline_options) as pipeline:
                chunks = pipeline | f'CreateChunks_{year}_{month}' >> beam.Create(month_chunks)
                
                results = chunks | f'LoadAndSave_{year}_{month}' >> beam.ParDo(
                    LoadAndSaveChunk(
                        prediction_loader,
                        target_loader,
                        output_path,
                        checkpoint_manager
                    )
                )
                
                results | f'LogResults_{year}_{month}' >> beam.Map(print)
            
            # Update checkpoint after successfully processing the month
            checkpoint["last_year"] = year
            checkpoint["last_month"] = month
            checkpoint_manager.save_checkpoint(checkpoint)
            logger.info(f"Completed processing {year}-{month:02d}")
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Completed all months from {start_year} to {end_year}")
    logger.info(f"{'='*50}")

def test_target_loading(target_path: str, variables: List[str]):
    """Test function to diagnose target loading issues"""
    
    logger.info("Running target loading diagnostic test...")
    
    try:
        # Create loader
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=FutureWarning)
            
            logger.info("Creating target data loader...")
            target_loader = xarray_loaders.TargetsFromXarray(
                path=target_path,
                variables=variables[:1]  # Just test with first variable
            )
        
        # Test with a single time
        test_time = np.array(['2018-01-02T12:00:00'], dtype='datetime64[ns]')
        
        logger.info(f"Attempting to load target data for time: {test_time[0]}")
        
        # Try different time formats and configurations
        try:
            # Method 1: Direct time
            target_data = target_loader.load_chunk(test_time)
            logger.info(f"Success! Target data shape: {dict(target_data.sizes)}")
            logger.info(f"Target variables: {list(target_data.data_vars)}")
            return True
        except Exception as e1:
            logger.error(f"Method 1 failed: {str(e1)}")
            
            # Method 2: Try as valid_time dimension
            try:
                # Open the dataset directly to inspect structure
                import fsspec
                logger.info("Attempting to open target dataset directly...")
                
                fs = fsspec.filesystem('gcs', token='anon')
                mapper = fs.get_mapper(target_path)
                
                ds = xr.open_zarr(mapper)
                logger.info(f"Target dataset dimensions: {list(ds.dims)}")
                logger.info(f"Target dataset coordinates: {list(ds.coords)}")
                logger.info(f"Target dataset variables: {list(ds.data_vars)}")
                
                # Check time dimension name
                time_dims = [d for d in ds.dims if 'time' in d.lower()]
                logger.info(f"Time-related dimensions: {time_dims}")
                
                return True
                
            except Exception as e2:
                logger.error(f"Direct inspection failed: {str(e2)}")
                return False
                
    except Exception as e:
        logger.error(f"Target loading test failed: {str(e)}")
        return False

def main():
    """Main function with improved error handling and checkpointing"""
    
    # Setup
    logger.info("Setting up anonymous access...")
    if not patch_gcsfs_for_anonymous():
        logger.error("Failed to setup anonymous access. Exiting.")
        return
    
    # Configuration
    # for pangu
    # prediction_path = 'gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr'
    # target_path = 'gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr'

    # for ifs
    prediction_path= "gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr"
    target_path= "gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr" 
    
    # Variables to download
    variables = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"]
    
    # Run diagnostic test first
    logger.info("\n" + "="*50)
    logger.info("Running diagnostic tests...")
    logger.info("="*50)
    
    if not test_target_loading(target_path, variables):
        logger.error("Target loading test failed - please check the logs above")
        logger.info("Continuing anyway - predictions will still be downloaded")
    
    # Setup directories
    dirs = setup_directories()
    output_path = os.path.join(dirs['raw'], 'ifs_raw_data')
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(dirs['checkpoint'])
    
    # Time tracking
    start_time = time.time()
    
    try:
        # Run the download process
        run_download_by_month(
            prediction_path=prediction_path,
            target_path=target_path,
            variables=variables,
            output_path=output_path,
            start_year=2018,
            end_year=2022,
            checkpoint_manager=checkpoint_manager
        )
        
        end_time = time.time()
        logger.info(f"\nPipeline completed successfully in {(end_time - start_time)/60:.2f} minutes")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        logger.error("You can restart the script and it will resume from the last checkpoint")
        raise

if __name__ == "__main__":
    main()