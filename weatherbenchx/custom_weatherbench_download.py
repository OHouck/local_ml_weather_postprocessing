# fixed_weatherbench_download.py
"""
Fixed version of the WeatherBench download script with proper error handling,
authentication setup, and path corrections.
"""

import apache_beam as beam
import numpy as np
import xarray as xr
import weatherbenchX
from weatherbenchX.data_loaders import xarray_loaders
from weatherbenchX import time_chunks
from datetime import datetime
import time
import os
import socket
import warnings
from typing import List, Tuple
from apache_beam.options.pipeline_options import PipelineOptions
import logging
import dask

dask.config.set({'distributed.comm.timeouts.tcp': '60s'})

def setup_directories():
    # Determine root directory based on environment.
    nodename = socket.gethostname()
    if nodename == "oMac.local":  # local laptop
        root = os.path.expanduser(
            "~/OneDrive - The University of Chicago/ai_weather_ag/data"
        )
    else:
        raise Exception("Unknown environment, please specify the root directory")

    dirs = {
        "root": root,
        "raw": os.path.join(root, "raw"),
        "processed": os.path.join(root, "processed"),
        "fig": os.path.join(root, "../figures/finetuning"),
        "external": os.path.join("Volumes" ,"wd_external_hd", "weatherbench")
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress the xarray FutureWarning
warnings.filterwarnings('ignore', category=FutureWarning, module='weatherbenchX.data_loaders.xarray_loaders')

def setup_gcs_authentication():
    """
    Set up Google Cloud Storage authentication for local development.
    """
    # For anonymous access to public buckets
    # This should work for WeatherBench2 public data
    try:
        import gcsfs
        # Create anonymous access
        fs = gcsfs.GCSFileSystem(token='anon')
        return fs
    except ImportError:
        logger.warning("gcsfs not installed. Install with: pip install gcsfs")
        return None


class LoadAndSaveChunk(beam.DoFn):
    """
    Custom DoFn to load data chunks and save them to disk.
    Fixed version with proper path handling and error management.
    """
    def __init__(self, prediction_loader, target_loader, output_path):
        self.prediction_loader = prediction_loader
        self.target_loader = target_loader
        self.output_path = output_path
        # Ensure output_path is a directory
        if self.output_path.endswith('.nc'):
            self.output_path = self.output_path[:-3]  # Remove .nc extension
    
    def setup(self):
        """Setup method called once per worker."""
        # Create output directory if it doesn't exist
        os.makedirs(self.output_path, exist_ok=True)
    
    def process(self, element):
        """Process a single time chunk."""
        init_times, lead_times = element
        
        # Convert to proper numpy datetime format if needed
        init_times = np.array(init_times, dtype='datetime64[ns]')
        lead_times = np.array(lead_times, dtype='timedelta64[ns]')
        
        try:
            # Load prediction data
            logger.info(f"Loading predictions for init times: {init_times[0]} to {init_times[-1]}")
            prediction_chunk = self.prediction_loader.load_chunk(init_times, lead_times)
            
            # Calculate actual times from init_times + lead_times
            valid_times = []
            for init_time in init_times:
                for lead_time in lead_times:
                    valid_time = init_time + lead_time
                    valid_times.append(valid_time)
            
            valid_times = np.unique(np.array(valid_times, dtype='datetime64[ns]'))
            
            # Create a simple time-based chunk identifier
            # Fix the datetime string formatting to be filesystem-safe
            chunk_id = f"{str(init_times[0])[:10]}_{str(init_times[-1])[:10]}".replace(":", "-")
            
            # Save prediction data
            pred_filename = f"predictions_{chunk_id}.nc"
            pred_output_path = os.path.join(self.output_path, pred_filename)
            
            logger.info(f"Saving predictions to: {pred_output_path}")
            # Add compression to reduce file size
            encoding = {var: {'zlib': True, 'complevel': 4} for var in prediction_chunk.data_vars}
            prediction_chunk.to_netcdf(pred_output_path, encoding=encoding)
            
            # Load and save target data
            logger.info(f"Loading targets for valid times: {valid_times[0]} to {valid_times[-1]}")
            target_chunk = self.target_loader.load_chunk(valid_times)
            
            target_filename = f"targets_{chunk_id}.nc"
            target_output_path = os.path.join(self.output_path, target_filename)
            
            logger.info(f"Saving targets to: {target_output_path}")
            encoding = {var: {'zlib': True, 'complevel': 4} for var in target_chunk.data_vars}
            target_chunk.to_netcdf(target_output_path, encoding=encoding)
            
            yield f"Successfully saved chunk {chunk_id}"
            
        except Exception as e:
            error_msg = f"Error processing chunk: {str(e)}"
            logger.error(error_msg)
            yield error_msg


def create_time_chunks(times: time_chunks.TimeChunks) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create a list of time chunks from TimeChunks object."""
    chunks = []
    
    # Get all chunk combinations
    init_time_chunks = np.array_split(times.init_times, 
                                     max(1, len(times.init_times) // times.init_time_chunk_size))
    lead_time_chunks = np.array_split(times.lead_times,
                                     max(1, len(times.lead_times) // times.lead_time_chunk_size))
    
    for init_chunk in init_time_chunks:
        for lead_chunk in lead_time_chunks:
            if len(init_chunk) > 0 and len(lead_chunk) > 0:
                chunks.append((init_chunk, lead_chunk))
    
    return chunks


def run_download_pipeline_with_auth(
    prediction_path: str,
    target_path: str,
    variables: List[str],
    init_times: np.ndarray,
    lead_times: np.ndarray,
    output_path: str,
    init_time_chunk_size: int = 2,
    lead_time_chunk_size: int = 1,
    runner: str = 'DirectRunner',
    beam_options: dict = None,
    use_anonymous_access: bool = True
):
    """
    Run pipeline with proper authentication and error handling.
    """
    
    # Fix output path if it ends with .nc
    if output_path.endswith('.nc'):
        output_path = output_path[:-3]
        logger.info(f"Removed .nc extension from output path. Using: {output_path}")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Set up authentication
    if use_anonymous_access:
        logger.info("Setting up anonymous access for public GCS buckets...")
        fs = setup_gcs_authentication()
    
    # Create data loaders with proper xarray options
    # Suppress the warning at the source
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning)
        
        target_data_loader = xarray_loaders.TargetsFromXarray(
            path=target_path,
            variables=variables,
        )
        
        prediction_data_loader = xarray_loaders.PredictionsFromXarray(
            path=prediction_path,
            variables=variables,
        )
    
    # Create time chunks
    times = time_chunks.TimeChunks(
        init_times,
        lead_times,
        init_time_chunk_size=init_time_chunk_size,
        lead_time_chunk_size=lead_time_chunk_size
    )
    
    # Generate list of chunks
    chunk_list = create_time_chunks(times)
    logger.info(f"Created {len(chunk_list)} chunks for processing")
    
    # Set up pipeline options
    if beam_options is None:
        beam_options = {}
    
    pipeline_options = PipelineOptions(**beam_options)
    
    # Create and run pipeline
    with beam.Pipeline(runner=runner, options=pipeline_options) as pipeline:
        
        # Create PCollection from chunk list
        chunks = pipeline | 'CreateChunks' >> beam.Create(chunk_list)
        
        # Process each chunk
        results = chunks | 'LoadAndSave' >> beam.ParDo(
            LoadAndSaveChunk(
                prediction_data_loader,
                target_data_loader,
                output_path
            )
        )
        
        # Log results
        results | 'LogResults' >> beam.Map(print)


def download_single_chunk_test(
    prediction_path: str,
    target_path: str,
    variables: List[str],
    output_path: str
):
    """
    Test function to download a single small chunk without beam.
    Useful for debugging authentication and data access issues.
    """
    import warnings
    
    logger.info("Running single chunk test...")
    
    # Fix output path
    if output_path.endswith('.nc'):
        output_path = output_path[:-3]
    os.makedirs(output_path, exist_ok=True)
    
    # Test with a single day and single lead time
    test_init_time = np.array(['2018-01-01T12:00:00'], dtype='datetime64[ns]')
    test_lead_time = np.array([24 * 3600 * 10**9], dtype='timedelta64[ns]')  # 24 hours
    
    try:
        # Create loaders
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=FutureWarning)
            
            logger.info("Creating data loaders...")
            pred_loader = xarray_loaders.PredictionsFromXarray(
                path=prediction_path,
                variables=variables[:2]  # Just test with first 2 variables
            )
            
            target_loader = xarray_loaders.TargetsFromXarray(
                path=target_path,
                variables=variables[:2]
            )
        
        # Load data
        logger.info("Loading prediction data...")
        pred_data = pred_loader.load_chunk(test_init_time, test_lead_time)
        logger.info(f"Prediction data shape: {dict(pred_data.dims)}")
        
        # Calculate target time
        target_time = test_init_time + test_lead_time
        
        logger.info("Loading target data...")
        target_data = target_loader.load_chunk(target_time)
        logger.info(f"Target data shape: {dict(target_data.dims)}")
        
        # Save test data
        pred_path = os.path.join(output_path, 'test_predictions.nc')
        target_path = os.path.join(output_path, 'test_targets.nc')
        
        pred_data.to_netcdf(pred_path)
        target_data.to_netcdf(target_path)
        
        logger.info(f"Test successful! Data saved to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        logger.error("This might be due to authentication issues or network connectivity.")
        logger.error("Try running: gcloud auth application-default login")
        return False


def main():
    """Example usage with fixed paths and authentication."""
    
    # Configuration
    prediction_path = 'gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr'
    target_path = 'gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr'
    
    full_surface_var_list = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"] 
    full_atm_var_list = ["geopotential", "v_component_of_wind", "u_component_of_wind", 
                         "specific_humidity", "temperature"]
    variables = full_surface_var_list + full_atm_var_list
    
    # Fix: Use a directory path, not a file path
    dirs = setup_directories()
    output_path = os.path.join(dirs['raw'], 'pangu2018_raw_data')
    
    # First, run a simple test to check if everything works
    logger.info("Running authentication and data access test...")
    if download_single_chunk_test(prediction_path, target_path, variables, output_path):
        logger.info("Test successful! Proceeding with full pipeline...")
        
        # Time configuration for full run
        start_date = datetime.strptime("2018-01-01", '%Y-%m-%d')
        end_date = datetime.strptime("2018-01-02", '%Y-%m-%d')
        
        date_list = np.arange(
            np.datetime64(start_date), 
            np.datetime64(end_date), 
            dtype='datetime64[D]'
        ) + np.timedelta64(12, 'h')
        
        init_times = np.array(date_list, dtype='datetime64[ns]')
        lead_times = np.array([24, 48, 72, 96, 120, 144, 168], dtype='timedelta64[h]').astype('timedelta64[ns]')
        
        # Run pipeline
        logger.info("Starting full data download pipeline...")
        start_time = time.time()
        
        run_download_pipeline_with_auth(
            prediction_path=prediction_path,
            target_path=target_path,
            variables=variables,
            init_times=init_times,
            lead_times=lead_times,
            output_path=output_path,
            init_time_chunk_size=1,
            lead_time_chunk_size=7,
            runner='DirectRunner',
            beam_options={
                'direct_num_workers': 2,
            },
            use_anonymous_access=True
        )
        
        end_time = time.time()
        logger.info(f"Pipeline completed in {(end_time - start_time)/60:.2f} minutes")
    else:
        logger.error("Test failed. Please check authentication and network connectivity.")
        logger.info("\nTo set up authentication:")
        logger.info("1. Install gcloud CLI: https://cloud.google.com/sdk/docs/install")
        logger.info("2. Run: gcloud auth application-default login")
        logger.info("3. Or for anonymous access: pip install gcsfs")


if __name__ == "__main__":
    main()