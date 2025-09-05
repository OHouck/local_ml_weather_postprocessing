# aifs_cleaning.py
# Helper file used to clean aifs files when downloading from download_aifs.sh

import xarray as xr
import sys
import os
import zarr
import traceback
import logging
from datetime import datetime
import numpy as np

def setup_logging(input_file):
    """Set up detailed logging for debugging."""
    log_dir = os.path.join(os.path.dirname(os.path.dirname(input_file)), 'error_logs')
    os.makedirs(log_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    log_file = os.path.join(log_dir, f'{base_name}_processing.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def validate_input_data(ds, logger):
    """Validate input dataset structure and content."""
    logger.info("Validating input data...")
    
    required_vars = ['tp', '2t']
    missing_vars = [var for var in required_vars if var not in ds.data_vars]
    if missing_vars:
        raise ValueError(f"Missing required variables: {missing_vars}")
    
    if 'step' not in ds.dims:
        raise ValueError("Missing 'step' dimension in dataset")
    
    # Check step values
    steps = ds.step.values
    logger.info(f"Available steps: {steps[:10]}... (showing first 10)")
    logger.info(f"Step range: {steps.min()} to {steps.max()}")
    logger.info(f"Total steps: {len(steps)}")
    
    # Check for expected step intervals (should be in hours)
    step_diffs = np.diff(steps)
    if len(step_diffs) > 0:
        logger.info(f"Step intervals: min={step_diffs.min()}, max={step_diffs.max()}, median={np.median(step_diffs)}")
    
    # Check data shapes
    for var in required_vars:
        shape = ds[var].shape
        logger.info(f"Variable '{var}' shape: {shape}")
        
        # Check for NaN values
        if hasattr(ds[var], 'values'):
            nan_count = np.isnan(ds[var].values).sum()
            total_count = ds[var].size
            logger.info(f"Variable '{var}' NaN values: {nan_count}/{total_count} ({100*nan_count/total_count:.2f}%)")
    
    return True
def process_daily_forecasts(ds, lead_time_days, logger):
    """
    Transform forecast data to daily aggregates for specified lead times.
    Enhanced with better error handling and validation.
    """
    logger.info(f"Processing daily forecasts for lead days: {lead_time_days}")
    
    steps_available = ds.step.values
    logger.info(f"Available steps: {steps_available}")
    
    # Convert steps to hours if they're in different units
    if hasattr(ds.step, 'units'):
        logger.info(f"Step units: {ds.step.units}")
    
    # Lists to collect processed data
    processed_datasets = []
    
    for lead_day in lead_time_days:
        logger.info(f"Processing lead day {lead_day}...")
        
        # Calculate step indices for this lead day (assuming steps are in hours)
        start_step = lead_day * 24 
        
        # For cumulative precipitation for a day - need steps at 6, 12, 18, 24 hours into the day
        tp_steps = [start_step + 6, start_step + 12, start_step + 18, start_step + 24]
        missing_tp_steps = [step for step in tp_steps if step not in steps_available]
        
        if missing_tp_steps:
            logger.warning(f"Missing required precipitation steps for lead day {lead_day}: {missing_tp_steps}")
            continue  # Skip this lead day
            
        # Get precipitation data for this day and sum it
        tp_selected = ds["tp"].sel(step=tp_steps)
        # Sum across the 4 time steps to get daily total, but don't keep the step dimension
        daily_tp_value = tp_selected.sum(dim="step")  # This removes the step dimension
        
        # Get temperature data for midnight and noon
        temp_steps = [start_step, start_step + 12]  # midnight and noon
        missing_temp_steps = [step for step in temp_steps if step not in steps_available]
        
        if missing_temp_steps:
            logger.warning(f"Missing required temperature steps for lead day {lead_day}: {missing_temp_steps}")
            continue  # Skip this lead day
            
        temp_selected = ds["2t"].sel(step=temp_steps)
        
        # Create dataset for this lead day
        # Create new step coordinates for midnight and noon
        temp_times = [np.timedelta64(step_val, 'h') for step_val in temp_steps]
        
        # Broadcast the daily precipitation to both time points (midnight and noon)
        # We need to expand daily_tp_value along a new step dimension with 2 values
        daily_tp_broadcast = daily_tp_value.expand_dims(
            dim={'step': temp_steps}
        )
        
        # Create the dataset with aligned dimensions
        lead_day_ds = xr.Dataset({
            'total_precipitation': daily_tp_broadcast.assign_coords(step=temp_times),
            '2m_temperature': temp_selected.assign_coords(step=temp_times)
        })
        
        processed_datasets.append(lead_day_ds)
    
    if not processed_datasets:
        raise ValueError("No lead days could be processed due to missing data")
    
    # Concatenate all lead days along step dimension
    result_ds = xr.concat(processed_datasets, dim="step")
    
    # Rename step to prediction_timedelta
    result_ds = result_ds.rename({"step": "prediction_timedelta"})

    # rename lat and lon to match other datasets
    result_ds = result_ds.rename({"lat": "latitude", "lon": "longitude"})
    
    # Update coordinate attributes
    result_ds.prediction_timedelta.attrs["long_name"] = "forecast lead time"
    result_ds.prediction_timedelta.attrs["description"] = "time since forecast initialization"
    
    # Update variable attributes
    result_ds.total_precipitation.attrs.update({
        "long_name": "Daily total precipitation",
        "units": "m",
        "description": "24-hour accumulated precipitation"
    })
    
    result_ds['2m_temperature'].attrs.update({
        "long_name": "2m temperature", 
        "units": ds["2t"].attrs.get("units", "K"),
        "description": "2m temperature"
    })
    
    # Add processing metadata
    result_ds.attrs.update({
        "processing_timestamp": datetime.now().isoformat(),
        "original_file": os.path.basename(sys.argv[1]) if len(sys.argv) > 1 else "unknown",
        "lead_days_requested": lead_time_days,
        "lead_days_processed": len(processed_datasets)
    })
    
    return result_ds

def save_with_retry(processed_data, output_file, logger, max_retries=3):
    """Save data with retry logic and different methods."""
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Saving to {output_file} (attempt {attempt + 1}/{max_retries})...")
            
            # Remove existing output if it exists
            if os.path.exists(output_file):
                logger.info(f"Removing existing output: {output_file}")
                if os.path.isdir(output_file):
                    import shutil
                    shutil.rmtree(output_file)
                else:
                    os.remove(output_file)
            
            # Primary method: direct to_zarr
            processed_data.to_zarr(output_file, mode='w', consolidated=True)
            
            # Verify the save was successful
            try:
                test_ds = xr.open_zarr(output_file)
                logger.info(f"Verification successful. Saved dataset shape: {dict(test_ds.dims)}")
                test_ds.close()
                return True
            except Exception as verify_error:
                logger.error(f"Verification failed: {verify_error}")
                raise verify_error
                
        except Exception as e:
            logger.error(f"Save attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                logger.info("Retrying...")
                continue
            else:
                # Try alternative save method as last resort
                logger.info("Trying alternative save method (compute first)...")
                try:
                    computed_data = processed_data.compute()
                    computed_data.to_zarr(output_file, mode='w', consolidated=True)
                    logger.info("Alternative save method succeeded")
                    return True
                except Exception as alt_error:
                    logger.error(f"Alternative save method also failed: {alt_error}")
                    raise alt_error
    
    return False

def main():

    if len(sys.argv) != 4:
        print("Usage: aifs_cleaning.py <input_file> <output_file> <lead_days>")
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2]  # This will be a .zarr path
    lead_days = [int(x) for x in sys.argv[3].split(',')]

    ############################################################################
    # # For testing purposes only - comment out when running from bash script
    # input_file = "/Users/ohouck/Desktop/init_2021031500.nc"
    # output_file = "/Users/ohouck/test.zarr"
    # lead_days = [1, 5, 9]
    ############################################################################
    
    # Set up logging
    logger = setup_logging(input_file)
    
    try:
        logger.info(f"Starting processing of {input_file}")
        logger.info(f"Output file: {output_file}")
        logger.info(f"Lead days: {lead_days}")
        
        # Check input file exists and get size
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        file_size = os.path.getsize(input_file) / (1024**3)  # GB
        logger.info(f"Input file size: {file_size:.2f} GB")
        
        # Load only required variables with chunking for large files
        logger.info(f"Loading {input_file}...")
        if file_size > 1.0:  # Use chunking for files > 1GB
            logger.info("Large file detected, using chunking...")
            ds = xr.open_dataset(input_file, chunks={'step': 100})[["tp", "2t"]]
        else:
            ds = xr.open_dataset(input_file)[["tp", "2t"]]
        
        logger.info(f"Dataset loaded successfully. Dimensions: {dict(ds.dims)}")

        # Validate input data
        validate_input_data(ds, logger)
        
        # Process the data
        logger.info(f"Processing with lead days: {lead_days}...")
        processed = process_daily_forecasts(ds, lead_days, logger)

        # Save to Zarr format with retry logic
        if save_with_retry(processed, output_file, logger):
            logger.info(f"SUCCESS: {output_file}")
            print(f"SUCCESS: {output_file}")
        else:
            raise RuntimeError("Failed to save processed data after all retry attempts")
        
    except Exception as e:
        error_msg = f"ERROR processing {input_file}: {e}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        
        # Print to stderr for the bash script to capture
        print(error_msg, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        
        # Additional debugging info
        logger.error(f"Python version: {sys.version}")
        logger.error(f"Xarray version: {xr.__version__}")
        try:
            logger.error(f"Available memory: {os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024**3):.2f} GB")
        except:
            pass
        
        sys.exit(1)

if __name__ == "__main__":
    main()