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
    
    processed_data = []
    steps_available = ds.step.values
    
    # Convert steps to hours if they're in different units
    if hasattr(ds.step, 'units'):
        logger.info(f"Step units: {ds.step.units}")
    
    for lead_day in lead_time_days:
        logger.info(f"Processing lead day {lead_day}...")
        
        # Calculate step indices for this lead day (assuming steps are in hours)
        start_step = lead_day * 24 
        end_step = start_step + 18 
        
        logger.info(f"Looking for steps between {start_step} and {end_step} hours")
        
        # Check if required steps are available
        required_steps = [start_step, start_step + 12]  # midnight and noon
        missing_steps = [step for step in required_steps if step not in steps_available]
        if missing_steps:
            logger.warning(f"Missing required steps for lead day {lead_day}: {missing_steps}")
            continue
            
        try:
            # Extract the relevant steps for this day
            day_data = ds.sel(step=slice(start_step, end_step))
            
            if len(day_data.step) == 0:
                logger.warning(f"No data found for lead day {lead_day} (steps {start_step}-{end_step})")
                continue
            
            logger.info(f"Found {len(day_data.step)} time steps for lead day {lead_day}")
            
            # For precipitation: sum over the day
            daily_tp = day_data["tp"].sum(dim="step", keepdims=True)
            daily_tp = daily_tp.assign_coords(step=[lead_day])
            
            # For temperature: extract specific times
            try:
                midnight_2t = day_data["2t"].sel(step=start_step)  # midnight
                noon_2t = day_data["2t"].sel(step=start_step + 12) # noon
            except KeyError as e:
                logger.error(f"Could not extract temperature data for lead day {lead_day}: {e}")
                continue
            
            # Expand dims and assign correct step coordinate
            midnight_2t = midnight_2t.expand_dims("step", axis=1).assign_coords(step=[lead_day])
            noon_2t = noon_2t.expand_dims("step", axis=1).assign_coords(step=[lead_day])
            
            # Create dataset for this lead day
            day_ds = xr.Dataset({
                "daily_tp": daily_tp,
                "midnight_2t": midnight_2t, 
                "noon_2t": noon_2t
            })
            
            # Validate the processed data
            for var_name, var_data in day_ds.data_vars.items():
                if np.isnan(var_data.values).all():
                    logger.warning(f"All NaN values in {var_name} for lead day {lead_day}")
                elif np.isnan(var_data.values).any():
                    nan_fraction = np.isnan(var_data.values).sum() / var_data.size
                    logger.info(f"{var_name} for lead day {lead_day}: {nan_fraction:.2%} NaN values")
            
            processed_data.append(day_ds)
            logger.info(f"Successfully processed lead day {lead_day}")
            
        except Exception as e:
            logger.error(f"Error processing lead day {lead_day}: {e}")
            logger.error(traceback.format_exc())
            continue
    
    if not processed_data:
        raise ValueError("No lead days were successfully processed")
    
    logger.info(f"Successfully processed {len(processed_data)} out of {len(lead_time_days)} lead days")
    
    # Concatenate all lead days
    result = xr.concat(processed_data, dim="step")
    
    # Update step coordinate attributes
    result.step.attrs["units"] = "days"
    result.step.attrs["long_name"] = "lead time in days"
    
    # Update variable attributes
    result.daily_tp.attrs.update({
        "long_name": "Daily total precipitation",
        "units": "m",
        "description": "24-hour accumulated precipitation"
    })
    
    result.midnight_2t.attrs.update({
        "long_name": "2m temperature at midnight", 
        "units": ds["2t"].attrs.get("units", "K"),
        "description": "2m temperature at 00:00 UTC"
    })
    
    result.noon_2t.attrs.update({
        "long_name": "2m temperature at noon",
        "units": ds["2t"].attrs.get("units", "K"), 
        "description": "2m temperature at 12:00 UTC"
    })
    
    # Add processing metadata
    result.attrs.update({
        "processing_timestamp": datetime.now().isoformat(),
        "original_file": os.path.basename(sys.argv[1]) if len(sys.argv) > 1 else "unknown",
        "lead_days_requested": lead_time_days,
        "lead_days_processed": [int(x) for x in result.step.values]
    })
    
    return result

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