#!/usr/bin/env python
"""
Combined AIFS processing script for cluster execution.
Cleans individual files and combines them into yearly files with valid_time dimension.
now done on dsi cluster

"""

import sys
import os
import glob
import argparse
import xarray as xr
import pandas as pd
import numpy as np
import logging
from datetime import datetime


def setup_logging(log_name="aifs_processing"):
    """Set up simple logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(log_name)


def process_single_file(input_file, output_dir, lead_days):
    """
    Process a single AIFS file: clean and save to zarr format.
    
    Parameters:
    -----------
    input_file : str
        Path to input .nc file
    output_dir : str
        Directory to save processed file
    lead_days : list
        List of lead days to process
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    logger = logging.getLogger()
    
    try:
        # Extract filename for output
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(output_dir, f"processed_{base_name}.zarr")
        
        # Skip if already processed
        if os.path.exists(output_file):
            logger.info(f"Skipping {base_name}: already processed")
            return True
        
        logger.info(f"Loading {input_file}...")
        
        # Load only required variables
        ds = xr.open_dataset(input_file)[["tp", "2t"]]
        
        # Process daily forecasts
        processed = process_daily_forecasts(ds, lead_days)
        
        # Convert time to init_time for later combination
        if 'time' in processed.dims:
            processed = processed.rename({'time': 'init_time'})
        
        # Save to zarr
        logger.info(f"Saving to {output_file}...")
        processed.to_zarr(output_file, mode='w', consolidated=True)
        
        # Close dataset
        ds.close()
        
        logger.info(f"Successfully processed {base_name}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {input_file}: {e}")
        return False


def process_daily_forecasts(ds, lead_time_days):
    """
    Transform forecast data to daily aggregates for specified lead times.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Input dataset with 'tp' and '2t' variables
    lead_time_days : list
        List of lead days (integers)
    
    Returns:
    --------
    xarray.Dataset
        Processed dataset with daily aggregates
    """
    logger = logging.getLogger()
    steps_available = ds.step.values
    processed_datasets = []
    
    for lead_day in lead_time_days:
        # Calculate required steps
        start_step = lead_day * 24
        
        # Precipitation steps (6-hour accumulations)
        tp_steps = [start_step + 6, start_step + 12, start_step + 18, start_step + 24]
        
        # Temperature steps (midnight and noon)
        temp_steps = [start_step, start_step + 12]
        
        # Check if all required steps are available
        if not all(step in steps_available for step in tp_steps + temp_steps):
            logger.warning(f"Skipping lead day {lead_day}: missing required steps")
            continue
        
        # Sum precipitation for daily total
        tp_daily = ds["tp"].sel(step=tp_steps).sum(dim="step")
        
        # Get temperature at midnight and noon
        temp_selected = ds["2t"].sel(step=temp_steps)
        
        # Create timedelta coordinates
        temp_times = [np.timedelta64(step, 'h') for step in temp_steps]
        
        # Expand precipitation to match temperature times
        tp_broadcast = tp_daily.expand_dims(dim={'step': temp_steps})
        
        # Create dataset for this lead day
        lead_day_ds = xr.Dataset({
            'total_precipitation': tp_broadcast.assign_coords(step=temp_times),
            '2m_temperature': temp_selected.assign_coords(step=temp_times)
        })
        
        processed_datasets.append(lead_day_ds)
    
    if not processed_datasets:
        raise ValueError("No lead days could be processed")
    
    # Concatenate and rename dimensions
    result = xr.concat(processed_datasets, dim="step")
    result = result.rename({
        "step": "prediction_timedelta",
        "lat": "latitude",
        "lon": "longitude"
    })
    
    # Add attributes
    result.total_precipitation.attrs = {
        "long_name": "Daily total precipitation",
        "units": "m",
        "description": "24-hour accumulated precipitation"
    }
    
    result['2m_temperature'].attrs = {
        "long_name": "2m temperature",
        "units": ds["2t"].attrs.get("units", "K"),
        "description": "2m temperature at surface"
    }
    
    return result


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Process AIFS files')
    parser.add_argument('input', help='Input file or directory')
    parser.add_argument('output', help='Output directory')
    parser.add_argument('lead_days', nargs='?', default='1,5,9',
                       help='Lead days to process (comma-separated)')
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    lead_days = [int(x) for x in args.lead_days.split(',')]
    success = process_single_file(args.input, args.output, lead_days)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
