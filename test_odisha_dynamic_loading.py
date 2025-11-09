#!/usr/bin/env python3
"""
Test script for dynamic data loading with the new Odisha region
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "finetuning"))

from finetuning.prepare_forecasts_and_targets import prepare_data_for_finetuning
import numpy as np

def test_odisha_dynamic_loading():
    """Test dynamic data loading for Odisha region"""

    print("="*70)
    print("TESTING DYNAMIC DATA LOADING FOR ODISHA REGION")
    print("="*70)

    # Define test parameters
    data_dir = os.path.expanduser("~/ai_weather_ag/data/raw")
    model_name = "pangu"
    ground_truth_source = ""  # Will use default (era5 for pangu)
    training_vars = ["2m_temperature"]
    output_vars = ["2m_temperature"]

    # Use a very short time period for testing (just 1 week in 2020)
    train_start = "2020-01-01"
    train_end = "2020-01-07"
    test_start = "2020-01-08"
    test_end = "2020-01-10"

    # Test with one lead time
    lead_time_hours = [24]

    # Define Odisha region (centered at 20, 84)
    region_lat = np.arange(18, 22, 0.25)
    region_lon = np.arange(82, 86, 0.25)

    print(f"\nTest Configuration:")
    print(f"  Region: Odisha (lat: {region_lat.min()}-{region_lat.max()}, "
          f"lon: {region_lon.min()}-{region_lon.max()})")
    print(f"  Center: ~20°N, 84°E")
    print(f"  Model: {model_name}")
    print(f"  Training period: {train_start} to {train_end}")
    print(f"  Test period: {test_start} to {test_end}")
    print(f"  Variables: {training_vars}")
    print(f"  Lead times: {lead_time_hours} hours")

    # Run the data preparation
    try:
        result = prepare_data_for_finetuning(
            data_dir=data_dir,
            model_name=model_name,
            ground_truth_source=ground_truth_source,
            training_vars=training_vars,
            output_vars=output_vars,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            lead_time_hours=lead_time_hours,
            region_lat=region_lat,
            region_lon=region_lon
        )

        print("\n" + "="*70)
        print("TEST RESULT: SUCCESS")
        print("="*70)
        print(f"Status: {result['status']}")
        print(f"Data directory: {result['data_dir']}")
        print(f"Forecast source: {result['forecast_source']}")
        print(f"Target source: {result['target_source']}")
        print(f"Years processed: {result['years']}")

        return True

    except Exception as e:
        print("\n" + "="*70)
        print("TEST RESULT: FAILED")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_odisha_dynamic_loading()
    sys.exit(0 if success else 1)
