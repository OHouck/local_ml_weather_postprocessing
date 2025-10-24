"""
Name: clean_forecasts.py
Author: Ozma Houck
Date created: 10/21/2025

Purpose: convert raw aurora weather forecast outputs from pytorch to xarray in a form
that can be used in finetuning
"""

import torch 
import xarray as xr
import numpy as np
import pandas as pd
from datetime import timedelta
from aurora import Batch
from typing import List

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from helper_funcs import setup_directories

def aurora_batch_to_xarray(preds: List[Batch]) -> xr.Dataset:
    """
    Convert Aurora predictions to a single xarray Dataset.
    
    Parameters
    ----------
    preds : List[Batch]
        List of Aurora Batch predictions from rollout. Each element represents
        one forecast lead time (6-hour increments starting at 12 hours).
        
    Returns
    -------
    xr.Dataset
        Combined dataset with both surface and atmospheric variables.
        
    Coordinates
    -----------
    init_time : datetime64
        Initialization time of the forecast
    lead_time : timedelta64
        Forecast lead time (12h, 18h, 24h, ...)
    latitude : float
        Latitude coordinates (degrees North)
    longitude : float
        Longitude coordinates (degrees East)
    level : int
        Atmospheric pressure levels (hPa)
        
    Variables
    ---------
    Surface variables (dimensions: init_time, lead_time, latitude, longitude):
        - 2t: 2-meter temperature (K)
        - 10u: 10-meter U wind component (m/s)
        - 10v: 10-meter V wind component (m/s)
        - msl: Mean sea level pressure (Pa)
        
    Atmospheric variables (dimensions: init_time, lead_time, level, latitude, longitude):
        - t: Temperature (K)
        - u: U wind component (m/s)
        - v: V wind component (m/s)
        - q: Specific humidity (kg/kg)
        - z: Geopotential (m²/s²)
    """
    
    if len(preds) == 0:
        raise ValueError("preds list is empty")
    
    # Extract basic information from first prediction
    first_pred = preds[0]
    n_steps = len(preds)
    
    # Get spatial coordinates
    lat = first_pred.metadata.lat.numpy()
    lon = first_pred.metadata.lon.numpy()
    
    # Get atmospheric levels
    levels = np.array(list(first_pred.metadata.atmos_levels))
    
    # Determine initialization time (6 hours before first prediction)
    # Aurora uses 2 time steps as input, the last one is at 06:00
    # The first prediction is 6 hours later (12:00)
    first_valid_time = first_pred.metadata.time[0]
    init_time = first_valid_time - timedelta(hours=6)
    
    # Create lead times array (12h, 18h, 24h, ...)
    lead_times = np.array([timedelta(hours=12 + 6*i) for i in range(n_steps)])
    
    # Convert to timedelta64 for xarray
    lead_times_td64 = pd.to_timedelta(lead_times).to_numpy()
    
    # Initialize data dictionaries
    surf_data = {}
    atmos_data = {}
    
    # Get variable names from first prediction
    surf_vars = list(first_pred.surf_vars.keys())
    atmos_vars = list(first_pred.atmos_vars.keys())
    
    # Extract surface variables
    # Original shape: [batch, time, lat, lon] -> we want [lead_time, lat, lon]
    for var_name in surf_vars:
        # Stack all lead times
        var_data = np.stack([
            pred.surf_vars[var_name][0, 0].numpy()  # Remove batch and time dims
            for pred in preds
        ], axis=0)  # Shape: [lead_time, lat, lon]
        
        # Add init_time dimension
        surf_data[var_name] = (
            ["init_time", "lead_time", "latitude", "longitude"],
            var_data[np.newaxis, ...]  # Add init_time dim: [1, lead_time, lat, lon]
        )
    
    # Extract atmospheric variables
    # Original shape: [batch, time, level, lat, lon] -> we want [lead_time, level, lat, lon]
    for var_name in atmos_vars:
        # Stack all lead times
        var_data = np.stack([
            pred.atmos_vars[var_name][0, 0].numpy()  # Remove batch and time dims
            for pred in preds
        ], axis=0)  # Shape: [lead_time, level, lat, lon]
        
        # Add init_time dimension
        atmos_data[var_name] = (
            ["init_time", "lead_time", "level", "latitude", "longitude"],
            var_data[np.newaxis, ...]  # Add init_time dim: [1, lead_time, level, lat, lon]
        )
    
    # Combine all variables
    all_data = {**surf_data, **atmos_data}
    
    # Create the dataset
    ds = xr.Dataset(
        all_data,
        coords={
            "init_time": [np.datetime64(init_time)],
            "lead_time": lead_times_td64,
            "latitude": lat,
            "longitude": lon,
            "level": levels,
        }
    )
    
    # Add metadata attributes
    ds.attrs.update({
        "title": "Aurora Weather Forecast",
        "source": "Aurora foundation model",
        "init_time": str(init_time),
        "n_lead_times": n_steps,
        "lead_time_increment": "6 hours",
        "first_lead_time": "12 hours",
        "spatial_resolution": "0.25 degrees",
    })
    
    # Add variable attributes
    var_attrs = {
        # Surface variables
        "2t": {
            "long_name": "2 metre temperature",
            "units": "K",
            "standard_name": "air_temperature",
        },
        "10u": {
            "long_name": "10 metre U wind component",
            "units": "m s-1",
            "standard_name": "eastward_wind",
        },
        "10v": {
            "long_name": "10 metre V wind component", 
            "units": "m s-1",
            "standard_name": "northward_wind",
        },
        "msl": {
            "long_name": "Mean sea level pressure",
            "units": "Pa",
            "standard_name": "air_pressure_at_mean_sea_level",
        },
        # Atmospheric variables
        "t": {
            "long_name": "Temperature",
            "units": "K",
            "standard_name": "air_temperature",
        },
        "u": {
            "long_name": "U component of wind",
            "units": "m s-1",
            "standard_name": "eastward_wind",
        },
        "v": {
            "long_name": "V component of wind",
            "units": "m s-1", 
            "standard_name": "northward_wind",
        },
        "q": {
            "long_name": "Specific humidity",
            "units": "kg kg-1",
            "standard_name": "specific_humidity",
        },
        "z": {
            "long_name": "Geopotential",
            "units": "m2 s-2",
            "standard_name": "geopotential",
        },
    }
    
    for var_name, attrs in var_attrs.items():
        if var_name in ds:
            ds[var_name].attrs.update(attrs)
    
    # Add coordinate attributes
    ds["init_time"].attrs.update({
        "long_name": "Forecast initialization time",
        "standard_name": "forecast_reference_time",
    })
    
    ds["lead_time"].attrs.update({
        "long_name": "Forecast lead time",
        "units": "hours",
    })
    
    ds["latitude"].attrs.update({
        "long_name": "Latitude",
        "units": "degrees_north",
        "standard_name": "latitude",
    })
    
    ds["longitude"].attrs.update({
        "long_name": "Longitude", 
        "units": "degrees_east",
        "standard_name": "longitude",
    })
    
    ds["level"].attrs.update({
        "long_name": "Pressure level",
        "units": "hPa",
        "standard_name": "air_pressure",
        "positive": "down",
    })
    
    return ds



def main():

    dirs = setup_directories()

    aurora_forecast_path = os.path.join(dirs['root'], "aurora_predictions.pt") # should eventually be in processed folder

    # Add Batch to safe globals
    torch.serialization.add_safe_globals([Batch])
    preds = torch.load(aurora_forecast_path, weights_only=False)

    ds = aurora_batch_to_xarray(preds)

if __name__ == '__main__':
    main()