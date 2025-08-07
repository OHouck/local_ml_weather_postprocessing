import xarray as xr
import sys
import os
import zarr

def process_daily_forecasts(ds, lead_time_days):
    """
    Transform forecast data to daily aggregates for specified lead times.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Input dataset with 'tp' (total precipitation) and '2t' (2m temperature)
        Step dimension should be in 6-hour increments
    lead_time_days : list
        List of lead time days to extract
    
    Returns:
    --------
    xarray.Dataset
        Processed dataset with daily aggregates
    """
    processed_data = []
    
    for lead_day in lead_time_days:
        # Calculate step indices for this lead day
        start_step = lead_day * 24 
        end_step = start_step + 18 
        
        # Extract the relevant steps for this day
        day_data = ds.sel(step=slice(start_step, end_step))
        
        # For precipitation: sum over the day
        daily_tp = day_data["tp"].sum(dim="step", keepdims=True)
        daily_tp = daily_tp.assign_coords(step=[lead_day])
        
        # For temperature: extract specific times
        midnight_2t = day_data["2t"].sel(step=start_step)  # midnight
        noon_2t = day_data["2t"].sel(step=start_step + 12) # noon
        
        # Expand dims and assign correct step coordinate
        midnight_2t = midnight_2t.expand_dims("step", axis=1).assign_coords(step=[lead_day])
        noon_2t = noon_2t.expand_dims("step", axis=1).assign_coords(step=[lead_day])
        
        # Create dataset for this lead day
        day_ds = xr.Dataset({
            "daily_tp": daily_tp,
            "midnight_2t": midnight_2t, 
            "noon_2t": noon_2t
        })
        processed_data.append(day_ds)
    
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
    
    return result

def main():
    if len(sys.argv) != 4:
        print("Usage: aifs_cleaning.py <input_file> <output_file> <lead_days>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]  # This will be a .zarr path
    lead_days = [int(x) for x in sys.argv[3].split(',')]
    
    try:
        # Load only required variables
        print(f"Loading {input_file}...")
        ds = xr.open_dataset(input_file)[["tp", "2t"]]
        
        # Process the data
        print(f"Processing with lead days: {lead_days}...")
        processed = process_daily_forecasts(ds, lead_days)
        
        # Save to Zarr format
        print(f"Saving to {output_file}...")
        processed.to_zarr(output_file, mode='w')
        print(f"SUCCESS: {output_file}")
        
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()