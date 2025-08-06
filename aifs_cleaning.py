import xarray as xr
import os

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
    
    # Assuming step is in hours (6-hour increments)
    # Convert step from hours to timestep index for days
    hours_per_day = 24
    steps_per_day = hours_per_day // 6  # 4 steps per day (6-hour increments)
    
    processed_data = []
    
    for lead_day in lead_time_days:
        # Calculate step indices for this lead day
        # Day 1 means steps 4-7 (hours 24-42), Day 5 means steps 20-23 (hours 120-138), etc.
        start_step = lead_day * 24 
        end_step = start_step + 18 
        
        # Extract the relevant steps for this day
        day_data = ds.sel(step=slice(start_step, end_step))
        
        # For precipitation: sum over the day (steps within the day)
        daily_tp = day_data["tp"].sum(dim="step", keepdims=True)
        # assign step coordinate as the lead day
        daily_tp = daily_tp.assign_coords(step=[lead_day])

        # For temperature: extract specific times
        midnight_2t = day_data["2t"].sel(step=start_step)  # midnight
        noon_2t = day_data["2t"].sel(step=start_step + 12) # noon (12:00)
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
    # aifs1 dataset 
    local_test_path = "/Users/ohouck/Documents/init_2024071100.nc"

    # Define the folder path
    directory_path = "/net/monsoon/marchakitus/model_data/AIFS/output_daily_march_15_october"

    # Get a list of all file paths in the folder
    file_paths = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, file))]

    for file_path in file_paths:

        # only save total precipitation and surface temp
        aifs = xr.open_dataset(file_path)[["tp", "2t"]]

        lead_time_days = [1, 5, 10]

        # Process the data
        processed_aifs = process_daily_forecasts(aifs, lead_time_days)


if __name__ == "__main__":
    main()
