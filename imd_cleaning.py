import xarray as xr
import numpy as np
import pandas as pd
import glob

def count_days_without_rainfall(ds, threshold=0.1):
    """
    Count the number of days in a row without rainfall exceeding a threshold.
    For each spell, keep track of the number of consecutive days without rainfall, 
    as well as the date when the spell started
    Outputs:
    - a pandas dataset with columns for spell start date, 
      number of consecutive days without rainfall
    """
    # Create boolean mask for dry days (rainfall <= threshold)
    dry_days = ds.rainfall <= threshold
    
    # Convert to numpy array for easier processing
    dry_array = dry_days.values
    time_coords = ds.time.values

    spell_data = []
    
    # Process each grid point
    for lat_idx in range(dry_array.shape[1]):
        for lon_idx in range(dry_array.shape[2]):
            # Extract time series for this grid point
            # time series of true/false values indicating dry days
            point_dry = dry_array[:, lat_idx, lon_idx]

            # Skip if all NaN
            if np.all(np.isnan(point_dry)):
                continue
            
            # Find dry spells
            spell_length = 0
            spell_start = None
            
            for i, is_dry in enumerate(point_dry):
                if np.isnan(is_dry):
                    # Handle NaN values - end current spell if active
                    if spell_length > 0:
                        spell_data.append({
                            'spell_start_date': spell_start,
                            'consecutive_dry_days': spell_length,
                            'latitude': ds.latitude.values[lat_idx],
                            'longitude': ds.longitude.values[lon_idx]
                        })
                        spell_length = 0
                        spell_start = None
                elif is_dry:
                    # Dry day
                    if spell_length == 0:
                        # Start new spell
                        spell_start = time_coords[i]
                        spell_length = 1
                    else:
                        # Continue existing spell
                        spell_length += 1
                else:
                    # Wet day - end current spell if active
                    if spell_length > 0:
                        spell_data.append({
                            'spell_start_date': spell_start,
                            'consecutive_dry_days': spell_length,
                            'latitude': ds.latitude.values[lat_idx],
                            'longitude': ds.longitude.values[lon_idx]
                        })
                        spell_length = 0
                        spell_start = None
            
            # Handle spell that extends to end of time series
            if spell_length > 0:
                spell_data.append({
                    'spell_start_date': spell_start,
                    'consecutive_dry_days': spell_length,
                    'latitude': ds.latitude.values[lat_idx],
                    'longitude': ds.longitude.values[lon_idx]
                })
    
    # Convert to pandas DataFrame
    if spell_data:
        df = pd.DataFrame(spell_data)
        # Convert datetime if needed
        if 'spell_start_date' in df.columns:
            df['spell_start_date'] = pd.to_datetime(df['spell_start_date'])
    else:
        # Return empty DataFrame with correct columns
        df = pd.DataFrame(columns=['spell_start_date', 'consecutive_dry_days', 'latitude', 'longitude'])
    
    return df

def main():
    imd_path = "/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/IMD/IMD_0p25deg"
    # start with subset of years to test with
    year_list = np.arange(2022, 2022 + 1)
    imd_patterns = [f"{imd_path}/data_{year}*.nc" for year in year_list]
    files = []
    for pattern in imd_patterns:
        files.extend(glob.glob(pattern))
    # merge in all files together
    ds = xr.open_mfdataset(files, combine='by_coords', parallel=True)
    # rename dimensions to be lowercase rainfall is in mm
    ds = ds.rename({'LATITUDE': 'latitude', 'LONGITUDE': 'longitude', 'TIME': 'time', 'RAINFALL': 'rainfall'})

    ds = ds.sel(time = "2022-07-11")
    lat0, lat1 = 17, 27
    lon0, lon1 = 72, 82
    ds = ds.sel(latitude=slice(lat0, lat1), longitude=slice(lon0, lon1))
    mean_rainfall = ds['rainfall'].mean(dim=['latitude', 'longitude']).values
    print(mean_rainfall)



    exit()
    
    # Count dry spells
    dry_spells = count_days_without_rainfall(ds, threshold=0.1)
    print(f"\nFound {len(dry_spells)} dry spells")
    print(f"Dry spell statistics:")
    print(dry_spells['consecutive_dry_days'].describe())

if __name__ == "__main__":
    main()