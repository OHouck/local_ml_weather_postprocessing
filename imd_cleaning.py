import xarray as xr
import os
import numpy as np
import pandas as pd
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from helper_funcs import setup_directories

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
    dry_days = ds.total_precipitation <= threshold
    
    # Convert to numpy array for easier processing
    dry_array = dry_days.values
    time_coords = ds.time.values

    spell_data = []
    
    na_count = 0
    # Process each grid point
    for lat_idx in range(dry_array.shape[1]):
        for lon_idx in range(dry_array.shape[2]):
            # Extract time series for this grid point
            # time series of true/false values indicating dry days
            point_dry = dry_array[:, lat_idx, lon_idx]

            # Skip if all NaN
            if np.all(np.isnan(point_dry)):
                na_count += 1
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

def standardize_imd_dataset(ds):
    """
    Standardize IMD dataset variable/coordinate names to a consistent format.
    Handles both old format (LATITUDE, LONGITUDE, TIME, RAINFALL) and
    new 2024 format (lat, lon, time, rf).
    """
    rename_dict = {}

    # Handle coordinate names
    if 'LATITUDE' in ds.coords:
        rename_dict['LATITUDE'] = 'latitude'
    elif 'lat' in ds.coords:
        rename_dict['lat'] = 'latitude'

    if 'LONGITUDE' in ds.coords:
        rename_dict['LONGITUDE'] = 'longitude'
    elif 'lon' in ds.coords:
        rename_dict['lon'] = 'longitude'

    if 'TIME' in ds.coords:
        rename_dict['TIME'] = 'time'
    # 'time' is already correct, no rename needed

    # Handle data variable names
    if 'RAINFALL' in ds.data_vars:
        rename_dict['RAINFALL'] = 'total_precipitation'
    elif 'rf' in ds.data_vars:
        rename_dict['rf'] = 'total_precipitation'

    if rename_dict:
        ds = ds.rename(rename_dict)

    return ds

def process_imd_data(dirs):
    """
    Process IMD data files: standardize variable names, 
    combine yearly files into a single dataset, and save.

    Currently just processes years 2022-2024 for testing.

    Args:
        dirs (dict): Dictionary of directory paths.
    """
    imd_path = os.path.join(dirs['raw'], "IMD_0p25deg")
    # start with subset of years to test with XX
    year_list = np.arange(2022, 2024 + 1)
    imd_patterns = [f"{imd_path}/data_{year}*.nc" for year in year_list]
    files = []
    for pattern in imd_patterns:
        files.extend(glob.glob(pattern))
    # merge in all files together, applying standardization to each file before combining
    if not files:
        raise FileNotFoundError(f"No IMD files found matching patterns: {imd_patterns}")
    datasets = [standardize_imd_dataset(xr.open_dataset(f)) for f in sorted(files)]
    ds = xr.concat(datasets, dim='time')

    # save combined ds
    output_path = os.path.join(dirs["processed"], "IMD")
    os.makedirs(output_path, exist_ok=True)
    file_name = f"imd_0p25deg_{year_list[0]}-{year_list[-1]}.nc"
    ds.to_netcdf(os.path.join(output_path, file_name))

def main():

    dirs = setup_directories()

    # process_imd_data(dirs) # Uncomment to process and save combined IMD data

    output_path = os.path.join(dirs["processed"], "IMD")
    file_name = "imd_0p25deg_2022-2024.nc" # change name if processing different years
    # read in combined ds for analysis
    ds = xr.open_dataset(os.path.join(output_path, file_name), chunks=None)

    # Subset to region of interest for dry spell analysis
    lat0, lat1 = 17, 27
    lon0, lon1 = 72, 82
    ds_region = ds.sel(latitude=slice(lat0, lat1), longitude=slice(lon0, lon1))

    # Example: check mean rainfall for a single day
    ds_single_day = ds_region.sel(time="2022-07-11")
    mean_rainfall = ds_single_day['total_precipitation'].mean(dim=['latitude', 'longitude']).values
    print(f"Mean rainfall on 2022-07-11: {mean_rainfall}")

    # Count dry spells using the full time series (not just one day)
    dry_spells = count_days_without_rainfall(ds_region, threshold=0.1)
    print(dry_spells.head())
    print(f"\nFound {len(dry_spells)} dry spells")
    print(f"Dry spell statistics:")
    print(dry_spells['consecutive_dry_days'].describe())

if __name__ == "__main__":
    main()