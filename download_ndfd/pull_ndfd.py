import subprocess
import os
from datetime import datetime
import sys
from pathlib import Path
import tempfile
import xarray as xr
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from helper_funcs import setup_directories

def check_data_availability(element='maxt', start_year=2020, end_year=2025):
    """Check which months have data for each year"""
    print(f"\n=== Data Availability for {element} ===\n")
    
    for year in range(start_year, end_year + 1):
        path = f"s3://noaa-ndfd-pds/wmo/{element}/{year}/"
        cmd = ["aws", "s3", "ls", "--no-sign-request", path]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and result.stdout.strip():
            months = [line.split()[1].replace('/', '').replace('PRE', '').strip() 
                     for line in result.stdout.split('\n') if 'PRE' in line]
            months = [m for m in months if m.isdigit()]
            
            if months:
                print(f"{year}: Months {min(months)}-{max(months)} ({len(months)} months)")
            else:
                print(f"{year}: No data or irregular structure")
        else:
            print(f"{year}: No data")

def extract_texas_from_grib(grib_file, output_dir):
    """Extract Texas bounding box from CONUS GRIB2 file and save as NetCDF.

    Handles Lambert Conformal projection with 2D lat/lon arrays.
    Skips files that don't cover Texas (e.g., Puerto Rico, Alaska, Hawaii).
    """
    # Texas bounds
    lat_min, lat_max = 25.8, 36.5
    lon_min, lon_max = -106.6, -93.5

    try:
        # Open with xarray + cfgrib
        ds = xr.open_dataset(str(grib_file), engine='cfgrib')

        lon = ds.longitude.values
        lat = ds.latitude.values

        # Check if this is a 2D grid (CONUS Lambert Conformal) or 1D (other regions)
        if lat.ndim == 1:
            # 1D coordinates - likely a non-CONUS regional file
            # Convert longitude and check if any points are in Texas
            lon_converted = np.where(lon > 180, lon - 360, lon)
            if not (lat.min() <= lat_max and lat.max() >= lat_min and
                    lon_converted.min() <= lon_max and lon_converted.max() >= lon_min):
                print(f"  - Skipping {grib_file.name}: region doesn't overlap Texas")
                ds.close()
                return None
            # For 1D grids, we'd need different handling - skip for now
            print(f"  - Skipping {grib_file.name}: 1D grid format not supported")
            ds.close()
            return None

        # 2D grid (CONUS Lambert Conformal projection)
        # Convert longitude from 0-360 to -180-180 format
        lon_converted = np.where(lon > 180, lon - 360, lon)

        # Create mask for Texas region
        mask = (
            (lat >= lat_min) & (lat <= lat_max) &
            (lon_converted >= lon_min) & (lon_converted <= lon_max)
        )

        # Find the bounding box indices for the mask
        y_indices, x_indices = np.where(mask)

        if len(y_indices) == 0:
            print(f"  - Skipping {grib_file.name}: no data points in Texas region")
            ds.close()
            return None

        y_min, y_max = y_indices.min(), y_indices.max() + 1
        x_min, x_max = x_indices.min(), x_indices.max() + 1

        # Subset to bounding box using isel
        ds_texas = ds.isel(y=slice(y_min, y_max), x=slice(x_min, x_max))

        # Update longitude to standard format (-180 to 180)
        ds_texas = ds_texas.assign_coords(
            longitude=(('y', 'x'), lon_converted[y_min:y_max, x_min:x_max])
        )

        # Create output filename
        output_file = os.path.join(output_dir, grib_file.stem + '_texas.nc')

        # Save as compressed NetCDF
        encoding = {var: {'zlib': True, 'complevel': 5}
                   for var in ds_texas.data_vars}
        ds_texas.to_netcdf(output_file, encoding=encoding)

        ds.close()
        ds_texas.close()

        return output_file

    except Exception as e:
        print(f"  ✗ Error processing {grib_file.name}: {e}")
        return None

def download_and_extract_texas_month(element, year, month, base_dir):
    """Download month of NDFD data, extract Texas, and delete full CONUS files"""
    
    # Create output directory for Texas NetCDF files
    output_dir = os.path.join(base_dir, element, str(year), f"{month:02d}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Use temporary directory for downloaded CONUS GRIB files
    with tempfile.TemporaryDirectory() as temp_dir:
        s3_path = f"s3://noaa-ndfd-pds/wmo/{element}/{year}/{month:02d}/"
        
        cmd = [
            "aws", "s3", "sync",
            "--no-sign-request",
            s3_path,
            temp_dir
        ]
        
        print(f"Downloading {element} for {year}-{month:02d} (full CONUS to temp)...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"✗ Download Error: {result.stderr}")
            return False
        
        # Find all downloaded files
        grib_files = [f for f in Path(temp_dir).rglob("*") if f.is_file()]
        
        if not grib_files:
            print(f"✗ No files downloaded for {element} {year}-{month:02d}")
            return False
        
        print(f"  Downloaded {len(grib_files)} files. Extracting Texas region...")
        
        # Extract Texas from each file
        successful = 0
        failed = 0
        
        for grib_file in grib_files:
            result = extract_texas_from_grib(grib_file, output_dir)
            if result:
                successful += 1
            else:
                failed += 1
        
        print(f"✓ Extracted Texas data: {successful} files successful, {failed} failed")
        print(f"  Saved to: {output_dir}")
        
        # Calculate space saved
        total_size_mb = sum(f.stat().st_size for f in grib_files) / (1024 * 1024)
        print(f"  Original CONUS files: {total_size_mb:.1f} MB (deleted)")
        
        # Check size of extracted files
        nc_files = list(Path(output_dir).glob("*.nc"))
        if nc_files:
            texas_size_mb = sum(f.stat().st_size for f in nc_files) / (1024 * 1024)
            print(f"  Texas NetCDF files: {texas_size_mb:.1f} MB")
            print(f"  Space savings: {((total_size_mb - texas_size_mb) / total_size_mb * 100):.1f}%")
        
        return True
        
        # temp_dir automatically cleaned up here

def download_year_data(year, elements, base_dir, texas_only=True):
    """Download data for an entire year, optionally extracting only Texas"""
    
    for element in elements:
        print(f"\n=== Processing {element} for {year} ===")
        for month in range(1, 13):
            if texas_only:
                download_and_extract_texas_month(element, year, month, base_dir)
            else:
                download_ndfd_month(element, year, month, base_dir)

def download_ndfd_month(element, year, month, base_dir):
    """Download entire month of NDFD data (full CONUS, no extraction)"""
    output_dir = os.path.join(base_dir, element, str(year), f"{month:02d}")
    os.makedirs(output_dir, exist_ok=True)
    
    s3_path = f"s3://noaa-ndfd-pds/wmo/{element}/{year}/{month:02d}/"
    
    cmd = [
        "aws", "s3", "sync",
        "--no-sign-request",
        s3_path,
        output_dir
    ]
    
    print(f"Downloading {element} for {year}-{month:02d}...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        # Count files downloaded
        file_count = sum(len(files) for _, _, files in os.walk(output_dir))
        print(f"✓ Downloaded {element} {year}-{month:02d}: {file_count} files")
        return True
    else:
        print(f"✗ Error: {result.stderr}")
        return False

def plot_texas_tmax_forecast(nc_file, step_index=0, output_file=None, units='F'):
    """Create a map of max temperature forecasts for Texas.

    Args:
        nc_file: Path to the Texas NetCDF file
        step_index: Which forecast step to plot (0, 1, 2, etc.)
        output_file: Path to save the figure (if None, displays interactively)
        units: Temperature units - 'F' for Fahrenheit, 'C' for Celsius, 'K' for Kelvin

    Returns:
        Path to saved figure or None if displayed interactively
    """
    import matplotlib.pyplot as plt

    ds = xr.open_dataset(nc_file)

    # Get the data for the specified step
    tmax = ds.tmax.isel(step=step_index).values
    lat = ds.latitude.values
    lon = ds.longitude.values
    valid_time = ds.valid_time.isel(step=step_index).values
    forecast_time = ds.time.values

    # Convert temperature units
    if units == 'F':
        tmax = (tmax - 273.15) * 9/5 + 32
        unit_label = '°F'
        cmap = 'RdYlBu_r'
        vmin, vmax = 20, 90
    elif units == 'C':
        tmax = tmax - 273.15
        unit_label = '°C'
        cmap = 'RdYlBu_r'
        vmin, vmax = -10, 35
    else:  # Kelvin
        unit_label = 'K'
        cmap = 'RdYlBu_r'
        vmin, vmax = 260, 310

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot the temperature data using pcolormesh for 2D coordinates
    mesh = ax.pcolormesh(lon, lat, tmax, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')

    # Add colorbar
    cbar = plt.colorbar(mesh, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(f'Max Temperature ({unit_label})', fontsize=12)

    # Format times for title
    forecast_dt = np.datetime64(forecast_time, 'h')
    valid_dt = np.datetime64(valid_time, 'h')
    forecast_str = str(forecast_dt)[:13].replace('T', ' ')
    valid_str = str(valid_dt)[:13].replace('T', ' ')

    ax.set_title(f'NDFD Max Temperature Forecast for Texas\n'
                 f'Forecast issued: {forecast_str} UTC | Valid: {valid_str} UTC',
                 fontsize=14)

    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)

    # Set axis limits to Texas bounds
    ax.set_xlim(-107, -93)
    ax.set_ylim(25.5, 37)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    ds.close()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved figure to: {output_file}")
        return output_file
    else:
        plt.show()
        return None


def main():
    dirs = setup_directories()
    
    # Configuration
    year = 2025
    texas_only = True  # Set to False if you want full CONUS files instead
    
    # Elements to download
    elements = [
        'maxt',      # Maximum temperature
        # Uncomment others when ready:
        # 'mint',      # Minimum temperature  
        # 'temp',      # Temperature
        # 'wspd',      # Wind speed
        # 'wdir',      # Wind direction
        # 'pop12',     # 12-hour probability of precipitation
        # 'qpf',       # Quantitative precipitation forecast
        # 'rhm',       # Relative humidity
    ]
    
    output_dir = os.path.join(dirs['raw'], 'ndfd_data')
    download = False

    if download:
        print(f"Starting download for {year}")
        print(f"Output directory: {output_dir}")
        print(f"Elements: {', '.join(elements)}")
        print(f"Texas extraction: {'ENABLED (saves ~70-80% space)' if texas_only else 'DISABLED (full CONUS)'}")
        print(f"\nExpected data: Jan-{datetime.now().strftime('%b')} {year}")
        
        # Confirm before starting
        response = input("\nProceed with download? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Download cancelled.")
            return
        
        download_year_data(year, elements, output_dir, texas_only=texas_only)
        
        print("\n=== Download Complete ===")
        print(f"Data saved to: {output_dir}")
        
        # Show summary
        print("\nSummary:")
        for element in elements:
            element_dir = os.path.join(output_dir, element, str(year))
            if os.path.exists(element_dir):
                total_files = sum(len(files) for _, _, files in os.walk(element_dir))
                total_size_mb = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, _, filenames in os.walk(element_dir)
                    for filename in filenames
                ) / (1024 * 1024)
                print(f"  {element}: {total_files} files, {total_size_mb:.1f} MB")

    # plot example forecast
    example_element = 'maxt'
    example_month = 1
    example_nc_dir = os.path.join(output_dir, example_element, str(year), f"{example_month:02d}")
    print(example_nc_dir)
    example_nc_files = list(Path(example_nc_dir).glob("*_texas.nc"))
    num_files = len(example_nc_files)
    print(f"\nFound {num_files} example Texas NetCDF files for {example_element}")
    if example_nc_files:
        example_nc_file = example_nc_files[0]
        print(f"\nPlotting example forecast from: {example_nc_file}")
        plot_texas_tmax_forecast(example_nc_file, step_index=0, output_file=None, units='F')

    

if __name__ == "__main__":
    main()