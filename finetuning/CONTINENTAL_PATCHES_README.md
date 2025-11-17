# Continental Patches Generation

## Overview

This module provides functionality to divide the world into 6x6 degree grid cells, filter them by land coverage, and organize them by continent.

## Features

1. **Global Grid Division**: Divides the world into 6x6 degree patches (24 grid points at 0.25° resolution)
2. **Land Filtering**: Keeps only patches where >50% of the area is land
3. **Antarctica Exclusion**: Automatically excludes patches south of -60° latitude
4. **Continental Classification**: Classifies patches into continents:
   - Africa
   - Asia
   - Europe
   - North America
   - South America
   - Oceania (Australia/Pacific Islands)
   - Unknown (for patches that don't fit continental boundaries)

## Usage

### Option 1: Using the standalone script

```bash
cd /home/user/ai_weather_ag/finetuning
python3 create_continental_patches.py
```

This will create `.npy` files in the `data/processed/` directory:
- `africa_patches.npy`
- `asia_patches.npy`
- `europe_patches.npy`
- `north_america_patches.npy`
- `south_america_patches.npy`
- `oceania_patches.npy`

### Option 2: Calling from main script

Uncomment the following line in `clean_and_sample_climate_zones.py`:

```python
# Line 674 in main():
create_global_land_patches(dirs, patch_size_deg=6, land_threshold=0.5)
```

Then run:
```bash
python3 clean_and_sample_climate_zones.py
```

### Option 3: Using as a library

```python
from helper_funcs import setup_directories
from finetuning.clean_and_sample_climate_zones import create_global_land_patches

dirs = setup_directories()
continent_patches = create_global_land_patches(
    dirs,
    patch_size_deg=6,      # Size of each patch in degrees
    land_threshold=0.5      # Minimum 50% land coverage
)

# Access patches by continent
africa_patches = continent_patches['africa']
for lat_slice, lon_slice in africa_patches:
    print(f"Patch: lat {lat_slice[0]:.2f} to {lat_slice[-1]:.2f}, "
          f"lon {lon_slice[0]:.2f} to {lon_slice[-1]:.2f}")
```

## Loading Saved Patches

```python
import numpy as np

# Load patches for a specific continent
africa_patches = np.load('data/processed/africa_patches.npy', allow_pickle=True)

# Each patch is a tuple: (lat_slice, lon_slice)
for lat_slice, lon_slice in africa_patches:
    # lat_slice: numpy array of latitude coordinates (length 24 for 6° patch at 0.25° resolution)
    # lon_slice: numpy array of longitude coordinates (length 24)

    # Use with xarray datasets:
    # data_patch = dataset.sel(latitude=lat_slice, longitude=lon_slice)
    pass
```

## Patch Format

Each patch is stored as a tuple:
```python
(lat_slice, lon_slice)
```

Where:
- `lat_slice`: numpy array of 24 latitude values (6° / 0.25° = 24 grid points)
- `lon_slice`: numpy array of 24 longitude values

Example:
```python
# Load patches
patches = np.load('data/processed/europe_patches.npy', allow_pickle=True)
lat_slice, lon_slice = patches[0]

# Use with xarray
import xarray as xr
dataset = xr.open_dataset('path/to/data.nc')
patch_data = dataset.sel(latitude=lat_slice, longitude=lon_slice)
```

## Continental Boundaries

The classification uses simplified continental boundaries:

| Continent | Latitude Range | Longitude Range |
|-----------|----------------|-----------------|
| **Africa** | -35° to 37° N | -18° to 52° E |
| **Asia** | -10° to 80° N | 25° to 180° E |
| **Europe** | 35° to 71° N | -25° to 60° E |
| **North America** | 15° to 83° N | -170° to -52° E |
| **South America** | -56° to 13° N | -82° to -34° E |
| **Oceania** | -47° to 10° N | 110° to 180° E |

**Note**: Some regions have overlapping boundaries (e.g., Europe/Asia). Classification priority is given in the order: Oceania → South America → North America → Africa → Europe → Asia.

## Land Coverage Calculation

Land coverage is calculated using the IMERG land-sea mask:
1. Load IMERG land-sea mask (0.1° resolution)
2. Regrid to 0.25° resolution using nearest-neighbor resampling
3. Binary mask: values <20 = land (1), otherwise = ocean (0)
4. For each 6x6° patch, calculate fraction of grid cells that are land
5. Keep patches where land fraction ≥ 0.5 (50%)

## Parameters

### `create_global_land_patches()`

```python
def create_global_land_patches(
    dirs: Dict,
    patch_size_deg: int = 6,
    land_threshold: float = 0.5
) -> dict
```

**Parameters:**
- `dirs`: Dictionary with 'raw' and 'processed' keys pointing to data directories
- `patch_size_deg`: Size of each patch in degrees (default: 6)
- `land_threshold`: Minimum fraction of land required (default: 0.5 = 50%)

**Returns:**
- Dictionary mapping continent names to lists of patches

**Requirements:**
- IMERG land-sea mask must exist at: `{dirs['raw']}/IMERG_land_sea_mask.nc`
- This file should be available from the aurora module download

## Integration with Existing Code

This function builds on the existing patch sampling infrastructure:
- Uses `regrid_to_025()` for consistent 0.25° resolution
- Uses same patch format as `sample_zone_patches()`
- Compatible with existing patch loading and visualization code

## Example: Visualizing Continental Patches

```python
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Load patches
africa_patches = np.load('data/processed/africa_patches.npy', allow_pickle=True)

# Load climate zones for background
climate_zones = xr.open_dataset('data/processed/climate_zones_0p25.nc')['climate_zones']

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
climate_zones.plot(ax=ax)

# Add patch rectangles
for lat_slice, lon_slice in africa_patches:
    min_lat, max_lat = lat_slice.min(), lat_slice.max()
    min_lon, max_lon = lon_slice.min(), lon_slice.max()

    rect = Rectangle((min_lon, min_lat), max_lon - min_lon, max_lat - min_lat,
                    facecolor='red', alpha=0.3, edgecolor='black', linewidth=1)
    ax.add_patch(rect)

plt.title('Africa Patches')
plt.savefig('africa_patches.png', dpi=300)
```

## Technical Details

### Grid Resolution
- Land mask resolution: 0.25° (~28 km at equator)
- Patch size: 6° × 6° (~670 km × 670 km at equator)
- Grid points per patch: 24 × 24 = 576 points

### Memory Usage
- Each patch: ~400 bytes (2 arrays × 24 values × 8 bytes/float64)
- Typical continental file: ~100-300 patches × 400 bytes = 40-120 KB
- All continents: ~500-800 KB total

### Processing Time
- Typical runtime: 10-30 seconds
- Depends on disk I/O speed for loading land mask
- Regridding land mask is the slowest step (~5-10 seconds)

## Troubleshooting

### "FileNotFoundError: IMERG_land_sea_mask.nc"
**Solution**: Download the IMERG land-sea mask:
```bash
cd aurora
python3 download_aurora.py  # This downloads era5_static.nc which includes land mask
```

### "No patches found for continent X"
**Possible causes**:
1. Land threshold too high (try 0.3 instead of 0.5)
2. Continental boundaries don't match your expectation
3. Check the `classify_patch_by_continent()` function

### Patches at continent boundaries classified as "unknown"
**Expected behavior**: Patches in ocean or at edges between continents may be classified as "unknown". This is intentional for ambiguous regions.

## Future Enhancements

Potential improvements:
1. Use higher-resolution continent boundaries (e.g., Natural Earth data)
2. Add overlapping patches (stride < patch_size)
3. Add metadata to saved files (creation date, parameters used)
4. Create visualization function for all continental patches
5. Add option to subdivide large continents (e.g., East Asia, West Asia)

## References

- IMERG Land-Sea Mask: https://gpm.nasa.gov/data/imerg
- ERA5 Static Variables: https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5
- WeatherBench2: https://weatherbench2.readthedocs.io/
