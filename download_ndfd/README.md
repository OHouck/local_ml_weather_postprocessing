# NDFD Data Download and Processing

This module downloads National Digital Forecast Database (NDFD) forecast data from NOAA's public S3 bucket and extracts regional subsets (currently Texas).

## Data Source

- **S3 Bucket**: `s3://noaa-ndfd-pds/wmo/`
- **Access**: Public, no authentication required (`--no-sign-request`)
- **Provider**: NOAA/National Weather Service

## File Structure

### S3 Source Structure

```
s3://noaa-ndfd-pds/wmo/
├── maxt/                          # Maximum temperature
│   ├── 2020/
│   │   ├── 01/
│   │   │   ├── 01/               # Day of month
│   │   │   │   ├── YGUZ98_KWBN_202001010046
│   │   │   │   ├── YGUZ98_KWBN_202001010146
│   │   │   │   └── ...
│   │   │   ├── 02/
│   │   │   └── ...
│   │   ├── 02/
│   │   └── ...
│   ├── 2021/
│   └── ...
├── mint/                          # Minimum temperature
├── temp/                          # Temperature
├── wspd/                          # Wind speed
├── wdir/                          # Wind direction
├── pop12/                         # 12-hour probability of precipitation
├── qpf/                           # Quantitative precipitation forecast
└── rhm/                           # Relative humidity
```

### Raw File Naming Convention

NDFD files follow the WMO header naming convention:

```
YGUZ98_KWBN_YYYYMMDDHHMM
│││││  │     └── Timestamp (UTC)
│││││  └── Originating center (KWBN = NWS Telecommunication Gateway)
│││└┴── Geographic region code
││└── Product type (Z = gridded binary)
│└── Data category (G = forecast)
└── WMO bulletin designator (Y = GRIB2)
```

**Region Codes:**
| Code | Region |
|------|--------|
| UZ | CONUS (Continental US) - **This is what we extract Texas from** |
| AZ | Puerto Rico |
| GZ | Guam |
| HZ | Hawaii |
| SZ | Alaska |
| TZ | US Pacific Islands |

**Product Suffix (97 vs 98):**
- `97`: Extended forecast (Days 4-7)
- `98`: Short-range forecast (Days 1-3)

### Local Output Structure (Texas Extraction)

After running `pull_ndfd.py`, data is saved as:

```
{data_dir}/ndfd_data/
├── maxt/
│   ├── 2025/
│   │   ├── 01/
│   │   │   ├── YGUZ98_KWBN_202501010046_texas.nc
│   │   │   ├── YGUZ98_KWBN_202501010146_texas.nc
│   │   │   └── ...
│   │   ├── 02/
│   │   └── ...
│   └── ...
├── mint/
└── ...
```

## NetCDF File Structure

Each extracted `*_texas.nc` file contains:

### Dimensions
| Dimension | Description |
|-----------|-------------|
| `step` | Forecast lead time steps (typically 3: Day 1, Day 2, Day 3) |
| `y` | Grid y-coordinate (~490 points for Texas) |
| `x` | Grid x-coordinate (~516 points for Texas) |

### Coordinates
| Coordinate | Type | Description |
|------------|------|-------------|
| `time` | datetime64 | Forecast initialization time (UTC) |
| `step` | timedelta64 | Lead time from initialization |
| `valid_time` | datetime64 | Valid time for each forecast step |
| `latitude` | float64 (y, x) | 2D latitude array (Lambert Conformal) |
| `longitude` | float64 (y, x) | 2D longitude array (converted to -180 to 180) |
| `heightAboveGround` | float64 | Height of measurement (2m for temperature) |

### Data Variables
| Variable | Units | Description |
|----------|-------|-------------|
| `tmax` | Kelvin | Maximum temperature forecast |
| `tmin` | Kelvin | Minimum temperature forecast (if downloading mint) |
| `t` | Kelvin | Temperature (if downloading temp) |

### Example: Reading the Data

```python
import xarray as xr

# Open a single file
ds = xr.open_dataset("ndfd_data/maxt/2025/01/YGUZ98_KWBN_202501010046_texas.nc")

# Access temperature data (in Kelvin)
tmax_kelvin = ds.tmax.values

# Convert to Fahrenheit
tmax_fahrenheit = (ds.tmax - 273.15) * 9/5 + 32

# Get coordinates
lat = ds.latitude.values  # 2D array
lon = ds.longitude.values  # 2D array

# Get forecast times
print(f"Forecast issued: {ds.time.values}")
print(f"Valid times: {ds.valid_time.values}")

ds.close()
```

## Grid Projection

NDFD CONUS data uses a **Lambert Conformal Conic** projection, which means:

1. Latitude and longitude are 2D arrays (not 1D)
2. Grid cells are not rectangular in lat/lon space
3. Use `pcolormesh` (not `imshow`) for plotting
4. Cannot use simple `.sel(latitude=slice(...))` for subsetting

The extraction code handles this by:
1. Creating a boolean mask for the Texas region
2. Finding the bounding box of the mask
3. Using `.isel()` with index slices to extract the region

## Texas Bounds

The extraction uses these geographic bounds:
- **Latitude**: 25.8°N to 36.5°N
- **Longitude**: -106.6°W to -93.5°W

## Usage

### Download Full Year of Data

```python
from download_ndfd.pull_ndfd import download_year_data

# Download max temperature for 2025, extracting only Texas
download_year_data(
    year=2025,
    elements=['maxt'],
    base_dir='/path/to/output',
    texas_only=True
)
```

### Download Single Month

```python
from download_ndfd.pull_ndfd import download_and_extract_texas_month

download_and_extract_texas_month(
    element='maxt',
    year=2025,
    month=1,
    base_dir='/path/to/output'
)
```

### Plot Forecast Map

```python
from download_ndfd.pull_ndfd import plot_texas_tmax_forecast

# Plot in Fahrenheit
plot_texas_tmax_forecast(
    nc_file='path/to/texas_data.nc',
    step_index=0,  # Day 1 forecast
    output_file='forecast_map.png',
    units='F'
)
```

## Available Elements

| Element | Description | Units (raw) |
|---------|-------------|-------------|
| `maxt` | Maximum temperature | Kelvin |
| `mint` | Minimum temperature | Kelvin |
| `temp` | Temperature | Kelvin |
| `wspd` | Wind speed | m/s |
| `wdir` | Wind direction | degrees |
| `pop12` | 12-hour probability of precipitation | % |
| `qpf` | Quantitative precipitation forecast | kg/m² |
| `rhm` | Relative humidity | % |

## Data Availability

- **Start**: ~2020 (varies by element)
- **End**: Current (updated continuously)
- **Frequency**: Multiple times per day (typically every 1-6 hours)
- **Forecast horizon**: Up to 7 days

## Dependencies

- `xarray`: Data handling
- `cfgrib`: GRIB2 file reading
- `eccodes`: ECMWF GRIB library (install via `brew install eccodes` on macOS)
- `matplotlib`: Plotting
- `numpy`: Array operations
- `awscli`: S3 data download

## Notes

1. **File sizes**: Raw CONUS GRIB files are ~3MB each. Texas NetCDF extracts are ~4MB (less compressed but smaller region).

2. **Non-CONUS regions**: Files for Puerto Rico, Hawaii, Alaska, etc. are automatically skipped during Texas extraction.

3. **Coordinate conversion**: Longitude is converted from 0-360° to -180-180° format during extraction.

4. **Temporary storage**: During download, full CONUS files are downloaded to a temp directory, Texas is extracted, then the temp files are deleted automatically.
