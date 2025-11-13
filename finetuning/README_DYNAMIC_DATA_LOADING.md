# Dynamic Data Loading for Finetuning

## Overview

The finetuning pipeline now supports **dynamic data loading**, which automatically downloads required data on-the-fly from weatherbench2 if it doesn't exist locally. This eliminates the need to pre-download large global datasets and allows for easier experimentation with different lead times, variables, and regions.

## How It Works

When you run `finetune.py`, the script now:

1. **Checks for existing data** - Scans the `data_dir` for required forecast and target data files
2. **Downloads missing data** - If data is missing, automatically downloads it from weatherbench2
3. **Proceeds with training** - Once all data is available, continues with the normal finetuning process

## New Module: `prepare_forecasts_and_targets.py`

This module provides the following functions:

### `check_data_exists(data_dir, data_source, years, variables)`
Checks if data files exist for the given parameters.

**Returns:** `(all_exist: bool, missing_years: list)`

### `download_forecast_data(data_dir, model_name, years, variables, lead_time_hours, ...)`
Downloads forecast data from weatherbench2 for the specified parameters.

**Supported models:**
- `pangu` - Pangu-Weather forecasts
- `ifs` - ECMWF IFS (HRES) forecasts
- `aifs` - ECMWF AIFS forecasts

**Features:**
- Uses Dask for parallel processing and optimization
- Downloads data year-by-year for memory efficiency
- Supports regional subsetting
- Converts init_time to valid_time for consistency

### `download_target_data(data_dir, model_name, ground_truth_source, years, variables, ...)`
Downloads target/observation data for the specified parameters.

**Supported targets:**
- `era5` - ERA5 reanalysis (default for Pangu and AIFS)
- `hres_t0` - ECMWF HRES T0 (default for IFS)

**Features:**
- Uses Dask for parallel processing
- Handles precipitation accumulation calculations
- Supports regional subsetting
- Filters for specific time steps (0, 6, 12 hours)

### `prepare_data_for_finetuning(data_dir, model_name, ground_truth_source, ...)`
Main function that orchestrates the entire data preparation process.

**Parameters:**
- `data_dir`: Directory for data storage
- `model_name`: Forecast model name
- `ground_truth_source`: Target data source (empty string for default)
- `training_vars`: List of training variables
- `output_vars`: List of output variables
- `train_start`, `train_end`: Training period
- `test_start`, `test_end`: Testing period
- `lead_time_hours`: List of lead times in hours
- `region_lat`, `region_lon`: Optional regional subsetting

**Returns:** Dictionary with status and paths

## Usage

No changes are required to how you run `finetune.py`. Simply use the same command-line arguments:

```bash
python3 finetuning/finetune.py \
    --data_dir="~/ai_weather_ag/data/raw/" \
    --output_dir="~/ai_weather_ag/data/fine_tuning_output" \
    --training_vars 2m_temperature \
    --output_vars 2m_temperature \
    --train_start="2018-01-01" --train_end="2021-12-31" \
    --test_start="2022-01-01" --test_end="2022-12-31" \
    --model_name="pangu" \
    --region="india" \
    --subregion="2x2" \
    --lead_time_hours 144 168
```

The script will automatically:
1. Check if data for years 2018-2022 exists
2. Download any missing data for the specified variables and lead times
3. Proceed with training

## Benefits

1. **No pre-downloading required** - Data is fetched on-demand
2. **Flexible experimentation** - Easily test different:
   - Lead times
   - Variables
   - Regions
   - Time periods
3. **Storage efficient** - Only download what you need
4. **Automatic caching** - Once downloaded, data is reused for future runs

## Performance

- Uses **Dask** for parallel downloads and memory-efficient processing
- Downloads data **year-by-year** to avoid memory issues
- Applies **regional subsetting** during download (not after) to reduce data size
- **Rechunking** optimizes storage format for fast loading

## Data Storage Format

Data is stored as **Zarr** files in the following structure:

```
data_dir/
├── pangu_2018.zarr/
├── pangu_2019.zarr/
├── era5_2018.zarr/
├── era5_2019.zarr/
└── ...
```

Each file contains:
- Variables requested
- Regional subset (if specified)
- Lead times (for forecasts)
- Time dimension (daily or 6-hourly)

## Requirements

The dynamic data loading requires the following packages:
- `xarray`
- `dask`
- `dask.distributed`
- `pandas`
- `numpy`
- `psutil`

These are the same requirements as the original `download_forecasts.py` and `download_targets.py` scripts.

## Notes

- **Wind speed**: If `10m_wind_speed` is requested, it will be computed from U and V components (not downloaded separately)
- **Precipitation**: Total precipitation is computed from 6-hourly values when needed
- **Time filtering**: Target data is filtered to hours 0, 6, and 12 for compatibility with forecast initialization times
- **Regional subsetting**: Applied during download to minimize data transfer and storage
