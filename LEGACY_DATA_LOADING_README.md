# Legacy Global Data Loading Feature

## Overview

This document describes the legacy data loading feature that allows loading from global yearly data files instead of region-specific files.

## How to Use

### 1. Edit the Flag

Open `finetuning/finetune.py` and find the flag at the start of the `main()` function (around line 1191):

```python
USE_LEGACY_GLOBAL_DATA = False  # <-- EDIT THIS FLAG
```

Set it to `True` to enable legacy mode:

```python
USE_LEGACY_GLOBAL_DATA = True  # <-- EDIT THIS FLAG
```

### 2. File Requirements

When legacy mode is enabled, the script expects files in this format:

```
{data_dir}/{model_name}/{model_name}_{year}.zarr
```

**Examples:**
- `~/data/pangu/pangu_2018.zarr`
- `~/data/pangu/pangu_2019.zarr`
- `~/data/pangu/pangu_2020.zarr`
- `~/data/era5/era5_2018.zarr`
- `~/data/era5/era5_2019.zarr`
- `~/data/era5/era5_2020.zarr`

These files should contain **global data** (not region-specific). The script will load these files and subset them to your region of interest.

### 3. What Happens in Legacy Mode

- **Data preparation is skipped** - No download or checking of region-specific files
- **Global files are loaded** - Reads `{model}_{year}.zarr` files
- **Spatial subsetting occurs** - Extracts only the lat/lon region you specified
- **Rest of pipeline is identical** - Training, evaluation, etc. work the same way

## When to Use

Use legacy mode when:
- You already have global yearly data files from previous runs
- You want to experiment with different regions without re-downloading
- You're testing with historical data that's in the old format

## How to Remove This Feature (When No Longer Needed)

When you're ready to remove legacy support completely, follow these steps:

### Step 1: Search for Legacy Markers

Search `finetuning/finetune.py` for these comment markers:

```python
# LEGACY
# TO REMOVE
```

### Step 2: Delete Marked Sections

Delete the following sections (marked with clear boundaries):

#### A. Delete the flag in `main()` (lines ~1186-1192)
```python
# ========================================================================
# LEGACY DATA LOADING FLAG - REMOVE THIS SECTION WHEN NO LONGER NEEDED
# ========================================================================
USE_LEGACY_GLOBAL_DATA = False
# ========================================================================
```

#### B. Delete `load_combined_dataset_legacy_global()` function (lines ~488-540)
```python
# ============================================================================
# LEGACY DATA LOADING - REMOVE THIS FUNCTION WHEN NO LONGER NEEDED
# ============================================================================
def load_combined_dataset_legacy_global(...):
    ...
# ============================================================================
# END LEGACY DATA LOADING
# ============================================================================
```

#### C. In `load_forecasts()`, replace the if/else with just the new loading (lines ~650-665)

**Delete this:**
```python
# ========================================================================
# LEGACY: Load datasets using appropriate method based on flag
# TO REMOVE: Remove this entire if/else block when legacy data not needed
# ========================================================================
if use_legacy_global_data:
    # Legacy: Load global files and subset spatially
    forecast_ds = load_combined_dataset_legacy_global(...)
    obs_ds = load_combined_dataset_legacy_global(...)
else:
    # New: Load region-specific files (already subsetted)
    forecast_ds = load_combined_dataset(...)
    obs_ds = load_combined_dataset(...)
# ========================================================================
# END LEGACY
# ========================================================================
```

**Replace with:**
```python
# Load region-specific files (already subsetted)
forecast_ds = load_combined_dataset(lat_values, lon_values, time_values_np, data_dir, args.model_name, args.region)
obs_ds = load_combined_dataset(lat_values, lon_values, time_values_np, data_dir, target, args.region)
```

#### D. In `main()`, remove the if/else around data preparation (lines ~1295-1324)

**Delete this:**
```python
# ========================================================================
# LEGACY: Skip data preparation if using legacy global data
# TO REMOVE: Remove this if/else block when legacy mode removed
# ========================================================================
if USE_LEGACY_GLOBAL_DATA:
    print("...")
else:
    prepare_data_for_finetuning(...)
# ========================================================================
# END LEGACY
# ========================================================================
```

**Replace with:**
```python
# Prepare data: check if exists, download if necessary
print("\nPreparing data for finetuning...")
prepare_data_for_finetuning(
    data_dir=args.data_dir,
    ...
)
print("Data preparation complete. Proceeding with finetuning...\n")
```

### Step 3: Remove Function Parameters

Remove `use_legacy_global_data` parameter from:

- `load_forecasts()` signature and all calls
- `run_subregion_experiment()` signature and all calls

### Step 4: Delete This README

Delete `LEGACY_DATA_LOADING_README.md`

### Step 5: Test

Run your experiments to ensure everything works with the new region-specific files.

## Summary

- **Enable**: Set `USE_LEGACY_GLOBAL_DATA = True` in `main()`
- **Disable**: Set `USE_LEGACY_GLOBAL_DATA = False` in `main()` (default)
- **Remove**: Follow the steps above to completely remove legacy code

All legacy code is clearly marked with comment boundaries for easy identification and removal.
