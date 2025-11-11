# Data Loading and Preparation Flags

## Overview

This document describes two optional flags that control data loading and preparation behavior in `finetuning/finetune.py`. Both flags are designed to be easily removable when no longer needed.

---

## Flag 1: USE_LEGACY_GLOBAL_DATA

### Purpose
Load from global yearly data files instead of region-specific files.

### Location
`finetuning/finetune.py` at the start of `main()` (around line 1270)

### Usage

```python
USE_LEGACY_GLOBAL_DATA = True  # <-- Set to True to enable
```

### File Format Expected

```
{data_dir}/{model_name}/{model_name}_{year}.zarr
```

**Examples:**
- `~/data/pangu/pangu_2018.zarr`
- `~/data/pangu/pangu_2019.zarr`
- `~/data/era5/era5_2018.zarr`

These files should contain **global data** (not region-specific). The script will load these files and subset them spatially to your region of interest.

### When to Use

- You already have global yearly data files from previous runs
- You want to experiment with different regions without re-downloading
- You're testing with historical data that's in the old format

### What Happens

- **Data preparation is skipped** - No download or checking
- **Global files are loaded** - Reads `{model}_{year}.zarr` files
- **Spatial subsetting occurs** - Extracts only the lat/lon region you specified
- **Training proceeds normally** - Rest of pipeline works the same

---

## Flag 2: SKIP_DOWNLOAD

### Purpose
Check for required data but skip saving it locally if missing (still pulls from weatherbench).

### Location
`finetuning/finetune.py` at the start of `main()` (around line 1278)

### Usage

```python
SKIP_DOWNLOAD = True  # <-- Set to True to enable
```

### File Format Expected

```
{data_dir}/{model_name}/{model_name}_{region}_{year}.zarr
```

**Examples:**
- `~/data/pangu/pangu_india_2018.zarr`
- `~/data/pangu/pangu_india_2019.zarr`
- `~/data/era5/era5_india_2018.zarr`

These files should contain **region-specific data** (already subsetted to your region).

### When to Use

- You want to verify data availability without actually downloading
- You're testing the data preparation pipeline
- You want to see what data would be downloaded without committing storage
- You're checking for missing data or variables

### What Happens

- **Data checking still runs** - Verifies what files and variables exist
- **Missing data is pulled from weatherbench** - Downloads/processes data in memory
- **Data is NOT saved to disk** - Skips the `.to_zarr()` save step
- **Training will use existing files** - Only pre-existing region-specific files are available for training

---

## Flag Behavior Summary

| Flag Combination | Data Checking | Data Downloading | Files Saved | Files Loaded | Spatial Subsetting |
|-----------------|---------------|------------------|-------------|--------------|-------------------|
| Both False (default) | ✓ Runs | ✓ If needed | ✓ Yes | Region-specific | Not needed |
| LEGACY = True | ✗ Skipped | ✗ Skipped | N/A | Global yearly | ✓ Required |
| SKIP_DOWNLOAD = True | ✓ Runs | ✓ But not saved | ✗ No | Existing only | Not needed |
| Both True | ✗ Skipped | ✗ Skipped | N/A | Global yearly | ✓ Required |

**Note:** If both flags are True, USE_LEGACY_GLOBAL_DATA takes precedence.

---

## How to Remove These Flags (When No Longer Needed)

When you're ready to remove flag support completely, follow these steps:

### Step 1: Search for Markers

Search `finetuning/finetune.py` for these comment markers:

```python
# LEGACY
# SKIP
# TO REMOVE
```

### Step 2: Delete Flag Definitions

In `main()` function, delete both flag sections (lines ~1265-1279):

```python
# ========================================================================
# LEGACY DATA LOADING FLAG - REMOVE THIS SECTION WHEN NO LONGER NEEDED
# ========================================================================
USE_LEGACY_GLOBAL_DATA = False
# ========================================================================

# ========================================================================
# SKIP DOWNLOAD FLAG - REMOVE THIS SECTION WHEN NO LONGER NEEDED
# ========================================================================
SKIP_DOWNLOAD = False
# ========================================================================
```

### Step 3: Simplify Data Preparation Logic

In `main()`, replace the if/elif/else block (lines ~1303-1336) with:

**Delete this:**
```python
# ========================================================================
# LEGACY/SKIP FLAGS: Conditionally skip data preparation
# TO REMOVE: Remove this entire if/elif/else block when flags removed
# ========================================================================
if USE_LEGACY_GLOBAL_DATA:
    print("\n[LEGACY MODE] ...")
elif SKIP_DATA_PREPARATION:
    print("\n[SKIP MODE] ...")
else:
    prepare_data_for_finetuning(...)
# ========================================================================
# END LEGACY/SKIP FLAGS
# ========================================================================
```

**Replace with:**
```python
# Prepare data: check if exists, download if necessary
print("\nPreparing data for finetuning...")
prepare_data_for_finetuning(
    data_dir=args.data_dir,
    model_name=args.model_name,
    ground_truth_source=args.ground_truth_source,
    region=args.region,
    training_vars=args.training_vars,
    output_vars=args.output_vars,
    train_start=args.train_start,
    train_end=args.train_end,
    test_start=args.test_start,
    test_end=args.test_end,
    lead_time_hours=args.lead_time_hours,
    region_lat=region_lat,
    region_lon=region_lon
)
print("Data preparation complete. Proceeding with finetuning...\n")
```

### Step 4: Remove Legacy Loading Function

Delete `load_combined_dataset_legacy_global()` function (lines ~488-540):

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

### Step 5: Remove skip_save Parameter from Download Functions

In `prepare_forecasts_and_targets.py`, remove the `skip_save` parameter from both download functions:

**In download_forecast_data() signature (line ~320):**
```python
# Before
def download_forecast_data(..., skip_save=False):

# After
def download_forecast_data(...):
```

**In download_target_data() signature (line ~580):**
```python
# Before
def download_target_data(..., skip_save=False):

# After
def download_target_data(...):
```

**Remove conditional save logic** in both functions - always save data:
```python
# Delete this:
if not skip_save:
    print(f"    Saving to {output_path}...")
    with ProgressBar():
        subset_flattened.to_zarr(...)
else:
    print(f"    [SKIP SAVE] Data pulled but not saved to disk")

# Replace with:
print(f"    Saving to {output_path}...")
with ProgressBar():
    subset_flattened.to_zarr(...)
```

### Step 6: Update load_forecasts()

Remove the `use_legacy_global_data` parameter and if/else logic:

**In function signature (line ~605):**
```python
# Before
def load_forecasts(..., use_legacy_global_data=False):

# After
def load_forecasts(...):
```

**In function body (lines ~650-665), delete:**
```python
# ========================================================================
# LEGACY: Load datasets using appropriate method based on flag
# TO REMOVE: Remove this entire if/else block when legacy data not needed
# ========================================================================
if use_legacy_global_data:
    forecast_ds = load_combined_dataset_legacy_global(...)
    obs_ds = load_combined_dataset_legacy_global(...)
else:
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

### Step 7: Update run_subregion_experiment()

Remove the `use_legacy_global_data` parameter:

**In function signature (line ~1126):**
```python
# Before
def run_subregion_experiment(..., use_legacy_global_data=False):

# After
def run_subregion_experiment(...):
```

**In function calls to load_forecasts() (2 places):**
```python
# Before
load_forecasts(..., use_legacy_global_data=use_legacy_global_data)

# After
load_forecasts(...)
```

### Step 8: Remove Parameter from All Calls

Remove `use_legacy_global_data=USE_LEGACY_GLOBAL_DATA` from all calls to `run_subregion_experiment()` (3 locations in main(), around lines 1364, 1378, 1388).

### Step 9: Delete This README

Delete `DATA_LOADING_FLAGS_README.md`

### Step 10: Test

Run your experiments to ensure everything works with automatic data preparation.

---

## Quick Reference

### To Use Legacy Global Data
```python
USE_LEGACY_GLOBAL_DATA = True
SKIP_DOWNLOAD = False  # Not needed, but okay to leave as False
```

### To Check Data But Skip Saving
```python
USE_LEGACY_GLOBAL_DATA = False
SKIP_DOWNLOAD = True
```

### Normal Operation (Automatic Data Management)
```python
USE_LEGACY_GLOBAL_DATA = False  # Default
SKIP_DOWNLOAD = False           # Default
```

All flag code is clearly marked with comment boundaries for easy identification and removal.
