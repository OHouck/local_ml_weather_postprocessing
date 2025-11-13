# Global Improvement Map Function

## Overview

The `map_global_improvements()` function has been added to [figures_finetuning.py](finetuning/figures_finetuning.py:67-336). This function creates global maps showing RMSE percent improvement for all post-processed weather forecast patches.

**Key Feature**: The function automatically generates **3 separate maps**, one for each lead time (24h, 120h, and 216h).

## Function Signature

```python
def map_global_improvements(
    dirs,
    model="pangu",
    variable="10m_wind_speed",
    zone_types=None,
    save_dir=None
)
```

## Parameters

- **dirs** (dict): Dictionary of directories from `setup_directories()`
- **model** (str): Model to use - `"pangu"` or `"ifs"`
- **variable** (str): Variable to plot:
  - `"2m_temperature"`
  - `"10m_wind_speed"`
  - `"total_precipitation"`
- **zone_types** (list, optional): List of zone types to include. Default includes all:
  - Climate zones: `["tropical", "arid", "temperate"]`
  - Topographic zones: `["flat", "hilly", "mountainous"]`
- **save_dir** (str, optional): Custom save directory. If None, saves to `dirs["fig"]/model/global_maps/`

## Returns

- **figs** (dict): Dictionary of matplotlib figures keyed by lead time (24, 120, 216)

## Output

The function generates 3 PNG files, one for each lead time:
- `global_improvement_map_{variable}_{model}_lt24h_{zones}.png`
- `global_improvement_map_{variable}_{model}_lt120h_{zones}.png`
- `global_improvement_map_{variable}_{model}_lt216h_{zones}.png`

Each map includes:
- **Colored patches**: Each 2x2° patch is colored based on RMSE percent improvement
  - Blue colors: Positive improvement (better forecasts)
  - Red colors: Negative values (worse forecasts)
  - Color scale centered at 0% improvement
- **Land/ocean outlines**: For geographic context
- **Statistics box**: Shows mean, median, and standard deviation of improvements
- **Colorbar**: Shows the improvement percentage scale
- **Title**: Includes model, variable, lead time, and number of patches

## Example Usage

### Basic Usage

```python
from helper_funcs import setup_directories
from finetuning.figures_finetuning import map_global_improvements

# Setup directories
dirs = setup_directories()

# Create 3 maps (24h, 120h, 216h) for PANGU 10m_wind_speed
# across all zone types
map_global_improvements(
    dirs=dirs,
    model="pangu",
    variable="10m_wind_speed"
)
```

### Specific Zone Types

```python
# Create maps for only climate zones
map_global_improvements(
    dirs=dirs,
    model="pangu",
    variable="2m_temperature",
    zone_types=["tropical", "arid", "temperate"]
)

# Create maps for only topographic zones
map_global_improvements(
    dirs=dirs,
    model="ifs",
    variable="10m_wind_speed",
    zone_types=["flat", "hilly", "mountainous"]
)
```

### Different Models and Variables

```python
# IFS model with 2m_temperature
map_global_improvements(
    dirs=dirs,
    model="ifs",
    variable="2m_temperature"
)

# PANGU model with total_precipitation
map_global_improvements(
    dirs=dirs,
    model="pangu",
    variable="total_precipitation"
)
```

## Test Script

A test script [test_global_map.py](test_global_map.py) is included demonstrating various usage patterns:

```bash
python test_global_map.py
```

This will create:
- 3 maps for PANGU 10m_wind_speed (all zones)
- 3 maps for IFS 2m_temperature (all zones)
- 3 maps for PANGU 10m_wind_speed (climate zones only)

Total: 9 map files

## Data Requirements

The function expects post-processed forecast data in the following directory structure:

```
{dirs["raw"]}/../processed/finetuning_output/{model}/{zone_type}/*.zarr
```

Each zarr dataset should contain:
- `{variable}_ground_truth_lt{lead_time}h`
- `{variable}_original_lt{lead_time}h`
- `{variable}_corrected_lt{lead_time}h`
- Coordinates: `latitude`, `longitude`, `time`

## Output Example

For PANGU 10m_wind_speed with all zones:

```
Processing tropical: found 50 files
Processing arid: found 50 files
Processing temperate: found 100 files
Processing flat: found 50 files
Processing hilly: found 50 files
Processing mountainous: found 50 files

Lead time 24h: 350 patches
  Improvement range: -15.5% to 33.8%

Lead time 120h: 350 patches
  Improvement range: 3.3% to 42.9%

Lead time 216h: 350 patches
  Improvement range: 7.0% to 41.1%

Creating map for lead time 24h...
  Saved: .../global_improvement_map_10m_wind_speed_pangu_lt24h_tropical_arid_temperate_flat_hilly_mountainous.png

Creating map for lead time 120h...
  Saved: .../global_improvement_map_10m_wind_speed_pangu_lt120h_tropical_arid_temperate_flat_hilly_mountainous.png

Creating map for lead time 216h...
  Saved: .../global_improvement_map_10m_wind_speed_pangu_lt216h_tropical_arid_temperate_flat_hilly_mountainous.png

All 3 global improvement maps created successfully!
```

## Implementation Details

The function:
1. Scans all zarr files in the specified zone directories
2. For each file, extracts data for all 3 lead times (24h, 120h, 216h)
3. Calculates RMSE for original and corrected forecasts
4. Computes percent improvement: `(RMSE_orig - RMSE_corr) / RMSE_orig * 100`
5. Creates separate global maps for each lead time
6. Uses a diverging colormap (RdBu) centered at 0% improvement
7. Adds land/ocean features using Cartopy
8. Saves high-resolution (300 dpi) PNG files

## Notes

- The function handles longitude conversion automatically (converts negative longitudes to 0-360 range)
- NaN values are properly filtered before RMSE calculation
- Each map is closed after saving to free memory
- Progress is printed to console for monitoring
- Zarr loading uses `consolidated=False` to handle non-consolidated metadata
