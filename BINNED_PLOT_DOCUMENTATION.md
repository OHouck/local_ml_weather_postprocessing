# Binned Weather Variable Analysis - Documentation

## Overview

The `plot_rmse_improvement_by_weather_bin` function creates visualizations showing how model performance (RMSE improvement or other metrics) varies across different ranges of weather variable values. This helps identify whether the model performs better or worse under specific weather conditions.

## Function Location

**File**: [finetuning/figures_finetuning.py](finetuning/figures_finetuning.py#L1666)

## What It Does

1. **Bins weather data**: Divides the range of weather variable values (e.g., temperature or wind speed) into evenly-spaced bins
2. **Calculates metrics per bin**: Computes RMSE improvement (or other metrics) for each bin
3. **Plots regional comparisons**: Creates line plots showing how each region performs across weather bins
4. **Shows data density**: Includes a histogram at the bottom showing the distribution of weather values

## Visual Output

The plot consists of two panels:
- **Top panel**: Line plot with metric on y-axis and binned weather values on x-axis (one line per region)
- **Bottom panel**: Histogram showing the frequency distribution of weather values (translucent bars)

## Function Signature

```python
plot_rmse_improvement_by_weather_bin(
    zarr_paths,
    dirs,
    variable,
    lead_time,
    model="pangu",
    regions=None,
    subregion="6x6",
    nn_architecture="mlp",
    metric="rmse_pct_improvement",
    n_bins=10,
    save_path=None
)
```

## Parameters

### Required Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `zarr_paths` | dict or str | Dictionary mapping region names to zarr file paths, OR single path for one region.<br>Example: `{'india': 'path.zarr', 'usa_south': 'path2.zarr'}` |
| `dirs` | dict | Dictionary of directories from `setup_directories()` |
| `variable` | str | Weather variable to analyze (e.g., `"2m_temperature"`, `"10m_wind_speed"`) |
| `lead_time` | int | Forecast lead time in hours (e.g., `24`, `72`, `144`) |

### Optional Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | `"pangu"` | Model name: `"pangu"`, `"ifs"`, `"aifs"`, etc. |
| `regions` | list or None | `None` | List of regions to include. If `None`, uses all regions in `zarr_paths` |
| `subregion` | str | `"6x6"` | Patch size identifier (for file naming) |
| `nn_architecture` | str | `"mlp"` | Architecture: `"mlp"` or `"unet"` |
| `metric` | str | `"rmse_pct_improvement"` | Metric to plot (see Supported Metrics below) |
| `n_bins` | int | `10` | Number of bins for weather variable |
| `save_path` | str | `None` | Custom save path. If `None`, auto-generates based on parameters |

## Supported Metrics

Currently supported values for the `metric` parameter:

1. **`"rmse_pct_improvement"`** (default)
   - Plots percentage improvement in RMSE: `(RMSE_original - RMSE_corrected) / RMSE_original * 100`
   - Positive values indicate improvement
   - Y-axis: "RMSE Improvement (%)"

2. **`"extreme_heat"`**
   - Plots percentage improvement in extreme heat RMSE using weighted loss function
   - Uses `calculate_extreme_heat_rmse()` from `process_forecasts.py`
   - Applies higher weights to under-predictions at high temperatures:
     - 25-30°C with negative error: 2x weight
     - ≥30°C with negative error: 10x weight
   - Formula: `(RMSE_extreme_orig - RMSE_extreme_corr) / RMSE_extreme_orig * 100`
   - Y-axis: "Extreme Heat RMSE Improvement (%)"

### Adding Custom Metrics

To add a new metric, modify the metric calculation section in the function (around line 1804):

```python
elif metric == "your_metric_name":
    # Your calculation here
    # Example: Calculate improvement percentage
    metric_orig = calculate_your_metric(orig_bin, gt_bin)
    metric_corr = calculate_your_metric(corr_bin, gt_bin)

    if metric_orig == 0:
        improvement = 0
    else:
        improvement = (metric_orig - metric_corr) / metric_orig * 100
    metric_values.append(improvement)
```

Also add the corresponding y-axis label (around line 1834):

```python
elif metric == "your_metric_name":
    ax_main.set_ylabel("Your Metric Label", fontsize=18)
    ax_main.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
```

Don't forget to import your metric function at the top of the file:

```python
from finetuning.process_forecasts import calculate_rmse, calculate_extreme_heat_rmse, calculate_your_metric
```

## Usage Examples

### Example 1: Single Region

```python
from helper_funcs import setup_directories
from finetuning.figures_finetuning import plot_rmse_improvement_by_weather_bin

dirs = setup_directories()

# Analyze India temperature at 72h lead time
plot_rmse_improvement_by_weather_bin(
    zarr_paths="/path/to/india_pangu_results.zarr",
    dirs=dirs,
    variable="2m_temperature",
    lead_time=72,
    model="pangu",
    regions=["india"],
    nn_architecture="mlp",
    n_bins=10
)
```

**Output**: Line plot showing RMSE improvement across 10 temperature bins for India

### Example 2: Multiple Regions Comparison

```python
# Compare three regions
zarr_paths = {
    'india': '/path/to/india_results.zarr',
    'usa_south': '/path/to/usa_south_results.zarr',
    'amazon': '/path/to/amazon_results.zarr'
}

plot_rmse_improvement_by_weather_bin(
    zarr_paths=zarr_paths,
    dirs=dirs,
    variable="2m_temperature",
    lead_time=24,
    model="pangu",
    regions=None,  # Uses all regions in zarr_paths
    nn_architecture="mlp",
    n_bins=10
)
```

**Output**: Multi-line plot with one line per region, showing how RMSE improvement varies with temperature

### Example 3: Wind Speed Analysis with More Bins

```python
# Analyze wind speed with finer resolution
plot_rmse_improvement_by_weather_bin(
    zarr_paths=zarr_paths,
    dirs=dirs,
    variable="10m_wind_speed",
    lead_time=144,
    model="ifs",
    regions=['india', 'usa_south'],
    nn_architecture="unet",
    n_bins=15  # More bins for finer detail
)
```

**Output**: Plot showing RMSE improvement across 15 wind speed bins

### Example 4: Custom Save Location

```python
# Save to custom location
plot_rmse_improvement_by_weather_bin(
    zarr_paths="/path/to/results.zarr",
    dirs=dirs,
    variable="2m_temperature",
    lead_time=72,
    model="pangu",
    regions=["india"],
    save_path="/custom/path/my_plot.png"
)
```

## Output Files

### Default Save Location

If `save_path=None`, files are saved to:
```
{dirs["fig"]}/{model}/binned_analysis/{subregion}/binned_{metric}_{variable}_lt{lead_time}h_{model}_{nn_architecture}_{region_str}_{n_bins}bins.png
```

Example:
```
~/ai_weather_ag/reports/figures/pangu/binned_analysis/6x6/
  binned_rmse_pct_improvement_2m_temperature_lt72h_pangu_mlp_india_10bins.png
```

### File Naming Convention

- `binned_` prefix indicates binned analysis
- `{metric}`: The metric being plotted
- `{variable}`: Weather variable name
- `lt{lead_time}h`: Lead time in hours
- `{model}`: Model name
- `{nn_architecture}`: Architecture type
- `{region_str}`: Single region name, or `climate_zones`/`topographic_zones`/`multi_region`
- `{n_bins}bins`: Number of bins used

## Data Requirements

### Zarr File Structure

The zarr files must contain the following variables for each lead time:

```python
# For variable "2m_temperature" and lead_time 72:
{variable}_ground_truth_lt{lead_time}h  # e.g., "2m_temperature_ground_truth_lt72h"
{variable}_original_lt{lead_time}h      # e.g., "2m_temperature_original_lt72h"
{variable}_corrected_lt{lead_time}h     # e.g., "2m_temperature_corrected_lt72h"
```

These are automatically created by the fine-tuning pipeline in [finetuning/finetune.py](finetuning/finetune.py).

### Data Dimensions

- **Coordinates**: `time`, `latitude`, `longitude`
- **Shape**: `(n_timesteps, n_lat, n_lon)`

## Technical Details

### Binning Algorithm

1. **Calculate global range**: Finds min/max across all regions to ensure consistent bins
2. **Create equal bins**: Uses `np.linspace()` to create `n_bins + 1` bin edges
3. **Assign data to bins**: Uses `np.digitize()` to classify each data point
4. **Calculate metrics**: Computes metric separately for data in each bin

### Histogram Alignment

- The histogram uses the **exact same bin edges** as the line plot
- Bars are aligned with x-axis tick marks
- Histogram shows combined data from all regions

### Color Coding

Regions are colored according to predefined schemes in `_get_color_schemes()`:

- **Climate zones**: tropical (green), arid (yellow), temperate (light green), cold (blue), polar (light blue)
- **Topographic zones**: Custom colors
- **Geographic regions**: India (orange), USA South (cyan), Amazon (pink), etc.

### Performance Optimization

- Uses `load_zarr_cached()` for efficient repeated loading
- Processes data in two passes:
  1. First pass: Calculate global min/max for binning
  2. Second pass: Bin data and calculate metrics

## Interpreting the Plots

### Y-Axis Interpretation (RMSE % Improvement)

- **Positive values**: Model improvement (corrected forecast better than original)
- **Negative values**: Model degradation (corrected forecast worse than original)
- **Zero line**: No improvement (corrected = original performance)

### X-Axis Interpretation

- Shows **actual weather values** (not normalized)
- Bins are evenly spaced across the data range
- Units auto-detected:
  - Temperature: °C (if min > -50) or K
  - Wind: m/s

### Histogram Interpretation

- Shows **data density** - which weather conditions occur most frequently
- Tall bars = common conditions (more training/test data)
- Short bars = rare conditions (less data, potentially less reliable metrics)

### Common Patterns

1. **U-shaped curves**: Model performs better at moderate values, worse at extremes
2. **Monotonic trends**: Performance improves/degrades consistently with variable value
3. **Flat lines**: Consistent performance across all weather conditions
4. **High variance with low histogram bars**: Unreliable metrics due to insufficient data

## Troubleshooting

### Error: "No zarr path provided for region"

**Cause**: Region specified in `regions` list but not in `zarr_paths` dictionary

**Solution**: Ensure all regions in `regions` have corresponding entries in `zarr_paths`

### Error: "Unknown metric"

**Cause**: Invalid `metric` parameter

**Solution**: Use `"rmse_pct_improvement"` or `"extreme_heat"`, or implement custom metric

### Empty bins (NaN values)

**Cause**: Insufficient data in extreme bins

**Solutions**:
- Reduce `n_bins` to group more data per bin
- Use longer training/test periods
- Accept that rare conditions may have sparse data

### Histogram doesn't align with line plot

**Cause**: This should not happen - both use same `bin_edges`

**Solution**: File a bug report

## Future Enhancements

Potential additions to consider:

1. **Confidence intervals**: Add error bars showing uncertainty in each bin
2. **Statistical significance**: Mark bins where improvement is statistically significant
3. **Weighted metrics**: Weight bins by sample count
4. **Quantile bins**: Use quantile-based binning instead of equal spacing
5. **Multiple lead times**: Overlay multiple lead times on same plot
6. **Seasonal analysis**: Separate plots for different seasons

## Related Functions

- [`plot_rmse_improvement`](finetuning/figures_finetuning.py#L1383): Baseline function showing RMSE improvement by lead time
- [`extract_forecast_data`](finetuning/figures_finetuning.py#L47): Extracts forecast data from zarr files
- [`calculate_rmse`](finetuning/figures_finetuning.py#L59): Computes RMSE between predictions and ground truth

## Example Output Description

A typical output image contains:

**Top Panel (70% of figure height)**:
- Multiple colored lines (one per region)
- Markers at each bin center
- Y-axis: RMSE Improvement (%)
- Horizontal dashed line at y=0
- Grid for easy reading
- Legend identifying regions
- Title with model, lead time, and architecture info

**Bottom Panel (30% of figure height)**:
- Translucent blue histogram bars
- X-axis shared with top panel
- Y-axis: Count
- X-axis label with weather variable and units

**Overall figure size**: 14" × 10" at 300 DPI (high quality for publications)

## Contact

For questions or issues with this function, contact the repository maintainer or file an issue on GitHub.

---

**Last Updated**: 2025-11-17
**Author**: Ozma Houck
**Function Version**: 1.0
