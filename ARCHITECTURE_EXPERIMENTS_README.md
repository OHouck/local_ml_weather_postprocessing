# Architecture Experiments for Pangu Forecast Improvement

## Overview

This experiment compares **4 different neural network architectures** to optimize post-processing of Pangu weather forecasts for the India region.

### Objective
Reduce RMSE of Pangu 2m temperature forecasts using atmospheric and surface variables as input.

### Region
- **Location**: India
- **Subregion**: 6x6 degrees (24 x 24 grid points at 0.25° resolution)
- **Center**: Approximately 20°N, 77°E

### Input Variables (6 total)
1. `2m_temperature` - Surface temperature
2. `10m_u_component_of_wind` - U wind component at 10m
3. `10m_v_component_of_wind` - V wind component at 10m
4. `temperature_1000hPa` - Temperature at 1000 hPa level
5. `specific_humidity_1000hPa` - Specific humidity at 1000 hPa
6. `geopotential_1000hPa` - Geopotential height at 1000 hPa

### Output Variable
- `2m_temperature` - Corrected surface temperature

### Lead Times
- **24h**: 1-day forecast
- **72h**: 3-day forecast
- **144h**: 6-day forecast

---

## Architectures Being Compared

### 1. MLP_Deep
**Description**: Deep MLP with 6 hidden layers

**Parameters**:
- Hidden dimension: 1024 neurons per layer
- Number of layers: 6
- Dropout rate: 0.25

**Rationale**: Deeper networks can learn more complex non-linear relationships between input variables and corrections needed.

### 2. MLP_Wide
**Description**: Wide MLP with 3 hidden layers

**Parameters**:
- Hidden dimension: 2048 neurons per layer
- Number of layers: 3
- Dropout rate: 0.30

**Rationale**: Wider layers can capture more features in parallel, potentially learning richer representations with fewer layers.

### 3. UNet_Light
**Description**: Lightweight U-Net with spatial structure

**Parameters**:
- Base hidden dimension: 64 channels
- Dropout rate: 0.10

**Rationale**: Lighter model with fewer parameters may generalize better and train faster while still leveraging spatial structure.

### 4. UNet_Deep
**Description**: Deep U-Net with more capacity

**Parameters**:
- Base hidden dimension: 128 channels
- Dropout rate: 0.15

**Rationale**: More channels allow the U-Net to learn richer spatial features, potentially capturing fine-grained atmospheric patterns.

---

## Quick Start

### 1. Run All Experiments

```bash
./run_architecture_experiments.sh
```

This will:
- Download data for India region (6x6) if not present
- Run all 4 architecture experiments sequentially
- Save logs to `~/ai_weather_ag/data/architecture_experiments/logs/`
- Save model outputs to `~/ai_weather_ag/data/architecture_experiments/`

**Estimated Time**:
- Data download: 30-60 minutes (if data doesn't exist)
- Each experiment: 1-3 hours (depending on hardware)
- Total: 5-12 hours for all 4 experiments

### 2. Analyze Results

```bash
python3 analyze_architecture_results.py
```

This will:
- Parse all log files
- Calculate RMSE and improvement percentages
- Generate comparison report
- Save results to:
  - `ARCHITECTURE_COMPARISON_REPORT.txt` (detailed text report)
  - `results_summary.json` (machine-readable summary)

---

## Running Individual Experiments

You can run individual experiments with custom parameters:

### MLP_Deep Example
```bash
python3 finetuning/finetune.py \
    --region=india \
    --subregion=6x6 \
    --model_name=pangu \
    --nn_architecture=mlp \
    --mlp_hidden_dim=1024 \
    --mlp_num_layers=6 \
    --mlp_dropout=0.25 \
    --training_vars 2m_temperature 10m_u_component_of_wind 10m_v_component_of_wind temperature_1000hPa specific_humidity_1000hPa geopotential_1000hPa \
    --output_vars 2m_temperature \
    --lead_time_hours 24 72 144 \
    --train_start=2020-01-01 --train_end=2020-12-31 \
    --test_start=2021-01-01 --test_end=2021-06-30 \
    --data_dir=~/ai_weather_ag/data/raw \
    --output_dir=~/ai_weather_ag/data/architecture_experiments
```

### UNet_Light Example
```bash
python3 finetuning/finetune.py \
    --region=india \
    --subregion=6x6 \
    --model_name=pangu \
    --nn_architecture=unet \
    --unet_hidden_dim=64 \
    --unet_dropout=0.1 \
    --training_vars 2m_temperature 10m_u_component_of_wind 10m_v_component_of_wind temperature_1000hPa specific_humidity_1000hPa geopotential_1000hPa \
    --output_vars 2m_temperature \
    --lead_time_hours 24 72 144 \
    --train_start=2020-01-01 --train_end=2020-12-31 \
    --test_start=2021-01-01 --test_end=2021-06-30 \
    --data_dir=~/ai_weather_ag/data/raw \
    --output_dir=~/ai_weather_ag/data/architecture_experiments
```

---

## New Command-Line Parameters

The `finetune.py` script now accepts architecture-specific parameters:

### MLP Parameters
- `--mlp_hidden_dim` (int, default=1024): Hidden dimension for MLP
- `--mlp_num_layers` (int, default=4): Number of hidden layers for MLP
- `--mlp_dropout` (float, default=0.25): Dropout rate for MLP

### UNet Parameters
- `--unet_hidden_dim` (int, default=32): Base hidden dimension for UNet
- `--unet_dropout` (float, default=0.1): Dropout rate for UNet

---

## Expected Results

Each experiment will output:

### During Training
```
Using SimpleMLP with 3 lead times and month encoding
  MLP hidden_dim: 1024
  MLP num_layers: 6
  MLP dropout: 0.25
...
Epoch 10/750 - Train Loss: X.XXXXXX, Val Loss: Y.YYYYYY, LR: Z.ZZe-06
...
Training complete in XX.XX minutes
```

### After Testing
```
Lead time 24h - MSE original: 8.234567, MSE corrected: 6.123456
Lead time 72h - MSE original: 12.345678, MSE corrected: 9.876543
Lead time 144h - MSE original: 15.678901, MSE corrected: 12.345678
```

### Metrics Explained
- **MSE original**: Mean Squared Error of original Pangu forecast
- **MSE corrected**: Mean Squared Error after applying correction
- **RMSE**: Square root of MSE (Root Mean Squared Error)
- **Improvement**: Percentage reduction in RMSE

**Formula**:
```
RMSE = sqrt(MSE)
Improvement (%) = (RMSE_original - RMSE_corrected) / RMSE_original × 100
```

---

## Comparison Report Format

The analysis script generates a detailed comparison report:

```
================================================================================
ARCHITECTURE EXPERIMENT RESULTS
================================================================================

Objective: Improve Pangu weather forecasts for India region
Region: India (6x6 degree subregion)
Output variable: 2m temperature
Lead times: 24h, 72h, 144h

================================================================================
INDIVIDUAL EXPERIMENT RESULTS
================================================================================

MLP Deep - Deep MLP (6 layers × 1024 neurons)
--------------------------------------------------------------------------------
  Architecture Parameters:
    hidden_dim: 1024
    num_layers: 6
    dropout: 0.25
  Training Time: 45.23 minutes

  Results by Lead Time:
    Lead Time    RMSE Orig    RMSE Corr    Improvement
    ------------ ------------ ------------ ------------
    24h          2.870123     2.475234          13.76%
    72h          3.512345     3.141234          10.57%
    144h         3.956789     3.678901           7.02%

...

================================================================================
COMPARISON SUMMARY
================================================================================

Lead Time: 24h
--------------------------------------------------------------------------------
Architecture         RMSE Original   RMSE Corrected  Improvement
-------------------- --------------- --------------- ---------------
⭐ UNet Deep          2.870123         2.415234          15.85%
   MLP Deep           2.870123         2.475234          13.76%
   MLP Wide           2.870123         2.523456          12.08%
   UNet Light         2.870123         2.589012           9.80%

...

================================================================================
BEST ARCHITECTURES
================================================================================

Rank   Architecture         Avg Improvement      Description
------ -------------------- -------------------- ----------------------------------------
🏆     UNet Deep                        14.23%  Deep UNet (128 base channels)
2.     MLP Deep                         12.45%  Deep MLP (6 layers × 1024 neurons)
3.     MLP Wide                         10.67%  Wide MLP (3 layers × 2048 neurons)
4.     UNet Light                        8.91%  Lightweight UNet (64 base channels)
```

---

## Customizing Experiments

### Testing Different Architectures

You can easily test different architectures by modifying the parameters:

```bash
# Test an even deeper MLP (8 layers)
python3 finetuning/finetune.py \
    --nn_architecture=mlp \
    --mlp_hidden_dim=1024 \
    --mlp_num_layers=8 \
    --mlp_dropout=0.3 \
    ...

# Test a very wide MLP (4096 neurons)
python3 finetuning/finetune.py \
    --nn_architecture=mlp \
    --mlp_hidden_dim=4096 \
    --mlp_num_layers=2 \
    --mlp_dropout=0.35 \
    ...

# Test an extra-deep UNet (256 channels)
python3 finetuning/finetune.py \
    --nn_architecture=unet \
    --unet_hidden_dim=256 \
    --unet_dropout=0.2 \
    ...
```

### Testing Different Time Periods

```bash
# Shorter training period (faster experiments)
--train_start=2020-06-01 --train_end=2020-12-31

# Longer test period (more robust evaluation)
--test_start=2021-01-01 --test_end=2021-12-31

# Different years
--train_start=2019-01-01 --train_end=2019-12-31 \
--test_start=2020-01-01 --test_end=2020-12-31
```

### Testing Different Variables

```bash
# Add more atmospheric levels
--training_vars 2m_temperature temperature_1000hPa temperature_850hPa temperature_500hPa

# Focus on specific variables
--training_vars 2m_temperature temperature_1000hPa specific_humidity_1000hPa
```

---

## File Structure

```
ai_weather_ag/
├── finetuning/
│   ├── finetune.py (updated with architecture parameters)
│   └── prepare_forecasts_and_targets.py
├── data/
│   ├── raw/
│   │   ├── pangu/
│   │   │   └── pangu_india_2020.zarr (auto-downloaded)
│   │   └── era5/
│   │       └── era5_india_2020.zarr (auto-downloaded)
│   └── architecture_experiments/
│       ├── logs/
│       │   ├── mlp_deep_20250109_120000.log
│       │   ├── mlp_wide_20250109_130000.log
│       │   ├── unet_light_20250109_140000.log
│       │   └── unet_deep_20250109_150000.log
│       ├── ARCHITECTURE_COMPARISON_REPORT.txt
│       ├── results_summary.json
│       └── [model output zarr files]
├── run_architecture_experiments.sh
├── analyze_architecture_results.py
├── ARCHITECTURE_EXPERIMENTS_README.md (this file)
└── setup_architecture_experiments.py
```

---

## Troubleshooting

### Data Download Issues
If data download fails:
```bash
# Check data directory exists
ls ~/ai_weather_ag/data/raw/

# Manually trigger data download
python3 finetuning/finetune.py --region=india --subregion=6x6 \
    --model_name=pangu --training_vars 2m_temperature \
    --output_vars 2m_temperature --lead_time_hours 24 \
    --train_start=2020-01-01 --train_end=2020-01-07 \
    --test_start=2020-01-08 --test_end=2020-01-10 \
    --data_dir=~/ai_weather_ag/data/raw \
    --output_dir=~/ai_weather_ag/data/test
```

### Memory Issues
If you run out of memory:
- Reduce training period (e.g., 6 months instead of 1 year)
- Reduce batch size in finetune.py (currently 128)
- Use lighter architectures (smaller hidden_dim)

### No Results Found
If analysis script can't find results:
```bash
# Check log directory
ls ~/ai_weather_ag/data/architecture_experiments/logs/

# Run analysis with verbose output
python3 analyze_architecture_results.py
```

---

## Next Steps

After completing the experiments:

1. **Review the comparison report** to identify the best architecture
2. **Use the best architecture** for production forecasts
3. **Experiment with variations**:
   - Different dropout rates
   - Different layer sizes
   - Different number of layers
4. **Test on other regions** (e.g., Odisha, USA South)
5. **Try different output variables** (e.g., precipitation)

---

## References

- Pangu-Weather: https://github.com/198808xc/Pangu-Weather
- WeatherBench2: https://weatherbench2.readthedocs.io/
- ERA5 Reanalysis: https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5
