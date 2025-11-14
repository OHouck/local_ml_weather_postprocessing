# CLAUDE.md - AI Weather AG Repository Guide for AI Assistants

**Version**: 1.0
**Last Updated**: 2025-11-14
**Target Audience**: AI assistants (Claude, GPT, etc.) working with this codebase

---

## TABLE OF CONTENTS

1. [Project Overview](#project-overview)
2. [Codebase Structure](#codebase-structure)
3. [Development Workflows](#development-workflows)
4. [Key Conventions](#key-conventions)
5. [Common Tasks](#common-tasks)
6. [Data Organization](#data-organization)
7. [Important Files Reference](#important-files-reference)
8. [Best Practices](#best-practices)
9. [Common Pitfalls](#common-pitfalls)
10. [Additional Resources](#additional-resources)

---

## PROJECT OVERVIEW

### Purpose
This repository implements **neural network-based post-processing** (bias correction) for AI weather forecasts. The primary goal is to reduce RMSE (Root Mean Squared Error) in temperature predictions from models like Pangu, ECMWF IFS, and AIFS for specific geographic regions.

### Core Technology Stack
- **Language**: Python 3.11+
- **Deep Learning**: PyTorch 2.1.0+, torchvision 0.16.0+
- **Data Processing**: xarray 2025+, Dask, zarr 3+
- **Weather Data**: WeatherBench2, ERA5 reanalysis
- **Visualization**: matplotlib, cartopy
- **Environment**: Works on both Mac (development) and Linux clusters (SLURM)

### Key Models
1. **SimpleMLP**: Multi-layer perceptron for flattened spatial data
2. **UNet**: Convolutional U-Net for preserving spatial structure
3. **Aurora**: Microsoft's foundation weather model (separate module)

### Expected Outcomes
- **Baseline**: SimpleMLP/UNet without improvements
- **Conservative improvements**: 4-5% RMSE reduction (residual connections, better normalization)
- **Moderate improvements**: 5-6% RMSE reduction (+ attention mechanisms)
- **Aggressive improvements**: Not recommended (over-engineered, worse performance)

---

## CODEBASE STRUCTURE

### Directory Organization

```
/home/user/ai_weather_ag/
├── finetuning/                     # PRIMARY MODULE - Main training pipeline
│   ├── finetune.py                # **MAIN ENTRY POINT** for training
│   ├── prepare_forecasts_and_targets.py  # Data loading with auto-download
│   ├── process_forecasts.py       # Forecast processing utilities
│   ├── figures_finetuning.py      # Visualization and analysis
│   ├── hyperparam_tuning.py       # Hyperparameter optimization
│   └── clean_and_sample_climate_zones.py  # Climate zone processing
│
├── aurora/                         # Aurora weather model integration
│   ├── run_aurora.py              # Main Aurora execution
│   ├── download_aurora.py         # Data download for Aurora
│   └── prepare_aurora_data.py     # Aurora data preparation
│
├── downloading_data/               # Raw data acquisition utilities
│   ├── download_forecasts.py      # Forecast data from WeatherBench2
│   └── download_targets.py        # ERA5 target data
│
├── neuralGCM_retraining/          # NeuralGCM decoder experiments (archived)
├── run_ECMWF_forecasts/           # ECMWF forecast tools (archived)
├── run_weatherbench2/             # WeatherBench2 evaluation (archived)
├── weatherbenchx/                 # WeatherBench metrics
├── gee_gencast/                   # Google Earth Engine GenCast
├── old_finetuning/                # Legacy code (archived)
│
├── hyperopt_results_*/            # Hyperparameter tuning results
├── reports/                        # Output reports
├── logs/                          # Execution logs
├── slides/                        # Presentations
│
├── helper_funcs.py                # Shared utilities (path setup, etc.)
├── setup_architecture_experiments.py  # Architecture comparison config
├── run_architecture_experiments.sh   # Run 4 architecture experiments
├── analyze_architecture_results.py   # Parse experiment results
│
└── Documentation Files
    ├── CLAUDE.md                  # This file
    ├── README.md                  # Basic project info
    ├── ARCHITECTURE_EXPERIMENTS_README.md  # Architecture comparison guide
    ├── UPDATED_FEATURES_SUMMARY.md        # Feature updates
    ├── BRANCH_SUMMARY.md          # Git branch guide
    ├── REPOSITORY_STRUCTURE_ANALYSIS.md   # Detailed technical analysis
    └── AI_ASSISTANT_QUICK_REFERENCE.md    # Quick lookup guide
```

### Module Sizes
- **finetuning/**: ~7,244 lines of Python (primary module)
- **neuralGCM_retraining/**: ~60+ test files and core modules
- **Root scripts**: ~1,439 lines across 9 files

---

## DEVELOPMENT WORKFLOWS

### Workflow 1: Standard Fine-tuning

**Goal**: Train a bias correction model for a specific region

```bash
# Example: Train MLP for India region
python3 finetuning/finetune.py \
    --output_dir ~/ai_weather_ag/data/fine_tuning_output \
    --model_name pangu \
    --region india \
    --subregion 6x6 \
    --training_vars 2m_temperature 10m_u_component_of_wind 10m_v_component_of_wind \
                    temperature_1000hPa specific_humidity_1000hPa geopotential_1000hPa \
    --output_vars 2m_temperature \
    --lead_time_hours 24 72 144 \
    --train_start 2020-01-01 --train_end 2020-12-31 \
    --test_start 2021-01-01 --test_end 2021-06-30 \
    --nn_architecture mlp \
    --mlp_hidden_dim 1024 \
    --mlp_num_layers 6 \
    --mlp_dropout 0.25 \
    --data_dir ~/ai_weather_ag/data/raw
```

**What happens**:
1. `finetune.py` calls `prepare_forecasts_and_targets.py`
2. System checks if data exists locally
3. If missing, auto-downloads from WeatherBench2 to `data/raw/{model}/{model}_{region}_{year}.zarr`
4. Data is loaded, preprocessed, and split into train/val/test
5. Model trains with early stopping (patience=50 epochs)
6. Applies correction to test data
7. Saves output to `{output_dir}/{model}_{region}_{architecture}_{timestamp}.zarr`
8. Prints RMSE metrics for original vs corrected forecasts

### Workflow 2: Architecture Comparison

**Goal**: Compare 4 different architectures on the same data

```bash
# Run all 4 experiments (MLP Deep, MLP Wide, UNet Light, UNet Deep)
./run_architecture_experiments.sh

# Analyze results
python3 analyze_architecture_results.py
```

**Output**: `ARCHITECTURE_COMPARISON_REPORT.txt` with detailed metrics

### Workflow 3: Aurora Forecasting

**Goal**: Generate forecasts using Microsoft Aurora foundation model

```bash
python3 aurora/run_aurora.py \
    --start_date 2020-01-01 \
    --end_date 2020-01-31 \
    --output_dir ~/aurora_output
```

### Workflow 4: Hyperparameter Tuning

**Goal**: Find optimal hyperparameters using Hyperopt

```bash
python3 finetuning/hyperparam_tuning.py \
    --region india \
    --model_name pangu \
    --max_evals 100
```

---

## KEY CONVENTIONS

### 1. Naming Conventions

#### Files and Modules
- **snake_case**: All Python files use snake_case (e.g., `prepare_forecasts_and_targets.py`)
- **Script pattern**: Entry points end in `.py` with no special prefix
- **Test pattern**: Test files use `*_test.py` pattern (NeuralGCM module)

#### Classes
- **PascalCase**: All classes use PascalCase (e.g., `SimpleMLP`, `UNet`, `ResidualBlock`)

#### Functions and Variables
- **snake_case**: All functions and variables (e.g., `load_combined_dataset()`, `hidden_dim`)
- **Private functions**: Prefix with `_` (e.g., `_validate_input()`)

#### Constants
- **UPPER_SNAKE_CASE**: Module-level constants (though not heavily used in this codebase)

### 2. Variable Naming Patterns

#### Weather Variables
- **Surface variables**: `2m_temperature`, `10m_u_component_of_wind`, `10m_v_component_of_wind`
- **Atmospheric variables**: `{variable}_{pressure}hPa` (e.g., `temperature_500hPa`, `geopotential_850hPa`)
- **Parsing**: Use `parse_atmospheric_variable(var_name)` from `prepare_forecasts_and_targets.py`

Example:
```python
from prepare_forecasts_and_targets import parse_atmospheric_variable

var, level = parse_atmospheric_variable("temperature_500hPa")
# Returns: ("temperature", 500)

var, level = parse_atmospheric_variable("2m_temperature")
# Returns: ("2m_temperature", None)
```

#### Region Names
Supported regions (defined in `prepare_forecasts_and_targets.py`):
```python
REGION_BOUNDS = {
    'india': (5, 35, 65, 100),              # (lat_min, lat_max, lon_min, lon_max)
    'usa_south': (25, 35, -100, -90),
    'amazon': (-10, 0, -70, -60),
    'pakistan': (24, 37, 60, 77),
    'china': (20, 40, 100, 120),
    'australia': (-35, -25, 135, 145),
    'europe': (40, 55, -5, 15),
    'africa': (0, 15, 10, 30),
    'odisha': (17.78, 22.57, 81.37, 87.53)
}
```

### 3. Argument Patterns

All main scripts use `argparse` with consistent patterns:

**Required Arguments**:
- `--output_dir`: Where to save results
- `--region`: Geographic region
- `--model_name`: Forecast model (pangu, ifs, aifs, aurora)
- `--training_vars`: Input variables (space-separated)
- `--output_vars`: Target variables (space-separated)
- `--lead_time_hours`: Forecast lead times (space-separated integers)
- `--train_start`, `--train_end`: Training date range (YYYY-MM-DD)
- `--test_start`, `--test_end`: Testing date range (YYYY-MM-DD)

**Optional Arguments**:
- `--data_dir`: Where raw data is stored (default: `~/ai_weather_ag/data/raw`)
- `--subregion`: Subregion size (e.g., `6x6` degrees)
- `--nn_architecture`: `mlp` or `unet` (default: `mlp`)
- `--mlp_hidden_dim`: MLP hidden layer size (default: 1024)
- `--mlp_num_layers`: MLP depth (default: 4)
- `--mlp_dropout`: MLP dropout rate (default: 0.25)
- `--unet_hidden_dim`: UNet base channels (default: 32)
- `--unet_dropout`: UNet dropout rate (default: 0.1)

### 4. Import Structure

Standard import order:
```python
# 1. Standard library
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional

# 2. Third-party packages
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 3. Local modules
from helper_funcs import setup_directories, generate_output_path
from prepare_forecasts_and_targets import load_forecasts
```

### 5. Data Flow Patterns

#### Temporal Encoding
All models include temporal features:
- **Day of year**: `sin(2π * day / 365)`, `cos(2π * day / 365)`
- **Month encoding**: One-hot encoded months (12 dimensions)
- **Hour of day** (if applicable): `sin(2π * hour / 24)`, `cos(2π * hour / 24)`

#### Lead Time Encoding
Models support multiple lead times simultaneously:
- Embedded as additional input features
- Allows single model to correct multiple forecast horizons
- Lead times typically: 24h, 72h, 144h, 168h

#### Spatial Flattening (MLP)
```python
# Input shape: (batch, channels, height, width)
# Flatten to: (batch, channels * height * width)
x_flat = x.view(batch_size, -1)
```

#### Spatial Preservation (UNet)
```python
# Maintains (batch, channels, height, width) throughout
# Uses encoder-decoder with skip connections
```

---

## COMMON TASKS

### Task 1: Add a New Region

**Steps**:
1. Open `finetuning/prepare_forecasts_and_targets.py`
2. Add region to `REGION_BOUNDS` dictionary:
```python
REGION_BOUNDS = {
    # ... existing regions ...
    'new_region': (lat_min, lat_max, lon_min, lon_max),  # Degrees
}
```
3. Use in commands: `--region new_region`

### Task 2: Add a New Atmospheric Variable

**Steps**:
1. Determine variable name from WeatherBench2/ERA5 catalog
2. Add to `--training_vars` with pressure level suffix:
```bash
--training_vars 2m_temperature temperature_850hPa specific_humidity_700hPa
```
3. System automatically downloads and extracts the correct pressure level

### Task 3: Modify Model Architecture

**For MLP**:
Edit `finetune.py`, class `SimpleMLP`:
```python
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, ...):
        # Modify layers here
```

**For UNet**:
Edit `finetune.py`, class `UNet`:
```python
class UNet(nn.Module):
    def __init__(self, in_channels, hidden_dim, dropout, ...):
        # Modify encoder/decoder here
```

**Best Practice**: Use command-line arguments for architecture params instead of hardcoding:
```bash
--mlp_hidden_dim 2048 --mlp_num_layers 8
```

### Task 4: Change Training Parameters

Modify `finetune.py` training loop:
```python
# Key parameters (around line 300-350)
num_epochs = 750
batch_size = 128
learning_rate = 1e-4
patience = 50  # Early stopping
```

Or add command-line arguments in `parse_args()`.

### Task 5: Add Visualization

**Option 1**: Use existing `figures_finetuning.py`
```python
from figures_finetuning import plot_forecast_comparison

plot_forecast_comparison(
    original_ds=original_zarr,
    corrected_ds=corrected_zarr,
    target_ds=era5_zarr,
    output_path="comparison.png"
)
```

**Option 2**: Create new plotting function
- Follow matplotlib + cartopy patterns in `figures_finetuning.py`
- Save figures to `{output_dir}/figures/`

### Task 6: Run on SLURM Cluster

Create SLURM script (e.g., `finetune_job.sh`):
```bash
#!/bin/bash
#SBATCH --job-name=finetune
#SBATCH --output=logs/finetune-%j.txt
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

source setup_server.sh  # Loads environment

python3 finetuning/finetune.py \
    --output_dir ~/data/output \
    --region india \
    --model_name pangu \
    --training_vars 2m_temperature \
    --output_vars 2m_temperature \
    --lead_time_hours 24 72 144 \
    --train_start 2020-01-01 --train_end 2020-12-31 \
    --test_start 2021-01-01 --test_end 2021-06-30 \
    --nn_architecture mlp
```

Submit: `sbatch finetune_job.sh`

---

## DATA ORGANIZATION

### Local Data Structure

```
~/ai_weather_ag/data/
├── raw/                           # Downloaded forecast and target data
│   ├── pangu/
│   │   ├── pangu_india_2020.zarr
│   │   ├── pangu_india_2021.zarr
│   │   └── pangu_usa_south_2020.zarr
│   ├── ifs/
│   │   └── ifs_india_2020.zarr
│   ├── aifs/
│   │   └── aifs_india_2020.zarr
│   └── era5/                      # Ground truth targets
│       ├── era5_india_2020.zarr
│       └── era5_india_2021.zarr
│
├── fine_tuning_output/            # Model outputs
│   ├── pangu_india_mlp_20250114_120000.zarr
│   ├── pangu_india_unet_20250114_130000.zarr
│   └── figures/
│
└── architecture_experiments/      # Architecture comparison outputs
    ├── logs/
    │   ├── mlp_deep_20250109_120000.log
    │   └── unet_light_20250109_140000.log
    ├── ARCHITECTURE_COMPARISON_REPORT.txt
    └── results_summary.json
```

### File Naming Convention

**Raw data**:
```
{model}_{region}_{year}.zarr
```

**Model outputs**:
```
{model}_{region}_{architecture}_{timestamp}.zarr
```

**Logs**:
```
{experiment_name}_{timestamp}.log
```

### Zarr Dataset Structure

**Forecast data** (e.g., `pangu_india_2020.zarr`):
```python
<xarray.Dataset>
Dimensions:           (time, latitude, longitude, level)
Coordinates:
  * time              (time) datetime64[ns]
  * latitude          (latitude) float32
  * longitude         (longitude) float32
  * level             (level) int32  # For atmospheric variables
Data variables:
    2m_temperature    (time, latitude, longitude) float32
    temperature       (time, level, latitude, longitude) float32  # Atmospheric
    geopotential      (time, level, latitude, longitude) float32
Attributes:
    model:            pangu
    region:           india
    lead_time_hours:  [24, 72, 144]
```

**Target data** (ERA5):
```python
<xarray.Dataset>
Dimensions:           (time, latitude, longitude)
Coordinates:
  * time              (time) datetime64[ns]
  * latitude          (latitude) float32
  * longitude         (longitude) float32
Data variables:
    2m_temperature    (time, latitude, longitude) float32
```

---

## IMPORTANT FILES REFERENCE

### Primary Entry Points

| File | Purpose | Key Functions |
|------|---------|---------------|
| `finetuning/finetune.py` | Main training script | `parse_args()`, `load_combined_dataset()`, `train_model()`, `apply_correction()` |
| `aurora/run_aurora.py` | Aurora model execution | `run_aurora_forecast()` |
| `run_architecture_experiments.sh` | Batch architecture comparison | N/A (bash script) |

### Core Modules

| File | Purpose | Key Functions |
|------|---------|---------------|
| `finetuning/prepare_forecasts_and_targets.py` | Data loading & download | `load_forecasts()`, `download_forecast_data()`, `parse_atmospheric_variable()`, `check_data_exists()` |
| `finetuning/process_forecasts.py` | Forecast utilities | `process_forecast_data()` |
| `finetuning/figures_finetuning.py` | Visualization | `plot_forecast_comparison()`, `create_rmse_maps()` |
| `finetuning/hyperparam_tuning.py` | Hyperparameter search | `objective()`, `run_hyperopt()` |
| `helper_funcs.py` | Shared utilities | `setup_directories()`, `generate_output_path()` |

### Analysis & Reporting

| File | Purpose |
|------|---------|
| `analyze_architecture_results.py` | Parse experiment logs and generate comparison reports |
| `setup_architecture_experiments.py` | Configure architecture comparison experiments |

### Configuration

| File | Purpose |
|------|---------|
| `pyproject.toml` | Python dependencies and project metadata |
| `.python-version` | Python version (3.11) |
| `setup_mac.sh` | Mac development environment setup |
| `setup_server.sh` | Server (Linux/SLURM) environment setup |

---

## BEST PRACTICES

### 1. Code Modification Guidelines

**DO**:
- ✅ Add command-line arguments for new parameters
- ✅ Test on small data subset first (short date range)
- ✅ Save intermediate outputs for debugging
- ✅ Use existing helper functions (`setup_directories()`, etc.)
- ✅ Follow existing naming conventions
- ✅ Add docstrings for new functions
- ✅ Validate inputs early (check shapes, data types)

**DON'T**:
- ❌ Hardcode file paths (use `data_dir` and `output_dir` arguments)
- ❌ Modify archived modules (`old_finetuning/`, `run_ECMWF_forecasts/`)
- ❌ Change core data loading logic without thorough testing
- ❌ Skip error handling for data downloads (network issues common)
- ❌ Remove existing command-line arguments (breaks backward compatibility)

### 2. Testing Strategy

**Before committing changes**:
1. Run small test (1-week train, 1-week test) to verify code runs
2. Check output shapes and data types
3. Verify RMSE improvements are reasonable (not too good to be true)
4. Test on both MLP and UNet architectures
5. Ensure output files are created correctly

**Example test command**:
```bash
python3 finetuning/finetune.py \
    --output_dir ~/test_output \
    --region india --subregion 2x2 \
    --model_name pangu \
    --training_vars 2m_temperature \
    --output_vars 2m_temperature \
    --lead_time_hours 24 \
    --train_start 2020-01-01 --train_end 2020-01-07 \
    --test_start 2020-01-08 --test_end 2020-01-14 \
    --nn_architecture mlp
```

### 3. Git Workflow

**Current branch**: `claude/claude-md-mhzdrqu2tkbkl2gm-0186xVGLrHNfJWYxfGXre8BT`

**Branch conventions**:
- `main`: Production baseline (original implementation)
- `model-architecture-improvements`: Conservative improvements (recommended)
- `moderate-architecture-improvements`: Moderate improvements (experimental)
- `aggressive-architecture-improvements`: Aggressive changes (not recommended)
- See `BRANCH_SUMMARY.md` for detailed comparison

**When committing**:
```bash
# Stage changes
git add <files>

# Commit with descriptive message
git commit -m "Add support for new atmospheric variable parsing"

# Push to remote (ALWAYS use -u origin)
git push -u origin claude/claude-md-mhzdrqu2tkbkl2gm-0186xVGLrHNfJWYxfGXre8BT
```

**IMPORTANT**: Branch names must start with `claude/` and end with session ID for push to succeed.

### 4. Performance Optimization

**Memory**:
- Use Dask for large datasets (`chunks={'time': 10}`)
- Process data in batches (current batch_size: 128)
- Free GPU memory with `torch.cuda.empty_cache()` between runs

**Compute**:
- Use GPU when available (automatically detected in code)
- Leverage DataLoader `num_workers` for parallel data loading
- Cache preprocessed data when running multiple experiments

**Storage**:
- Use zarr compression for output files
- Download only required pressure levels (not all atmospheric data)
- Use region-based subsetting (don't download global data)

### 5. Documentation

**When adding features**:
1. Update this `CLAUDE.md` file
2. Add docstrings to new functions
3. Update relevant README files (e.g., `ARCHITECTURE_EXPERIMENTS_README.md`)
4. Document command-line arguments in `parse_args()` help text

**Docstring format**:
```python
def new_function(param1: str, param2: int) -> np.ndarray:
    """
    Brief one-line description.

    Longer description with more details about what the function does,
    edge cases, and usage examples.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When param2 is negative
    """
```

---

## COMMON PITFALLS

### 1. Data Loading Issues

**Problem**: Missing variables error
```
KeyError: 'temperature_500hPa'
```

**Solution**: Variable doesn't exist in downloaded data. Check:
- Is variable name correct? (see WeatherBench2 catalog)
- Does the pressure level exist in source data?
- Run `xr.open_zarr(path).info()` to see available variables

### 2. Memory Errors

**Problem**: Out of memory during training
```
RuntimeError: CUDA out of memory
```

**Solutions**:
- Reduce batch size (edit `batch_size` in `finetune.py`)
- Use smaller region (`--subregion 2x2` instead of `6x6`)
- Use shorter training period
- Use lighter architecture (`--mlp_num_layers 3` instead of `6`)

### 3. Network Download Failures

**Problem**: Data download times out
```
ConnectionError: Failed to download from WeatherBench2
```

**Solutions**:
- Retry the command (downloads are incremental)
- Use smaller date ranges (download year by year)
- Check network connection
- For large downloads, run on server with better bandwidth

### 4. Shape Mismatches

**Problem**: Model input/output shape errors
```
RuntimeError: Expected input shape (batch, 1440) but got (batch, 576)
```

**Cause**: Changing region size or variables without adjusting model

**Solutions**:
- Retrain model from scratch (don't load old checkpoints)
- Ensure `input_dim` matches actual data dimensions
- Check spatial dimensions after preprocessing

### 5. Incorrect Lead Time Handling

**Problem**: Model treats all lead times the same

**Cause**: Not properly encoding lead time as input feature

**Solution**: Ensure lead time is embedded correctly:
```python
# In model forward pass
lead_time_emb = self.lead_time_embedding(lead_time_tensor)
x = torch.cat([x, lead_time_emb], dim=1)
```

### 6. Zarr Consolidation Issues

**Problem**: Zarr reads are slow or fail
```
ValueError: Failed to open zarr group
```

**Solution**: Consolidate metadata after writing:
```python
import zarr
zarr.consolidate_metadata(zarr_path)
```

### 7. Date Range Errors

**Problem**: No data found for requested dates
```
ValueError: No forecast data found for period 2023-01-01 to 2023-12-31
```

**Cause**: Data not available for those dates in WeatherBench2

**Solution**: Check available years:
- Pangu: 2018-2022
- IFS: 2020-2022
- ERA5: 1979-present (but limited in WeatherBench2)

---

## ADDITIONAL RESOURCES

### External Documentation
- **WeatherBench2**: https://weatherbench2.readthedocs.io/
- **Pangu-Weather**: https://github.com/198808xc/Pangu-Weather
- **Aurora**: https://www.microsoft.com/en-us/research/project/aurora/
- **ERA5 Reanalysis**: https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5

### Internal Documentation
- `ARCHITECTURE_EXPERIMENTS_README.md`: Detailed guide to architecture comparison
- `UPDATED_FEATURES_SUMMARY.md`: Recent feature updates (dynamic loading, atmospheric variables)
- `BRANCH_SUMMARY.md`: Git branch comparison and recommendations
- `REPOSITORY_STRUCTURE_ANALYSIS.md`: Comprehensive technical analysis (632 lines)
- `AI_ASSISTANT_QUICK_REFERENCE.md`: Quick lookup guide

### Key Research Papers
- Pangu-Weather (2023): https://arxiv.org/abs/2211.02556
- WeatherBench2 (2024): https://arxiv.org/abs/2308.15560
- NeuralGCM (2024): https://arxiv.org/abs/2311.07222

### Contacts
- **Repository Owner**: Ozma Houck
- **GitHub**: https://github.com/OHouck/ai_weather_ag

---

## QUICK COMMAND REFERENCE

### Most Common Commands

```bash
# 1. Standard training (MLP)
python3 finetuning/finetune.py \
    --output_dir ~/output --region india --model_name pangu \
    --training_vars 2m_temperature --output_vars 2m_temperature \
    --lead_time_hours 24 72 144 \
    --train_start 2020-01-01 --train_end 2020-12-31 \
    --test_start 2021-01-01 --test_end 2021-06-30 \
    --nn_architecture mlp --mlp_hidden_dim 1024 --mlp_num_layers 6

# 2. Training with UNet
python3 finetuning/finetune.py \
    --output_dir ~/output --region india --model_name pangu \
    --training_vars 2m_temperature --output_vars 2m_temperature \
    --lead_time_hours 24 72 144 \
    --train_start 2020-01-01 --train_end 2020-12-31 \
    --test_start 2021-01-01 --test_end 2021-06-30 \
    --nn_architecture unet --unet_hidden_dim 128

# 3. Architecture comparison
./run_architecture_experiments.sh
python3 analyze_architecture_results.py

# 4. Hyperparameter tuning
python3 finetuning/hyperparam_tuning.py \
    --region india --model_name pangu --max_evals 100

# 5. Aurora forecasting
python3 aurora/run_aurora.py \
    --start_date 2020-01-01 --end_date 2020-01-31 \
    --output_dir ~/aurora_output
```

### Git Commands

```bash
# Check current branch
git branch

# Stage and commit changes
git add <files>
git commit -m "Description of changes"

# Push to remote (use -u origin with full branch name)
git push -u origin claude/claude-md-mhzdrqu2tkbkl2gm-0186xVGLrHNfJWYxfGXre8BT

# Switch branches
git checkout model-architecture-improvements

# View changes
git status
git diff
```

---

## VERSION HISTORY

- **v1.0** (2025-11-14): Initial comprehensive documentation created
  - Full codebase structure analysis
  - Development workflow documentation
  - Key conventions and best practices
  - Common tasks and pitfalls

---

**End of CLAUDE.md**

For detailed technical analysis, see `REPOSITORY_STRUCTURE_ANALYSIS.md`.
For quick reference, see `AI_ASSISTANT_QUICK_REFERENCE.md`.
