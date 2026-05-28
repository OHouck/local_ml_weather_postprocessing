# AI Assistant Quick Reference Guide

**For detailed information, see: `REPOSITORY_STRUCTURE_ANALYSIS.md`**

---

## PROJECT AT A GLANCE

**Purpose**: Fine-tune neural networks to post-process weather forecasts (Pangu, IFS, AIFS)  
**Language**: Python 3.11+  
**Primary Libraries**: PyTorch, xarray, Dask  
**Main Output**: Regional temperature forecast bias correction models  
**Environment Management**: uv using local .venv

---

## CRITICAL DIRECTORIES & FILES

| Path | Purpose |
|------|---------|
| `/finetuning/post_process.py` | **MAIN ENTRY POINT** - Train bias correction models |
| `/finetuning/prepare_forecasts_and_targets.py` | Data loading with auto-download from WeatherBench2 |
| `/helper_funcs.py` | Shared utilities (path generation, setup) |
| `/aurora/run_aurora.py` | Aurora weather model execution |
| `/downloading_data/download_forecasts.py` | Raw data acquisition |
| `/setup_architecture_experiments.py` | Config for architecture comparison |
| `/analyze_architecture_results.py` | Parse experiment results |

---

## MOST COMMON COMMAND PATTERNS

### Running Fine-tuning
```bash
python3 finetuning/post_process.py \
    --output_dir ~/output \
    --model_name pangu \
    --region india \
    --training_vars 2m_temperature \
    --output_vars 2m_temperature \
    --lead_time_hours 24 72 144 \
    --train_start 2020-01-01 --train_end 2020-12-31 \
    --test_start 2021-01-01 --test_end 2021-06-30 \
    --nn_architecture mlp \
    --mlp_hidden_dim 1024 --mlp_num_layers 6
```

### Running Architecture Comparison
```bash
./run_architecture_experiments.sh
python3 analyze_architecture_results.py
```

### Understanding Data Input
```python
# Data automatically downloads to:
# ~/data/raw/{model}/{model}_{region}_{year}.zarr
# e.g., ~/data/raw/pangu/pangu_india_2020.zarr
```

---

## KEY CLASSES & FUNCTIONS

### Models (in post_process.py)
```python
class SimpleMLP(nn.Module)          # For 1D flattened inputs
class UNet(nn.Module)               # For 2D spatial grids
```

### Main Functions (in post_process.py)
```python
def parse_args()                    # CLI argument parsing
def load_combined_dataset()         # Load forecast + target data
def create_dataloader()             # Create PyTorch DataLoader
def train_model()                   # Main training loop
def apply_correction()              # Inference on test data
def save_output()                   # Save predictions to .zarr
```

### Data Loading (in prepare_forecasts_and_targets.py)
```python
def load_forecasts()                # Main data loading function
def parse_atmospheric_variable()    # Parse "temperature_500hPa" -> (temperature, 500)
def flatten_atmospheric_variables() # Extract specific pressure levels
def check_data_exists()             # Verify data files locally
def download_forecast_data()        # Download from WeatherBench2
```

### Utilities (in helper_funcs.py)
```python
def setup_directories()             # Environment-based path setup
def generate_output_path()          # Construct output file paths
```

---

## CRITICAL ARGUMENT PATTERNS

### Required Arguments
- `--output_dir`: Where to save trained models
- `--model_name`: pangu | ifs | aifs | aurora

### Important Optional Arguments
```
--region                    : india (default), usa_south, amazon, pakistan, etc.
--nn_architecture          : mlp (default) or unet
--mlp_hidden_dim           : 512-2048 (default 1024)
--mlp_num_layers           : 2-8 (default 4)
--mlp_dropout              : 0.0-0.3 (default 0.25)
--unet_hidden_dim          : 32-256 (default 256)
--unet_dropout             : 0.0-0.2 (default 0.1)
--lead_time_hours          : Space-separated list (default 24)
--training_vars            : Space-separated variable names
--output_vars              : Space-separated (typically 1 var)
--train_start/end          : YYYY-MM-DD format
--test_start/end           : YYYY-MM-DD format
```

---

## SUPPORTED VARIABLES

### Surface Variables (no suffix)
- `2m_temperature` (most common)
- `10m_u_component_of_wind`
- `10m_v_component_of_wind`
- `total_precipitation`
- `mean_sea_level_pressure`

### Atmospheric Variables (with _XhPa suffix)
- `temperature_500hPa`, `temperature_850hPa`, `temperature_1000hPa`
- `geopotential_500hPa`, `geopotential_850hPa`
- `specific_humidity_500hPa`, `specific_humidity_850hPa`, `specific_humidity_1000hPa`
- `u_component_of_wind_500hPa`, `u_component_of_wind_850hPa`
- `v_component_of_wind_500hPa`, `v_component_of_wind_850hPa`

---

## DATA FLOW OVERVIEW

```
1. User runs: python3 finetuning/post_process.py --args
                    ↓
2. parse_args() → Reads CLI arguments
                    ↓
3. load_forecasts() → Check local data → Download if missing
                    ↓
4. get_region_grid() → Extract spatial bounds
                    ↓
5. create_dataloader() → Prepare PyTorch tensors
                    ↓
6. select model (SimpleMLP or UNet)
                    ↓
7. train_model() → Training loop with validation
                    ↓
8. apply_correction() → Run model on test data
                    ↓
9. save_output() → Write results to .zarr file
                    ↓
10. Logs appear in output directory with metrics
```

---

## EXPECTED DIRECTORY STRUCTURE

```
/home/user/ai_weather_ag/
├── finetuning/              # Core training code
│   ├── post_process.py          # MAIN FILE - start here
│   ├── prepare_forecasts_and_targets.py
│   ├── process_forecasts.py
│   ├── figures_finetuning.py
│   ├── hyperparam_tuning.py
│   └── clean_and_sample_climate_zones.py
├── aurora/                  # Aurora model code
├── downloading_data/        # Data utils
├── neuralGCM_retraining/    # NeuralGCM integration
├── helper_funcs.py          # SHARED UTILITIES
├── analyze_architecture_results.py
├── setup_architecture_experiments.py
└── run_architecture_experiments.sh
```

---

## KEY CONFIGURATION PARAMETERS

### Project Configuration (pyproject.toml)
- Python >= 3.11
- torch >= 2.1.0
- xarray > 2025
- zarr > 3
- Optional CUDA for GPU

### Environment Setup
- Mac: Use `setup_mac.sh`
- Server: Use `setup_server.sh`
- Detection via `socket.gethostname()`

### Data Paths
- Uses environment detection to set data root
- For known environments (oMac.local, midway3)
- Can override with `--data_dir` argument

---

## HYPERPARAMETER PATTERNS

### Conservative Settings (Recommended)
```python
# MLP
mlp_hidden_dim=1024, mlp_num_layers=6, mlp_dropout=0.25

# UNet
unet_hidden_dim=128, unet_dropout=0.15
```

### Wide Network Settings
```python
# MLP
mlp_hidden_dim=2048, mlp_num_layers=3, mlp_dropout=0.30
```

### Lightweight Settings
```python
# UNet
unet_hidden_dim=64, unet_dropout=0.1
```

---

## ARCHITECTURE COMPARISON RESULTS

Expected improvements (vs baseline Pangu):
- **MLP Deep**: 4-5% RMSE reduction
- **MLP Wide**: 3-4% RMSE reduction
- **UNet Deep**: 5-6% RMSE reduction (best)
- **UNet Light**: 2-3% RMSE reduction

---

## IMPORT PATHS FOR CODE

```python
# From root of repo
sys.path.insert(0, str(Path(__file__).parent.parent))

# Typical imports
from helper_funcs import setup_directories, generate_output_path
from finetuning.prepare_forecasts_and_targets import load_forecasts
from aurora import Batch, Metadata, Aurora, rollout
```

---

## TESTING

### Test Files Location
- `neuralGCM_retraining/local_neuralGCM/*_test.py` (15 files)
- `weatherbenchx/wbx_test.py`, `wbx_setup_test.py`
- `wb_test.py` (root level)

### Running Tests
Tests use `_test.py` naming convention; run via Python import or pytest

---

## COMMON ISSUES & SOLUTIONS

| Issue | Solution |
|-------|----------|
| Data not found | Check `--data_dir` argument; script auto-downloads |
| Unknown region | See supported regions list above |
| Memory error | Reduce `--lead_time_hours` or use smaller patch |
| Missing variables | Ensure variable name spelling is correct |
| Environment detection fails | Edit `helper_funcs.py` setup_directories() |

---

## OUTPUT FILE NAMING

Format: `{model}/{region}/train_{train_vars}_test_{out_vars}_dim{size}_{lead_times}_{dates}_{arch}.zarr`

Example: `pangu/india/train_2m_temperature_test_2m_temperature_dim2x2_leadtime_24_72_144h_train2020-01-01-2020-12-31_test2021-01-01-2021-06-30_mlp.zarr`

---

## WHEN TO USE WHICH ARCHITECTURE

**MLP**:
- Single-location predictions
- Small spatial domains
- Fast training/inference
- Default choice

**UNet**:
- Spatial structure important
- Larger regions (>2x2°)
- Can capture fine-scale patterns
- Slower but often better results

---

## KEY METRICS

**RMSE** = Root Mean Squared Error (lower is better)  
**Improvement %** = (RMSE_original - RMSE_corrected) / RMSE_original × 100  

By lead time (longer = harder):
- 24h: ~15% improvement possible
- 72h: ~10% improvement possible
- 144h: ~5-7% improvement possible

---

## ENVIRONMENT VARIABLES & PATHS

Code uses:
- `socket.gethostname()` for environment detection
- `os.path.expanduser()` for home directory
- Default paths:
  - Mac: `/Users/ohouck/globus/forecast_data`
  - Midway3: `/project/jfranke/ozma/forecast_data`

---

## FOR DETAILED INFORMATION

**See full analysis**: `REPOSITORY_STRUCTURE_ANALYSIS.md`

Topics covered:
- Complete directory structure
- All module functions with signatures
- Data organization details
- Testing infrastructure
- Code conventions
- Workflow patterns
- Dependency documentation
- Performance patterns

---
