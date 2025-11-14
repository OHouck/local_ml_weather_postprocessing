# AI Weather AG Repository - Comprehensive Analysis

## 1. DIRECTORY STRUCTURE

### Top-Level Organization
```
/home/user/ai_weather_ag/
├── finetuning/                 # Main fine-tuning pipeline (7,244 lines of Python)
├── aurora/                     # Aurora weather model implementation
├── downloading_data/           # Data download and preprocessing utilities
├── neuralGCM_retraining/       # NeuralGCM decoder retraining
├── run_ECMWF_forecasts/        # ECMWF forecast execution (archived)
├── run_weatherbench2/          # WeatherBench2 evaluation tools (archived)
├── weatherbenchx/              # WeatherBenchX metrics and evaluation
├── gee_gencast/                # Google Earth Engine GenCast integration
├── old_finetuning/             # Legacy fine-tuning code (archived)
├── hyperopt_results_*/         # Hyperparameter optimization results
├── reports/                    # Output reports and results
├── logs/                        # Execution logs
├── slides/                      # Presentation slides
└── Users/ohouck/               # User-specific directory
```

### Size Metrics
- **Finetuning module**: 7,244 lines across 7 Python files
- **NeuralGCM retraining**: ~60+ test files and core modules
- **Root-level scripts**: 1,439 lines across 9 main Python files

---

## 2. FILE ORGANIZATION & MODULE STRUCTURE

### A. FINETUNING MODULE (Primary)
**Location**: `/home/user/ai_weather_ag/finetuning/`

Core files:
- **finetune.py** (43,745 bytes) - Main training entry point
- **prepare_forecasts_and_targets.py** (65,185 bytes) - Data loading & preprocessing
- **process_forecasts.py** (31,336 bytes) - Forecast processing utilities
- **figures_finetuning.py** (112,366 bytes) - Visualization and analysis
- **hyperparam_tuning.py** (17,042 bytes) - Hyperparameter search
- **clean_and_sample_climate_zones.py** (25,311 bytes) - Climate zone processing

### B. AURORA MODULE
**Location**: `/home/user/ai_weather_ag/aurora/`

- **run_aurora.py** - Main Aurora forecast execution (supports rollout predictions)
- **download_aurora.py** - Aurora data download utilities
- **prepare_aurora_data.py** - Aurora data preparation pipeline
- **clean_forecasts.py** - Forecast cleaning and validation
- **create_aurora_forecasts.sh** - Bash wrapper for batch processing

### C. DATA DOWNLOADING MODULE
**Location**: `/home/user/ai_weather_ag/downloading_data/`

- **download_forecasts.py** - Forecast data acquisition from WeatherBench2
- **download_targets.py** - Observational target data download
- **combine_aifs.py** - AIFS data combination utilities
- **aifs_cleaning.py** - AIFS-specific preprocessing
- **download_aifs.sh** - Automated AIFS download script

### D. NEURALGCM RETRAINING
**Location**: `/home/user/ai_weather_ag/neuralGCM_retraining/`

- **local_neuralGCM/** - Local copy of NeuralGCM public repository
  - Core modules: `api.py`, `model_builder.py`, `layers.py`, `encoders.py`
  - 15+ test files (*_test.py pattern)
  - Reference implementations: `reference_code/`
- **retrain_neuralGCM_decoder.py** - Decoder customization
- **neuralGCM_inference.py** - Inference pipeline

---

## 3. KEY ENTRY POINTS

### Primary Workflows

#### A. Fine-tuning Workflow
```bash
# Main entry point
python3 finetuning/finetune.py \
    --region=india \
    --model_name=pangu \
    --nn_architecture=mlp|unet \
    --training_vars 2m_temperature [additional vars...] \
    --output_vars 2m_temperature \
    --lead_time_hours 24 72 144 \
    --train_start=YYYY-MM-DD --train_end=YYYY-MM-DD \
    --test_start=YYYY-MM-DD --test_end=YYYY-MM-DD \
    --data_dir ~/path/to/data \
    --output_dir ~/path/to/output \
    [--nn_architecture=mlp --mlp_hidden_dim=1024 --mlp_num_layers=6] \
    [--nn_architecture=unet --unet_hidden_dim=128]
```

**Defines**: `parse_args()` in finetune.py (lines 413-448)

#### B. Architecture Experiment Workflow
```bash
# Run all 4 architecture comparisons
./run_architecture_experiments.sh

# Analyze results
python3 analyze_architecture_results.py
```

**Compares**:
- MLP_Deep (6 layers × 1024 neurons)
- MLP_Wide (3 layers × 2048 neurons)
- UNet_Light (64 base channels)
- UNet_Deep (128 base channels)

#### C. Aurora Forecast Workflow
```bash
python3 aurora/run_aurora.py [parameters]
```

#### D. Data Preparation Workflow
```bash
# Downloads and processes forecast data
python3 finetuning/prepare_forecasts_and_targets.py
```

---

## 4. CONFIGURATION FILES

### A. Project Configuration
- **pyproject.toml** - Project metadata and dependencies
  - Python >= 3.11
  - 123+ direct dependencies
  - Optional CUDA dependencies for GPU clusters

### B. Python Version Management
- **.python-version** - Specifies Python version (5 bytes)

### C. Environment & Ignore Files
- **.gitignore** - Version control configuration
- **.git/** - Full git repository history

### D. Setup Scripts
- **setup_mac.sh** - Mac-specific environment setup (160 bytes)
- **setup_server.sh** - Server environment setup (202 bytes)
- **setup_architecture_experiments.py** - Experiment configuration (325 lines)

### E. Shell Execution Wrappers
- **run_architecture_experiments.sh** - Orchestrates 4 experiments
- **marimo.sh** - Marimo notebook launcher
- **hyperparam_tuning.sh** - Hyperparameter search automation

---

## 5. DATA ORGANIZATION

### A. Expected Directory Structure
```
~/ai_weather_ag/data/
├── raw/
│   ├── pangu/
│   │   ├── pangu_india_2020.zarr
│   │   ├── pangu_odisha_2020.zarr
│   │   └── pangu_usa_south_2020.zarr
│   ├── era5/
│   │   ├── era5_india_2020.zarr
│   │   ├── era5_odisha_2020.zarr
│   │   └── era5_usa_south_2020.zarr
│   ├── ifs/
│   ├── aifs/
│   └── ...
├── processed/
│   ├── finetuning_output/
│   └── architecture_experiments/
│       ├── logs/
│       │   ├── mlp_deep_YYYYMMDD_HHMMSS.log
│       │   ├── mlp_wide_YYYYMMDD_HHMMSS.log
│       │   ├── unet_light_YYYYMMDD_HHMMSS.log
│       │   └── unet_deep_YYYYMMDD_HHMMSS.log
│       ├── results_summary.json
│       └── ARCHITECTURE_COMPARISON_REPORT.txt
└── figures/
```

### B. Supported Models
- **Pangu-Weather** (pangu)
- **ECMWF IFS** (ifs)
- **ECMWF AIFS** (aifs)
- **Aurora** (aurora)

### C. Supported Regions
- india (17-27°N, 72-82°E)
- usa_south (30-40°N, -105--95°W)
- amazon (-10-0°N, -70--60°W)
- british_columbia (48.25-58°N, -130--120°W)
- pakistan (25-34°N, 60-70°E)
- ethiopia (4-14°N, 34-44°E)
- corn_belt (36-46°N, -95--85°W)
- global (-90-90°N, 0-360°E)
- climate_zones (tropical, arid, temperate, cold, polar)
- topographic_zones (flat, hilly, mountainous)

### D. Variable Organization

#### Surface Variables
- 2m_temperature
- 10m_u_component_of_wind
- 10m_v_component_of_wind
- total_precipitation
- mean_sea_level_pressure

#### Atmospheric Variables (with pressure level suffix)
- temperature_{level}hPa (e.g., temperature_500hPa)
- geopotential_{level}hPa
- specific_humidity_{level}hPa
- u_component_of_wind_{level}hPa
- v_component_of_wind_{level}hPa

---

## 6. TESTING INFRASTRUCTURE

### Test Files Location
```
/home/user/ai_weather_ag/
├── neuralGCM_retraining/local_neuralGCM/
│   ├── api_test.py
│   ├── decoders_test.py
│   ├── encoders_test.py
│   ├── features_test.py
│   ├── layers_test.py
│   ├── models_test.py
│   ├── transforms_test.py
│   └── 10+ more test files (15 total)
├── weatherbenchx/
│   ├── wbx_test.py
│   ├── wbx_setup_test.py
│   └── test_diagnostics.py
└── wb_test.py (root level)
```

### Testing Patterns
- **Unit tests**: Located alongside modules with `_test.py` suffix
- **Integration tests**: `weatherbenchx/` contains WeatherBench2 integration tests
- **Manual test scripts**: Standalone test files in `weatherbenchx/`

### Test Execution
- Tests appear to be module-specific (testing NeuralGCM components)
- No centralized pytest configuration found
- Tests likely run via Python import (based on naming convention)

---

## 7. DOCUMENTATION PATTERNS

### A. README Files
- **README.md** (1,793 bytes) - High-level project overview
- **ARCHITECTURE_EXPERIMENTS_README.md** (12,369 bytes) - Detailed experiment guide
- **BRANCH_SUMMARY.md** (4,529 bytes) - Git branch management guide
- **UPDATED_FEATURES_SUMMARY.md** (9,301 bytes) - Feature documentation
- **finetuning/README_DYNAMIC_DATA_LOADING.md** - Data loading documentation

### B. In-Code Documentation
All major Python files include:
- **Module-level docstrings** with purpose and usage
- **Author and date information** at file top
- **Example command-line usage** in comments
- **Class docstrings** with parameter descriptions
- **Function docstrings** with Parameters/Returns sections

### C. Documentation Examples
From `finetune.py`:
```python
"""
Author: Ozma Houck 
Filename: finetune/finetune.py

Purpose: use a simple MLP to post-process weather forecasts trained on
specific regions and variables. Call this script from command line or with 
1_run_experiments.sh script.

# example call
python3 finetuning/finetune.py \
    --data_dir="..." \
    --output_dir="..." \
    --training_vars 2m_temperature \
    --output_vars 2m_temperature \
    ...
"""
```

### D. Log & Report Generation
Scripts generate:
- **Training logs**: Real-time epoch metrics (Loss, LR)
- **Test reports**: RMSE comparisons and improvements
- **Architecture comparison reports**: JSON summaries and text analysis
- **Experiment logs**: Timestamped with {experiment_name}_{YYYYMMDD}_{HHMMSS}.log

---

## 8. DEPENDENCIES & LIBRARIES

### Core Machine Learning Stack
- **torch** >= 2.1.0
- **torchvision** >= 0.16.0
- **timm** - Transformer implementations
- **pytorch-lightning** (implied from aurora)

### Data Processing
- **xarray** > 2025 - N-dimensional array processing
- **zarr** > 3 - Chunked storage
- **dask** - Distributed computing
- **pandas** - Tabular data
- **numpy** >= 1.26.0, < 2.0
- **netcdf4**, **h5py**, **h5netcdf** - File I/O

### Scientific Computing
- **scipy**, **scikit-learn** (implied)
- **numba** - JIT compilation
- **numcodecs** - Compression codecs

### Cloud & Storage
- **gcsfs** - Google Cloud Storage
- **google-cloud-storage** - GCS interface
- **azure-storage-blob** - Azure Blob Storage
- **ecmwf-datastores-client** - ECMWF data access

### Visualization & Notebooks
- **matplotlib** - Plotting
- **cartopy** - Geographical plotting
- **marimo** >= 0.17.6 - Interactive notebooks
- **pillow** - Image processing

### Utilities
- **loguru** - Logging
- **pydantic** - Data validation
- **click**, **argparse** - CLI interfaces
- **hyperopt** >= 0.2.7 - Hyperparameter optimization
- **psutil** - System monitoring

### Optional Dependencies
- **CUDA 12.x packages** - GPU acceleration (optional)
- **triton** - GPU kernel language (optional)

---

## 9. WORKFLOW PATTERNS

### A. Data Pipeline Workflow
```
1. Argument Parsing
   └─> parse_args() in finetune.py

2. Data Loading
   └─> load_forecasts() [prepare_forecasts_and_targets.py]
       ├─> Check if data exists locally
       ├─> Download missing data from WeatherBench2
       ├─> Parse atmospheric variables (e.g., temperature_500hPa)
       └─> Return xarray.Dataset with proper dimensions

3. Data Preprocessing
   └─> get_region_grid() - Extract region bounds
   └─> get_patch_shape() - Convert degree specs to grid points
   └─> sort_lat_lon() - Ensure proper coordinate ordering
   └─> create_dataloader() - PyTorch DataLoader creation

4. Model Training
   └─> Select architecture (SimpleMLP or UNet)
   └─> train_model() with callbacks
       ├─> Forward pass with lead_time_idx and day_of_year_features
       ├─> Loss computation (MSE, quantile_loss, or extreme_heat_loss)
       ├─> Backward pass
       └─> Learning rate scheduling

5. Inference & Evaluation
   └─> apply_correction() - Apply model to test data
   └─> Compute RMSE improvements
   └─> save_output() - Save corrected forecasts as .zarr

6. Reporting
   └─> Generate comparison metrics
   └─> Save results to JSON
```

### B. Architecture Experiments Workflow
```
1. Setup (setup_architecture_experiments.py)
   ├─> Define 4 architectures with parameters
   ├─> Set common training/test periods
   └─> Create output directory structure

2. Execution (run_architecture_experiments.sh)
   ├─> Run each experiment sequentially
   ├─> Log output to timestamped files
   └─> Save model predictions

3. Analysis (analyze_architecture_results.py)
   ├─> Parse all log files
   ├─> Extract RMSE values by lead time
   ├─> Calculate improvements
   └─> Generate comparison report with rankings
```

### C. Model Class Hierarchy
```
SimpleMLP(nn.Module)
├─> __init__: Set up layers with optional lead_time embedding
├─> forward: Process input + day_of_year features + lead_time embedding
└─> Usage: For 1D flattened input data

UNet(nn.Module)
├─> __init__: Build encoder/decoder + FiLM conditioning layers
├─> _build_encoder: Downsampling path with conv blocks
├─> _build_decoder: Upsampling path with transposed conv
├─> _calculate_num_levels: Adaptive depth based on spatial dims
├─> forward: Spatial processing with FiLM modulation
└─> Usage: For 2D grid-based input data
```

---

## 10. CODE CONVENTIONS & PATTERNS

### A. Naming Conventions
**Files**: snake_case.py
- `finetune.py`, `prepare_forecasts_and_targets.py`
- `run_aurora.py`, `download_forecasts.py`

**Classes**: PascalCase
- `SimpleMLP`, `UNet`, `Batch`, `Metadata`

**Variables & Functions**: snake_case
- `lead_time_hours`, `training_vars`, `n_hidden_layers`
- `parse_args()`, `get_region_grid()`, `create_dataloader()`

**Constants**: UPPER_CASE
- `CLIMATE_ZONE_MAP`, `TOPO_ZONE_MAP`, `EXPERIMENTS` (in setup scripts)

### B. Argument Pattern
All main scripts use `argparse`:
```python
parser = argparse.ArgumentParser(description='...')
parser.add_argument('--data_dir', type=str, default="~/...")
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--region', type=str, default="india")
parser.add_argument('--training_vars', type=str, nargs='+', default=[...])
parser.add_argument('--lead_time_hours', type=int, nargs='+', default=[24])
parser.add_argument('--nn_architecture', type=str, choices=['mlp', 'unet'])
parser.add_argument('--mlp_hidden_dim', type=int, default=1024)
parser.add_argument('--unet_hidden_dim', type=int, default=256)
# ... more args
```

### C. Import Patterns
**Root-level imports**:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from helper_funcs import setup_directories, generate_output_path
```

**Module imports**:
```python
from finetuning.prepare_forecasts_and_targets import load_forecasts
from aurora import Batch, Metadata, Aurora, rollout
```

**Standard library grouping**:
1. Built-in modules (os, sys, argparse, etc.)
2. Third-party libraries (torch, xarray, numpy, etc.)
3. Local modules (helper_funcs, module imports)

### D. Data Flow Patterns
```python
# 1. Configuration from args
args = parse_args()

# 2. Directory setup based on environment
dirs = setup_directories()  # Uses socket.gethostname()

# 3. Path generation
output_path = generate_output_path(args)

# 4. Data loading with automatic download
data = load_forecasts(
    data_dir=args.data_dir,
    model_name=args.model_name,
    region=args.region,
    variables=args.training_vars,
    years=[2020, 2021],
    lead_time_hours=args.lead_time_hours
)

# 5. Model creation
model = SimpleMLP(input_dim, **model_params)

# 6. Training loop
train_model(model, train_loader, valid_loader, ...)

# 7. Inference
corrected_output = apply_correction(model, test_data)

# 8. Output saving
save_output(output_path, corrected_output)
```

### E. Error Handling Patterns
- **Path handling**: `os.path.expanduser()` for home directory
- **File checks**: `os.path.exists()` before operations
- **Directory creation**: `os.makedirs(path, exist_ok=True)`
- **Assertions**: Minimal; relies on exception raising
- **Warnings**: Uses `warnings.filterwarnings('ignore')` to suppress known issues

### F. Hyperparameter Patterns
**MLP hyperparameters**:
- `mlp_hidden_dim` (int): 512-2048 neurons per layer
- `mlp_num_layers` (int): 2-8 layers
- `mlp_dropout` (float): 0.0-0.3 dropout rate

**UNet hyperparameters**:
- `unet_hidden_dim` (int): 32-256 base channels
- `unet_dropout` (float): 0.0-0.2 dropout rate

**Training hyperparameters** (embedded in code):
- Batch size: 128
- Learning rate: Varies (uses scheduler)
- Epochs: 750 (configurable)
- Optimizer: Adam with learning rate decay

### G. Feature Engineering Patterns
**Temporal encoding** (in forward pass):
```python
# Day-of-year sinusoidal encoding
day_of_year_idx = [0-365]
day_sin = sin(2π * day_of_year_idx / 365)
day_cos = cos(2π * day_of_year_idx / 365)
# Added to input features

# Lead time embedding (for multiple lead times)
lead_time_idx = [0 to n_lead_times-1]
lead_time_emb = nn.Embedding(n_lead_times, embedding_dim)
```

**Architecture-specific**:
- MLP: Works with flattened inputs
- UNet: Works with 2D spatial grids, includes:
  - Encoder (downsampling)
  - Bottleneck
  - Decoder (upsampling)
  - Skip connections
  - FiLM conditioning (Feature-wise Linear Modulation)

---

## 11. ADDITIONAL PATTERNS & PRACTICES

### A. Output Path Naming Convention
Format: `{model_name}/{ground_truth_source}{region}/train_{training_vars}_test_{output_vars}_dim{subregion}_{lead_times}{grow_str}_{dates}_{model_str}.zarr`

Example: `pangu/india/train_2m_temperature_test_2m_temperature_dim2x2_leadtime_24_72_144h_train2020-01-01-2020-12-31_test2021-01-01-2021-06-30_mlp.zarr`

### B. Region Grid Coordinate System
- **Latitude**: 0.25-degree resolution
- **Longitude**: 0.25-degree resolution
- **Mapping**: Grid point size = degree_size / 0.25

### C. Loss Function Options
- **MSE** (default): Mean Squared Error
- **quantile_loss**: Emphasizes high-percentile predictions (0.95)
- **extreme_heat_loss**: Custom loss for temperature extremes

### D. Memory & Performance Monitoring
```python
def print_time_and_memory(step_name, start_time):
    """Tracks execution time and RAM usage"""
    elapsed = time.time() - start_time
    memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024
    print(f"  {step_name}: {elapsed:.2f}s | Memory: {memory:.2f} GB")
```

### E. Cloud Storage Integration
- **GCS**: Uses `gcsfs` and `google-cloud-storage`
- **Azure**: Uses `azure-storage-blob`
- **Local**: Standard file I/O with zarr/netcdf4

### F. Distributed Computing
- **Dask**: Used for parallel data loading
- **Dask Distributed**: Optional for cluster execution
- **Thread pools**: For concurrent downloads
- **Multiprocessing**: For multi-core operations

---

## 12. PROJECT MATURITY INDICATORS

### Active Development Areas
- Architecture experiments (conservative, moderate, aggressive branches)
- Dynamic data loading with incremental variable support
- Regional fine-tuning at 0.25° resolution
- Multi-lead-time training (24h, 72h, 144h)
- Hyperparameter optimization

### Archived/Legacy Code
- ECMWF forecast runners (run_ECMWF_forecasts/)
- WeatherBench2 evaluation (run_weatherbench2/)
- Old finetuning approaches (old_finetuning/)

### Well-Maintained Areas
- Core finetuning pipeline
- Aurora model integration
- Data downloading and preprocessing
- Documentation and examples

---

## 13. KEY METRICS & OUTPUTS

### Training Outputs
- **Per-epoch metrics**: Loss, validation loss, learning rate
- **Lead-time performance**: RMSE original vs. corrected
- **Improvement percentage**: (RMSE_orig - RMSE_corr) / RMSE_orig × 100

### Expected Improvements
- **Conservative architecture**: ~4-5% RMSE reduction
- **Moderate architecture**: ~5-6% RMSE reduction
- **Aggressive architecture**: Degrades performance

### Benchmark Performance (India 2x2, 2022 test data)
- MLP Deep (24h): 13.76% improvement
- UNet Deep (24h): 15.85% improvement
- Lead time correlation: Longer lead times show smaller improvements

