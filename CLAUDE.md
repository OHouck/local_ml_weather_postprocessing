# CLAUDE.md — AI Weather Forecast Post-Processing

**Project**: "Tailoring machine learning weather predictions for local impacts"
**Authors**: Ozma Houck & James Franke (University of Chicago)
**Purpose**: Train computationally cheap neural networks to post-process weather forecast errors, improving regional forecast skill for local applications.

---

## Project Summary

The core idea: train lightweight neural networks (MLP or U-Net) to predict the **error** in existing global weather forecasts (Pangu-Weather, ECMWF IFS) and subtract that error. This is applied independently to 6×6 degree regional patches across global land surface.

Key findings in the paper:
- Mean RMSE improvement of ~10% for 2m temperature, ~14% for 10m wind speed
- Improvement is larger near the equator and in high-topography areas
- The simple MLP is as good as the U-Net and trains ~25× faster
- Adding more input variables or larger training domains does not improve accuracy

---

## Codebase Structure

```
ai_weather_ag/
├── finetuning/                              # PRIMARY MODULE
│   ├── finetune.py                          # Main training entry point
│   ├── prepare_forecasts_and_targets.py     # Data loading (load_forecasts)
│   ├── figures_finetuning.py                # All paper figure generation
│   ├── process_forecasts.py                 # Compute statistics across output files
│   ├── custom_loss_fns.py                   # Alternative loss functions
│   ├── hyperparam_tuning.py                 # Bayesian hyperparameter search
│   ├── hyperparam_tuning.sh                 # SLURM script for hyperopt
│   ├── clean_and_sample_climate_zones.py    # Bootstrap zone sampling
│   │
│   ├── run_experiments.sh                   # SLURM: main global experiment runs (continents)
│   ├── run_region_size_experiments.sh       # SLURM: region size ablation (Finland/Amazon)
│   ├── run_custom_loss_experiments.sh       # SLURM: custom loss experiments (Ethiopia/India)
│   ├── run_arch_experiments.sh              # SLURM: architecture experiment driver
│   ├── run_arch_experiments_eval.py         # Python: run arch experiments on 5% eval cell sample
│   ├── run_improvement_regression.py        # OLS regression of improvement on geographic features
│   │
│   ├── plot_maps_and_binscatters.py         # Script → paper Figs 1–3
│   ├── plot_arch_experiment_results.py      # Script → paper Fig 4
│   ├── plot_region_size_results.py          # Script → paper Fig 5
│   └── plot_custom_loss.py                  # Script → custom loss evaluation figures
│
├── helper_funcs.py                          # setup_directories(), generate_output_path(),
│                                            #   load_all_continent_patches(), sample_continent_patches()
├── hyperopt_results_*/                      # Saved Bayesian hyperopt results (JSON)
└── CLAUDE.md                                # This file
```

---

## Environment Setup

Data root is determined by hostname in `helper_funcs.setup_directories()`:
- **Mac (`oMac.local`)**: `/Users/ohouck/globus/forecast_data`
- **Midway3 cluster**: `/project/jfranke/ozma/forecast_data`

Adding a new machine requires editing `helper_funcs.py`.

Directory layout under the data root:
```
forecast_data/
├── raw/                    # Downloaded forecast zarrs (pangu/, ifs/, aifs/, era5/)
├── processed/
│   └── finetuning_output/  # Output zarrs, organized by model/region/
└── figures/                # Saved figure files (figs/pangu/, figs/ifs/, etc.)
```

---

## Three Key Files

### 1. `finetuning/finetune.py` — Training Script

Entry point for training a post-processing model on a region.

**Model classes defined here**:
- `SimpleMLP` — flattens spatial patch, concatenates day-of-year sin/cos and learned lead-time embedding, passes through fully connected layers; supports `small_output_init`
- `UNet` — encoder-decoder with skip connections; caps channels at 128; number of pooling levels auto-calculated from patch size
- `PooledFiLMMLP` — global model trained across all patches simultaneously; uses FiLM conditioning on a 4-dim region descriptor (sin/cos lat/lon); designed for multi-patch pooled training experiments
- `ClassifierMLP` — used only for classification-based loss experiments (e.g., heatwave duration)

**Key functions**:
- `parse_args()` — defines all CLI flags (see below)
- `get_region_grid(args)` — returns lat/lon arrays for named regions or global grids
- `train_model(...)` — training loop with Adam optimizer, ReduceLROnPlateau scheduler, early stopping, AMP on CUDA
- `train_model_cosine(...)` — cosine annealing training (used for snapshot ensembles)
- `train_model_weighted(...)` — training with optional lead-time loss weights and C-Mixup augmentation
- `train_swa_ensemble(...)` — stochastic weight averaging ensemble training
- `apply_correction(...)` — inference: predicts error and adds to raw forecast; supports MC dropout
- `save_output(...)` — writes corrected+original+ground_truth to zarr, organized by lead time
- `load_optimal_hyperparameters(arch, training_vars, output_vars, alternate_loss_fn, use_snapshot, use_block_ltho, use_per_lt)` — reads best params from appropriate `hyperopt_results_*/optimization_results_{arch}.json`
- `run_subregion_experiment(...)` — trains and evaluates a single patch; called by `run_arch_experiments_eval.py`

**The model predicts forecast error, not the weather value directly**:
```
corrected = raw_forecast + model(forecast_fields, lead_time, day_of_year)
```

**Named regions** with fixed center lat/lon (expanded by subregion size):
```python
REGION_CENTERS = {
    'india': (22.0, 77.0),
    'usa_south': (35.0, 260.0),
    'amazon': (-5.0, 295.0),
    'pakistan': (29.5, 65.0),
    'ethiopia': (9.0, 39.0),
    'corn_belt': (41.0, 270.0),
    'finland': (65.0, 29.0),
    ...
}
```

**Special region keywords** (use full global grid):
- `global`, climate zones (`tropical`, `arid`, `temperate`, `cold`, `polar`), topographic zones (`flat`, `hilly`, `mountainous`), continents (`africa`, `asia`, `europe`, `north_america`, `south_america`, `oceania`)

**CLI flags**:
```
--data_dir               Raw data directory
--output_dir             Where to write output zarrs (REQUIRED)
--model_name             pangu | ifs | aifs (REQUIRED)
--ground_truth_source    Alternate ground truth source (default: "")
--region                 Region name (default: india)
--subregion              Patch size, e.g. 6x6 (default: 2x2)
--lead_time_hours        List of lead times in hours, e.g. 24 120 216
--training_vars          Input variable(s), e.g. 2m_temperature
--output_vars            Variable(s) to correct, e.g. 2m_temperature
--train_start/end        Date range YYYY-MM-DD
--test_start/end         Date range YYYY-MM-DD
--nn_architecture        mlp | unet | gated_mlp (default: mlp)
--alternate_loss_fn      extreme_heat_loss | mortality_weighted_loss | quantile_loss |
                         heatwave_loss | joint_temp_wind_loss
--bootstrap              N  (run N bootstrap samples of subregions)
--growing_season_only    Filter training to growing season only
--pca_components         N  (reduce input dim via PCA before training; 0 = disabled)

# Ensemble methods
--ensemble               N  (train N seed-diverse MLPs, average predictions)
--snapshot_ensemble      N  (train N snapshot ensemble runs, recommended: 3)
--snapshot_epochs        Total epochs per snapshot run (default: 210)
--snapshot_T0            Cosine cycle period for snapshots (default: auto)
--snapshot_T_mult        Cosine annealing multiplier (default: 1)
--swa_ensemble           N  (stochastic weight averaging ensemble runs)
--swa_warmup_epochs      Warmup epochs before SWA (default: 150)
--swa_epochs             SWA averaging epochs (default: 60)
--swa_T0                 SWA cosine cycle period (default: 20)
--mc_dropout_samples     N  (MC dropout inference samples; 0 = disabled)

# Block leave-time-holdout ensemble
--block_ensemble         Train separate model per held-out year block
--block_holdout          N  years held out per block (default: 3)

# Per-lead-time training
--per_lead_time          Train separate model per lead time (works with snapshot and block ensembles)

# Advanced
--lead_time_loss_weights  Per-lead-time loss weights (space-separated floats)
--cmixup_alpha           C-Mixup data augmentation alpha (0 = disabled)
--small_output_init      Initialize output layer with small weights
--mlp_hidden_dim         (default: 1024)
--mlp_num_layers         (default: 2)
--mlp_dropout            (default: 0.244)
--unet_hidden_dim        (default: 64, max channels capped at 128)
--unet_dropout           (default: 0.1)
```

**Standard training periods by model**:
- Pangu / IFS: train 2018–2021, test 2022
- AIFS: train 2022–2023, test 2024

### 2. `finetuning/figures_finetuning.py` — Figure Generation

All paper figures come from functions in this file. The `plot_*.py` scripts call these functions.

**Functions that produce paper figures**:

| Function | Paper Figure | Description |
|----------|-------------|-------------|
| `map_global_improvements(pixel_level=True)` | Fig 1, Appendix maps | Global map of RMSE % improvement per pixel |
| `lead_time_compare_binscatter()` | Figs 2, 3 | Binscatter of improvement vs equator distance or SDOR, by lead time |
| `plot_rmse_improvement()` | Fig 4 | Bar chart comparing architectures/input configs |
| `generate_subregion_comparison_plots()` | Fig 5 | RMSE improvement vs training domain size (Finland/Amazon) |
| `model_compare_boxplot()` | Appendix Fig 6 | IFS vs Pangu improvement comparison boxplot |

**Additional figure functions**:
- `map_forecasts(...)` — maps of original vs corrected forecasts for a region; used for joint temp-wind model visualizations
- `plot_improvement_by_weather_bin(...)` — improvement vs weather value bin (evaluates custom loss functions; called by `plot_custom_loss.py`)
- `plot_arch_experiment_results(...)` — aggregated bar chart from eval-cell architecture experiments
- `map_arch_exeriment_regions(...)` — map showing which patches were used in arch experiments
- `plot_raw_forecast_values(...)` — raw forecast value distributions
- `plot_error_cutoff(...)` — error frequency above cutoff threshold
- `plot_scatter_forecast_improvement(...)` — scatter: improvement vs geographic features
- `model_compare_binscatter(...)` — binscatter comparing IFS vs Pangu

**Supporting functions**:
- `load_region_data(dirs, model, variable, regions, ...)` — loads all matching zarr files for given model/arch/subregion config, returns dict keyed by lead time
- `filter_patch_zarr_files(zone_dir, variable, ...)` — matches zarr files by filename pattern (dates, subregion, arch, loss fn)
- `validate_non_overlapping_patches()` — used in pixel-level map plotting to ensure tiles don't overlap

**Key dependencies**:
- `binsreg` library for binscatter plots (Figures 2 and 3)
- `cartopy` for map projections
- SDOR (standard deviation of orography) data from ERA5 for Figure 3

### 3. `finetuning/process_forecasts.py` — Statistics Aggregation

Reads output zarr files across all region/model/variable combinations and aggregates into a summary CSV. Used for structured comparison tables.

**Main function**: `calculate_and_save_statistics(dirs, models, variable_configs, ...)` → returns `pd.DataFrame`

Computes per-file: RMSE original, RMSE corrected, % improvement, extreme-heat RMSE, mean forecast values, error frequency above cutoff threshold.

For bootstrap regions (climate/topographic zones): aggregates across bootstrap samples with 95% CIs via t-distribution.

---

## Output File Naming Convention

Outputs are written to `{output_dir}/{model_name}/{region}/` with filename:
```
train_{training_vars}_test_{output_vars}_dim{subregion}_leadtime_{lead_times}h_{dates}_{arch}[_{loss_fn}][_{bootstrap_info}].zarr
```

Example:
```
pangu/india/train_2m_temperature_test_2m_temperature_dim6x6_leadtime_24_120_216h_train2018-01-01-2021-12-31_test2022-01-01-2022-12-31_mlp.zarr
```

`helper_funcs.generate_output_path(args)` generates this path. `filter_patch_zarr_files()` in `figures_finetuning.py` parses it back when loading results.

---

## Data Variables

**Primary variables for the paper**:
- `2m_temperature` — 2-meter air temperature (data in K; custom loss functions convert to Celsius internally)
- `10m_wind_speed` — 10-meter wind speed (m/s)

**Additional input variables supported**:
- `10m_u_component_of_wind`, `10m_v_component_of_wind`
- `temperature_1000hPa`, `specific_humidity_1000hPa`, `geopotential_1000hPa`
- Any variable with pattern `{variable}_{pressure}hPa` (parsed by `parse_atmospheric_variable()`)

**Output zarr variable naming** (organized by lead time):
```
{var}_original_lt{N}h       (time, latitude, longitude)
{var}_corrected_lt{N}h      (time, latitude, longitude)
{var}_ground_truth_lt{N}h
{var}_mean_corrected_lt{N}h  (mean-bias-corrected baseline)
```

---

## Hyperparameter Search

Hyperparameters are tuned via Bayesian optimization in `hyperparam_tuning.py`, run on a random 10% sample of continent patches. Results saved to mode-specific directories:

```
hyperopt_results_temperature_mlp/         # Single MLP (temperature)
hyperopt_results_wind_mlp/                # Single MLP (wind speed)
hyperopt_results_temperature_unet/        # UNet (temperature)
hyperopt_results_wind_speed_unet/         # UNet (wind speed)
hyperopt_results_snapshot_temperature_mlp/  # Snapshot ensemble (temperature)
hyperopt_results_snapshot_wind_mlp/         # Snapshot ensemble (wind speed)
hyperopt_results_multivar_temperature_mlp/  # Multi-variable input
hyperopt_results_multivar_temperature_unet/
hyperopt_results_per_lt_temperature_mlp/    # Per-lead-time MLP (temperature)
hyperopt_results_per_lt_wind_mlp/           # Per-lead-time MLP (wind speed)
hyperopt_results_block_ltho_temperature_mlp/ # Block leave-time-holdout ensemble
hyperopt_results_block_ltho_wind_mlp/
hyperopt_results_joint_wind_temperature_24h_mlp/  # Joint temp+wind model
```

`finetune.py` loads the appropriate set automatically via `load_optimal_hyperparameters()` based on the architecture and training mode flags, unless architecture flags are passed explicitly on the CLI.

---

## Loss Functions (`finetuning/custom_loss_fns.py`)

| Name | Use Case |
|------|----------|
| MSE (default) | Standard mean squared error |
| `extreme_heat_loss` | Penalizes errors on hot days more heavily (T>25°C: 6×, T>30°C: 11×) |
| `mortality_weighted_loss` | Weights errors by mortality risk curve (Carleton et al. 2022) |
| `quantile_loss` | Quantile regression |
| `heatwave_loss` | Duration-weighted MSE based on consecutive days above threshold |
| `joint_temp_wind_loss` | Jointly optimizes temperature and wind speed |

Custom losses that operate on Celsius (not normalized) values use `is_normalized=True` during training and `is_normalized=False` for evaluation.

---

## Common Workflows

### Train a post-processing model
```bash
python3 finetuning/finetune.py \
    --output_dir ~/data/fine_tuning_output \
    --model_name pangu \
    --region india \
    --subregion 6x6 \
    --training_vars 2m_temperature \
    --output_vars 2m_temperature \
    --lead_time_hours 24 120 216 \
    --train_start 2018-01-01 --train_end 2021-12-31 \
    --test_start 2022-01-01 --test_end 2022-12-31 \
    --nn_architecture mlp \
    --snapshot_ensemble 3 \
    --per_lead_time
```

### Run global experiments (continent patches, IFS)
```bash
sbatch finetuning/run_experiments.sh
```

### Run architecture experiments
```bash
sbatch finetuning/run_arch_experiments.sh   # calls run_arch_experiments_eval.py
```

### Run custom loss experiments
```bash
sbatch finetuning/run_custom_loss_experiments.sh
```

### Run region size ablation
```bash
sbatch finetuning/run_region_size_experiments.sh
```

### Run improvement regression analysis
```bash
python3 finetuning/run_improvement_regression.py
```

### Regenerate paper figures
```bash
python3 finetuning/plot_maps_and_binscatters.py   # Figs 1, 2, 3
python3 finetuning/plot_arch_experiment_results.py # Fig 4
python3 finetuning/plot_region_size_results.py     # Fig 5
python3 finetuning/plot_custom_loss.py             # Custom loss evaluation figures
```

### Run hyperparameter tuning
```bash
python3 finetuning/hyperparam_tuning.py \
    --model_name pangu --region india --subregion 6x6 \
    --training_vars 2m_temperature --output_vars 2m_temperature \
    --nn_architecture mlp --max_evals 100
```

---

## Important Notes

- **Don't hardcode data paths** — always use `setup_directories()` from `helper_funcs.py`
- **Adding a new machine**: edit the hostname check in `helper_funcs.setup_directories()`
- **MLP with snapshot ensemble + per_lead_time is the recommended approach**: `--snapshot_ensemble 3 --per_lead_time` trains separate models per lead time with 3-run snapshot ensembles. Single MLP trains ~25× faster than U-Net with equivalent accuracy on 6×6 patches
- **Extra input variables hurt or are neutral**: paper shows single-variable input is best for correcting 2m_temperature
- **Bootstrap regions**: climate/topographic zones use `--bootstrap N`; filenames contain `bs*` and `filter_patch_zarr_files` matches on that pattern
- **SDOR data** (standard deviation of orography from ERA5) must be loaded separately before calling `lead_time_compare_binscatter` with `x_metric="sdor"`
- **Paper uses 6×6 degree patches globally** — all continent-based training runs use `--subregion 6x6`
- **Continent patch sampling**: `helper_funcs.sample_continent_patches()` samples a reproducible fraction of patches from continent zarr outputs; `split='hyperopt'` and `split='eval'` produce disjoint subsets so hyperopt and arch-experiment evaluation don't overlap
