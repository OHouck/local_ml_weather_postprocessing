# Architecture Experiments for Pangu Forecast Improvement (v2.0)

## Overview

This comprehensive experiment suite compares **9 different configurations** to optimize post-processing of Pangu weather forecasts for the India region. The experiments test architecture choices (MLP vs UNet), architectural variations (wide/shallow vs skinny/deep, different channel sizes), and input variable requirements (full set vs minimal).

**New in v2.0:**
- Expanded from 4 to 9 experiments
- GPU training optimizations (mixed precision, faster data loading)
- Tests MLP depth vs width tradeoffs
- Tests UNet channel size variations
- Compares full variable set (6 vars) vs minimal (1 var)
- Enhanced metrics output (raw MSE + RMSE improvement + training time)

### Objective
Reduce RMSE of Pangu 2m temperature forecasts and understand the tradeoffs between:
1. **Architecture type**: MLP (flattens spatial data) vs UNet (preserves spatial structure)
2. **MLP shape**: Wide & shallow vs Skinny & deep
3. **UNet capacity**: Light (32ch) → Medium (64ch) → Heavy (128ch) → Very Heavy (256ch)
4. **Input complexity**: Full atmospheric profile vs temperature-only

### Region
- **Location**: India
- **Subregion**: 6x6 degrees (24 x 24 grid points at 0.25° resolution)
- **Center**: Approximately 20°N, 77°E

### Input Variables

**Full Set (6 variables):**
1. `2m_temperature` - Surface temperature
2. `10m_u_component_of_wind` - U wind component at 10m
3. `10m_v_component_of_wind` - V wind component at 10m
4. `temperature_1000hPa` - Temperature at 1000 hPa level
5. `specific_humidity_1000hPa` - Specific humidity at 1000 hPa
6. `geopotential_1000hPa` - Geopotential height at 1000 hPa

**Minimal Set (1 variable):**
1. `2m_temperature` - Surface temperature only

### Output Variable
- `2m_temperature` - Corrected surface temperature

### Lead Times
- **24h**: 1-day forecast
- **120h**: 5-day forecast
- **216h**: 9-day forecast

---

## Experiment Suite (9 Total)

### Group 1: MLP Architecture Variations (Full Variables)

#### 1.1 MLP Wide Shallow
**Configuration**: 3 layers × 2048 neurons, dropout 0.3

**Hypothesis**: Wide layers capture many features in parallel. Fewer layers means faster training and potentially better generalization through simplicity.

**Tradeoff**: Higher memory usage per layer, but fewer gradient steps.

#### 1.2 MLP Moderate (Baseline)
**Configuration**: 6 layers × 1024 neurons, dropout 0.25

**Hypothesis**: Balanced depth and width provides good expressiveness without being too complex. This is our baseline MLP configuration.

**Tradeoff**: Middle ground between width and depth.

#### 1.3 MLP Skinny Deep
**Configuration**: 8 layers × 512 neurons, dropout 0.2

**Hypothesis**: Deep networks with narrow layers can learn hierarchical representations with fewer parameters total. Better for learning complex transformations through composition.

**Tradeoff**: More gradient propagation steps, potential for vanishing gradients, but more parameter-efficient.

---

### Group 2: UNet Architecture Variations (Full Variables)

All UNets preserve spatial structure through encoder-decoder architecture with skip connections. They vary only in the number of base channels (which doubles at each downsampling level).

#### 2.1 UNet Light
**Configuration**: 32 base channels, dropout 0.1

**Hypothesis**: Lightweight UNet may be sufficient for the 6x6 degree region. Faster training and less overfitting risk.

**Capacity**: ~32 → 64 → 128 channels through encoder levels

#### 2.2 UNet Medium
**Configuration**: 64 base channels, dropout 0.1

**Hypothesis**: Medium capacity balances expressiveness and efficiency. Good starting point for spatial models.

**Capacity**: ~64 → 128 → 256 channels through encoder levels

#### 2.3 UNet Heavy
**Configuration**: 128 base channels, dropout 0.15

**Hypothesis**: More channels can capture finer spatial patterns and more complex atmospheric features.

**Capacity**: ~128 → 256 → 512 channels through encoder levels

#### 2.4 UNet Very Heavy
**Configuration**: 256 base channels, dropout 0.2

**Hypothesis**: Maximum capacity for learning detailed spatial corrections. May be overkill for this problem but establishes upper bound.

**Capacity**: ~256 → 512 channels through encoder levels (capped at 512)

---

### Group 3: Input Variable Comparison

Uses the best-performing architectures from Groups 1 & 2 but with minimal input (2m_temperature only).

#### 3.1 MLP Moderate (Minimal)
**Configuration**: 6 layers × 1024 neurons, 2m_temperature only

**Hypothesis**: Perhaps atmospheric variables don't add much value. Testing if surface temperature alone suffices for bias correction.

**Comparison**: Compare with Experiment 1.2 to quantify value of additional variables.

#### 3.2 UNet Medium (Minimal)
**Configuration**: 64 channels, 2m_temperature only

**Hypothesis**: Spatial structure (UNet) with minimal input may work well if corrections are primarily spatial rather than variable-dependent.

**Comparison**: Compare with Experiment 2.2 to quantify value of additional variables.

---

## Performance Optimizations (New in v2.0)

### GPU Training Enhancements

The training code has been optimized for modern GPUs while maintaining compatibility with CPU and Apple Silicon (MPS):

1. **Mixed Precision Training (AMP)**
   - Automatically enabled for CUDA GPUs
   - Uses float16 for forward/backward passes, float32 for critical operations
   - **Expected speedup**: 1.5-2x on modern GPUs (V100, A100, RTX 30xx/40xx)
   - No accuracy loss in testing

2. **Optimized Data Loading**
   - `pin_memory=True` for CUDA (faster CPU→GPU transfers)
   - `num_workers=4` for parallel data loading on GPU
   - `non_blocking=True` for asynchronous transfers
   - **Expected speedup**: 10-20% from reduced data loading bottlenecks

3. **cuDNN Benchmarking**
   - `torch.backends.cudnn.benchmark = True`
   - Auto-selects fastest convolution algorithms for your specific input sizes
   - **Expected speedup**: 5-15% for UNet architectures

4. **Backward Compatibility**
   - All optimizations automatically disabled for CPU/MPS
   - No code changes needed for different hardware

### Estimated Training Times

Based on 4 years training data (2018-2021), 1 year test data (2022):

| Hardware | MLP (per experiment) | UNet (per experiment) | Total (9 experiments) |
|----------|----------------------|-----------------------|-----------------------|
| **Modern GPU** (A100/V100) | 15-25 min | 30-45 min | 3-5 hours |
| **Consumer GPU** (RTX 3080) | 25-40 min | 50-75 min | 5-8 hours |
| **Apple M1/M2** (MPS) | 40-60 min | 80-120 min | 8-14 hours |
| **CPU** (16 cores) | 90-150 min | 180-300 min | 20-35 hours |

**Note**: Times include data loading. First experiment downloads data (~20-30 min), subsequent experiments reuse cached data.

---

## Quick Start

### 1. Run All Experiments

```bash
./run_architecture_experiments.sh
```

This will:
- Download data for India region (6x6) if not present
- Run all 9 experiments sequentially:
  - Group 1: MLP variations (3 experiments)
  - Group 2: UNet variations (4 experiments)
  - Group 3: Variable comparison (2 experiments)
- Automatically use GPU optimizations if CUDA is available
- Save logs to `~/ai_weather_ag/data/architecture_experiments/logs/`
- Save model outputs to `~/ai_weather_ag/data/architecture_experiments/`

### 2. Analyze Results

```bash
python3 analyze_architecture_results.py
```

This generates:
- `ARCHITECTURE_COMPARISON_REPORT.txt` - Comprehensive text report with:
  - Individual experiment results (grouped by category)
  - MSE and RMSE for each lead time
  - RMSE improvement percentages
  - Training time for each experiment
  - Comparison tables ranked by performance
  - Overall best architecture recommendations
- `results_summary.json` - Machine-readable summary for further analysis

### 3. Monitor Running Experiments

While experiments are running:
```bash
# Watch progress of current experiment
tail -f ~/ai_weather_ag/data/architecture_experiments/logs/[experiment_name]_*.log

# Check GPU usage (if using CUDA)
watch -n 1 nvidia-smi
```

---

## Understanding Results

### Metrics Explained

For each experiment and lead time, you'll see:

1. **MSE Original**: Mean Squared Error of raw Pangu forecast
2. **MSE Corrected**: MSE after applying the trained correction model
3. **RMSE Original**: Root Mean Squared Error (√MSE) of raw forecast
4. **RMSE Corrected**: RMSE after correction
5. **Improvement %**: `(RMSE_original - RMSE_corrected) / RMSE_original × 100%`
6. **Training Time**: Wall-clock time for model training (minutes)

### Example Output

```
Lead Time: 24h
Architecture               MSE Orig     MSE Corr     RMSE Orig    RMSE Corr    Improvement  Train Time
⭐ UNet Medium              8.234567     6.123456     2.869123     2.474567       13.76%      42.3 min
   MLP Moderate             8.234567     6.345678     2.869123     2.519234       12.19%      28.1 min
   MLP Wide Shallow         8.234567     6.456789     2.869123     2.541567       11.42%      31.5 min
```

### Key Questions to Answer

1. **MLP vs UNet**: Which architecture type performs better? Is spatial structure important?
2. **MLP Shape**: For MLPs, is it better to go wide & shallow or skinny & deep?
3. **UNet Capacity**: At what point do more channels stop helping (diminishing returns)?
4. **Variable Importance**: How much do atmospheric variables improve over temperature-only?
5. **Efficiency**: What's the best performance/training-time tradeoff?

---

## Running Individual Experiments

You can run individual experiments for testing or focused comparisons:

```bash
# Example: MLP Wide Shallow
python3 finetuning/finetune.py \
    --region=india \
    --subregion=6x6 \
    --model_name=pangu \
    --nn_architecture=mlp \
    --mlp_hidden_dim=2048 \
    --mlp_num_layers=3 \
    --mlp_dropout=0.3 \
    --training_vars 2m_temperature 10m_u_component_of_wind 10m_v_component_of_wind \
                    temperature_1000hPa specific_humidity_1000hPa geopotential_1000hPa \
    --output_vars 2m_temperature \
    --lead_time_hours 24 120 216 \
    --train_start=2018-01-01 --train_end=2021-12-31 \
    --test_start=2022-01-01 --test_end=2022-12-31 \
    --data_dir=~/ai_weather_ag/data/raw \
    --output_dir=~/ai_weather_ag/data/architecture_experiments
```

```bash
# Example: UNet Medium with minimal variables
python3 finetuning/finetune.py \
    --region=india \
    --subregion=6x6 \
    --model_name=pangu \
    --nn_architecture=unet \
    --unet_hidden_dim=64 \
    --unet_dropout=0.1 \
    --training_vars 2m_temperature \
    --output_vars 2m_temperature \
    --lead_time_hours 24 120 216 \
    --train_start=2018-01-01 --train_end=2021-12-31 \
    --test_start=2022-01-01 --test_end=2022-12-31 \
    --data_dir=~/ai_weather_ag/data/raw \
    --output_dir=~/ai_weather_ag/data/architecture_experiments
```

---

## Customization

### Testing Different Periods

```bash
# Shorter training for faster experimentation
--train_start=2020-01-01 --train_end=2020-12-31

# Longer test period for more robust evaluation
--test_start=2022-01-01 --test_end=2022-12-31
```

### Testing Different Regions

Edit `run_architecture_experiments.sh`:
```bash
REGION=odisha          # Other options: usa_south, pakistan, corn_belt, etc.
SUBREGION=4x4          # Smaller = faster training
```

### Custom Architecture Configurations

Add new experiments to `run_architecture_experiments.sh`:
```bash
# Example: Extra Deep MLP
run_experiment \
    "mlp_extra_deep" \
    "MLP Extra Deep: 12 layers × 512 neurons (Full vars)" \
    "mlp" \
    "${TRAINING_VARS_FULL}" \
    "--mlp_hidden_dim=512 --mlp_num_layers=12 --mlp_dropout=0.25"
```

Don't forget to update `analyze_architecture_results.py` EXPERIMENTS dictionary to include the new experiment!

---

## Troubleshooting

### Out of Memory (GPU)

If you hit GPU OOM errors:
1. Reduce batch size in `finetune.py` (currently 128)
2. Use lighter architectures (UNet Light, MLP Skinny Deep)
3. Reduce region size (`--subregion=4x4` instead of `6x6`)
4. Use CPU/MPS fallback (automatically detected)

### Slow Training

1. **Verify GPU usage**: Run `nvidia-smi` to confirm GPU is being used
2. **Check data loading**: First experiment downloads data (~20-30 min), others are faster
3. **Monitor GPU utilization**: Should be 70-95% during training
4. **Check I/O bottleneck**: SSD recommended for data storage

### No Results in Analysis

```bash
# Check if experiments completed
ls -lh ~/ai_weather_ag/data/architecture_experiments/logs/

# Verify log files contain results
grep "Lead time" ~/ai_weather_ag/data/architecture_experiments/logs/*.log

# Check for errors
grep -i "error\|failed" ~/ai_weather_ag/data/architecture_experiments/logs/*.log
```

---

## Technical Details

### Model Architectures

**SimpleMLP**:
- Input: Flattened spatial data + temporal features (day-of-year sin/cos) + lead time embedding
- Hidden layers: Fully connected with ReLU and dropout
- Output: Predicted error to add to forecast
- Training: Residual learning (predicts correction, not absolute value)

**UNet**:
- Input: Spatial data (channels × height × width) + temporal conditioning via FiLM
- Encoder: Progressive downsampling with skip connections
- Decoder: Progressive upsampling with skip connections
- FiLM conditioning: Feature-wise linear modulation of each encoder level using temporal features
- Output: Predicted spatial error field
- Training: Residual learning with spatial structure preservation

### Training Configuration

- **Optimizer**: Adam (lr=8.67e-6, weight_decay=5.21e-6)
- **Scheduler**: ReduceLROnPlateau (patience=10, factor=0.5, min_lr=1e-7)
- **Early Stopping**: Patience=50 epochs, min_delta=1e-5
- **Loss Function**: MSE (Mean Squared Error)
- **Validation Split**: 80/20 train/val split
- **Data Augmentation**: None (temporal shuffling only)

### GPU Optimizations Technical Details

**Mixed Precision (AMP)**:
```python
# Automatically applied for CUDA
with torch.cuda.amp.autocast():
    predictions = model(inputs)
    loss = criterion(predictions, targets)
scaler.scale(loss).backward()
```

**Data Loading**:
```python
DataLoader(
    dataset,
    batch_size=128,
    pin_memory=True if cuda else False,  # Faster CPU→GPU
    num_workers=4 if cuda else 0,         # Parallel loading
    shuffle=True
)
```

**Non-blocking Transfers**:
```python
inputs = inputs.to(device, non_blocking=True)  # Async GPU transfer
```

---

## References

- **Pangu-Weather**: [https://github.com/198808xc/Pangu-Weather](https://github.com/198808xc/Pangu-Weather)
- **WeatherBench2**: [https://weatherbench2.readthedocs.io/](https://weatherbench2.readthedocs.io/)
- **ERA5 Reanalysis**: [https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5)
- **Mixed Precision Training**: [https://pytorch.org/docs/stable/amp.html](https://pytorch.org/docs/stable/amp.html)
- **FiLM (Feature-wise Linear Modulation)**: [https://arxiv.org/abs/1709.07871](https://arxiv.org/abs/1709.07871)

---

## Changelog

### v2.0 (Current)
- Expanded from 4 to 9 experiments
- Added MLP depth vs width comparison
- Added UNet channel size sweep
- Added input variable ablation study
- Implemented GPU training optimizations (AMP, optimized data loading)
- Enhanced metrics output (MSE + RMSE + training time)
- Updated analysis script with grouped results and comprehensive comparisons

### v1.0
- Initial 4-experiment setup
- Basic MLP and UNet comparison
- Manual hyperparameter selection

---

**Version**: 2.0
**Last Updated**: 2025-01-17
**Author**: Ozma Houck
