# Continent-Based Patch Training Guide

## Overview

The finetuning system now supports training models on continent-based 6×6 degree patches, similar to how climate and topographic zones are handled. This allows you to train separate models for each land patch within a continent.

## Changes Made

### 1. Added Continent Mapping (`finetune.py`)

```python
CONTINENT_MAP = {
    'africa': 1,
    'asia': 2,
    'europe': 3,
    'north_america': 4,
    'south_america': 5,
    'oceania': 6,
}
```

### 2. Updated Region Handling

- Modified `get_region_grid()` to recognize continents and return global grid
- Updated `main()` to load continent patches from `{continent}_patches.npy` files
- Patches are loaded from: `{processed_dir}/{continent}_patches.npy`

### 3. Patch File Naming Convention

Unlike climate zones which use a subregion suffix (e.g., `climate_zone_patches_tropical_2x2.npy`), continent patches are saved as:
- `africa_patches.npy`
- `asia_patches.npy`
- `europe_patches.npy`
- `north_america_patches.npy`
- `south_america_patches.npy`
- `oceania_patches.npy`

## Prerequisites

Before running continent-based training, ensure you have created the continent patches by running:

```bash
python3 finetuning/clean_and_sample_climate_zones.py
```

This will:
1. Load the land-sea mask
2. Divide the world into 6×6 degree patches
3. Filter patches with >50% land coverage
4. Classify each patch by continent
5. Save patches to `{processed_dir}/{continent}_patches.npy`

## Usage

### Example Command for Africa

```bash
python3 finetuning/finetune.py \
    --output_dir ~/ai_weather_ag/data/fine_tuning_output \
    --model_name pangu \
    --region africa \
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

### Example Command for Asia

```bash
python3 finetuning/finetune.py \
    --output_dir ~/ai_weather_ag/data/fine_tuning_output \
    --model_name pangu \
    --region asia \
    --subregion 6x6 \
    --training_vars 2m_temperature \
    --output_vars 2m_temperature \
    --lead_time_hours 24 72 144 \
    --train_start 2020-01-01 --train_end 2020-12-31 \
    --test_start 2021-01-01 --test_end 2021-06-30 \
    --nn_architecture unet \
    --unet_hidden_dim 64 \
    --unet_dropout 0.1 \
    --data_dir ~/ai_weather_ag/data/raw
```

## What Happens During Training

When you specify a continent as the region:

1. **Load Patches**: The script loads all patches for that continent from `{continent}_patches.npy`
2. **Iterate Over Patches**: For each patch (numbered 1 to N):
   - Extract the lat/lon bounds from the patch
   - Load forecast and target data for that specific patch
   - Train a model on that patch
   - Save the output as `{output_path}_{continent}_bs{patch_number}.zarr`

3. **Output Files**: Results are saved with the continent name and patch number, e.g.:
   - `pangu_africa_mlp_..._bs1.zarr`
   - `pangu_africa_mlp_..._bs2.zarr`
   - ...
   - `pangu_africa_mlp_..._bsN.zarr`

## Number of Patches Per Continent

The number of patches varies by continent based on land coverage:
- **Africa**: ~70-100 patches
- **Asia**: ~150-200 patches
- **Europe**: ~50-80 patches
- **North America**: ~80-120 patches
- **South America**: ~60-90 patches
- **Oceania**: ~30-50 patches

(Exact numbers depend on the 6×6 degree grid and >50% land threshold)

## Differences from Climate Zones

| Feature | Climate Zones | Continents |
|---------|---------------|------------|
| Patch size | 2×2 degrees (configurable) | 6×6 degrees (fixed) |
| Number of patches | 50 (sampled) | Variable (all land patches) |
| File naming | `climate_zone_patches_{zone}_{size}.npy` | `{continent}_patches.npy` |
| Coverage | Global, zone-based sampling | Continent-specific, exhaustive |
| Use case | Climate-specific training | Geographic coverage |

## Batch Processing All Continents

To train models for all continents, you can create a batch script:

```bash
#!/bin/bash
# run_continent_experiments.sh

CONTINENTS=("africa" "asia" "europe" "north_america" "south_america" "oceania")
OUTPUT_DIR=~/ai_weather_ag/data/fine_tuning_output
DATA_DIR=~/ai_weather_ag/data/raw

for continent in "${CONTINENTS[@]}"; do
    echo "Starting training for ${continent}..."

    python3 finetuning/finetune.py \
        --output_dir ${OUTPUT_DIR} \
        --data_dir ${DATA_DIR} \
        --model_name pangu \
        --region ${continent} \
        --subregion 6x6 \
        --training_vars 2m_temperature \
        --output_vars 2m_temperature \
        --lead_time_hours 24 72 144 \
        --train_start 2020-01-01 --train_end 2020-12-31 \
        --test_start 2021-01-01 --test_end 2021-06-30 \
        --nn_architecture mlp \
        --mlp_hidden_dim 1024 \
        --mlp_num_layers 6 \
        --mlp_dropout 0.25

    echo "Completed training for ${continent}"
done
```

## Monitoring Progress

The script will:
1. Print the number of patches loaded for the continent
2. For each patch, print the lat/lon range being processed
3. Skip patches that already have output files (resume capability)
4. Print training progress, validation loss, and test metrics for each patch

## Verification

To verify the patches were created correctly:

```bash
# Check if continent patch files exist
ls -lh ~/ai_weather_ag/data/processed/*_patches.npy

# Load and inspect a continent's patches in Python
python3 << 'EOF'
import numpy as np
patches = np.load('~/ai_weather_ag/data/processed/africa_patches.npy', allow_pickle=True)
print(f"Number of Africa patches: {len(patches)}")
print(f"First patch lat range: {patches[0][0].min():.2f} to {patches[0][0].max():.2f}")
print(f"First patch lon range: {patches[0][1].min():.2f} to {patches[0][1].max():.2f}")
EOF
```

## Troubleshooting

### Issue: "Unknown region" error
**Solution**: Ensure you're using exact continent names: `africa`, `asia`, `europe`, `north_america`, `south_america`, `oceania` (lowercase, underscores for multi-word)

### Issue: Patch file not found
**Solution**: Run `clean_and_sample_climate_zones.py` first to generate the patch files

### Issue: Running out of memory
**Solution**:
- Use smaller batch sizes (edit `batch_size` in `finetune.py`)
- Use lighter architectures (`--mlp_num_layers 3` or `--unet_hidden_dim 32`)
- Process continents sequentially rather than in parallel

### Issue: Training takes too long
**Solution**:
- Use shorter training periods (e.g., 1 year instead of 4 years)
- Reduce number of epochs (edit `num_epochs` in `finetune.py`)
- Use GPU if available

## Future Enhancements

Potential improvements to consider:
1. Add support for sub-continental regions (e.g., East Africa, South Asia)
2. Implement parallel processing of patches using multiprocessing
3. Add visualization of patch coverage and training results by continent
4. Support for custom patch sizes (currently fixed at 6×6 degrees)

## Related Files

- `finetuning/finetune.py`: Main training script (modified)
- `finetuning/clean_and_sample_climate_zones.py`: Patch creation script
- `finetuning/prepare_forecasts_and_targets.py`: Data loading module
- `helper_funcs.py`: Directory setup utilities

## Questions?

For issues or questions about continent-based training, check:
1. CLAUDE.md - Main documentation
2. UPDATED_FEATURES_SUMMARY.md - Feature updates
3. GitHub issues at https://github.com/OHouck/ai_weather_ag
