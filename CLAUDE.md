# CLAUDE.md — AI Weather Forecast Post-Processing

**Project**: "Tailoring machine learning weather predictions for local impacts" (submitted to *Environmental Research Letters* — "the ERL paper")
**Authors**: Ozma Houck & James Franke (University of Chicago)
**Purpose**: train computationally cheap neural networks to post-process weather forecast errors, improving regional forecast skill, with a focus on the global inequality in forecast quality (high-income vs. low/middle-income countries).

---

## Project summary

Lightweight networks predict the **error** of a global forecast (Pangu-Weather, ECMWF IFS HRES) and add it back:
```
corrected = raw_forecast + model(forecast, lead_time, day_of_year)
```

Two analyses:
- **Gridded** (main paper): one network per 6x6 degree land patch, trained against ERA5 (Pangu) or the HRES t=0 analysis (IFS). Mean RMSE improvement ~10% (2m temperature), ~14% (10m wind); larger near the equator and in high topography; the simple MLP is as good as a U-Net.
- **Station** (referee response): one network per HadISD station, trained against the station's observations. ERA5 verification understates the rich/poor skill gap ~3x; ERA5-trained post-processing is calibration only, while station-trained post-processing adds information. [`station_post_processing/README.md`](station_post_processing/README.md) is the source of truth for this analysis.

---

## Layout and how to run

`uv run python main.py` runs the nine sections in order; each section is one script with a `run()` function (comment lines out of `main()` to skip them, or run a script directly). See [`README.md`](README.md) for the table of sections, scripts and run times.

```
main.py                     runs every section
common.py                   setup_directories(), study design constants, gridded output naming,
                            land patch lists (load_continent_patches, sample_continent_patches)
income_groups.py            World Bank FY25 income groups via Natural Earth polygons
data_preparation/           1 download_forecasts, create_land_patches
                            2 download_reanalysis (ERA5, HRES t0, era5_static.nc via CDS)
                            3 select_stations (HadISD manifest), prepare_station_data
gridded_post_processing/    models.py (MultilayerPerceptron, UNet), training_data.py (load_patch_data),
                            training.py (variants, week split, output dataset),
                            tuned_hyperparameters.json
                            4 hyperparameter_search, architecture_comparison
                            5 run_post_processing (production: mlp_snapshot3, Pangu + IFS)
station_post_processing/    station_training.py (features, network, error decomposition),
                            tuned_specifications.json
                            6 specification_search   7 run_station_post_processing
figures/                    8 gridded_figures   9 station_figures  -> <root>/erc_figures/
```

Python: always `uv run python ...`; add packages with `uv add`.

---

## Data

`common.setup_directories()` picks the data root by hostname (`oMac.local` → `/Users/ohouck/globus/forecast_data`, Midway3 → `/project/jfranke/ozma/forecast_data`). Keys: `raw`, `processed`, `gridded_output` (`processed/finetuning_output`), `station_output` (`processed/station_finetuning`), `figures` (`<root>/erc_figures`). Never hardcode paths.

- `raw/{pangu,ifs}_{year}.zarr`: forecasts indexed by **valid time**, dims (valid_time, prediction_timedelta, latitude, longitude). Older archives also hold 12 UTC valid times and 36/132/228 h leads; the code only uses 00 UTC and 24/120/216 h.
- `raw/{era5,hres_t0}_{year}.zarr`: gridded truth, dims (time, latitude, longitude).
- `processed/{continent}_patches.npy`: 373 land patches, (n, 2, 24) lat/lon arrays; patch numbers are 1-based.
- Gridded outputs: `finetuning_output/{model}/{continent}/train_{var}_test_{var}_dim6x6_leadtime_24_120_216h_train2018-01-01-2021-12-31_test2022-01-01-2022-12-31_{variant}_{continent}_bs{N}.zarr` (built by `common.gridded_output_path`), with `{var}_{original,corrected,mean_corrected,ground_truth}_lt{N}h`. Outputs from the current pipeline carry a `validation_split` attribute; older ones do not and are retrained by section 5.

Train 2018–2021, test 2022, lead times 24/120/216 h, everything at 00 UTC.

---

## Conventions to preserve

- **Validation is never random days.** Gridded models hold out 20% of whole weeks of valid days (`training.split_training_and_validation_by_week`); block LTHO holds out whole years; station models hold out whole weeks (`station_training.validation_week_blocks`).
- **Model variants** are the keys of `common.MODEL_VARIANTS` (which also name the output files): `mlp`, `unet`, `mlp_snapshot3` (production, `common.PRODUCTION_MODEL_SUFFIX`), `mlp_snapshot3_perlt`, `mlp_blockk3_snapshot1`. Their hyperparameters live in `gridded_post_processing/tuned_hyperparameters.json`; IFS uses the Pangu-tuned values.
- **Study design constants** (years, lead times, variables, coverage bar, neighbourhood width) live only in `common.py`; import them rather than redefining.
- The architecture-comparison evaluation patches are `sample_continent_patches(fraction=0.05, seed=PATCH_SAMPLE_SEED, split="eval")`; they share files with the production run for `mlp_snapshot3`. Note they are a subset of the hyperparameter-search sample (fraction 0.25), because the two splits are sized from different fractions.
- Station↔grid matching always goes through `prepare_station_data.snap_to_grid_cell` (nearest cell, ties round up). Temperature is moved to station height with the dry adiabatic lapse rate (raw forecast only).
- The station neighbourhood `.npy` must stay paired with its `.npz`/`station_dataset_stations.csv`; the station IDs are checked on load.
- Station networks train with mean squared error loss; each station trains a station-target and an ERA5-target network with the same specification (the controlled contrast).

## Code style for this repo

Human-readable over clever: long descriptive names, no abbreviations, inline comments describing each block, a function only for snippets over ~10 lines, a header docstring in every script and a docstring (what / inputs / outputs) on every function. Scripts are self-contained; shared code lives in `common.py`, `income_groups.py`, `gridded_post_processing/training*.py` and `station_post_processing/station_training.py`.
