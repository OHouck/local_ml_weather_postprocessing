# Tailoring machine learning weather predictions for local impacts

Code to reproduce "Tailoring machine learning weather predictions for local
impacts at a global scale" by Ozma Houck and James Franke (University of
Chicago).

Lightweight neural networks are trained to predict, and remove, the error of
global weather forecasts (Pangu-Weather and IFS HRES) on 6x6 degree land
patches, verified against gridded truth (ERA5 / the HRES analysis). A second
analysis trains the same kind of network at individual weather stations and
verifies against the station observations (see
[station_post_processing/README.md](station_post_processing/README.md)).

## Running it

```bash
uv run python main.py
```

`main.py` runs the nine sections below in order; comment out a line in
`main()` to skip a section. Every script can also be run on its own with
`uv run python <script>`. Data paths come from `common.setup_directories()`,
which recognises the machine by its hostname (add a new machine there).
Figures are written to `<data root>/erc_figures/`.

| # | Section | Script(s) | Approximate time (M-series Mac) |
|---|---|---|---|
| 1 | Download and prepare the gridded forecasts | `data_preparation/download_forecasts.py`, `data_preparation/create_land_patches.py` | skipped when the files exist |
| 2 | Download ERA5 and the IFS HRES analysis | `data_preparation/download_reanalysis.py` | skipped when the files exist |
| 3 | Download and prepare the station data | `data_preparation/select_stations.py`, `data_preparation/prepare_station_data.py` | ~10 min (~7 GB written) |
| 4 | Gridded hyperparameter search and architecture comparison | `gridded_post_processing/hyperparameter_search.py`, `gridded_post_processing/architecture_comparison.py` | days; off by default |
| 5 | Gridded post-processing | `gridded_post_processing/run_post_processing.py` | ~0.9 min per patch, ~22 h in total |
| 6 | Station specification search | `station_post_processing/specification_search.py` | ~1.5 h |
| 7 | Station post-processing | `station_post_processing/run_station_post_processing.py` | ~8 s per station, ~9 h in total |
| 8 | Gridded figures | `figures/gridded_figures.py` | ~4 min |
| 9 | Station figures | `figures/station_figures.py` | ~1 min |

Sections 5 and 7 resume where they stopped if interrupted.

## Layout

```
main.py                     runs every section
common.py                   data directories, study design, output file naming, land patches
income_groups.py            World Bank income groups of countries, grid cells and stations

data_preparation/           sections 1-3: downloads and dataset preparation
gridded_post_processing/    sections 4-5: patch-level networks trained against ERA5 / HRES t0
    models.py                   the MLP and U-Net architectures
    training_data.py            loads one patch's forecasts and truth as training arrays
    training.py                 training loops and the five model variants
    tuned_hyperparameters.json  best hyperparameters of every variant (written by section 4)
station_post_processing/    sections 6-7: per-station networks trained against observations
    station_training.py         features, network, error decomposition
    tuned_specifications.json   network specification per variable (written by section 6)
figures/                    sections 8-9: every paper figure
```

## Methods in brief

- **Gridded models** predict the forecast error of a whole 6x6 degree patch
  from the raw forecast patch, the day of year and the lead time (1, 5 and 9
  days). They train on 2018-2021 and are tested on 2022. The production model
  is an MLP snapshot ensemble of 3 runs. Validation samples are whole weeks,
  never single days, because weather is autocorrelated from day to day.
- **Station models** predict the forecast error at one station from the 5x5
  grid-cell forecast neighbourhood around it, trained with a mean squared
  error loss on the station's own observations (or, for comparison, on ERA5).
- Every forecast verifies at 00 UTC.
