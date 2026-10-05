# Station-Level Post-Processing

Training forecast post-processing models against **real surface station
observations** instead of ERA5, and measuring how much of the rich/poor forecast
quality gap that closes.

---

## Why this exists

Three referees raised the same objection to the main paper: it verifies against
ERA5, and ERA5 is itself a model product that is weakest exactly where we claim
the largest gains — the data-sparse tropics. This folder is the analysis that
answers that objection, and it ended up reframing the paper.

The headline results:

1. **ERA5 verification understates the forecast inequality by roughly 3x.**
2. **Post-processing trained against ERA5 adds no information at all** — it is
   pure calibration.
3. **Post-processing trained against station observations does add information,
   and closes 33–45% of the inequality gap** for temperature.

---

## Metrics: why not just RMSE

RMSE conflates three different things, and the distinction turned out to drive
every conclusion here. Following Bonavita & Geer (2026, QJRMS):

```
RMSE² = bias² + information_error² + noise_error²
```

- **bias** — a constant offset. Any mean correction removes it.
- **noise error** — error perpendicular to the true anomaly. Shrinks when a
  forecast is *damped or smoothed*, which lowers RMSE without the forecast
  becoming more skilful.
- **information error** — error along the true anomaly. Measures whether the
  forecast can discriminate outcomes. Only a real gain in predictive
  information reduces it.

Anomaly correlation (ACC) is reported alongside because that is the metric in
which Linsenmeier & Shrader (2025) measure the income gap. ACC is invariant to
any constant offset, which makes it a clean test: a method that only removes
bias cannot move it.

---

## Findings

### 1. ERA5 hides most of the inequality

Raw Pangu ACC at 24h, 370 globally sampled stations:

| Variable | Truth | High income | LMIC | Gap |
|---|---|---|---|---|
| 2m temperature | **Station** | 0.773 | 0.622 | **+0.151** (p=5e-10) |
| 2m temperature | ERA5 | 0.964 | 0.916 | +0.048 |
| 10m wind | **Station** | 0.486 | 0.269 | **+0.217** (p=2e-17) |
| 10m wind | ERA5 | 0.893 | 0.790 | +0.103 |

Verifying against ERA5 shrinks the measured gap by about 3x for temperature and
2x for wind. At 9 days the ERA5-measured gap even flips sign. The station
numbers land almost exactly on Linsenmeier & Shrader's reported values (~0.74
high income, ~0.55 low income at 1 day), from a different model and a different
year — a useful independent check that the metric is computed comparably.

### 2. ERA5-trained post-processing is calibration, not information

With identical architecture and inputs, changing only the training target:

| Metric (temperature, 24h, LMIC) | raw | MLP, ERA5 target | MLP, station target |
|---|---|---|---|
| ACC | 0.788 | 0.789 | **0.849** |
| information error | 0.900 | **0.947 (worse)** | **0.692 (better)** |
| noise error | 1.281 | 1.246 | 1.161 |
| \|bias\| | 1.187 | 1.174 | 0.193 |
| RMSE | 2.159 | 2.149 | **1.399** |

The ERA5-trained model leaves ACC statistically unchanged and *raises*
information error — the Bonavita & Geer signature of a forecast that has been
damped rather than improved. Across every region, season and lead we tested, its
ΔACC was indistinguishable from zero or significantly negative, while forecast
activity dropped sharply (e.g. 0.98 → 0.79 at 9 days).

This is why the main paper's ERA5-measured gains did not transfer: the model was
learning ERA5's local bias, which in LMIC regions differs from the station's.

### 3. Station-trained post-processing closes a third of the gap

Held-out test year 2022, 307 stations, trained 2018–2021:

**2m temperature**

| Lead | raw | station-trained (tuned) | RMSE raw → tuned |
|---|---|---|---|
| 1 day | 0.8481 | **0.8932** | 2.104 → 1.494 K |
| 5 day | 0.7654 | **0.8178** | 2.763 → 2.155 K |
| 9 day | 0.5176 | **0.5725** | 3.999 → 3.301 K |

**10m wind speed**

| Lead | raw | station-trained (tuned) | RMSE raw → tuned |
|---|---|---|---|
| 1 day | 0.5238 | **0.6559** | 1.780 → 1.214 m/s |
| 5 day | 0.4028 | **0.5374** | 1.972 → 1.430 m/s |
| 9 day | 0.1791 | **0.2958** | 2.255 → 1.668 m/s |

Overall ΔACC vs raw: temperature **+0.0508** (p=7e-66), wind **+0.1278**
(p=4e-117), paired over 921 station-leads.

Both variables use forecast fields only. A 7-day lagged-bias predictor was
tested and gave wind a further +0.008 ACC, but it was dropped: it needs
*real-time* station data at forecast time, not just a historical record, which
would strengthen the infrastructure requirement the paper argues about.

**Effect on the income gap:**

| Variable | Lead | raw gap | tuned gap | closed |
|---|---|---|---|---|
| temperature | 1 day | 0.1188 | 0.0794 | **33.2%** |
| temperature | 5 day | 0.0975 | 0.0531 | **45.5%** |
| wind | 1 day | 0.2212 | 0.1788 | **19.2%** |
| wind | 5 day | 0.1363 | 0.0995 | **27.0%** |

The gap narrows because **LMIC stations gain about 2.5x more than high-income
ones** (+0.062 vs +0.024 ACC for temperature). A station's observations are
worth more, in skill terms, where the raw forecast is currently weakest.

> Caveat worth stating in the paper: roughly half the temperature gain is
> reachable with a simple day-of-year varying bias correction (which closes 21%
> on its own). The neural network adds the rest. For wind the network's margin
> over that baseline is much larger.

### 4. Negative results

Two hypotheses were tested and did not survive. Both are worth reporting.

**Teleconnection indices do not help.** Adding AO, NAO, PNA and Niño3.4 at
initialisation time degraded ACC in every region, season, lead and variable
(−0.003 to −0.010, mostly significant), including in the NH cold season where
Rust et al. (2015) find the strongest links. A control arm explains why: giving
the model a wider window of *Pangu's own forecast* beat the indices by 0.024 ACC
at 9 days. An index is a lossy, stale summary of a circulation field the model
already forecasts skilfully.

**Wider input windows do not help either.** Holding feature count constant, ACC
falls monotonically with window extent — mildly for temperature (optimum ≈ ±1°)
and steeply for wind (−0.08 ACC at ±22.75°, narrowest window best). Large-scale
context survives only as a small *supplement* at long lead (+0.005 ACC at 9 days
when added alongside local detail), never as a substitute for local resolution.

### 5. Tuned specification

Selected on a **development year (2021)**, never on the 2022 test set.

| | 2m temperature | 10m wind speed |
|---|---|---|
| Input | 5×5 grid-cell neighbourhood of t2m, u, v, wind speed | same |
| Hidden units | **32** | **128** |
| Layers | 2 | 2 |
| Dropout | 0.2 | 0.2 |
| Loss | Huber | Huber |
| Ensemble | 5 seeds averaged | 5 seeds averaged |
| Lead handling | joint, with one-hot lead indicator | same |

Gain over the untuned baseline: temperature **+0.0049** ACC (p=3e-55), wind
**+0.0075** (p=4e-11).

What the search taught us:

- **Ensembling is the most reliable single gain.** 5 seeds beat 1 everywhere;
  10 seeds added nothing further.
- **Capacity preferences are opposite by variable.** Temperature wants a *small*
  network (h128 and h256 were significantly worse than h32); wind wants a large
  one. Temperature error is close to a smooth local calibration; wind carries
  more learnable local structure.
- **Huber loss helps both**, consistent with station records containing outliers
  that squared error over-weights.
- **Per-lead-time models hurt** (−0.002 temperature, −0.007 wind), contradicting
  the ERA5-target finding in the main paper. With only ~1,450 station-days per
  lead, splitting the data costs more than specialisation gains.
- **Feature engineering mostly failed.** Persistence features significantly hurt
  temperature, and the lagged-bias term that helped wind was dropped for the
  real-time-data reason given above.

---

## Pipeline

```bash
# 1. Select stations, download HadISD, extract forecast/ERA5/climatology,
#    and attach each station's World Bank income group.
uv run python station_finetuning/prepare_station_dataset.py \
    --regions_file station_finetuning/configs/study_regions.json \
    --model_name pangu \
    --output_prefix station_dataset

# 2. Train every method at every qualifying station and score on the test year.
uv run python station_finetuning/train_station_models.py \
    --dataset   <processed>/station_finetuning/station_dataset.npz \
    --station_metadata <processed>/station_finetuning/station_dataset_stations.csv \
    --specifications station_finetuning/configs/tuned_specifications.json \
    --output    <processed>/station_finetuning/station_results.csv

# 3. Draw the paper figures from the sampled run (run_station_sample.sh).
uv run python station_finetuning/station_figures.py
```

Step 1 writes three files: `<prefix>.npz` (the small arrays),
`<prefix>_neighbourhoods.npy` (the forecast windows, memory-mapped by step 2)
and `<prefix>_stations.csv` (the metadata the figures and the income-gap
summary read).

Step 2 writes one row per station, variable, method and lead time. Feed that csv
to `verification.summarise_income_gap` to get the gap table in Finding 3.

| File | Purpose |
|---|---|
| `station_observations.py` | HadISD station discovery, download, 00 UTC parsing |
| `station_figures.py` | The four paper figures: ACC map + station-verified income gap (one per variable), ACC-change map (one per variable), gap for raw vs ERA5-trained vs station-trained, RMSE decomposition |
| `station_sampling.py` | Continent / income classification and balanced sampling |
| `select_station_sample.py` | Global stage: screens coverage, writes the sample manifest |
| `run_station_sample.sh` | Laptop driver: one continent at a time, frees disk between |
| `run_station_experiments.sh` | SLURM driver for the full global run |
| `forecast_features.py` | Pangu neighbourhoods, ERA5 values, climatology, grid elevation |
| `verification.py` | ACC / information error / noise error decomposition; income-gap summary |
| `train_station_models.py` | Per-station training, reference baselines, evaluation |
| `configs/` | Study-region boxes and tuned network specifications |

`train_station_models.py` fits every specification against **both** targets
(`mlp_station_*` and `mlp_era5_*`), so the controlled contrast in Finding 2 is
reproduced by default. Three reference methods bracket the networks:
`mean_bias`, `linear_mos` (both affine, so ACC must be unchanged — a built-in
check on the metric) and `seasonal_bias` (not affine, so it can move ACC).

A specification may carry a `variables` list to restrict it to one variable,
which is how the two different tuned architectures are bound to temperature and
wind in `configs/tuned_specifications.json`. Omitting the field runs it on both.

The station axis is positional, so the metadata csv must come from the same
`prepare_station_dataset.py` run. The identifiers are stored in the npz and
checked on load; a mismatch raises rather than silently pairing one station's
forecast with another's observations. The neighbourhood array is station-major
and lives in a `.npy` beside the npz rather than inside it, so training can
memory-map it and read one station at a time; a dataset built before that
change has to be rebuilt.

---

## Station data: HadISD

Observations come from **HadISD v3.4.3.2025f** (Met Office Hadley Centre; Dunn
et al. 2012, 2016), a selected, quality-controlled subset of NOAA's ISD. It
replaced raw ISD-Lite in October 2026 so the observations rest on a published,
citable QC suite rather than our own filtering. Results quoted above this
section were produced with ISD-Lite; the old manifest is kept as
`station_sample_manifest_isd_lite.csv`.

- Same `{usaf}-{wban}` identifiers as ISD, so everything downstream is keyed
  unchanged. Names and countries are joined from `isd-history.csv`.
- One netCDF per station (whole record, 2–5 MB compressed). On download each is
  reduced to its 00 UTC temperature and wind speed, with QC-rejected values
  (`-2e30`) set to NaN, and cached as `hadisd/{usaf}-{wban}.csv.gz`; the netCDF
  is deleted. The netCDFs are opened one at a time — HDF5 is not thread-safe.
- Not homogenised, and final: ISD stopped updating on 29 August 2025.
- Measured in Africa against ISD-Lite at 00 UTC: temperature QC removes almost
  nothing (values agree to 0.05 °C in 99.8% of shared observations); wind QC
  removes >5% of observations at 46 of 289 well-covered stations. HadISD's
  station selection is the real cost — 275 African stations clear the 70% bar,
  versus 390 in ISD-Lite.

**Regional boost.** On top of the 1,000-per-income-group balanced draw,
`select_station_sample.py` takes every usable station in Africa and in South
America outside Argentina (`sample_stratum == "regional_boost"`). The balanced
draw alone left most of these out because Asia dominates the
low-and-middle-income pool. Filter to `sample_stratum == "balanced"` for the
like-for-like income comparison.

## Scaling the station sample

The 307-station result above came from a draw restricted to the study-region
boxes. That restriction, not any quality filter, is what held the sample down.
Measured directly — 800 stratified ISD-Lite station-years downloaded and scored
with this module's own `load_station_observations`:

| | Active 2018-2022 | Clear the 70% 00 UTC bar | Usable |
|---|---|---|---|
| Tropics (\|lat\| < 23.5) | 2,801 | 49.2% ±4.9% | ~1,380 |
| Extratropics | 10,776 | 74.8% ±4.3% | ~8,055 |
| **Global** | **13,577** | — | **~9,150** |

Reweighted to the true population that is roughly 6,500 high-income and 2,650
LMIC stations, of which about 1,170 are tropical LMIC.

Relaxing the coverage bar is *not* the lever: dropping it from 70% to 50% adds
only about 5 percentage points, because the distribution is bimodal — 28% of
LMIC stations never report at 00 UTC at all, and most of the rest report
reliably. Widening the draw is the lever.

```bash
# Global draw, ~9,000 stations
uv run python station_finetuning/prepare_station_dataset.py \
    --station_selection global --model_name pangu \
    --parallel_downloads 32 --output_prefix station_dataset_global

# Tropics only, which is what the referee comments actually ask about
uv run python station_finetuning/prepare_station_dataset.py \
    --station_selection global --latitude_band -23.5 23.5 \
    --model_name pangu --output_prefix station_dataset_tropics
```

### The sampled run: 1,000 per income group, one continent at a time

Taking every usable station is not necessary to measure the income gap, and
15 GB of forecast neighbourhoods is awkward on a laptop. The default route is
instead a balanced sample — an equal number of high-income and
low-and-middle-income stations worldwide — built and trained one continent at a
time.

```bash
bash station_finetuning/run_station_sample.sh select    # once, globally
bash station_finetuning/run_station_sample.sh all       # each continent in turn
bash station_finetuning/run_station_sample.sh combine
```

`select` is the only global stage. It classifies every active station by income
group and continent, downloads HadISD for an oversampled pool, measures 00 UTC
coverage over the training years, and writes a manifest naming the chosen
stations. The quota has to be global — decided before any chunk is processed —
or each chunk would fill a quota of its own and the worldwide totals would not
hold. It reads no forecast data, and it screens on the training years only, so
the test year is fetched later for the stations that actually make the cut.

`all` then loops the chunks named in the manifest, and for each one prepares
its slice, trains it, and **deletes that chunk's neighbourhood array** (via
`train_station_models.py --drop_neighbourhoods`, so the pipeline owns the file's
lifetime rather than the shell). That deletion is the reason for the split: peak
disk is one chunk, not the whole sample. It is resumable — a chunk whose results
csv exists is skipped — and `continent <name>` runs a single one. `KEEP_ARRAYS=1`
keeps the arrays if you want to re-train against a different specification.

Chunk names are read from the manifest rather than hardcoded. Usually that is
the six continents; a handful of offshore stations that match no continent are
routed to an `unassigned` chunk rather than dropped, since the income quota has
already counted them.

**The only quota is income group.** `--stations_per_income_group 1000` gives a
2,000-station worldwide sample, 1,000 high income and 1,000 low and middle
income. Continents are a processing unit, not a stratum: no quota is applied to
them, so the chunks come out deliberately unequal. They could not be balanced
anyway — ISD has no high-income stations in Africa at all, and only 70 in South
America.

| Continent | High income | Low and middle income |
|---|---|---|
| Africa | 0 | 1,070 |
| Asia | 471 | 2,166 |
| Europe | 3,473 | 181 |
| North America | 3,768 | 229 |
| Oceania | 726 | 64 |
| South America | 70 | 940 |
| **Total candidates** | **8,509** | **4,653** |

### Will it run on a laptop?

Yes. Measured on an M-series Mac against the real archives:

| Stage | Cost |
|---|---|
| Forecast extraction, 5 years | **~1.5 min per continent** (~9 min for all six) |
| ERA5 extraction, 5 years | ~12 s per continent |
| HadISD download (`select`, one-off) | ~4,100 stations, ~0.3 MB cached each |
| Training | ~12 s per station, so ~1–3 h per continent |
| Peak disk | **~1 GB**, versus ~3 GB in one piece or 15 GB for the full global draw |

Extraction cost barely moves with the continent split because it is set by
reading the archive once per pass, not by the station count — which is also why
splitting into six passes costs six reads of the archive rather than one. At
~1.5 minutes a pass that is a fine trade for holding disk to a single continent;
it would not be on a slower filesystem.

Training dominates the wall clock and is the reason to go continent by
continent: each one finishes in a sitting, and the results csvs accumulate.

What this costs, measured rather than estimated:

- **Archive reads: unchanged.** `extract_forecast_neighbourhoods` reads one
  global slab per (valid time, lead, variable) and gathers every station from
  it, so 400 stations and 9,000 stations read the same bytes.
- **Training: ~76 core-hours** worst case for 9,150 stations (24 networks per
  station, 150 epochs, early stopping disabled), under two hours on a 40-core
  node. Stations are independent, so `run_station_experiments.sh` shards them
  across a SLURM array via `--shard_index` / `--shard_count`; the array size
  is the only knob.
- **Storage: ~1.6 MB per station**, so ~15 GB for the global draw. The array is
  written station-major straight into a memory-mapped `.npy`, so preparation
  holds one year at a time and training pages in only the station it is
  fitting. Peak resident memory stays in the low gigabytes at either scale.
- **Downloads: ~46,000 files.** `list_available_station_years` reads NOAA's
  per-year index once, so station-years that were never published are never
  requested; without it most of the wall time goes to waiting on 404s.

```bash
sbatch station_finetuning/run_station_experiments.sh prepare
sbatch station_finetuning/run_station_experiments.sh train
bash   station_finetuning/run_station_experiments.sh combine
```

## Method notes and caveats

- **Grid-to-station matching** is nearest neighbour, with a dry adiabatic lapse
  rate (0.0098 K/m) applied to temperature for the elevation difference between
  the grid cell mean and the station, following Trotta et al. (2025). This
  shifts the mean only, so it affects bias and RMSE but never ACC, IE or NE.
  Every lookup (forecast, ERA5, elevation, climatology, gridded patches) goes
  through `forecast_features.snap_to_grid_cell`, which rounds ties up and wraps
  longitude at 360, so a station exactly between two cells is read at the same
  cell in every archive whatever order it stores latitude in.
- **Forecast arrays are indexed by valid time**, not initialisation time — the
  archive convention was verified directly against raw ERA5.
- **Validation blocks are whole weeks.** Random daily splits leak, because
  weather is strongly autocorrelated day to day.
- **Anomalies use a 30-year (1990–2019) ERA5 climatology**: WeatherBench2's
  precomputed 00 UTC day-of-year climatology
  (`era5-hourly-climatology/1990-2019_6h_1440x721.zarr`), read at each
  station's nearest grid cell and applied to forecast and observation alike.
  `prepare_station_dataset.py` caches its 00 UTC slice once to
  `raw/era5_climatology_1990-2019_00utc.zarr` (~9 GB of reads). This follows
  Linsenmeier & Shrader, with two differences: they use 1991–2020 and a 7-day
  moving average, where WeatherBench2 uses a 61-day running window. A
  seasonally-varying ERA5-vs-station bias would leak into the station anomaly;
  a constant one cancels, because anomalies are centred.
- **00 UTC only.** All forecasts verify at 00 UTC, so stations reporting only at
  03/12 UTC are unusable. This restricts the sample to near-hourly reporters,
  mostly airports — probably the best-maintained sites in each country, which
  likely makes the measured inequality a *lower* bound.
- **One test year (2022)**, which was a persistent La Niña. Slowly varying
  indices are near-constant within it, so the teleconnection result tests daily
  regime conditioning rather than ENSO.
- **307 stations** clear the 70% training-coverage bar (156 high income, 150
  LMIC, 1 unlabelled). Attrition is mild and roughly symmetric across income
  groups, because these stations already passed the hourly-reporting filter.

## References

- Bonavita, M. & Geer, A.J. (2026). Forecast verification using information and
  noise. *QJRMS* 152:e70109.
- Linsenmeier, M. & Shrader, J. (2025). Global inequalities in weather forecasts.
- Rust, H.W. et al. (2015). Linking teleconnection patterns to European
  temperature. *Meteorologische Zeitschrift* 24(4).
- Trotta, J. et al. (2025). Statistical post-processing yields accurate
  probabilistic forecasts from AI weather models. arXiv:2504.12672.
- Xu, W. et al. (2024). WeatherReal: a benchmark based on in-situ observations.
  arXiv:2409.09371.
