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

> The numbers in this section come from earlier runs (ISD-Lite observations,
> Huber loss, Pangu only). The current pipeline uses HadISD, a mean squared
> error loss, and runs both Pangu and IFS; rerun sections 6-7 of `main.py`
> and refresh these tables from `station_results_{pangu,ifs}.csv`.

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

### 5. Network specification

The specification of each variable is chosen by
`specification_search.py` (section 6) on a **development year (2021)**,
never on the 2022 test set: every combination of hidden-layer width (32, 64,
128, 256), number of hidden layers (1-3) and dropout (0.1, 0.2) is trained on
2018-2020 as a 5-seed ensemble, with a mean squared error loss, and scored by
its mean ACC over an income-balanced sample of 100 stations. The best one per
variable is written to `tuned_specifications.json`, which section 7 reads.

Until the search is rerun, that file holds the earlier tuned values:

| | 2m temperature | 10m wind speed |
|---|---|---|
| Input | 5x5 grid-cell neighbourhood of t2m, u, v, wind speed | same |
| Hidden units | 32 | 128 |
| Layers | 2 | 2 |
| Dropout | 0.2 | 0.2 |
| Loss | mean squared error | mean squared error |
| Ensemble | 5 seeds averaged | 5 seeds averaged |
| Lead handling | joint, with one-hot lead indicator | same |

What the earlier search (on Huber loss) found:

- **Ensembling is the most reliable single gain.** 5 seeds beat 1 everywhere;
  10 seeds added nothing further.
- **Capacity preferences are opposite by variable.** Temperature wants a small
  network; wind wants a large one.
- **Per-lead-time models hurt** (-0.002 temperature, -0.007 wind): with only
  ~1,450 station-days per lead, splitting the data costs more than
  specialisation gains.
- **Feature engineering mostly failed.** Persistence features hurt
  temperature, and a lagged-bias term that helped wind was dropped because it
  needs real-time station data at forecast time.

---

## Pipeline

Sections 3, 6, 7 and 9 of `main.py`:

| Section | Script | What it does |
|---|---|---|
| 3 | `data_preparation/select_stations.py` | Lists every active HadISD station, labels it with its income group and continent, downloads and screens an oversampled pool for 00 UTC coverage, and writes the sample manifest (1,000 stations per income group plus the regional boost) |
| 3 | `data_preparation/prepare_station_data.py` | For Pangu and IFS: the 5x5 forecast neighbourhood at every station, ERA5 at the station, the station observations, the ERA5 climatology and the grid-cell elevation |
| 6 | `station_post_processing/specification_search.py` | Chooses each variable's network specification on the 2021 development year |
| 7 | `station_post_processing/run_station_post_processing.py` | Trains every qualifying station on 2018-2021 and scores raw, linear MOS, and the network trained on stations and on ERA5, on 2022 |
| 9 | `figures/station_figures.py` | The four station figures (and their appendix versions) |

Shared pieces (features, network, error decomposition) are in
`station_post_processing/station_training.py`.

Outputs, in `processed/station_finetuning/`:

- `station_sample_manifest.csv`: the chosen stations
- `station_dataset_stations.csv`, `station_dataset_{model}.npz` and
  `station_dataset_{model}_neighbourhoods.npy`: the training data. The station
  axis is positional; the station identifiers are stored with the arrays and
  checked on load. The neighbourhoods (~1.6 MB per station and model) are
  memory-mapped, so training reads one station at a time.
- `station_results/{model}_{continent}.csv`, combined into
  `station_results_{model}.csv`: one row per station, variable, method and
  lead time.

The two networks are identical except for their training target, which
reproduces the controlled contrast in Finding 2. Linear MOS is affine, so it
cannot change ACC: a built-in check on the metric.

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
`select_stations.py` takes every usable station in Africa and in South
America outside Argentina (`sample_stratum == "regional_boost"`). The balanced
draw alone left most of these out because Asia dominates the
low-and-middle-income pool. Filter to `sample_stratum == "balanced"` for the
like-for-like income comparison.

## Method notes and caveats

- **Grid-to-station matching** is nearest neighbour, with a dry adiabatic lapse
  rate (0.0098 K/m) applied to temperature for the elevation difference between
  the grid cell mean and the station, following Trotta et al. (2025). This
  shifts the mean only, so it affects bias and RMSE but never ACC, IE or NE.
  Every lookup (forecast, ERA5, elevation, climatology, gridded patches) goes
  through `prepare_station_data.snap_to_grid_cell`, which rounds ties up and wraps
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
