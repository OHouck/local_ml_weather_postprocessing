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
# 1. Select stations, download ISD-Lite, extract forecast/ERA5/climatology,
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
```

Step 2 writes one row per station, variable, method and lead time. Feed that csv
to `verification.summarise_income_gap` to get the gap table in Finding 3.

| File | Purpose |
|---|---|
| `station_observations.py` | ISD station discovery, download, 00 UTC parsing |
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

The station axis of the npz is positional, so the metadata csv must come from
the same `prepare_station_dataset.py` run. The identifiers are stored in the
npz and checked on load; a mismatch raises rather than silently pairing one
station's forecast with another's observations.

---

## Method notes and caveats

- **Grid-to-station matching** is nearest neighbour, with a dry adiabatic lapse
  rate (0.0098 K/m) applied to temperature for the elevation difference between
  the grid cell mean and the station, following Trotta et al. (2025). This
  shifts the mean only, so it affects bias and RMSE but never ACC, IE or NE.
- **Forecast arrays are indexed by valid time**, not initialisation time — the
  archive convention was verified directly against raw ERA5.
- **Validation blocks are whole weeks.** Random daily splits leak, because
  weather is strongly autocorrelated day to day.
- **Anomalies use a 4-year (2018–2021) ERA5 climatology** with a 31-day smoothing
  window, applied to forecast and observation alike. Linsenmeier & Shrader use
  30 years. A seasonally-varying ERA5-vs-station bias would leak into the
  station anomaly; a constant one cancels, because anomalies are centred.
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
