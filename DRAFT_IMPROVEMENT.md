# Draft Improvement Plan

**Target journal**: Environmental Research Letters
**Draft location**: `/Users/ohouck/globus/forecast_data/draft.typ`
**Date**: 2026-04-01

---

## Change 1: Add Mean-Bias-Correction Baseline to Figure 4

**What**: Add a mean-bias-correction (climatological mean error subtraction) baseline bar to the architecture comparison bar chart (Figure 4). This shows how much of the NN improvement comes from correcting the mean bias vs learning nonlinear error patterns.

**Why**: Reviewers will ask whether a simple mean correction achieves most of the improvement. The data already exists — output zarr files contain `{var}_mean_corrected_lt{N}h` variables.

**Code changes**:
- **`finetuning/figures_finetuning.py`** → `plot_arch_experiment_results()` (line ~1862):
  - After loading each experiment's zarr, also extract the `mean_corrected` variable via `extract_forecast_data()` (which already returns it as the 4th element).
  - Compute RMSE for mean-corrected forecasts and their % improvement.
  - Add a single "Mean Bias Correction" bar group (gray, no hatch) to the plot alongside the 4 existing experiment bars.
  - Add it to the legend.
- **`finetuning/plot_arch_experiment_results.py`**: No changes needed (calls the function above).

**Data needed**: Already available in existing zarr files. The `mean_corrected` variable is computed during training and saved by `save_output()` in `finetune.py`.

**Draft text changes** (`draft.typ`):
- Update Section 3.2 text to mention the mean-bias-correction baseline and interpret the gap between the mean correction bar and the NN bars.
- Update Figure 4 caption to describe the new baseline bar.

---

## Change 2: Regression of Improvement on Geographic Features

**What**: Run OLS regression: `improvement_pct ~ abs(latitude) + SDOR + climate_zone_dummies + baseline_rmse + lead_time_dummies`. Report results in a table in the paper.

**Why**: Formalizes the spatial heterogeneity claims currently made only via binscatter visual inspection (Figures 2–3). Provides coefficients, standard errors, and significance levels. Strengthens the paper's central interpretive claim that local weather patterns drive post-processing skill.

**Code changes**:
- **New script**: `finetuning/run_improvement_regression.py`
  - Uses `load_region_data()` and `_extract_pixel_level_data()` from `figures_finetuning.py` to get pixel-level improvement, RMSE, latitude, and SDOR for both variables across all lead times.
  - Assigns each pixel a Köppen climate zone using `clean_and_sample_climate_zones.py` utilities or a simple latitude-band heuristic.
  - Runs `statsmodels.api.OLS` with robust (HC1) standard errors.
  - Outputs a formatted regression table (saved as a CSV/LaTeX file) and prints summary.
  - Runs separate regressions for 2m_temperature and 10m_wind_speed.

**Data needed**:
- Pixel-level data already loadable via `load_region_data()` + `_extract_pixel_level_data()`.
- SDOR from `era5_static.nc` (already used by binscatter code).
- Climate zone classification: either from existing `clean_and_sample_climate_zones.py` or derived from latitude bands.

**Draft text changes** (`draft.typ`):
- Add a new paragraph in Section 3.1 (after the binscatter discussion) presenting the regression results.
- Add a regression table (Table 1 in main text, or in Appendix).
- Reference the table when discussing spatial heterogeneity.

---

## Change 3: Rewrite Figure Captions

**What**: Replace all placeholder/terse captions with proper scientific figure captions.

**Figures to update**:

| Figure | Current caption | Needed |
|--------|----------------|--------|
| Fig 1 | Decent but needs polish | Add variable units, note the color scale, clarify "improvement" = (RMSE_orig - RMSE_corrected)/RMSE_orig × 100 |
| Fig 2 | "Binscatter by lead time equator distance" | Full description: 2×2 panel, rows = variables, cols = original RMSE vs improvement, colors = lead times, explain binscatter method |
| Fig 3 | "Binscatter by lead time sdor" | Same structure as Fig 2 but with SDOR on x-axis; define SDOR |
| Fig 4 | "MLP with one variable is best" | Describe bar chart, mention mean-bias baseline (after Change 1), explain hatch = multi-variable, colors = architecture, training times in legend |
| Fig 5 | "Effect of Expanding Training Region: 2m Temperature" | Describe what x-axis shows (training domain size), two panels = Finland/Amazon, colors = lead times |
| Fig 6 (appendix) | Partial | Describe boxplot comparing IFS vs Pangu |
| Appendix maps | Missing captions | Add captions describing variable, model, lead time |
| Appendix binscatters (IFS) | "Binscatter by lead time..." | Mirror main-text caption style |
| Appendix region size figs | Wrong captions (say "Wind Speed" for UNet temp) | Fix label errors and add proper captions |

---

## Change 4: Fix Typos and Grammar

**Specific fixes in `draft.typ`**:

| Location | Current | Fix |
|----------|---------|-----|
| Line 54 | "easy to implementable" | "easy-to-implement" |
| Line 57 | "discretive" | "discretize" |
| Line 65 | "out proposed" | "our proposed" |
| Line 65 | "We believe that given the simplicity..." (grammatically broken) | "Given the simplicity and generality of our proposed post-processing approach, we believe it is a feasible additional step..." |
| Line 69 | "they important impacts" | "they have important impacts" |
| Line 79 | "bencharmakring" | "benchmarking" |
| Line 103 | "we use early stopping apply L2" | "we use early stopping, apply L2" |
| Line 116 | "pursists" | "persists" |
| Line 116 | "errors structure" | "error structure" |
| Line 150 | "taxes" | "takes" |
| Line 150 | "and MLP" | "an MLP" |
| Line 153 | "about about" | "about" |
| Line 171 | "the the" | "the" |
| Line 175 | "dieing" | "dying" |
| Line 175 | "improvements in forecast accuracy in would lead" | "improvement in forecast accuracy would lead to" |
| Line 177 | "characteries" | "characterizes" |
| Line 177 | "large sections lower" | "large sections of lower" |
| Line 180 | "have effective" | "have been effective" |
| Line 180 | "for and and adapt" | "for and adapt" |
| Line 182 | "However, most cases" | "However, in most cases" |
| Line 54 | "forecasts models" | "forecast models" |
| Line 125 | Remove `#text(red)[XX ...]` placeholders — replace with actual content |
| Line 127 | Remove `#text(red)[XX ...]` placeholder |
| Line 150 | Remove `#text(red)[XX ...]` placeholder and fill in citation |
| Line 153 | Remove `#text(red)[XX ...]` placeholder — use "synoptic-scale" or "baroclinic" |

**XX Placeholder resolutions**:
- Line 125 (why 40-50° improvement spike): Suggest this may reflect mid-latitude baroclinic zones where synoptic-scale storm systems create systematic forecast biases amenable to local correction. Or it may reflect denser training data quality in these regions.
- Line 127 (SDOR interpretation): Note that wind speed is more sensitive to unresolved subgrid orography than temperature, explaining why SDOR predicts wind improvement but not temperature improvement.
- Line 150 (cite for humidity+temp predicting temp): Reference standard NWP texts or remove the claim and simply state the empirical finding.
- Line 153 (name for large-scale weather): Use "synoptic-scale" or "large-scale baroclinic" dynamics.

---

## Change 5: Explicit Comparison to Trotta et al. 2025

**What**: Add 2–3 sentences in the Introduction (after line 69 where Trotta et al. is cited) and a paragraph in the Discussion comparing scope and findings.

**Key distinctions to highlight**:
1. Trotta et al. focus on a single region (Australia) with station-level verification; this paper covers global land surface with gridded verification against ERA5/IFS analysis.
2. Trotta et al. evaluate multiple ML models (Pangu, GraphCast, etc.) but don't systematically study how improvement varies spatially or what drives that variation.
3. This paper's contribution is the global spatial analysis and the finding that improvement heterogeneity is informative about local vs large-scale forecast error sources.
4. Both papers find that simple post-processing methods substantially improve ML forecasts, reinforcing the robustness of this approach.

**Draft text changes** (`draft.typ`):
- Add comparison sentences in Introduction after the Trotta citation.
- Add a paragraph in Discussion (after the architecture discussion, before the welfare estimates) explicitly positioning relative to Trotta et al.

---

## Change 6: Single-Test-Year Limitation Discussion

**What**: Add a short paragraph (3–4 sentences) in the Discussion acknowledging that all out-of-sample evaluation uses 2022 only.

**Content**:
- 2022 featured La Niña conditions, which may affect the representativeness of tropical forecast errors.
- A single test year limits the ability to assess whether post-processing gains are stable across different climate regimes (e.g., El Niño years).
- Future work should evaluate robustness using multiple test years or rolling cross-validation.
- Note that the training period (2018–2021) spans both El Niño and La Niña phases, so the model has been exposed to diverse conditions.

**Draft text changes** (`draft.typ`):
- Add paragraph in Discussion section, after the paragraph about dissemination challenges and before the custom loss function paragraph.

---

## Implementation Order

1. **Figure 4 mean-bias baseline** — code change in `figures_finetuning.py`, regenerate figure
2. **Regression analysis** — new script, generate table
3. **Figure captions** — edit `draft.typ`
4. **Typos and grammar** — edit `draft.typ`
5. **Trotta comparison** — edit `draft.typ`
6. **Single-test-year limitation** — edit `draft.typ`

Steps 3–6 are all text edits to `draft.typ` and can be done in sequence.
