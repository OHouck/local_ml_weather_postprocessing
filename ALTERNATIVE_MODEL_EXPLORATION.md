# Alternative Model Exploration for Forecast Post-Processing

**Context.** The current experiments in [run_arch_experiments_eval.py](finetuning/run_arch_experiments_eval.py) are all variants of a single family: an MSE-trained MLP with Block Leave-Time-Holdout (LTHO) ensembling, plus tweaks (lead-time loss weighting, per-lead-time training, small output-layer init). They differ in ensembling / weighting / initialization but not in **model family** or **loss function**. This document surveys the literature on ML-based weather forecast post-processing and recommends three methods that (a) are genuinely distinct from the current experiments, (b) are small enough to train in under 5 minutes per 6×6 patch on an M3 Max, and (c) plausibly improve skill given only 4 years of training data (2018–2021) across many regions.

## Research constraints

- **Task:** post-process a single global forecast (Pangu / IFS / AIFS) to correct 2 m temperature or 10 m wind speed in 6×6° patches, at lead times 24 h / 120 h / 216 h.
- **Data budget:** 4 years train (2018–2021), 1 year test (2022). ~1,460 samples per patch per lead time.
- **Compute budget:** ≤5 min per region on an M3 Max laptop (MPS / CPU).
- **Generalization:** method must work across many regions globally.
- **Current best on record:** Block-LTHO MLP ensemble with MSE loss and learned lead-time embedding.

## Literature landscape (brief)

Post-processing ML methods for NWP cluster into roughly five families:

1. **Distributional Regression Networks (DRN).** A NN outputs the parameters of a parametric predictive distribution (usually Gaussian) and is trained with CRPS. Introduced for post-processing by Rasp & Lerch 2018 (*"Neural Networks for Postprocessing Ensemble Weather Forecasts"*, *MWR* 146(11), 3885–3900). This is the dominant baseline in the probabilistic post-processing literature and outperforms classical EMOS in most studies.
2. **Bernstein Quantile Networks (BQN).** Bremnes 2020 (*"Ensemble Postprocessing Using Quantile Functions Based on Bernstein Polynomials"*, *MWR* 148(1), 403–414). The NN outputs monotonic Bernstein coefficients, yielding a full non-parametric quantile function with very few parameters. In the systematic comparison of Schulz & Lerch 2022 (*MWR* 150(1)) BQN ranked first or tied-first for wind gusts against DRN, QRF, GBM, EMOS.
3. **CNN / U-Net spatial post-processors.** Grönquist et al. 2021 (*"Deep Learning for Post-Processing Ensemble Weather Forecasts"*, *Phil. Trans. R. Soc. A*). Higher ceiling on gridded improvements, but heavier and data-hungry — likely a poor fit for a 4-year, 6×6° budget.
4. **Transformer / attention post-processors.** Finn 2021 ("Self-attention for post-processing"), Ashkboos et al. 2022 ("ENS-10"). Much more compute; typically need more data than we have per patch.
5. **Pooled / globally-shared models with location conditioning.** Rasp & Lerch 2018 and Schulz & Lerch 2022 both showed that pooling stations into a single NN with an *embedding* for station identity (or lat/lon/climatology) substantially outperforms training one model per location when data per location is small. This is the single most impactful design choice documented in the literature for small-per-region data regimes, and we are in exactly that regime.

Methods that can be ruled out up front for this project:

- **Generative / diffusion post-processors** (Li et al. 2023; Finn 2023) — far too expensive per region, and need more data.
- **Graph NN / transformer** variants — training budget busts 5 min easily on M3 Max.
- **Full analog / QRF / GBM** baselines — not neural; already covered by Schulz & Lerch as a reference point, not a new direction.

## Top 3 recommendations

### 1. CRPS-trained Distributional Regression Network (DRN)

**Source.** Rasp & Lerch 2018, *MWR* 146(11), 3885–3900. https://doi.org/10.1175/MWR-D-18-0187.1

**What it is.** Keep exactly the current MLP backbone, but change the head and the loss:

- Head predicts `(μ, log σ)` for each output pixel × lead time (two scalars, not one).
- The correction is `μ` (so point-forecast RMSE is directly comparable to the current setup), but training minimizes the **closed-form Gaussian CRPS** of `(μ, σ)` against the ERA5 target. The Gaussian CRPS has a simple analytic form and is differentiable.

**Why it is promising here.**
- The current work optimizes MSE, which is scoring-rule-inconsistent for anything but the mean and wastes capacity that could be spent on heteroscedasticity. CRPS gives the network a reason to learn *when* the forecast is uncertain (e.g. high-topography, summer convective regimes) and pull `μ` more aggressively towards climatology when σ is large, which directly helps RMSE in those regions — exactly the regions the paper already flags as hardest.
- Parameter count is essentially unchanged (one extra output head), so training time is unchanged — well inside 5 min on M3 Max.
- It is the de-facto baseline in the probabilistic post-processing literature, so its behavior in the small-data regime is well-characterized.

**Modifications for our data/compute budget.**
- **Share σ across space, not time.** With only ~1,460 samples per patch, letting σ vary per-pixel per-lead-time is too flexible. Predict a single `log σ(lead_time, day_of_year)` scalar per sample (via a tiny 2-layer head reading the lead-time embedding and the DOY sin/cos), and a per-pixel `μ`. This gives most of the CRPS benefit with almost no extra parameters.
- **Warm-start from the current MSE model** for the first 20 epochs, then switch to CRPS. This is the standard "MSE pre-train → CRPS fine-tune" trick from the downscaling literature and stabilizes σ when data is scarce.
- **Clamp `log σ ∈ [−3, 3]`** to stop degenerate collapses, which are a known failure mode of Gaussian CRPS on tiny datasets.

---

### 2. Bernstein Quantile Network (BQN)

**Source.** Bremnes 2020, *MWR* 148(1), 403–414. https://doi.org/10.1175/MWR-D-19-0227.1. Validated as state-of-the-art for surface wind post-processing by Schulz & Lerch 2022, *MWR* 150(1), 235–257.

**What it is.** Instead of a Gaussian head, the MLP outputs `d+1` Bernstein polynomial coefficients `α_0 ≤ α_1 ≤ … ≤ α_d` (monotonicity enforced via a cumulative softplus parameterization). This defines a full quantile function `Q(τ) = Σ α_k · B_k,d(τ)`. Training uses the quantile loss averaged over a fixed grid of quantile levels, which is an unbiased CRPS estimator.

**Why it is promising here.**
- **Distribution-free.** Unlike DRN, BQN does not assume Gaussian errors, which is important for 10 m wind speed (right-skewed, bounded below by 0) — the second target variable in this project. Schulz & Lerch explicitly found BQN beats DRN on wind gusts for this reason.
- **Very parameter-efficient.** A degree-`d=8` polynomial needs only 9 extra outputs per pixel. Strictly smaller than a per-pixel mean head + per-pixel variance head, so this is the *cheapest* probabilistic option.
- **Point-forecast correction** is taken as `Q(0.5)`, directly comparable to the current MSE setup.

**Modifications for our data/compute budget.**
- **Low polynomial degree.** Schulz & Lerch use `d=12`. With 4 years of data per patch, start at `d=6` and only raise it if validation improves. Fewer coefficients ⇒ less overfitting and faster training.
- **Pixel-tied coefficients + per-pixel offset.** Predict one set of Bernstein coefficients per *patch-level forecast*, plus a small per-pixel additive offset. Empirically most of the shape information in 6×6° is shared across pixels; this is a ~10× parameter reduction vs. full per-pixel Bernstein outputs.
- **Evaluate the quantile loss on only 19 quantiles** (0.05, 0.10, …, 0.95) rather than 99 to keep the CRPS estimator cheap on MPS.

---

### 3. Globally-pooled MLP with region/climatology conditioning (shared backbone + FiLM)

**Sources.**
- Core idea: Rasp & Lerch 2018 (embedding-conditioned pooled NN, Section 3.b).
- Systematic validation across methods: Schulz & Lerch 2022, *MWR* 150(1), 235–257.
- FiLM conditioning mechanism: Perez et al. 2018, *AAAI*, "FiLM: Visual Reasoning with a General Conditioning Layer".

**What it is.** Instead of training an independent model per 6×6° patch, train **one shared MLP** on *all* patches pooled together, and condition every hidden layer via Feature-wise Linear Modulation (FiLM) on a small region descriptor:
- Region descriptor: `[sin(lat), cos(lat), sin(lon), cos(lon), land-sea fraction, elevation mean, elevation stdev (SDOR), Köppen zone one-hot]`.
- A tiny hypernetwork maps the region descriptor to per-layer `(γ, β)` gain/bias vectors which modulate the shared MLP's activations.

At inference, the same shared weights apply to every region; only the FiLM conditioning changes. This is the single most data-efficient architectural change available and matches the project's stated goal of "working across multiple regions".

**Why it is promising here.**
- **Effectively multiplies the training set by the number of patches.** With ~100+ 6×6 cells on land, the shared backbone sees ~100× more samples than any per-patch model currently does, which is decisive at n≈1,460.
- **Directly addresses the paper's finding that improvement is weakest in high-topography and near-equator regions.** Those are exactly the regions where a single patch has too little data to learn a correction; pooling lets them borrow strength from statistically similar patches (via the climatology descriptor), which is precisely what Rasp & Lerch 2018 observed for stations in data-sparse regions.
- **Training is done once**, so the 5-minute budget applies to the *entire global model*, not per-patch. After training, inference on a new patch is essentially free — a better match to the project's compute constraint than the current per-patch retraining loop.
- **Orthogonal to recommendation 1 and 2.** FiLM conditioning can wrap either a CRPS/Gaussian head (recommendation 1) or a Bernstein head (recommendation 2), so the three recommendations compose.

**Modifications for our data/compute budget.**
- **Small shared backbone.** Hidden dim 256, depth 4 is enough when the data is pooled. Do not use the current hidden=1024, depth=6 — that was tuned for a single small patch and over-parameterizes a pooled model.
- **Stratified mini-batches.** Each batch samples patches proportional to climate zone frequency to prevent the model from collapsing onto the dominant (mid-latitude temperate) regime.
- **Optional per-region residual head.** After pooled training, optionally fit a tiny (1-layer, 32-unit) residual MLP per patch for 30 epochs as a second-stage refinement. Total compute is still well under 5 min per patch because the backbone is frozen.
- **Cache the FiLM-modulated features** at eval time: since the region descriptor is static per patch, γ and β can be pre-computed once and reused for every sample in that patch.

---

## How these differ from the current experiments

| | Current (all 4 experiments) | Rec 1 (DRN) | Rec 2 (BQN) | Rec 3 (Pooled + FiLM) |
|---|---|---|---|---|
| Loss | MSE | Gaussian CRPS | Quantile loss | MSE / CRPS (orthogonal) |
| Output | Point estimate | Mean + variance | Full quantile function | Any |
| Training scope | Per patch | Per patch | Per patch | Global pooled |
| Data seen per region | ~1,460 | ~1,460 | ~1,460 | ~1,460 × N_patches |
| Heteroscedastic | No | Yes | Yes (non-parametric) | Inherits from head |
| Extra params vs. current | 0 | ~tiny head | ~9 outputs/pixel | FiLM hypernet (small) |

The current experiments are all *optimization/ensembling* variations on a fixed model family. Recommendations 1 and 2 change the **loss and output head** (probabilistic calibration); recommendation 3 changes the **training topology** (pooling across regions). These are non-overlapping axes.

---

## Step-by-step implementation plan for a future agent

The goal is to add three new experiments to `EXPERIMENTS` in [run_arch_experiments_eval.py](finetuning/run_arch_experiments_eval.py) that correspond to the three recommendations above. Because recommendation 3 changes the training topology (global pool rather than per-patch), it requires a separate driver.

### Preliminaries (read before touching code)

1. Read [finetune.py](finetuning/finetune.py) end-to-end with particular attention to:
   - `SimpleMLP.__init__` / `forward` — this is the backbone you will reuse.
   - `train_model(...)` — the single-patch training loop. Note how loss is computed and where `loss_fn` is dispatched.
   - `apply_correction(...)` — inference: `corrected = raw_forecast + model(...)`. For probabilistic heads you return `μ` here.
   - `run_subregion_experiment(...)` — per-patch driver that `run_arch_experiments_eval.py` already calls.
2. Read [custom_loss_fns.py](finetuning/custom_loss_fns.py) to see the convention for custom losses (input normalization flag, signature `loss_fn(pred, target, ...)`).
3. Confirm on MPS that the current `Block LTHO Ensemble` baseline trains in well under 5 min on a single 6×6 patch (`python3 finetuning/run_arch_experiments_eval.py` with the eval sample restricted to 1 patch). This is your speed reference.

### Recommendation 1 — DRN (Gaussian CRPS head)

1. **Add the CRPS loss.** In `custom_loss_fns.py`, add:
   ```python
   def gaussian_crps_loss(pred, target, is_normalized=True):
       # pred: (..., 2) = (mu, log_sigma); target: (...,)
       mu = pred[..., 0]
       log_sigma = pred[..., 1].clamp(-3.0, 3.0)
       sigma = log_sigma.exp()
       z = (target - mu) / sigma
       # Closed-form Gaussian CRPS (Gneiting & Raftery 2007)
       from math import sqrt, pi
       pdf = torch.exp(-0.5 * z * z) / sqrt(2 * pi)
       cdf = 0.5 * (1 + torch.erf(z / sqrt(2)))
       crps = sigma * (z * (2 * cdf - 1) + 2 * pdf - 1 / sqrt(pi))
       return crps.mean()
   ```
   Register it in the `alternate_loss_fn` dispatch path in `finetune.py`.
2. **Add a Gaussian head to `SimpleMLP`.** Add constructor arg `probabilistic_head: str = "none"`. When `"gaussian"`, append one extra output dimension (size `2 * output_dim`) and reshape in `forward` to `(..., output_dim, 2)`. Store `head_type` so `apply_correction` knows to slice `[..., 0]` as the point estimate.
3. **Gate on a new CLI flag.** Add `--probabilistic_head {none, gaussian}` to `parse_args`. Thread it through `run_subregion_experiment → train_model → SimpleMLP`.
4. **Warm-start the sigma branch.** In `train_model`, when `probabilistic_head="gaussian"`, train with plain MSE on `μ` for the first 20 epochs (treat the σ branch as frozen at `log σ = 0`), then swap to `gaussian_crps_loss`. This is 3 extra lines in the epoch loop.
5. **New experiment entry.** Add to `EXPERIMENTS` in `run_arch_experiments_eval.py`:
   ```python
   {
       'name': 'Block LTHO + DRN (Gaussian CRPS)',
       'nn_architecture': 'mlp',
       'block_ensemble': True, 'block_holdout': 3,
       'snapshot_ensemble': 1, 'snapshot_epochs': 210,
       'snapshot_T0': 10, 'snapshot_T_mult': 1,
       'ensemble': None, 'swa_ensemble': None,
       'probabilistic_head': 'gaussian',
   },
   ```
   and add `args.probabilistic_head = exp.get('probabilistic_head', 'none')` in the arg-building loop.
6. **Smoke-test** on 1 patch, confirm training finishes in <5 min and the saved `*_corrected_lt{N}h` variable is finite.

### Recommendation 2 — BQN (Bernstein Quantile Network head)

1. **Add monotonic Bernstein head.** In `SimpleMLP`, when `probabilistic_head="bernstein"`, the final layer outputs `(d+1)` values per pixel; apply `softplus` and cumulative sum to get monotonic `α_0 ≤ … ≤ α_d`. Default `d=6`.
2. **Add the quantile loss.** In `custom_loss_fns.py`:
   ```python
   def bernstein_quantile_loss(alphas, target, degree=6, n_quantiles=19):
       # alphas: (..., d+1); target: (...,)
       taus = torch.linspace(0.05, 0.95, n_quantiles, device=alphas.device)
       # Bernstein basis B_{k,d}(tau)
       k = torch.arange(degree + 1, device=alphas.device)
       from math import comb
       coefs = torch.tensor([comb(degree, int(ki)) for ki in k],
                            device=alphas.device, dtype=alphas.dtype)
       basis = coefs * taus[:, None]**k * (1 - taus[:, None])**(degree - k)  # (Q, d+1)
       quantiles = (alphas.unsqueeze(-2) * basis).sum(-1)  # (..., Q)
       diff = target.unsqueeze(-1) - quantiles
       ql = torch.maximum(taus * diff, (taus - 1) * diff)
       return ql.mean()
   ```
3. **Point-forecast extraction for zarr output.** In `apply_correction`, when `head_type="bernstein"`, evaluate the quantile function at `τ=0.5` and use that as `μ`. Everything downstream (RMSE computation, plotting) stays identical.
4. **New experiment entry:**
   ```python
   {
       'name': 'Block LTHO + BQN (d=6)',
       # ... same ensembling fields as above ...
       'probabilistic_head': 'bernstein',
       'bernstein_degree': 6,
   },
   ```
   Thread `bernstein_degree` through the same path as `probabilistic_head`.
5. **Sanity check:** plot the learned quantile function `Q(τ)` for 3 random samples on 1 patch and confirm it is monotone and not degenerate (all coefficients equal).

### Recommendation 3 — Pooled MLP + FiLM conditioning

This one needs a **new driver**, not just a new `EXPERIMENTS` entry, because it trains a single global model rather than one per patch.

1. **New file `finetuning/run_pooled_film_experiment.py`:**
   - Call `sample_continent_patches` with the full eval set (no 5% restriction — we want all land patches).
   - For each patch, call `load_forecasts` once to get `(X_patch, y_patch, lead_time, doy)`. Concatenate all patches into one big tensor, and build a parallel tensor `R_patch` of region descriptors (sin/cos lat, sin/cos lon, elevation mean, SDOR from ERA5 — already used for Figure 3, see `figures_finetuning.py::lead_time_compare_binscatter`).
   - Store a `patch_id` column so you can split the held-out test year per patch later.
2. **New model class `PooledFiLMMLP` in `finetune.py`:**
   ```python
   class PooledFiLMMLP(nn.Module):
       def __init__(self, input_dim, region_dim, output_dim,
                    hidden_dim=256, num_layers=4):
           super().__init__()
           self.layers = nn.ModuleList(
               [nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim)
                for i in range(num_layers)])
           self.film = nn.ModuleList(
               [nn.Linear(region_dim, 2 * hidden_dim) for _ in range(num_layers)])
           self.out = nn.Linear(hidden_dim, output_dim)
       def forward(self, x, r):
           for layer, film in zip(self.layers, self.film):
               h = layer(x)
               gamma, beta = film(r).chunk(2, dim=-1)
               h = (1 + gamma) * h + beta
               x = F.gelu(h)
           return self.out(x)
   ```
   The `(1 + γ)` form is the standard FiLM init so that training begins as an identity modulation.
3. **Training loop.** Reuse the existing `train_model` but with a pooled `DataLoader` that yields `(x, r, y)` triples. Use stratified sampling: weight each sample by `1 / count(koppen_zone(patch))` so rare zones are not drowned out.
4. **Evaluation.** For each patch in the eval sample, run inference with the patch's fixed region descriptor and write a zarr with the exact same naming convention `train_...dim6x6..._pooled_film.zarr` so `process_forecasts.py` and the plotting scripts pick it up without modification. Use a distinct `nn_architecture` tag (`"pooled_film"`) in `generate_output_path`.
5. **Optional second-stage per-patch head.** After pooled training, iterate over eval patches and fit a tiny (1 hidden layer, 32 units) residual MLP on top of the frozen pooled output for 30 epochs each. Save as a separate `..._pooled_film_refined.zarr` so the comparison with the pure pooled model is clean.
6. **Budget check.** Target ≤10 min total for pooled training across all land patches on M3 Max (the 5-min-per-region budget is amortized across patches; this is still favorable compared to the current setup which trains ~hundreds of independent models). If it blows the budget, halve `hidden_dim` before halving the training set.

### Comparison / reporting

After all three new methods run, extend `plot_arch_experiment_results.py` to group results into two panels:

- **Panel A: Per-patch methods.** Current 4 baselines + DRN + BQN. Y-axis: RMSE % improvement at each lead time.
- **Panel B: Pooled methods.** Pooled-FiLM (and its per-patch-refined variant) vs. the baseline Block LTHO for reference.

Report both point RMSE improvement (for head-to-head with the existing paper results) and, for DRN/BQN, CRPS improvement vs. raw forecast climatology. The CRPS numbers are the main selling point of those two methods and should not be left on the table.

---

## Experiment results (run April 2026)

All six Block LTHO variants were evaluated on the 5% eval split (18 cells, globally distributed across all continents). MLP, Snapshot ×3, and UNet baselines are evaluated on the India 6×6 single-region zarr (an important caveat for comparing across groups — India is a relatively easy region). Results are RMSE % improvement over the raw Pangu forecast, averaged across eval cells where applicable.

| Method | 1 day | 5 days | 9 days | Notes |
|---|---|---|---|---|
| Mean Bias Correction | — | — | — | From India eval |
| **MLP (single var)** | 3.1% | 8.3% | 12.2% | Eval-cell average |
| MLP (3 vars) | 4.1% | 7.6% | 9.8% | India only |
| **MLP Snapshot ×3 (single var)** | **5.8%** | **11.4%** | **15.2%** | India only — best overall |
| MLP Snapshot ×3 (3 vars) | 5.4% | 9.3% | 11.5% | India only |
| UNet (single var) | 4.8% | 8.6% | 10.6% | India only |
| UNet (3 vars) | 3.3% | 2.7% | 2.1% | India only — notably poor |
| Block LTHO Ensemble | 1.7% | 8.8% | 12.9% | Eval-cell average |
| Block LTHO + LT-Weighted | 3.6% | 8.2% | 11.1% | Eval-cell average |
| Per-LT Block LTHO | 4.3% | 8.7% | 12.2% | Eval-cell average |
| Block LTHO + SmallInit | 1.9% | 8.4% | 12.2% | Eval-cell average |
| Block LTHO + DRN | 3.4% | 8.8% | 12.2% | Eval-cell average |
| Block LTHO + BQN d=6 | 3.5% | 8.7% | 11.8% | Eval-cell average |

### Key findings

**MLP Snapshot ×3 is the clear winner.** It outperforms every other method at every lead time by a substantial margin (15.2% at 9 days vs. 12.9% for the next best, Block LTHO). This result comes from India only, so it may not generalize uniformly, but the advantage is large enough to be robust. It trains in under 1 minute (0.8 min on M3 Max) and requires no architectural changes — just three warm-restart cycles with cosine annealing. **This is the recommended method going forward.**

**Adding extra input variables consistently hurts.** The 3-variable variants (+ 1000 hPa T and q) underperform the single-variable models at every lead time except 1 day where the gain is small (~1%). This is consistent with the main paper finding and extends it to the Snapshot setting. The UNet 3-variable result (2.1% at 9 days) is particularly striking — adding more inputs to a UNet appears to actively harm generalization, likely because the larger input space is harder to regularize with only 4 years of training data.

**Block LTHO underperforms Snapshot at 1-day lead time.** The baseline Block LTHO ensemble is the worst-performing method at 24h (1.7%), worse even than a plain MLP. The block holdout scheme withholds temporal blocks during training that happen to overlap with the 24h verification window, starving the model of close-in data. The three variants designed to address this have mixed success:
- **Per-LT Block LTHO** (4.3% at 1 day) helps the most, but at 9 days it matches plain Block LTHO rather than exceeding it.
- **LT-Weighted** (3.6% at 1 day) partially recovers short-range skill, but trades off 9-day performance (11.1% vs. 12.9%).
- **SmallInit** has essentially no effect — it barely moves any lead time relative to the baseline.

**DRN and BQN do not improve point RMSE over baseline Block LTHO.** DRN (12.2% at 9 days) and BQN (11.8%) are on par with the Block LTHO baseline (12.9%), not better. This is expected — these methods are designed to improve probabilistic calibration (CRPS), not point RMSE. Their value will only show up in CRPS evaluation, which has not yet been measured. The paper currently reports only RMSE, so these methods do not yet have a natural place in the results.

### What the new methods do (for the record)

**Block LTHO + LT-Weighted.** Identical architecture and training to Block LTHO, but during snapshot training the loss function applies a 5× weight to the 24-hour lead time and 0.5× to 216h. The intent is to make each snapshot specialize harder on short-range corrections while still benefiting from block diversity. In practice it improves 24h but degrades 9-day performance, suggesting the weights are too aggressive and interfere with mid-/long-range learning.

**Per-LT Block LTHO.** Trains a completely separate Block LTHO ensemble for each lead time (three independent models: one for 24h, one for 120h, one for 216h). By decoupling the lead times, the 24h model sees 100% of gradient signal from 1-day verification pairs, eliminating the gradient competition from longer lead times. This is the conceptually cleanest fix to the 24h problem and achieves the best 1-day result (4.3%) among the Block LTHO family. The downside is ~3× the training time and storage, and at 5 and 9 days it offers no improvement over the baseline.

**Block LTHO + SmallInit.** Identical to baseline Block LTHO but the final output layer is initialized near zero (weights ∼ 0.01 × normal). The idea is that the model starts by predicting near-zero corrections, which is closer to the true 24h correction (small) than a random initialization, providing a better gradient signal early in training. Results show essentially no effect at any lead time. The finding suggests that initialization is not a binding constraint relative to the block holdout design or model capacity.

**Block LTHO + DRN (Distributional Regression Network).** Adds a Gaussian probabilistic head to the Block LTHO MLP backbone. The final layer outputs `(μ, log σ)` per pixel rather than a single point correction. Training uses closed-form Gaussian CRPS loss after a 20-epoch MSE warm-start. Point-forecast RMSE uses `μ`. As expected, this does not improve RMSE — the benefit is probabilistic calibration, measurable only via CRPS. This method should be evaluated on CRPS before drawing conclusions.

**Block LTHO + BQN d=6 (Bernstein Quantile Network).** Replaces the point-estimate head with 7 Bernstein polynomial coefficients per pixel, enforcing a monotone quantile function via cumulative softplus. Training uses the average pinball loss over 19 quantile levels (τ = 0.05 to 0.95). Point RMSE uses the median (τ = 0.5). Like DRN, this method does not improve point RMSE. It is specifically designed for distributional calibration and is the most promising method for post-processing 10m wind speed (right-skewed, non-Gaussian), where the Gaussian DRN assumption breaks down.

### Recommendations going forward

1. **Use MLP Snapshot ×3 as the new paper baseline** and re-run it on the 18-cell global eval set to get a fair apples-to-apples comparison with the Block LTHO variants. The current Snapshot ×3 result is India-only.
2. **Evaluate DRN and BQN on CRPS** before deciding whether to include them. Their point RMSE results are uninformative about their value.
3. **Do not pursue SmallInit or LT-Weighted further** — neither adds value over the baseline, and Per-LT is strictly better at 1 day at equivalent compute.
4. **Consider Per-LT Block LTHO** if short-range (24h) skill is a paper priority; otherwise it is not worth the 3× compute overhead.
5. **The Pooled FiLM model (Recommendation 3 above) remains untested** and is still the highest-upside unexplored direction, particularly for data-sparse regions the paper already flags as weak.

---

### Guardrails for the implementing agent

- **Do not remove or rename the current 4 baseline experiments in `EXPERIMENTS`.** They are the reference point for the paper.
- **Do not change the output-file naming scheme** beyond appending new architecture tags — `filter_patch_zarr_files` in [figures_finetuning.py](finetuning/figures_finetuning.py) parses filenames and will silently drop renamed files.
- **Re-use `load_optimal_hyperparameters`** where it applies (DRN and BQN share the backbone with the existing MLP, so the learned hidden_dim / num_layers / lr are valid starting points). The pooled FiLM model has a different topology and should be hyperparameter-searched separately — but do that only after the pooled baseline beats the per-patch baseline at default settings. Do not spend compute on hyperparameter search for a method that is not yet winning.
- **Smoke-test each new method on a single patch before launching the full eval sweep.** A silent numerical bug in the Bernstein basis or the CRPS formula will not be caught by existing tests.


## Summary and explanation of the settled models

This section describes every model variant evaluated in the architecture experiments, covering architecture, training method, and the rationale for including it.

---

### Plain MLP

**Architecture.** `SimpleMLP` flattens the spatial patch into a 1-D feature vector and passes it through a stack of fully-connected ReLU layers (default: hidden dim 1024, 6 layers, dropout 0.25). Day-of-year is encoded as sin/cos and appended to the input. Lead-time is represented via a learned embedding of dimension 4 (tuned by hyperopt) that is also concatenated to the input. The output is a flat vector matching the spatial patch; added to the raw forecast to produce the corrected value.

**Training method.** Adam optimizer with ReduceLROnPlateau scheduler and early stopping. A single 80/20 random train/validation split. Standard MSE loss on the normalized correction target. No ensembling: one model, one run.

**Why it was tested.** The plain MLP is the paper's original baseline and the simplest post-processing model in the experiment suite. Every other method is compared against it to isolate the contribution of each architectural or training innovation. It also trains in approximately 0.3 min per patch on an M3 Max, providing the speed floor for all comparisons.

---

### MLP with 3-variable input

**Architecture.** Identical to the plain MLP except the input also includes 1000 hPa temperature and specific humidity from the same forecast model, concatenated as additional spatial channels alongside the 2 m temperature. Input dimension increases accordingly; output is still only 2 m temperature.

**Training method.** Same as the plain MLP.

**Why it was tested.** Including pressure-level variables provides the model with information about the free-tropospheric state that can modulate surface temperature errors—particularly at 5- and 9-day lead times where boundary-layer decoupling makes surface fields less predictive of their own future errors. The experiment tests whether the additional predictors offset the cost of the larger input space on a 4-year training dataset.

---

### MLP Snapshot Ensemble ×3

**Architecture.** Same `SimpleMLP` backbone as the plain MLP. The only change is in the training procedure.

**Training method.** Three independent snapshot ensemble runs are launched in sequence. Within each run the scheduler is `CosineAnnealingWarmRestarts` with T₀ = 30 and T_mult = 1 (seven cosine cycles over 210 epochs). At each cosine cycle minimum the current model weights are checkpointed; all checkpoints across all three runs are averaged at inference time. AdamW optimizer with gradient clipping (clip norm 1.0) is used in place of Adam+ReduceLROnPlateau. A fresh random 80/20 train/validation split is drawn for each run to add additional diversity. Model selection within each cycle uses best validation loss.

**Why it was tested.** Snapshot ensembling (Huang et al. 2017) achieves ensemble-level variance reduction at roughly the cost of a single training run: the cosine warm restarts drive the model through multiple distinct loss-basin neighborhoods before each checkpoint. Ensembling then averages over those diverse solutions. The ×3 multiplier adds further diversity by re-running the warm-restart schedule from a fresh random initialization. This was predicted to help most at long lead times where forecast error variance is high, and is the single architectural change with the largest expected benefit for a small compute budget.

---

### UNet

**Architecture.** `UNet` encodes the 6×6 spatial patch as a 2-D image where input variables form the channel dimension. A series of convolutional encoder blocks (Conv→BatchNorm→ReLU→Dropout2d, ×2 per level) halve the spatial resolution at each level via MaxPool2d; the decoder symmetrically upsamples with ConvTranspose2d and skip connections that concatenate encoder activations. Channel width doubles at each encoder level up to a maximum of 128. Day-of-year and lead-time embeddings are tiled as extra input channels rather than being appended after flattening. The number of pooling levels is determined automatically to maintain at least a 2×2 bottleneck (capped at 5 levels). Final 1×1 conv maps back to n_output_vars channels.

**Training method.** Same schedule as the plain MLP: Adam optimizer, ReduceLROnPlateau, early stopping, single 80/20 split, MSE loss. No ensembling.

**Why it was tested.** U-Nets explicitly preserve and exploit local spatial structure through skip connections and the encoder-decoder bottleneck. The hypothesis was that spatial correlations in forecast error—evident in topographic gradients, coastlines, and mesoscale circulation patterns—would be better captured by a convolutional model than by a fully-flattened MLP. The U-Net had already been established as a reasonable alternative to the MLP in earlier experiments; this comparison placed it alongside the snapshot and block-ensemble variants to see whether spatial inductive bias adds value at the 6×6 patch scale.

---

### Block Leave-Three-Out (LTHO) Ensemble

**Architecture.** Same `SimpleMLP` backbone as the plain MLP, with snapshot training applied within each block.

**Training method.** The four training years (2018–2021) define C(4,3) = 4 distinct held-out blocks, each consisting of three years. One snapshot ensemble run is trained per block using only the held-in year as training data; the three held-out years serve as the validation set for that block. All snapshot checkpoints from all four blocks are pooled and weighted by inverse validation loss before averaging. Within each block, the snapshot scheduler is CosineAnnealingWarmRestarts with T₀ = 10 (producing exactly 21 snapshots over 210 epochs), chosen so many cycles fit within the single-year training period.

**Why it was tested.** The block holdout scheme creates temporally diverse ensemble members: each member has never seen the training data pattern of its validation years, so the ensemble spans a wider range of climate-variability regimes than a random-split snapshot ensemble. This is expected to improve generalization in years with unusual anomaly patterns and to reduce over-fitting to the specific weather of 2018–2021. The method is the main novel training contribution of the paper.

---

### Per-Lead-Time MLP Snapshot ×3 (Per-LT Snapshot)

**Architecture.** Three independent `SimpleMLP` models, one for each lead time (24 h, 120 h, 216 h). Because each model sees only data from a single lead time, the lead-time embedding is omitted (n_lead_times = 1). *Note: the results table labels this "Per-LT Block LTHO," which reflects an earlier experiment variant that used block holdout per lead time (`block_ensemble=True`). The current `_ARCH_TEMPLATES` entry (`'Per-LT MLP Snapshot x3'`) uses plain snapshot ensembling (`block_ensemble=False`, `snapshot_ensemble=3`). Both variants decouple lead times; the key difference is whether temporal diversity comes from block holdout or from random train/val splits.*

**Training method.** For each lead time, three independent snapshot ensemble runs are launched (T₀ = 30, 210 epochs, CosineAnnealingWarmRestarts). Each run uses a fresh random 80/20 train/val split drawn from the single-lead-time data subset. All snapshots across all three runs are pooled and weighted by inverse validation loss. At inference the three models are applied separately to their corresponding test samples and the results are concatenated.

**Why it was tested.** Training a single model jointly on all three lead times creates gradient competition: the 120-h and 216-h errors dominate training loss because they are numerically larger, starving the 24-h head of useful gradient signal. Training a completely separate model per lead time eliminates this interference—the 24-h model devotes 100% of its capacity to 1-day corrections. The cost is roughly 3× the training time and storage.

---

### Block LTHO + Distributional Regression Network (DRN)

**Architecture.** `SimpleMLP` backbone with an expanded output head: instead of a single correction value per pixel, the final layer outputs two values per pixel—`μ` (the mean correction) and `σ` (the predictive standard deviation, parameterized as `exp(log σ)` internally). `σ` is clamped to a minimum of `1e-6` to prevent degenerate collapse. At inference, `μ` is used as the point-forecast correction and `σ` provides a per-sample uncertainty estimate.

**Training method.** The loss is the closed-form Gaussian CRPS (Gneiting & Raftery 2007, Rasp & Lerch 2018), implemented as `gaussian_crps_loss` in `custom_loss_fns.py`. Training uses a 20-epoch MSE warm-start on `μ` alone (σ branch frozen at `log σ = 0`) before switching to full CRPS loss, to stabilize σ when data is scarce. The Block LTHO outer loop is otherwise identical to the standard Block LTHO setup.

**Why it was tested.** MSE training is a consistent scoring rule only for the conditional mean; it gives the network no reason to learn heteroscedastic uncertainty. The Gaussian CRPS jointly optimizes the mean and spread, which can pull `μ` more aggressively toward climatology when `σ` is large (e.g., high-topography cells with large forecast variance). This was expected to improve RMSE in the hardest cells even though the primary benefit of CRPS training is probabilistic calibration. In practice it did not improve point RMSE over the Block LTHO baseline (see Key Findings above); its value lies in the calibrated uncertainty estimates, which have not yet been evaluated against CRPS. The DRN is the dominant baseline in the weather post-processing literature (Rasp & Lerch 2018) and is a natural comparison.

---

### Block LTHO + Bernstein Quantile Network (BQN, d = 6)

**Architecture.** `SimpleMLP` backbone with an expanded output head: instead of one value per pixel, the final layer outputs `(degree + 1) = 7` raw values per pixel. These are transformed via `softplus` followed by `cumsum` to produce monotone Bernstein polynomial coefficients `α₀ ≤ α₁ ≤ … ≤ α₆`, defining a non-parametric quantile function. The mean of the coefficients is subtracted and the forecast value is added, so the coefficients represent the corrected-forecast quantile function centered on the raw forecast. At inference, the median (`τ = 0.5`) is evaluated as the point-forecast correction.

**Training method.** The loss is the average pinball (quantile) loss over 19 quantile levels τ ∈ {0.05, 0.10, …, 0.95}, which is an unbiased Monte Carlo estimator of CRPS. Implemented as `bernstein_quantile_loss` in `custom_loss_fns.py`. The Block LTHO outer loop is otherwise identical to the standard Block LTHO setup; there is no MSE warm-start because the median of the Bernstein function is a stable estimator from the first epoch.

**Why it was tested.** BQN is distribution-free: unlike DRN it makes no Gaussian assumption, which makes it the preferred probabilistic method for right-skewed, bounded-below variables such as 10 m wind speed. Schulz & Lerch 2022 found BQN ranked first or tied-first against DRN, QRF, GBM, and EMOS on wind gust post-processing specifically because the Gaussian assumption breaks down for wind. Degree 6 (7 coefficients) was chosen as the minimum that can represent a unimodal skewed distribution without overfitting on the ~1,460 training samples per patch.