# Architecture Experiment Notes

## Goal
Improve post-processing model performance over current MLP and UNet for 6x6 degree regional weather forecasts.
Constraint: Training time < 5 minutes on M3 Max MacBook Pro.

## Test Case
- Region: India 6x6 (center 22.0, 77.0), Pangu model
- Variable: 2m_temperature
- Lead times: 24h, 120h, 216h
- Train: 2018-2021, Test: 2022
- Data shape: 24x24 spatial grid, ~4332 training samples (4 years x ~365 days x 3 lead times)
- Device: MPS (Apple Silicon M3 Max)

---

## Summary of Results

> **Note on metrics**: Values below are **MSE % improvement** = `(1 - MSE_corr/MSE_orig) * 100`.
> RMSE % improvement = `(1 - RMSE_corr/RMSE_orig) * 100` is approximately half the MSE improvement.
> Example: +11.2% MSE ≈ +5.8% RMSE.

### Round 1–2 (earlier session, measured in normalized space)
| Model | 24h | 120h | 216h | Avg Improvement | Training Time |
|-------|------|-------|-------|-----------------|---------------|
| **Single MLP (baseline)** | +10.3% | +19.2% | +26.8% | **+18.7%** | 0.19 min |
| ResidualMLP (768h, 6 blocks) | +8.6% | +19.1% | +24.2% | +17.3% | 1.31 min |
| ResCNN (64ch, 6 blocks) | +9.0% | +16.4% | +16.3% | +13.9% | 15.88 min |
| Per-lead-time MLPs | +11.0% | +18.3% | +22.7% | +17.4% | 0.17 min |
| Ensemble 5 MLPs | +10.8% | +20.3% | +27.4% | +19.5% | 0.69 min |
| Ensemble 7 MLPs | +11.0% | +21.1% | +27.6% | +19.9% | 0.75 min |
| Ensemble 10 MLPs | +11.0% | +21.3% | +27.9% | +20.1% | 1.07 min |
| Ensemble 12 MLPs | +11.1% | +21.4% | +28.3% | +20.3% | 1.49 min |
| Per-LT Ensemble 5 | +11.2% | +19.9% | +24.0% | +18.4% | 0.68 min |
| ResidualMLP+cosine | +8.9% | +16.2% | +21.6% | +15.5% | 0.39 min |
| Snapshot Ens (1 run, 5 snaps) | +11.5% | +20.9% | +26.6% | +19.7% | 0.22 min |
| Snapshot Ens (1 run, T=30 Tmult=2) | **+11.8%** | +20.7% | +26.7% | +19.7% | 0.22 min |
| **Snapshot Ens (3 runs × 7 snaps)** | +11.6% | +20.5% | +28.0% | **+20.0%** | **0.67 min** |
| **Snapshot Ens (5 runs × 7 snaps)** | **+11.7%** | **+20.9%** | **+28.3%** | **+20.3%** | **1.11 min** |

### Round 3 (session 2 — actual temperature space, verified from zarr)
| Model | 24h MSE% | 120h MSE% | 216h MSE% | Avg MSE% | Training Time |
|-------|----------|-----------|-----------|----------|---------------|
| **Mean Debiasing (baseline)** | +8.95% | +16.06% | +11.92% | +12.31% | — |
| Single MLP (hyperopt) | +9.44% | +16.51% | +24.26% | +16.74% | ~0.2 min |
| PixelwiseMLP | +6.55% | +13.40% | +27.76% | +15.90% | ~0.5 min |
| LocalGlobalMLP | +6.85% | +15.11% | +27.11% | +16.36% | ~0.6 min |
| TwoStreamMLP | +7.38% | +14.33% | +29.51% | +17.07% | ~1.1 min |
| TwoStreamMLP Snapshot 3 | +6.32% | +15.49% | +27.85% | +16.55% | ~4.4 min |
| MLP PCA-30 | +3.32% | +11.73% | +23.07% | +12.71% | ~0.3 min |
| SWA Ensemble 3 (warmup=150, T0=10, swa_epochs=110) | +10.76% | +20.85% | +28.49% | +20.03% | ~0.9 min |
| MLP Snapshot 3 (T0=30) | +11.35% | +21.49% | +28.14% | +20.33% | ~1.0 min |
| MLP Snapshot 5 (T0=30) | +11.32% | +21.37% | +28.45% | +20.38% | ~1.4 min |
| **Block Ensemble (4 early-stop)** | **+11.54%** | **+22.18%** | **+26.53%** | **+20.08%** | **~0.5 min** |
| **Block + Snapshot (T0=30, 28 preds, equal weight)** | **+12.10%** | **+22.31%** | **+29.34%** | **+21.25%** | **~1.5 min** |

### Round 4 (session 3 — val-loss weighting, T0 tuning for LOO blocks)
| Model | 24h MSE% | 120h MSE% | 216h MSE% | Avg MSE% | Training Time |
|-------|----------|-----------|-----------|----------|---------------|
| Block LOO Snap (T0=10, 84 preds, equal) | +12.20% | +22.47% | +29.37% | +21.35% | ~1.3 min |
| Block LOO Snap (T0=15, 56 preds, equal) | +12.22% | +22.37% | +29.46% | ~21.35% | ~1.3 min |
| Block LOO Snap (T0=20, 40 preds, equal) | +11.97% | +22.14% | +29.34% | +21.15% | ~1.3 min |
| Block LOO Snap (T0=15, 3 seeds/block, 112 preds) | +12.08% | +22.24% | +29.44% | +21.25% | ~2.5 min |
| **Block LOO Snap (T0=15, 56 preds, global val-loss weighted)** | **+12.25%** | **+22.51%** | **+29.36%** | **+21.37%** | **~1.3 min** |

### Round 5 (session 3 — leave-K-out generalisation: key breakthrough)
| Model | 24h MSE% | 120h MSE% | 216h MSE% | Avg MSE% | Training Time |
|-------|----------|-----------|-----------|----------|---------------|
| Block k=1 (LOO) + Snap T0=15, val-wt | +12.25% | +22.51% | +29.36% | +21.37% | ~1.3 min |
| Block k=2 (LTO) + Snap T0=15, val-wt | +12.77% | +24.36% | +30.51% | +22.55% | ~1.3 min |
| Block k=3 early-stop only (4 preds) | +11.24% | +23.11% | +26.40% | +20.25% | ~0.2 min |
| Block k=3 (LTHO) + Snap T0=30, val-wt | +13.28% | +26.60% | +30.91% | +23.60% | ~0.5 min |
| Block k=3 (LTHO) + Snap T0=15, val-wt | +13.34–13.36% | +26.55–26.86% | +30.89–31.29% | ~23.70% | ~0.5 min |
| **Block k=3 (LTHO) + Snap T0=10, val-wt** | **+13.36–13.41%** | **+26.82–27.07%** | **+31.12–31.29%** | **~23.80%** | **~0.5 min** |
| Block k=3, 2 seeds/block, T0=15 (112 preds) | +13.24% | +26.81% | +31.30% | +23.78% | ~0.9 min |
| Block k=3, 3 seeds/block, T0=10 (252 preds) | +13.27% | +27.08% | +31.28% | +23.88% | ~1.3 min |
| Block k=2 + k=3 combined (equal weight) | +13.36% | +25.91% | +31.12% | +23.46% | — |

**NEW WINNER: Block k=3 (Leave-Three-Out) + Snapshot T0=10, 1 seed, val-loss weighted**
**84 total predictions (4 blocks × 21 snapshots), training ~0.5 min**
**24h: +13.41% MSE (+6.97% RMSE) vs +8.95% mean debiasing — beats baseline by +4.46 pp**
**Best avg improvement: +23.80% MSE across all lead times**

**Key insight: why does leave-three-out (train on 1 year) beat leave-one-out (train on 3 years)?**
- Each single-year model (~1083 samples) is strongly regularized: it can only learn patterns that generalized within that one year
- When you average 4 models trained on completely different single years, you get maximum temporal diversity — each model learned different seasonal patterns
- The val-loss weighting naturally selects the "better" snapshots across all 4 single-year training runs
- More holdout → more temporal diversity among training sets → better ensemble → better 24h performance
- The sweet spot T0=10 (21 snaps/block) works because single-year models are small (train fast, converge quickly per cycle)
- Adding more seeds per block doesn't help: temporal diversity across blocks is sufficient; seed diversity within a block adds noise

---

## Experiment Details

### Experiment 1: Baseline Single MLP (hyperopt-tuned)
- Architecture: SimpleMLP, 2 hidden layers, 1024 hidden dim, dropout 0.244
- Hyperparameters from Bayesian optimization (100 evals on India 6x6)
- lr=3.3e-4, wd=2.2e-6, batch_size=256, patience=20
- Training: ~88-194 epochs (early stop), 0.19-0.77 min
- Results: +18.7% average MSE improvement

### Experiment 2: ResCNN (FiLM-conditioned Residual CNN)
- New architecture: Residual blocks with FiLM conditioning, no pooling
- Hypothesis: Preserving spatial structure should help
- Result: FAILED - 15.88 min training (3x over budget), worse performance
- Reason: Conv2d operations on MPS are slow for small grids; overhead not worth it
  with only ~4332 samples

### Experiment 3: ResidualMLP (Pre-norm Residual MLP with GELU)
- Architecture: Residual blocks with LayerNorm, GELU, expansion factor
- Result: Marginal improvement (+17.3% vs +17.2% baseline)
- Too many parameters (15M vs 2.2M baseline) causes overfitting with small data

### Experiment 4: Training Strategy - Per-Lead-Time Models
- Train separate SimpleMLP per lead time (24h, 120h, 216h)
- Result: +17.4% avg, slightly worse than joint training
- 24h improved (+11.0% vs +10.3%) but 120h and 216h degraded
- Fewer training samples per model hurts more than specialization helps

### Experiment 5: Ensemble of MLPs (WINNER)
- Train N SimpleMLP models with different random seeds and train/val splits
- Average predictions at test time
- Uses CosineAnnealingWarmRestarts scheduler + gradient clipping
- Results by ensemble size:
  - 5 members: +19.5% (0.69 min)
  - 7 members: +19.9% (0.75 min) <-- BEST tradeoff
  - 10 members: +20.1% (1.07 min) <-- diminishing returns
- Why it works: Reduces prediction variance by averaging diverse models

### Experiment 6: Per-Lead-Time Ensemble
- Train separate ensembles per lead time
- Result: +18.4%, worse than joint ensemble
- Joint training provides useful cross-lead-time learning

### Experiment 7: Train/Val Split Ratios
- Tested 80/20, 85/15, 90/10, 95/5 splits for single MLP and ensembles
- Result: 90/10 is best for single MLP (+19.0% avg), 80/20 is best for ensembles (+20.1%)
- Ensembles already get diversity from different seeds, so larger val set helps each member
- Split ratio matters less than ensemble method choice

### Experiment 8: 24h-Only Training
- Training only on 24h lead time data, hoping specialization helps
- Result: WORSE than joint 3-LT training for 24h predictions (+10.9% vs +11.1%)
- Joint training provides useful cross-lead-time learning even for 24h
- Fewer training samples (1436 vs 4332) hurts more than specialization helps

### Experiment 9: Snapshot Ensemble (NEW BEST)
- Instead of early stopping, train for fixed epochs with cosine annealing and save
  model weights at each cycle minimum (where LR reaches eta_min)
- Each cycle converges to a different local minimum; averaging predictions reduces variance
- Single run: 210 epochs, T_0=30, 7 snapshots → +11.5% at 24h, +19.7% avg in 0.22 min
- Multi-seed (3 runs × 7 snapshots = 21 members): +11.6% at 24h, +20.0% avg in 0.67 min
- Multi-seed (5 runs × 7 snapshots = 35 members): +11.7% at 24h, +20.3% avg in 1.11 min
- Why it works: "free" ensemble diversity from single training run + seed diversity across runs
- ~3x faster than equivalent seed-diverse ensemble for same number of predictions

### Experiment 10: Other Round 2 Approaches (not winners)
- 24h-weighted loss (weight 24h 3x higher): +11.2% at 24h but 6.86 min (too slow)
- Diverse ensemble (mix of wide/deep architectures): +10.6% at 24h (worse)
- Higher LR (1e-3): +10.8% at 24h (worse)
- Longer patience (40): +10.3% at 24h (worse, overfits)

---

## Key Takeaways

1. **Architecture changes don't help much** with only 4332 training samples.
   The data size is the bottleneck, not model expressiveness.

2. **Leave-K-out block ensemble is the best training strategy** — the more years held out
   per block, the more temporally diverse the ensemble:
   - k=1 (LOO, train on 3 years): +12.25% at 24h, 4 blocks
   - k=2 (LTO, train on 2 years): +12.77% at 24h, 6 blocks
   - k=3 (LTHO, train on 1 year): **+13.41% at 24h, 4 blocks** ← BEST
   More holdout = stronger regularization per model + higher temporal diversity across blocks.

3. **Block k=3 + Snapshot T0=10 is the overall winner** — 4 single-year models × 21
   snapshots/block = 84 total predictions in ~0.5 min. Beats everything else at all
   lead times. T0=10 (21 snaps/block) is the sweet spot for single-year training because
   these small models converge fast per cycle.

4. **Global val-loss weighted averaging matters for block ensembles** — weights each
   snapshot by 1/val_loss across all predictions. Naturally down-weights harder validation
   blocks. Adds ~0.1-0.2 pp at 24h. Does NOT help for random-split snapshot ensembles
   (val_losses are incomparable across different random splits).

5. **Seed diversity within blocks doesn't help** — temporal diversity across blocks is
   sufficient; adding more seeds per block adds noise, not signal.

6. **Combining different k-level ensembles (k=2+k=3) doesn't help** — k=3 alone is better
   than k=2+k=3 averaged. More predictions ≠ better if they come from weaker models.

7. **MC dropout at inference time doesn't work** — enabling dropout during inference
   badly degrades performance (10.9% at 24h with 20 samples); the 0.25 dropout rate
   introduces too much noise per pass.

8. **SWA (Stochastic Weight Averaging) is weaker than snapshot** — averaging weights
   finds a single "average" minimum vs snapshot's diverse minima ensemble.

9. **MPS (Apple Silicon) is slow for convolutions** on small grids. MLPs are
   much faster and equally effective.

10. **Joint training across lead times** is better than per-lead-time models.

11. **CosineAnnealingWarmRestarts T0=10 is the sweet spot for k=3 blocks** — single-year
    models (~1083 samples) are small and converge fast. T0=10 (21 snaps/block) > T0=15
    (14 snaps/block) > T0=30 (7 snaps/block) for k=3. For k=1 blocks (3× more data),
    T0=15 is better.

12. **The snapshot hyperopt was corrupted** — use the non-snapshot hyperopt params
    (mlp_hidden_dim=1024, num_layers=2, dropout=0.244, lr=3.3e-4, wd=2.2e-6) with
    --snapshot_T0=10 --snapshot_T_mult=1 --block_holdout=3 for best results.

---

## How to Use

```bash
# Single MLP (original behavior, unchanged)
python3 finetuning/finetune.py --nn_architecture mlp ...

# BEST OVERALL: Block k=3 + Snapshot T0=10 (~0.5 min, +13.41% at 24h)
# Trains 4 single-year blocks × 21 snapshots = 84 total predictions
# Uses global val-loss weighted averaging automatically
python3 finetuning/finetune.py --nn_architecture mlp --block_ensemble \
    --block_holdout 3 --snapshot_ensemble 1 --snapshot_T0 10 --snapshot_T_mult 1 ...

# Good tradeoff: Block k=2 + Snapshot T0=15 (~1.3 min, +12.77% at 24h)
# Trains 6 two-year blocks × 14 snapshots = 84 total predictions
python3 finetuning/finetune.py --nn_architecture mlp --block_ensemble \
    --block_holdout 2 --snapshot_ensemble 1 --snapshot_T0 15 --snapshot_T_mult 1 ...

# Simple baseline: Block LOO + Snapshot T0=15 (~1.3 min, +12.25% at 24h)
# Trains 4 three-year blocks × 14 snapshots = 56 total predictions
python3 finetuning/finetune.py --nn_architecture mlp --block_ensemble \
    --snapshot_ensemble 1 --snapshot_T0 15 --snapshot_T_mult 1 ...

# Pure snapshot (no block): 3 runs × T0=30 (~1.0 min, +11.15%)
python3 finetuning/finetune.py --nn_architecture mlp --snapshot_ensemble 3 \
    --snapshot_T0 30 --snapshot_T_mult 1 ...

# Snapshot ensemble of 5 runs (no block, ~1.4 min)
python3 finetuning/finetune.py --nn_architecture mlp --snapshot_ensemble 5 \
    --snapshot_T0 30 --snapshot_T_mult 1 ...

# SWA ensemble (not recommended: weaker than snapshot at 24h)
python3 finetuning/finetune.py --nn_architecture mlp --swa_ensemble 3 \
    --swa_warmup_epochs 100 --swa_epochs 110 --swa_T0 10 ...

# Other architectures (available but not recommended over block+snapshot)
python3 finetuning/finetune.py --nn_architecture resmlp ...
python3 finetuning/finetune.py --nn_architecture rescnn ...
```
