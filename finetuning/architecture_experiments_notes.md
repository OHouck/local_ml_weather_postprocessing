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

**Winner: Multi-seed Snapshot Ensemble of 5 runs (+20.3% avg, +1.6 pp over single MLP, 1.11 min)**
**Best 24h: Snapshot Ens with T=30/Tmult=2 (+11.8% MSE, 0.22 min)**
**Best tradeoff: Snapshot Ens 3 runs (+20.0% avg, +11.6% at 24h, 0.67 min)**

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

2. **Snapshot ensemble is the best approach** — it provides "free" ensemble diversity
   from a single training run by saving weights at each cosine cycle minimum. Combined
   with multi-seed runs, it achieves the best results in the least time.

3. **Ensemble is the most reliable way to improve** — it reduces variance without
   increasing bias, and works with any base model.

4. **MPS (Apple Silicon) is slow for convolutions** on small grids. MLPs are
   much faster and equally effective.

5. **Joint training across lead times** is better than per-lead-time models —
   the shared structure helps all lead times. Even for optimizing 24h specifically,
   training jointly with 120h and 216h produces better 24h predictions.

6. **CosineAnnealingWarmRestarts** helps over ReduceLROnPlateau for the MLP,
   as the periodic LR restarts help escape local minima. With T_mult=1, each cycle
   converges to a different local minimum that can be saved as a snapshot.

7. **Train/val split ratio matters less for ensembles** — 80/20 is fine for
   ensembles since they already get diversity from different seeds.

---

## How to Use

```bash
# Single MLP (original behavior, unchanged)
python3 finetuning/finetune.py --nn_architecture mlp ...

# Snapshot ensemble of 3 runs (RECOMMENDED - best accuracy/speed tradeoff)
# Each run saves ~7 snapshots → 21 total predictions averaged
python3 finetuning/finetune.py --nn_architecture mlp --snapshot_ensemble 3 ...

# Snapshot ensemble of 5 runs (best overall accuracy, ~1 min)
python3 finetuning/finetune.py --nn_architecture mlp --snapshot_ensemble 5 ...

# Seed-diverse ensemble of 7 MLPs (previous best, still good)
python3 finetuning/finetune.py --nn_architecture mlp --ensemble 7 ...

# Custom snapshot settings
python3 finetuning/finetune.py --nn_architecture mlp --snapshot_ensemble 3 \
    --snapshot_epochs 210 --snapshot_T0 30 ...

# Other architectures (available but not recommended over snapshot ensemble)
python3 finetuning/finetune.py --nn_architecture resmlp ...
python3 finetuning/finetune.py --nn_architecture rescnn ...
```
