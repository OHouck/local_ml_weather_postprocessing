---
name: Architecture improvements for weather post-processing
description: New architectures/methods implemented to improve 24h forecast correction
type: project
---

Added three new methods to finetuning/finetune.py to improve 24h forecast RMSE:

**Why:** 24h lead time is hardest because errors are small and dominated by mean bias;
the training loss is dominated by larger errors at longer lead times (120h, 216h).

## Implemented

### 1. GatedMLP (new architecture)
- `GatedMLP` class + `SwiGLUBlock` in finetune.py
- SwiGLU gating (gate × value, SiLU activation), LayerNorm, residual connections
- Small output initialization (std=0.01) so model starts near-zero correction
- Auto-scales hidden_dim to match SimpleMLP param count via `_compute_gated_mlp_hidden_dim()`
  (binary search to match params; at production config h=256/layers=2, GatedMLP h=167 ≈ 364K params)

### 2. Lead-Time-Weighted Loss (`--lead_time_loss_weights`)
- New `train_model_weighted()` function in finetune.py
- Weights MSE loss by lead time: e.g., 3.0x for 24h, 1.0x for 120h, 0.5x for 216h
- Benchmark config: `lead_time_loss_weights=[3.0, 1.0, 0.5]`

### 3. C-Mixup Data Augmentation (`--cmixup_alpha`)
- `cmixup_batch()` function in finetune.py
- Label-aware mixup: pairs samples with similar targets (Gaussian kernel on label distance)
- Benchmark config: `cmixup_alpha=0.4`
- Used together with lead_time_loss_weights in the "full combo" experiment

## Files changed
- `finetuning/finetune.py`: GatedMLP, SwiGLUBlock, cmixup_batch, train_model_weighted,
  _compute_gated_mlp_hidden_dim; updated model init, _create_model, CLI args
- `helper_funcs.py`: generate_output_path handles gated_mlp, _ltw, _cmix suffixes
- `finetuning/run_arch_experiments_eval.py`: EXPERIMENTS list with 4 configs
- `finetuning/figures_finetuning.py`: 4 new experiments in plot_arch_experiment_results
- `finetuning/plot_arch_experiment_results.py`: updated experiment list printout

## Benchmark experiments
- Block LTHO Ensemble (baseline, already run)
- GatedMLP Block LTHO (auto-scaled hidden_dim=167 for param parity)
- MLP + LT-Weighted Loss (3x weight on 24h, no block LTHO)
- GatedMLP + LT-Weighted + C-Mixup (full combo, no block LTHO)

**How to apply:** Run `python3 finetuning/run_arch_experiments_eval.py` then 
`python3 finetuning/plot_arch_experiment_results.py` to see results.
The benchmark run takes ~5-6 hours on MPS Mac (3 experiments × 37 cells × ~3 min/cell).
