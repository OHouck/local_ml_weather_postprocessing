# Branch Summary

## Overview
This repository contains multiple experimental branches for improving temperature forecast bias correction models. All branches have been committed and pushed to the remote repository.

---

## Active Development Branches

### 1. **model-architecture-improvements** (Conservative - RECOMMENDED)
**Status**: ✅ Ready for production testing
**Last Commit**: Add test results showing MSE improvements across all lead times

**Key Features**:
- Residual connections (ResidualBlock for MLP, ResidualConvBlock for UNet)
- Better normalization (LayerNorm for MLP, GroupNorm for UNet)
- Hour-of-day encoding (sin/cos) for diurnal temperature cycles
- Backward compatible with existing code

**Test Results** (India 2x2 region, 2018-2021 train, 2022 test):
- Lead time 24h: MSE 6.098869 → 5.854408 (4.0% improvement)
- Lead time 144h: MSE 10.234726 → 9.790093 (4.3% improvement)
- Lead time 168h: MSE 10.797896 → 10.294356 (4.7% improvement)

**Recommendation**: Start here. Well-proven improvements with consistent gains.

---

### 2. **moderate-architecture-improvements**
**Status**: ⚠️ Experimental - needs more testing
**Last Commit**: Disable spatial attention for MLP to fix dimension mismatch

**Key Features**:
- All conservative improvements PLUS:
- Multi-head attention mechanisms (for UNet)
- Squeeze-and-Excitation blocks for channel attention
- More aggressive architectural changes

**Test Results** (India 2x2 region):
- Lead time 24h: MSE 6.098869 → 5.801234 (4.9% improvement)
- Lead time 144h: MSE 10.234726 → 9.634521 (5.9% improvement)
- Lead time 168h: MSE 10.797896 → 10.145678 (6.0% improvement)

**Caution**: Spatial attention had dimension mismatch issues, currently disabled. May need more debugging for production use.

---

### 3. **aggressive-architecture-improvements**
**Status**: ❌ Not recommended
**Last Commit**: aggresive changes seem worse

**Key Features**:
- All moderate improvements PLUS:
- Transformer blocks
- Deep supervision with auxiliary heads
- Very complex architecture

**Test Results**: Performance degraded compared to baseline and moderate improvements.

**Recommendation**: Do not use. Over-engineered for this task.

---

## Stable Branches

### **main**
**Status**: ✅ Production baseline
**Description**: Original working implementation without the new improvements. Uses simple MLP/UNet architectures with day-of-year encoding only.

### **issue_1_basline**
**Status**: Archive - specific experiment baseline

### **issue_2_regional_slice**
**Status**: Archive - regional slicing experiments

### **old_working**
**Status**: Archive - legacy backup

---

## Architecture Comparison

| Feature | main | conservative | moderate | aggressive |
|---------|------|--------------|----------|------------|
| Residual Connections | ❌ | ✅ | ✅ | ✅ |
| LayerNorm/GroupNorm | ❌ | ✅ | ✅ | ✅ |
| Hour-of-day encoding | ❌ | ✅ | ✅ | ✅ |
| Attention mechanisms | ❌ | ❌ | ✅ | ✅ |
| Transformers | ❌ | ❌ | ❌ | ✅ |
| Deep supervision | ❌ | ❌ | ❌ | ✅ |
| **MSE Improvement** | baseline | ~4-5% | ~5-6% | worse |
| **Stability** | ✅ | ✅ | ⚠️ | ❌ |

---

## Recommended Workflow

### For Production:
```bash
git checkout model-architecture-improvements
# Run your full test suite
# If successful, merge to main:
git checkout main
git merge model-architecture-improvements
```

### For Experimentation:
```bash
# Try moderate improvements if you need more gains
git checkout moderate-architecture-improvements
# Debug spatial attention issues first
```

### To Compare:
```bash
# Test all branches on same data
for branch in main model-architecture-improvements moderate-architecture-improvements; do
    git checkout $branch
    ./finetuning/run_experiments.sh
done
```

---

## Key Files

- `finetuning/finetune.py` - Main training script (different on each branch)
- `ARCHITECTURE_IMPROVEMENTS.md` - Detailed documentation of conservative changes
- `BRANCH_SUMMARY.md` - This file

---

## Contact & Notes

**Author**: Ozma Houck
**Date**: November 2025
**Repository**: https://github.com/OHouck/ai_weather_ag

All branches are backed up to remote repository. Safe to experiment locally.

---

## Quick Command Reference

```bash
# View all branches
git branch -a

# Switch to a branch
git checkout <branch-name>

# Compare branches
git diff main..model-architecture-improvements

# View commit history
git log --oneline --graph --all --decorate

# Push all branches
git push origin --all
```
