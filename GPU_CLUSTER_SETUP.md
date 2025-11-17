# GPU Cluster Setup Guide (UChicago Midway3)

## Problem: Slow GPU Performance

If your GPU training is **slower than your laptop**, you likely have a **CPU bottleneck** due to insufficient CPU core allocation.

### Why This Happens

GPU training has several stages:
1. **Data loading** (CPU) - Reading from disk
2. **Data preprocessing** (CPU) - Normalization, batching, tensor creation
3. **CPU→GPU transfer** (CPU/GPU memory bandwidth)
4. **Forward/backward pass** (GPU) - Actual neural network computation
5. **GPU→CPU transfer** (for metrics, logging)

When you allocate **only 1 CPU core**, stages 1-3 become severe bottlenecks, and the GPU sits **idle** most of the time waiting for data.

---

## ✅ Recommended Resource Allocation

### For Architecture Experiments (Full Dataset)

```bash
sinteractive \
    --account=pi-id \
    --partition=gpu \
    --gres=gpu:1 \
    --time=04:00:00 \
    --ntasks-per-node=1 \
    --cpus-per-task=8 \
    --mem=32G
```

**Why these settings:**
- `--cpus-per-task=8`: Provides sufficient CPU cores for data loading (uses 4 workers + 1 main process)
- `--mem=32G`: Enough memory for large datasets
- `--time=04:00:00`: 4 hours per experiment (9 experiments = ~36 hours total, run in batches)

### For Quick Testing (1 month data)

```bash
sinteractive \
    --account=pi-id \
    --partition=gpu \
    --gres=gpu:1 \
    --time=01:00:00 \
    --ntasks-per-node=1 \
    --cpus-per-task=4 \
    --mem=16G
```

### For Batch Jobs (Submit All 9 Experiments)

Create `submit_experiments.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=arch_exp
#SBATCH --account=pi-id
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --output=logs/arch_exp_%j.out
#SBATCH --error=logs/arch_exp_%j.err

# Load modules (adjust for your cluster)
module load python/3.11
module load cuda/11.8

# Activate environment
source ~/envs/weather/bin/activate

# Run experiments
cd /path/to/ai_weather_ag
./run_architecture_experiments.sh
```

Submit with:
```bash
mkdir -p logs
sbatch submit_experiments.sbatch
```

---

## 🔍 Checking Your Current Bottleneck

### 1. Monitor GPU Utilization

In another terminal while training:
```bash
watch -n 1 nvidia-smi
```

**What to look for:**
- **GPU Utilization < 50%**: CPU bottleneck (not enough cores for data loading)
- **GPU Utilization 70-95%**: Good! GPU is being utilized
- **GPU Memory < 50%**: Can increase batch size for better efficiency

### 2. Check CPU Usage

```bash
htop
```

**What to look for:**
- **All cores at 100%**: Need more CPU cores
- **Most cores idle**: Likely good, GPU is the limiting factor

### 3. Check DataLoader Settings

Look for this line in your training output:
```
DataLoader settings: num_workers=0, pin_memory=True
  CPU cores available: 1
```

**What it means:**
- `num_workers=0`: Single-threaded data loading (slow, but necessary with 1 core)
- `num_workers=4`: Parallel data loading (fast, needs 5+ cores)
- `pin_memory=True`: Using pinned memory for faster CPU→GPU transfers

---

## ⚡ Expected Performance

With **proper CPU allocation** (8 cores):

### Full Dataset (2018-2021 train, 2022 test)

| GPU Type | MLP (each) | UNet (each) | Total (9 experiments) |
|----------|------------|-------------|-----------------------|
| **V100** | 15-25 min | 30-45 min | **3-5 hours** |
| **A100** | 10-20 min | 25-35 min | **2.5-4 hours** |
| **RTX 4090** | 12-22 min | 28-40 min | **3-4.5 hours** |

### Small Test (1 month train, 1 month test)

| GPU Type | MLP (each) | UNet (each) | Total (9 experiments) |
|----------|------------|-------------|-----------------------|
| **V100** | 1-2 min | 2-4 min | **15-30 min** |
| **A100** | 0.5-1.5 min | 1.5-3 min | **10-20 min** |

---

## 🐛 Troubleshooting

### GPU is slower than CPU/Laptop

**Symptoms:**
- GPU training takes longer than CPU on the same data
- `nvidia-smi` shows low GPU utilization (<30%)

**Causes:**
1. **Too few CPU cores** (most common with `--cpus-per-task=1`)
2. **Slow storage** (if data is on a slow filesystem)
3. **Small dataset** (GPU overhead dominates for tiny datasets)

**Solutions:**
1. Increase `--cpus-per-task` to at least 4, ideally 8
2. Copy data to local/scratch storage: `rsync -av data/ $TMPDIR/data/`
3. For small tests (<1 week data), CPU might actually be faster

### Out of GPU Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce batch size in `finetune.py` (change `batch_size=128` to `batch_size=64` or `32`)
2. Use lighter architecture (UNet Light or MLP Skinny Deep)
3. Reduce region size: `--subregion=4x4` instead of `6x6`
4. Request GPU with more memory: `--gres=gpu:v100:1` (32GB) or `--gres=gpu:a100:1` (40GB)

### DataLoader Workers Crashing

**Symptoms:**
```
ERROR: DataLoader worker (pid X) is killed by signal: Killed
```

**Causes:**
- Not enough memory allocated
- Too many workers for available cores

**Solutions:**
1. Increase memory: `--mem=32G` or `--mem=64G`
2. Code now auto-detects and limits workers based on available cores

---

## 📊 Understanding the Auto-Detection

The updated code automatically adjusts to your resource allocation:

```python
# Auto-detect available CPU cores
cpu_count = os.cpu_count() or 1

if cpu_count <= 2:
    num_workers = 0  # Too few cores, use main process
    pin_memory = True
else:
    num_workers = min(cpu_count - 1, 4)  # Leave 1 core free, cap at 4
    pin_memory = True
```

**What this means:**
- **1-2 cores allocated**: Uses main process for data loading (no parallelism, but no overhead)
- **3-5 cores allocated**: Uses 2-4 parallel workers
- **6+ cores allocated**: Uses 4 parallel workers (optimal, more provides diminishing returns)

---

## 🎯 Quick Reference

### Current Allocation (Slow)
```bash
--cpus-per-task=1  ⚠️ CPU bottleneck!
```
→ GPU sits idle waiting for CPU to prepare data

### Recommended Allocation (Fast)
```bash
--cpus-per-task=8  ✅ Balanced GPU/CPU
```
→ GPU stays busy, data loading happens in parallel

### Check Your Settings
```bash
# Before training starts, you'll see:
DataLoader settings: num_workers=4, pin_memory=True
  CPU cores available: 8
```

---

## 💡 Pro Tips

1. **Start with a small test**: Use 1 month of data to verify everything works before running full experiments

2. **Use scratch storage**: Copy data to `$TMPDIR` (local SSD) for faster I/O:
   ```bash
   cp -r ~/data $TMPDIR/
   python3 finetune.py --data_dir $TMPDIR/data ...
   ```

3. **Monitor resource usage**: Keep `nvidia-smi` and `htop` running to identify bottlenecks

4. **Batch submit jobs**: For 9 experiments, submit as one batch job instead of 9 separate jobs (saves queue time)

5. **Choose the right GPU**:
   - V100: Good balance of speed and availability
   - A100: Fastest, but may have longer queue times
   - T4: Slowest, but more available

6. **Request appropriate time**:
   - Add 20% buffer to estimated time
   - If job times out, you lose all progress

---

## 📝 Example: Complete Workflow

```bash
# 1. Request interactive session with proper resources
sinteractive --account=pi-id --partition=gpu --gres=gpu:1 \
    --time=01:00:00 --cpus-per-task=8 --mem=32G

# 2. Load environment
module load python/3.11 cuda/11.8
source ~/envs/weather/bin/activate

# 3. Navigate to project
cd ~/ai_weather_ag

# 4. Run quick test (1 month)
python3 finetuning/finetune.py \
    --region=india --subregion=6x6 \
    --model_name=pangu \
    --nn_architecture=mlp \
    --mlp_hidden_dim=1024 --mlp_num_layers=6 --mlp_dropout=0.25 \
    --training_vars 2m_temperature 10m_u_component_of_wind 10m_v_component_of_wind \
                    temperature_1000hPa specific_humidity_1000hPa geopotential_1000hPa \
    --output_vars 2m_temperature \
    --lead_time_hours 24 120 216 \
    --train_start=2020-01-01 --train_end=2020-01-31 \
    --test_start=2020-02-01 --test_end=2020-02-28 \
    --data_dir=~/ai_weather_ag/data/raw \
    --output_dir=~/test_output

# 5. Check output shows good settings
# Should see: DataLoader settings: num_workers=4, pin_memory=True
#             CPU cores available: 8

# 6. If test works well, submit full experiments
sbatch submit_experiments.sbatch
```

---

**Questions?** Check the UChicago RCC documentation: https://rcc-uchicago.github.io/user-guide/
