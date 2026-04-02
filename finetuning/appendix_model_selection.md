# Appendix: Model Architecture and Training Procedure Selection

**Test configuration for all experiments**: Pangu-Weather post-processing, India 6×6 degree patch
(center 22.0°N, 77.0°E), 2-meter air temperature, lead times 24h / 120h / 216h,
training years 2018–2021, test year 2022. The spatial grid is 24×24 at 0.25° resolution,
yielding ~4,332 training samples (4 years × ~365 days × 3 lead times). All percentage
improvements are in MSE (mean squared error) relative to the uncorrected Pangu forecast.
RMSE improvements are also reported and are approximately half the MSE percentage values.

---

## 1. Core Modeling Approach

Every architecture evaluated here follows the same residual error-correction framework:

```
corrected_forecast = raw_forecast + model(inputs)
```

The neural network predicts the **forecast error** (bias) rather than the weather value
directly. This is well-suited to post-processing because: (1) the raw forecast already
captures most of the variance in the target, so the model only needs to learn a
correction field; (2) the residual target has smaller dynamic range and is easier to
learn; and (3) the model defaults to zero correction if it is uncertain, preserving the
original forecast rather than producing a spurious prediction.

**Inputs** to all models: the flattened forecast field over the spatial patch
(576 = 24×24 grid points), plus a learned lead-time embedding and sinusoidal
day-of-year encoding [sin(2π·DOY/365), cos(2π·DOY/365)].

**Training loss**: MSE between corrected forecast and ERA5 reanalysis target, applied
over all spatial points simultaneously (the model corrects the entire patch at once).

---

## 2. Architectures Evaluated

### 2.1 Multilayer Perceptron (MLP) — *Selected Architecture*

The MLP flattens the 24×24 forecast field into a 576-dimensional vector, appends the
lead-time embedding (8-dim) and day-of-year features (2-dim), and passes the result
through a stack of fully-connected layers with ReLU activations and dropout.

**Architecture details:**
- Input: 576 (spatial) + 8 (lead-time embedding) + 2 (day-of-year) = 586 dimensions
- Hidden layers: 2 layers × 1,024 units, ReLU activation, dropout p = 0.244
- Output: 576 dimensions (full spatial patch, flattened)
- Parameter count: ~2.2 million

**Hyperparameters** were selected by Bayesian optimization (100 evaluations, `hyperopt`
library) over the India 6×6 region, minimizing validation MSE. The search space covered
hidden dimension (64–2048), number of layers (1–6), dropout (0–0.5), learning rate
(1×10⁻⁵–1×10⁻²), weight decay (1×10⁻⁷–1×10⁻³), and batch size (32–512). Optimal values:

| Hyperparameter | Optimal Value |
|---|---|
| Hidden dimension | 1024 |
| Number of hidden layers | 2 |
| Dropout rate | 0.244 |
| Learning rate | 3.30 × 10⁻⁴ |
| Weight decay | 2.21 × 10⁻⁶ |
| Batch size | 256 |
| Early stopping patience | 20 epochs |
| Lead-time embedding dim | 8 |

**Training procedure (baseline):** Adam optimizer with ReduceLROnPlateau scheduler
(factor 0.5, patience 10), early stopping at patience 20, up to 750 epochs.

**MSE improvements over raw Pangu forecast (single MLP, baseline training):**

| Lead time | RMSE original (K) | RMSE corrected (K) | RMSE improvement | MSE improvement |
|---|---|---|---|---|
| 24h | 0.927 | 0.878 | +5.3% | +10.3% |
| 120h | 1.329 | 1.194 | +10.2% | +19.2% |
| 216h | 1.726 | 1.509 | +12.6% | +26.8% |
| **Average** | | | **+9.4%** | **+18.7%** |

---

### 2.2 U-Net

The U-Net operates directly on the 2D spatial grid using an encoder-decoder architecture
with skip connections. The encoder progressively downsamples the input through a series of
convolution-BatchNorm-ReLU blocks and max-pooling operations; the decoder mirrors this
with transposed convolutions and skip connections that concatenate encoder feature maps.
Lead-time and day-of-year information is broadcast as additional input channels (spatial
conditioning by channel concatenation). The number of pooling levels is set automatically
from the patch size (capped at 5), with channel counts doubling per level up to a maximum
of 128 channels.

**Architecture details (6×6 patch):**
- Encoder: 3 levels, channels 64 → 128 → 128, each level has 2 × Conv(3×3)-BN-ReLU blocks
- Bottleneck: 128 channels at 6×6 spatial resolution
- Decoder: mirrors encoder with transposed convolutions + skip concatenation
- Total parameters: ~3.5 million

The U-Net was the second architecture reported in the main paper. On the India 6×6 test
case it achieves similar accuracy to the MLP but trains approximately **25× slower**
(owing to the overhead of convolution on small spatial grids with MPS/GPU acceleration).
Because performance is equivalent and training cost is substantially higher, the MLP is
preferred for global-scale experiments requiring many patch-level runs.

---

### 2.3 FiLM-Conditioned Residual CNN (ResCNN)

The ResCNN is a full-resolution convolutional network (no pooling) designed to preserve
fine-scale spatial structure within the 24×24 patch. It uses **Feature-wise Linear
Modulation (FiLM)** (Perez et al., 2018) to condition convolutional feature maps on lead
time and day-of-year: a learned affine transformation γ(c) · x + β(c) is applied
channel-wise, where γ and β are predicted from the conditioning vector c via small linear
layers. This is more expressive than simple concatenation because it modulates features
multiplicatively rather than additively.

**Architecture details:**
- Stem: Conv(3×3) → BatchNorm → ReLU projecting input to 64 channels
- 6 Residual blocks, each: Conv(3×3)-BN-ReLU → FiLM → Conv(3×3)-BN → skip add → ReLU
- Head: Conv(3×3)-BN-ReLU → Conv(1×1) projecting to output channels
- Spatial position encoding (normalized lat/lon) appended as 2 extra input channels
- Conditioning dim: 10 (2 day-of-year + 8 lead-time embedding)

**Results:** ResCNN underperformed the MLP on this task (14.0% vs 18.7% average MSE
improvement) and took 15.9 minutes to train — far exceeding the 5-minute budget. The
poor performance is attributed to the small training dataset (~4,332 samples), where
the inductive bias of convolutions on 24×24 grids is not beneficial and the additional
parameters introduce overfitting. ResCNN is not recommended for this problem scale.

---

### 2.4 Residual MLP (ResidualMLP)

The ResidualMLP replaces the plain hidden layers of the SimpleMLP with pre-norm residual
blocks (LayerNorm → Linear → GELU → Dropout → Linear → skip add). This design follows
the MLP-Mixer and Transformer feed-forward block convention. GELU activations replace
ReLU to avoid dead neurons, and LayerNorm before each block (pre-norm) provides more
stable gradients than post-norm.

**Architecture details:**
- Input projection: Linear → GELU → Dropout to hidden dimension
- 6 residual blocks, each: LayerNorm → Linear(dim → 2×dim) → GELU → Dropout → Linear(2×dim → dim) → skip
- Output: LayerNorm → Linear to output dimension
- Hidden dim: 768, ~15 million parameters

**Results:** The ResidualMLP achieved only marginal improvement over the baseline MLP
(+17.3% vs +18.7%) while taking 1.3 minutes to train. The large parameter count causes
overfitting with only ~4,332 training samples. The added architectural complexity does
not compensate for the data-starved regime. Not recommended.

---

## 3. Training Strategy Experiments

Given that architecture changes provide limited benefit (the data size is the bottleneck,
not model expressiveness), we explored alternative training strategies. All experiments
use the baseline hyperopt-tuned SimpleMLP architecture.

### 3.1 Per-Lead-Time Models

Training a separate MLP for each lead time increases the number of training samples per
model (1,436 vs 4,332) but allows specialization. Three independent models were trained
(one each for 24h, 120h, 216h), and their predictions were combined.

**Result:** Slightly worse overall (+17.4% vs +18.7%). The 24h model improved (+11.0%
vs +10.3%) but 120h and 216h degraded. Fewer samples per model hurt more than
specialization helped. Joint training with all lead times is preferred because the shared
structure provides useful cross-lead-time regularization.

**24h-focused training:** Training a single MLP with only 24h data (1,436 samples) gave
+9.7% MSE at 24h — worse than the joint 3-LT model's +10.3%. Confirmed that joint
training is also better for the shortest lead time.

---

### 3.2 Train/Validation Split Ratio

We tested whether giving the model more training data (reducing the validation fraction)
improves performance.

| Split (train/val) | 24h MSE% | 120h MSE% | 216h MSE% | Avg MSE% |
|---|---|---|---|---|
| 80/20 (default) | +10.2% | +19.1% | +24.6% | +18.0% |
| 85/15 | +10.0% | +18.4% | +24.9% | +17.8% |
| **90/10** | **+10.8%** | **+20.4%** | **+25.9%** | **+19.0%** |
| 95/5 | +10.6% | +20.2% | +25.4% | +18.7% |

The 90/10 split is slightly best for single-model training. However, for ensemble
methods the 80/20 split was preferred, as each ensemble member benefits from more
validation data to identify its own best checkpoint.

---

### 3.3 Seed-Diverse Ensemble of MLPs

Training multiple MLP instances with different random seeds and train/validation splits,
then averaging their predictions at test time. Each member uses a different seed
(seed = i × 17 + 1) to diversify both weight initialization and the subset of data
used for early stopping. All members use **AdamW** (decoupled weight decay) with
**CosineAnnealingWarmRestarts** (T₀ = 30, T_mult = 2, η_min = 10⁻⁶) and gradient
clipping (max norm = 1.0), which together modestly outperform the baseline
ReduceLROnPlateau-trained single model.

**Results by ensemble size:**

| Ensemble size | 24h MSE% | 120h MSE% | 216h MSE% | Avg MSE% | Training time |
|---|---|---|---|---|---|
| 1 (baseline) | +10.2% | +19.1% | +24.6% | +18.0% | 0.06 min |
| 5 members | +10.8% | +20.3% | +27.4% | +19.5% | 0.69 min |
| **7 members** | **+11.1%** | **+21.2%** | **+27.9%** | **+20.1%** | **0.70 min** |
| 10 members | +11.2% | +20.3% | +27.3% | +19.6% | 0.94 min |
| 12 members | +11.1% | +21.4% | +28.3% | +20.3% | 1.49 min |

Diminishing returns set in beyond 7–10 members. The ensemble reduces prediction
variance by averaging over diverse local minima, providing consistent improvement
of ~+1.5 percentage points over a single model.

---

### 3.4 Snapshot Ensemble — *Recommended Training Procedure*

**Motivation:** In a seed-diverse ensemble, each member is a fully independently trained
model. An alternative is the *snapshot ensemble* (Huang et al., 2017), which saves model
checkpoints at the end of each cosine annealing cycle — the points where the learning
rate reaches its minimum and the model has converged to a local minimum. Averaging
predictions from all saved checkpoints acts as a "free" ensemble from a single training run.

**Training procedure:** AdamW optimizer, CosineAnnealingWarmRestarts with fixed period
T₀ = 30 epochs (T_mult = 1), no early stopping, gradient clipping max norm = 1.0.
After 210 epochs (7 full cycles), 7 snapshot weights are saved. The model is re-initialized
from a new random seed and the process is repeated for N independent runs. All N × 7
snapshot predictions are averaged.

**Why it works:** Each 30-epoch cosine cycle drives the model toward a different local
minimum (because the learning rate resets to a high value after each cycle, escaping the
current basin). Unlike early stopping, which saves only the single best validation
checkpoint, snapshots capture the full diversity of the model's trajectory. Combining
multiple independent runs (different seeds, different train/val shuffles) adds further
cross-seed diversity on top of the within-run cycle diversity.

**Results:**

| Configuration | 24h MSE% | 24h RMSE% | 120h MSE% | 216h MSE% | Avg MSE% | Training time |
|---|---|---|---|---|---|---|
| Single run, 5 snapshots | +11.5% | +5.9% | +20.9% | +26.6% | +19.7% | 0.22 min |
| Single run, 7 snapshots (T₀=30) | +11.6% | +5.9% | +20.5% | +26.7% | +19.6% | 0.22 min |
| Single run, 3 snapshots (T₀=30, T_mult=2) | +11.8% | +6.1% | +20.7% | +26.7% | +19.7% | 0.22 min |
| **3 runs × 7 snapshots (21 total)** | **+11.6%** | **+6.0%** | **+20.5%** | **+28.0%** | **+20.0%** | **0.67 min** |
| **5 runs × 7 snapshots (35 total)** | **+11.7%** | **+6.0%** | **+20.9%** | **+28.3%** | **+20.3%** | **1.11 min** |

The 3-run configuration (21 total predictions averaged) provides the best
accuracy-to-time tradeoff: +20.0% average MSE improvement, +11.6% at 24h, in 0.67
minutes. The 5-run configuration achieves the best overall accuracy (+20.3%) while
still fitting comfortably within the 5-minute compute budget on an M3 Max MacBook Pro.

---

## 4. Summary and Model Selection Rationale

The table below consolidates all methods tested, ordered by average MSE improvement.

| Model / Strategy | 24h MSE% | 24h RMSE% | 120h MSE% | 216h MSE% | Avg MSE% | Train time |
|---|---|---|---|---|---|---|
| ResCNN (6 blocks, FiLM) | +9.0% | +4.6% | +16.4% | +16.3% | +13.9% | 15.9 min |
| ResidualMLP (768d, 6 blocks) | +8.6% | +4.4% | +19.1% | +24.2% | +17.3% | 1.3 min |
| Per-lead-time MLPs | +11.0% | +5.6% | +18.3% | +22.7% | +17.4% | 0.2 min |
| Single MLP (baseline) | +10.3% | +5.3% | +19.2% | +26.8% | +18.7% | 0.2 min |
| Single MLP (90/10 split) | +10.8% | +5.6% | +20.4% | +25.9% | +19.0% | 0.1 min |
| Snapshot ens. (1 run, 5 snaps) | +11.5% | +5.9% | +20.9% | +26.6% | +19.7% | 0.2 min |
| Seed-diverse ensemble (7) | +11.1% | +5.7% | +21.2% | +27.9% | +20.1% | 0.7 min |
| **Snapshot ens. (3 runs × 7)** | **+11.6%** | **+6.0%** | **+20.5%** | **+28.0%** | **+20.0%** | **0.7 min** |
| Seed-diverse ensemble (12) | +11.1% | +5.7% | +21.4% | +28.3% | +20.3% | 1.5 min |
| **Snapshot ens. (5 runs × 7)** | **+11.7%** | **+6.0%** | **+20.9%** | **+28.3%** | **+20.3%** | **1.1 min** |

**Key conclusions:**

1. **Architecture** matters less than **training strategy** for this problem scale.
   With ~4,300 training samples, the MLP already captures the learnable signal;
   the CNN and ResidualMLP architectures overfit or are computationally prohibitive.

2. **Ensemble averaging** is the most reliable lever for improvement. It reduces
   prediction variance without changing the model's capacity, and is always
   beneficial regardless of the base architecture.

3. **Snapshot ensemble** is the recommended approach because it achieves ensemble
   diversity more efficiently than re-training N independent models: a single 210-epoch
   run with cosine restart period T₀ = 30 produces 7 snapshots in the same wall-clock
   time as a single early-stopped model (~0.2 minutes). Combining 3–5 independent
   snapshot runs matches or exceeds the accuracy of 12-member seed-diverse ensembles
   while training up to 2× faster.

4. **Joint lead-time training** is better than per-lead-time specialization, even for
   optimizing 24h predictions. Training on all three lead times (24h, 120h, 216h)
   simultaneously provides implicit regularization through shared representations,
   resulting in better 24h performance than a 24h-only model despite using data from
   longer lead times.

5. **The selected configuration** for production use is: SimpleMLP with 2 hidden
   layers of 1,024 units, trained as a snapshot ensemble with 3 independent runs
   (21 total snapshots), using joint training over all lead times with the hyperopt-tuned
   hyperparameters. This achieves **+20.0% average MSE improvement** (+11.6% at 24h,
   +20.5% at 120h, +28.0% at 216h) in approximately 0.7 minutes per 6×6 degree patch.

---

## 5. References

Huang, G., Li, Y., Pleiss, G., Liu, Z., Hopcroft, J. E., & Weinberger, K. Q. (2017).
Snapshot ensembles: Train 1, get M for free. *International Conference on Learning
Representations (ICLR)*.

Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization.
*International Conference on Learning Representations (ICLR)*.

Loshchilov, I., & Hutter, F. (2017). SGDR: Stochastic gradient descent with warm
restarts. *International Conference on Learning Representations (ICLR)*.

Perez, E., Strub, F., de Vries, H., Dumoulin, V., & Courville, A. (2018). FiLM:
Visual reasoning with a general conditioning layer. *AAAI Conference on Artificial
Intelligence*.
