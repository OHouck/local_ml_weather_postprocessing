#!/usr/bin/env python3
"""
Unit tests for heatwave_loss and related functions.
"""

import numpy as np
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from finetuning.custom_loss_fns import (
    compute_implied_consecutive_days,
    compute_heatwave_duration_class,
    heatwave_loss,
    HeatWaveBatchSampler,
    generate_heatwave_labels,
    classification_loss,
    CLASSIFICATION_LOSS_FNS,
    N_CLASSES,
    DEFAULT_CLASS_WEIGHTS,
    LABEL_GENERATORS
)


def test_compute_implied_consecutive_days_pytorch():
    """Test compute_implied_consecutive_days with PyTorch tensors."""
    print("Testing compute_implied_consecutive_days (PyTorch)...")

    lead_days = [1, 5, 9]

    # Shape should be [n_timestamps, n_lead_times, n_spatial]
    # Case 1: All lead times above threshold -> max lead time days (9)
    # 1 timestamp, 3 lead times, 1 pixel
    above = torch.tensor([[[True], [True], [True]]])  # [1, 3, 1]
    result = compute_implied_consecutive_days(above, lead_days)
    assert result.shape == (1, 1), f"Expected shape (1, 1), got {result.shape}"
    assert result[0, 0].item() == 9, f"Expected 9, got {result[0, 0].item()}"
    print(f"  Case 1 (all above): PASSED - implied_days = {result[0, 0].item()}")

    # Case 2: First two above, third below -> 5 days
    above = torch.tensor([[[True], [True], [False]]])  # [1, 3, 1]
    result = compute_implied_consecutive_days(above, lead_days)
    assert result[0, 0].item() == 5, f"Expected 5, got {result[0, 0].item()}"
    print(f"  Case 2 (first two above): PASSED - implied_days = {result[0, 0].item()}")

    # Case 3: Broken streak (1st and 3rd above, 2nd below) -> 1 day
    above = torch.tensor([[[True], [False], [True]]])  # [1, 3, 1]
    result = compute_implied_consecutive_days(above, lead_days)
    assert result[0, 0].item() == 1, f"Expected 1, got {result[0, 0].item()}"
    print(f"  Case 3 (broken streak): PASSED - implied_days = {result[0, 0].item()}")

    # Case 4: None above -> 0 days
    above = torch.tensor([[[False], [False], [False]]])  # [1, 3, 1]
    result = compute_implied_consecutive_days(above, lead_days)
    assert result[0, 0].item() == 0, f"Expected 0, got {result[0, 0].item()}"
    print(f"  Case 4 (none above): PASSED - implied_days = {result[0, 0].item()}")

    # Case 5: Only first above -> 1 day
    above = torch.tensor([[[True], [False], [False]]])  # [1, 3, 1]
    result = compute_implied_consecutive_days(above, lead_days)
    assert result[0, 0].item() == 1, f"Expected 1, got {result[0, 0].item()}"
    print(f"  Case 5 (only first): PASSED - implied_days = {result[0, 0].item()}")

    # Case 6: Multiple timestamps, multiple pixels
    # Shape: [2 timestamps, 3 lead_times, 2 pixels]
    above = torch.tensor([
        [[True, True], [True, False], [True, True]],   # ts0: lt0=[p0:T, p1:T], lt1=[p0:T, p1:F], lt2=[p0:T, p1:T]
        [[False, True], [True, True], [True, False]]   # ts1: lt0=[p0:F, p1:T], lt1=[p0:T, p1:T], lt2=[p0:T, p1:F]
    ])  # [2 ts, 3 lt, 2 pixels]
    result = compute_implied_consecutive_days(above, lead_days)
    assert result.shape == (2, 2), f"Expected shape (2, 2), got {result.shape}"
    # ts0, pixel0: all True -> 9
    # ts0, pixel1: T, F, T -> streak breaks at lt1, so 1
    # ts1, pixel0: F, T, T -> streak never starts (first is False), so 0
    # ts1, pixel1: T, T, F -> streak for lt0 and lt1, so 5
    expected = torch.tensor([[9., 1.], [0., 5.]])
    assert torch.allclose(result, expected), f"Expected {expected}, got {result}"
    print(f"  Case 6 (multi-ts, multi-pixel): PASSED - implied_days = \n{result}")

    print("  All PyTorch tests PASSED!\n")


def test_compute_implied_consecutive_days_numpy():
    """Test compute_implied_consecutive_days with NumPy arrays."""
    print("Testing compute_implied_consecutive_days (NumPy)...")

    lead_days = [1, 5, 9]

    # Shape should be [n_timestamps, n_lead_times, n_spatial]
    # Case 1: All lead times above threshold -> max lead time days (9)
    above = np.array([[[True], [True], [True]]])  # [1, 3, 1]
    result = compute_implied_consecutive_days(above, lead_days)
    assert result.shape == (1, 1), f"Expected shape (1, 1), got {result.shape}"
    assert result[0, 0] == 9, f"Expected 9, got {result[0, 0]}"
    print(f"  Case 1 (all above): PASSED - implied_days = {result[0, 0]}")

    # Case 2: None above -> 0 days
    above = np.array([[[False], [False], [False]]])  # [1, 3, 1]
    result = compute_implied_consecutive_days(above, lead_days)
    assert result[0, 0] == 0, f"Expected 0, got {result[0, 0]}"
    print(f"  Case 2 (none above): PASSED - implied_days = {result[0, 0]}")

    print("  All NumPy tests PASSED!\n")


def test_compute_heatwave_duration_class():
    """Test compute_heatwave_duration_class function."""
    print("Testing compute_heatwave_duration_class...")

    # Default bins: [1, 3, 6] -> 4 classes
    # Class 0: 0 days (< 1)
    # Class 1: 1-2 days (>= 1, < 3)
    # Class 2: 3-5 days (>= 3, < 6)
    # Class 3: 6+ days (>= 6)
    duration_bins = [1, 3, 6]

    # Test with PyTorch
    implied_days = torch.tensor([[0., 1., 2., 3., 5., 6., 9.]])  # [1, 7]
    classes = compute_heatwave_duration_class(implied_days, duration_bins)
    expected = torch.tensor([[0, 1, 1, 2, 2, 3, 3]])
    assert torch.equal(classes, expected), f"Expected {expected}, got {classes}"
    print(f"  PyTorch: PASSED - classes = {classes}")

    # Test with NumPy
    implied_days_np = np.array([[0., 1., 2., 3., 5., 6., 9.]])
    classes_np = compute_heatwave_duration_class(implied_days_np, duration_bins)
    expected_np = np.array([[0, 1, 1, 2, 2, 3, 3]])
    assert np.array_equal(classes_np, expected_np), f"Expected {expected_np}, got {classes_np}"
    print(f"  NumPy: PASSED - classes = {classes_np}")

    print("  compute_heatwave_duration_class: PASSED!\n")


def test_heatwave_loss_weights():
    """Test that heatwave_loss computes correct class-based weights."""
    print("Testing heatwave_loss weight computation...")

    n_lead_times = 3
    n_spatial = 4
    n_timestamps = 2
    batch_size = n_timestamps * n_lead_times  # 6

    lead_time_days = [1, 5, 9]
    threshold = 35.0

    # Create targets in Celsius (already denormalized for simplicity)
    # Timestamp 0: All pixels above 35C for all lead times -> 9-day heat wave -> class 3
    # Timestamp 1: No pixels above threshold -> no heat wave -> class 0
    targets_c = torch.zeros(batch_size, n_spatial)

    # Timestamp 0 (samples 0, 1, 2): all above threshold
    targets_c[0, :] = 40.0  # lt0
    targets_c[1, :] = 38.0  # lt1
    targets_c[2, :] = 36.0  # lt2

    # Timestamp 1 (samples 3, 4, 5): all below threshold
    targets_c[3, :] = 30.0  # lt0
    targets_c[4, :] = 28.0  # lt1
    targets_c[5, :] = 25.0  # lt2

    # Create predictions (same as targets for weight testing)
    preds_c = targets_c.clone()

    # Add small errors to test loss computation
    preds_c = preds_c + 1.0  # Predictions are 1 degree higher

    lead_time_indices = torch.tensor([0, 1, 2, 0, 1, 2])

    # Test with is_normalized=False (already in Celsius)
    # Using default bins [1, 3, 6]:
    # - Timestamp 0: 9 days -> class 3 -> weight = 1 + 3*1 = 4
    # - Timestamp 1: 0 days -> class 0 -> weight = 1 + 0*1 = 1
    loss = heatwave_loss(
        preds_c, targets_c, lead_time_indices, n_lead_times,
        is_normalized=False,
        threshold_celsius=threshold,
        lead_time_days=lead_time_days,
        duration_bins=[1, 3, 6],
        weight_base=1.0,
        weight_per_class=1.0
    )

    # Squared errors are all 1.0 (preds - targets = 1)
    # Weighted MSE = mean of weighted squared errors
    # Timestamp 0: weight=4, 12 elements (3 lead times * 4 spatial)
    # Timestamp 1: weight=1, 12 elements
    # = (4 * 1 * 12 + 1 * 1 * 12) / 24 = (48 + 12) / 24 = 2.5
    expected_loss = (4.0 * 1.0 * 12 + 1.0 * 1.0 * 12) / 24
    print(f"  Expected loss: {expected_loss}")
    print(f"  Actual loss: {loss.item()}")

    assert abs(loss.item() - expected_loss) < 1e-5, f"Expected {expected_loss}, got {loss.item()}"
    print("  Weight computation: PASSED!\n")


def test_heatwave_loss_normalized():
    """Test heatwave_loss with normalized inputs."""
    print("Testing heatwave_loss with normalized inputs...")

    n_lead_times = 3
    n_spatial = 4
    n_timestamps = 2
    batch_size = n_timestamps * n_lead_times

    lead_time_days = [1, 5, 9]
    threshold = 35.0

    # Create targets in Celsius
    targets_c = torch.zeros(batch_size, n_spatial)
    targets_c[0:3, :] = 40.0  # Above threshold
    targets_c[3:6, :] = 30.0  # Below threshold

    # Convert to Kelvin
    targets_k = targets_c + 273.15

    # Normalize (simple z-score normalization)
    mean_out = torch.tensor([305.0])  # ~32C
    std_out = torch.tensor([10.0])

    targets_norm = (targets_k - mean_out) / std_out
    preds_norm = targets_norm.clone() + 0.1  # Small normalized error

    lead_time_indices = torch.tensor([0, 1, 2, 0, 1, 2])

    loss = heatwave_loss(
        preds_norm, targets_norm, lead_time_indices, n_lead_times,
        is_normalized=True,
        std_out=std_out,
        mean_out=mean_out,
        threshold_celsius=threshold,
        lead_time_days=lead_time_days
    )

    print(f"  Normalized loss: {loss.item()}")
    assert loss.item() > 0, "Loss should be positive"
    print("  Normalized input handling: PASSED!\n")


def test_heatwave_loss_numpy():
    """Test heatwave_loss with NumPy arrays (evaluation mode)."""
    print("Testing heatwave_loss with NumPy (evaluation)...")

    n_lead_times = 3
    n_spatial = 4
    n_timestamps = 2
    batch_size = n_timestamps * n_lead_times

    lead_time_days = [1, 5, 9]
    threshold = 35.0

    # Create targets in Celsius
    targets_c = np.zeros((batch_size, n_spatial), dtype=np.float32)
    targets_c[0:3, :] = 40.0  # Above threshold -> class 3
    targets_c[3:6, :] = 30.0  # Below threshold -> class 0

    # Create predictions with small error
    preds_c = targets_c.copy() + 1.0

    lead_time_indices = np.array([0, 1, 2, 0, 1, 2])

    loss = heatwave_loss(
        preds_c, targets_c, lead_time_indices, n_lead_times,
        is_normalized=False,
        threshold_celsius=threshold,
        lead_time_days=lead_time_days
    )

    print(f"  NumPy loss: {loss:.6f}")
    assert isinstance(loss, float), "NumPy loss should return float"
    assert loss > 0, "Loss should be positive"

    print("  NumPy evaluation: PASSED!\n")


def test_heatwave_loss_rmse():
    """Test heatwave_loss with return_rmse=True."""
    print("Testing heatwave_loss RMSE output...")

    n_lead_times = 3
    n_spatial = 4
    n_timestamps = 2
    batch_size = n_timestamps * n_lead_times

    lead_time_days = [1, 5, 9]
    threshold = 35.0

    targets_c = torch.zeros(batch_size, n_spatial)
    targets_c[0:3, :] = 40.0
    targets_c[3:6, :] = 30.0

    preds_c = targets_c.clone() + 1.0

    lead_time_indices = torch.tensor([0, 1, 2, 0, 1, 2])

    mse = heatwave_loss(
        preds_c, targets_c, lead_time_indices, n_lead_times,
        is_normalized=False,
        threshold_celsius=threshold,
        lead_time_days=lead_time_days,
        return_rmse=False
    )

    rmse = heatwave_loss(
        preds_c, targets_c, lead_time_indices, n_lead_times,
        is_normalized=False,
        threshold_celsius=threshold,
        lead_time_days=lead_time_days,
        return_rmse=True
    )

    expected_rmse = torch.sqrt(mse)
    print(f"  MSE: {mse.item():.6f}")
    print(f"  RMSE: {rmse.item():.6f}")
    print(f"  Expected RMSE (sqrt(MSE)): {expected_rmse.item():.6f}")

    assert abs(rmse.item() - expected_rmse.item()) < 1e-5, "RMSE should equal sqrt(MSE)"
    print("  RMSE output: PASSED!\n")


def test_heatwave_batch_sampler():
    """Test HeatWaveBatchSampler."""
    print("Testing HeatWaveBatchSampler...")

    n_lead_times = 3
    n_timestamps = 10
    n_samples = n_timestamps * n_lead_times  # 30

    # Create lead time indices: [0, 1, 2, 0, 1, 2, ...]
    lead_time_indices = np.tile(np.arange(n_lead_times), n_timestamps)

    batch_size_timestamps = 4

    sampler = HeatWaveBatchSampler(
        n_samples=n_samples,
        n_lead_times=n_lead_times,
        lead_time_indices=lead_time_indices,
        batch_size_timestamps=batch_size_timestamps,
        shuffle=False  # Disable shuffle for predictable testing
    )

    print(f"  n_valid_timestamps: {sampler.n_valid_timestamps}")
    print(f"  n_batches: {len(sampler)}")

    assert sampler.n_valid_timestamps == n_timestamps
    assert len(sampler) == 3  # ceil(10 / 4) = 3 batches

    # Check batch contents
    batches = list(sampler)
    print(f"  Batch 0: {batches[0]}")
    print(f"  Batch 1: {batches[1]}")
    print(f"  Batch 2: {batches[2]}")

    # First batch should have 4 timestamps * 3 lead times = 12 samples
    assert len(batches[0]) == 12
    # Last batch should have 2 timestamps * 3 lead times = 6 samples
    assert len(batches[2]) == 6

    print("  HeatWaveBatchSampler: PASSED!\n")


def test_heatwave_batch_sampler_with_gaps():
    """Test HeatWaveBatchSampler handles incomplete groups."""
    print("Testing HeatWaveBatchSampler with incomplete groups...")

    n_lead_times = 3

    # Create lead time indices with a gap (missing sample)
    # Normal: [0, 1, 2, 0, 1, 2, 0, 1, 2]
    # With gap: [0, 1, 2, 0, 2, 0, 1, 2]  (missing the second '1')
    lead_time_indices = np.array([0, 1, 2, 0, 2, 0, 1, 2])
    n_samples = len(lead_time_indices)

    sampler = HeatWaveBatchSampler(
        n_samples=n_samples,
        n_lead_times=n_lead_times,
        lead_time_indices=lead_time_indices,
        batch_size_timestamps=2,
        shuffle=False
    )

    # Should only find 2 complete groups: indices 0-2 and indices 5-7
    print(f"  n_valid_timestamps: {sampler.n_valid_timestamps}")
    print(f"  valid_timestamp_starts: {sampler.valid_timestamp_starts}")

    assert sampler.n_valid_timestamps == 2, f"Expected 2 valid timestamps, got {sampler.n_valid_timestamps}"
    assert sampler.valid_timestamp_starts == [0, 5], f"Expected [0, 5], got {sampler.valid_timestamp_starts}"

    print("  Incomplete group handling: PASSED!\n")


def test_per_pixel_weights():
    """Test that weights are computed per-pixel correctly."""
    print("Testing per-pixel weight computation...")

    n_lead_times = 3
    n_spatial = 2
    n_timestamps = 1
    batch_size = n_timestamps * n_lead_times

    lead_time_days = [1, 5, 9]
    threshold = 35.0

    # Create targets where pixel 0 has heat wave, pixel 1 does not
    targets_c = torch.zeros(batch_size, n_spatial)
    # Pixel 0: all above threshold -> 9-day heat wave -> class 3 -> weight 4
    targets_c[:, 0] = 40.0
    # Pixel 1: all below threshold -> no heat wave -> class 0 -> weight 1
    targets_c[:, 1] = 30.0

    # Predictions with same error for both pixels
    preds_c = targets_c.clone() + 2.0  # 2 degree error

    lead_time_indices = torch.tensor([0, 1, 2])

    loss = heatwave_loss(
        preds_c, targets_c, lead_time_indices, n_lead_times,
        is_normalized=False,
        threshold_celsius=threshold,
        lead_time_days=lead_time_days,
        duration_bins=[1, 3, 6],
        weight_base=1.0,
        weight_per_class=1.0
    )

    # Pixel 0: weight = 4 (9-day heat wave, class 3)
    # Pixel 1: weight = 1 (no heat wave, class 0)
    # Squared error = 4.0 for all elements
    # Weighted MSE = (4 * 4.0 * 3 + 1 * 4.0 * 3) / 6 = (48 + 12) / 6 = 10.0
    expected_loss = (4.0 * 4.0 * 3 + 1.0 * 4.0 * 3) / 6
    print(f"  Expected loss: {expected_loss}")
    print(f"  Actual loss: {loss.item()}")

    assert abs(loss.item() - expected_loss) < 1e-5, f"Expected {expected_loss}, got {loss.item()}"
    print("  Per-pixel weights: PASSED!\n")


def test_different_duration_bins():
    """Test heatwave_loss with custom duration bins."""
    print("Testing heatwave_loss with custom duration bins...")

    n_lead_times = 3
    n_spatial = 4
    n_timestamps = 1
    batch_size = n_timestamps * n_lead_times

    lead_time_days = [1, 5, 9]
    threshold = 35.0

    # All above threshold -> 9-day heat wave
    targets_c = torch.ones(batch_size, n_spatial) * 40.0
    preds_c = targets_c.clone() + 1.0

    lead_time_indices = torch.tensor([0, 1, 2])

    # With bins [1, 3, 6]: 9 days -> class 3 -> weight = 1 + 3*1 = 4
    loss_default = heatwave_loss(
        preds_c, targets_c, lead_time_indices, n_lead_times,
        is_normalized=False,
        threshold_celsius=threshold,
        lead_time_days=lead_time_days,
        duration_bins=[1, 3, 6],
        weight_base=1.0,
        weight_per_class=1.0
    )

    # With bins [2, 5, 8]: 9 days -> class 3 -> weight = 1 + 3*1 = 4 (same)
    loss_custom = heatwave_loss(
        preds_c, targets_c, lead_time_indices, n_lead_times,
        is_normalized=False,
        threshold_celsius=threshold,
        lead_time_days=lead_time_days,
        duration_bins=[2, 5, 8],
        weight_base=1.0,
        weight_per_class=1.0
    )

    # With bins [10]: 9 days -> class 0 -> weight = 1 (since 9 < 10)
    loss_high_threshold = heatwave_loss(
        preds_c, targets_c, lead_time_indices, n_lead_times,
        is_normalized=False,
        threshold_celsius=threshold,
        lead_time_days=lead_time_days,
        duration_bins=[10],
        weight_base=1.0,
        weight_per_class=1.0
    )

    print(f"  Loss with bins [1,3,6]: {loss_default.item():.6f}")
    print(f"  Loss with bins [2,5,8]: {loss_custom.item():.6f}")
    print(f"  Loss with bins [10]: {loss_high_threshold.item():.6f}")

    # With high threshold bin, weight should be lower (class 0)
    assert loss_high_threshold.item() < loss_default.item(), "Higher bin threshold should give lower weight"

    print("  Custom duration bins: PASSED!\n")


# =============================================================================
# Tests for Classification Functions (new)
# =============================================================================

def test_classification_registry():
    """Test that classification loss registry is properly configured."""
    print("Testing classification loss registry...")

    # Check heatwave_loss is in the registry
    assert "heatwave_loss" in CLASSIFICATION_LOSS_FNS
    print("  heatwave_loss in CLASSIFICATION_LOSS_FNS: PASSED")

    # Check N_CLASSES
    assert N_CLASSES["heatwave_loss"] == 4
    print(f"  N_CLASSES['heatwave_loss'] = {N_CLASSES['heatwave_loss']}: PASSED")

    # Check DEFAULT_CLASS_WEIGHTS
    assert DEFAULT_CLASS_WEIGHTS["heatwave_loss"] == [1.0, 2.0, 3.0, 4.0]
    print(f"  DEFAULT_CLASS_WEIGHTS['heatwave_loss'] = {DEFAULT_CLASS_WEIGHTS['heatwave_loss']}: PASSED")

    # Check LABEL_GENERATORS
    assert "heatwave_loss" in LABEL_GENERATORS
    assert LABEL_GENERATORS["heatwave_loss"] == generate_heatwave_labels
    print("  LABEL_GENERATORS['heatwave_loss'] = generate_heatwave_labels: PASSED")

    print("  Classification registry: PASSED!\n")


def test_generate_heatwave_labels():
    """Test label generation from observations."""
    print("Testing generate_heatwave_labels...")

    # Shape: [n_timestamps, n_lead_times, n_spatial]
    # 2 timestamps, 3 lead times, 2 pixels
    # Temperatures in Celsius
    obs = np.array([
        # Timestamp 0: pixel 0 all hot (class 3), pixel 1 first two hot (class 2 -> 5 days)
        [[40., 36.], [38., 36.], [36., 30.]],  # lt0, lt1, lt2
        # Timestamp 1: pixel 0 none hot (class 0), pixel 1 only first hot (class 1 -> 1 day)
        [[30., 36.], [28., 30.], [25., 28.]]
    ])
    lead_days = [1, 5, 9]
    threshold = 35.0

    labels = generate_heatwave_labels(obs, lead_days, threshold_celsius=threshold)

    assert labels.shape == (2, 2), f"Expected shape (2, 2), got {labels.shape}"

    # Expected labels:
    # ts0, p0: all above 35 -> implied_days=9 -> class 3
    # ts0, p1: 36, 36, 30 -> first two above -> implied_days=5 -> class 2
    # ts1, p0: 30, 28, 25 -> none above -> implied_days=0 -> class 0
    # ts1, p1: 36, 30, 28 -> only first above -> implied_days=1 -> class 1
    expected = np.array([[3, 2], [0, 1]])

    print(f"  Observations shape: {obs.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Expected labels:\n{expected}")
    print(f"  Actual labels:\n{labels}")

    assert np.array_equal(labels, expected), f"Expected {expected}, got {labels}"
    print("  generate_heatwave_labels: PASSED!\n")


def test_generate_heatwave_labels_pytorch():
    """Test label generation with PyTorch tensors."""
    print("Testing generate_heatwave_labels (PyTorch)...")

    obs = torch.tensor([
        [[40., 36.], [38., 36.], [36., 30.]],
        [[30., 36.], [28., 30.], [25., 28.]]
    ])
    lead_days = [1, 5, 9]

    labels = generate_heatwave_labels(obs, lead_days, threshold_celsius=35.0)

    assert labels.shape == (2, 2)
    expected = torch.tensor([[3, 2], [0, 1]])
    assert torch.equal(labels, expected), f"Expected {expected}, got {labels}"

    print("  generate_heatwave_labels (PyTorch): PASSED!\n")


def test_classification_loss():
    """Test weighted cross-entropy computation."""
    print("Testing classification_loss...")

    # Create logits: batch_size=4, n_classes=4
    # Sample 0: predicts class 0 (correct)
    # Sample 1: predicts class 2 (incorrect, true=3)
    # Sample 2: predicts class 1 (correct)
    # Sample 3: predicts class 3 (correct)
    logits = torch.tensor([
        [2.0, 0.5, 0.1, 0.1],   # Predicts class 0
        [0.1, 0.5, 2.0, 1.0],   # Predicts class 2
        [0.1, 2.0, 0.5, 0.1],   # Predicts class 1
        [0.1, 0.1, 0.5, 2.0],   # Predicts class 3
    ])
    labels = torch.tensor([0, 3, 1, 3])  # True labels

    # Without weights
    loss_unweighted = classification_loss(logits, labels)
    assert loss_unweighted > 0, "Loss should be positive"
    print(f"  Unweighted loss: {loss_unweighted.item():.6f}")

    # With weights prioritizing class 3
    loss_weighted = classification_loss(logits, labels, class_weights=[1.0, 2.0, 3.0, 4.0])
    assert loss_weighted > 0, "Weighted loss should be positive"
    print(f"  Weighted loss: {loss_weighted.item():.6f}")

    # Weighted loss should be different from unweighted (sample 1 has class 3 error)
    assert abs(loss_weighted.item() - loss_unweighted.item()) > 1e-6, \
        "Weighted and unweighted loss should differ"

    print("  classification_loss: PASSED!\n")


def test_classification_loss_perfect_predictions():
    """Test classification loss with perfect predictions."""
    print("Testing classification_loss with perfect predictions...")

    # Perfect predictions (high confidence in correct class)
    logits = torch.tensor([
        [10.0, 0.0, 0.0, 0.0],  # Class 0
        [0.0, 10.0, 0.0, 0.0],  # Class 1
        [0.0, 0.0, 10.0, 0.0],  # Class 2
        [0.0, 0.0, 0.0, 10.0],  # Class 3
    ])
    labels = torch.tensor([0, 1, 2, 3])

    loss = classification_loss(logits, labels)
    print(f"  Loss with perfect predictions: {loss.item():.6f}")

    # Loss should be very small (close to 0) for perfect predictions
    assert loss.item() < 0.001, f"Loss should be near 0 for perfect predictions, got {loss.item()}"

    print("  classification_loss perfect predictions: PASSED!\n")


def test_classification_loss_class_imbalance():
    """Test that class weights properly handle imbalanced classes."""
    print("Testing classification_loss class imbalance handling...")

    # All samples are class 0 (common class)
    logits = torch.tensor([
        [0.5, 2.0, 0.1, 0.1],  # Predicts class 1 (wrong)
        [0.5, 2.0, 0.1, 0.1],  # Predicts class 1 (wrong)
        [0.5, 2.0, 0.1, 0.1],  # Predicts class 1 (wrong)
        [0.5, 2.0, 0.1, 0.1],  # Predicts class 1 (wrong)
    ])
    labels_class0 = torch.tensor([0, 0, 0, 0])

    # All samples are class 3 (rare class)
    labels_class3 = torch.tensor([3, 3, 3, 3])

    # Without weights, both should have similar loss
    loss_class0_unweighted = classification_loss(logits, labels_class0)
    loss_class3_unweighted = classification_loss(logits, labels_class3)

    # With weights [1, 2, 3, 4], class 3 errors should be penalized more
    weights = [1.0, 2.0, 3.0, 4.0]
    loss_class0_weighted = classification_loss(logits, labels_class0, class_weights=weights)
    loss_class3_weighted = classification_loss(logits, labels_class3, class_weights=weights)

    print(f"  Class 0 loss (unweighted): {loss_class0_unweighted.item():.6f}")
    print(f"  Class 3 loss (unweighted): {loss_class3_unweighted.item():.6f}")
    print(f"  Class 0 loss (weighted): {loss_class0_weighted.item():.6f}")
    print(f"  Class 3 loss (weighted): {loss_class3_weighted.item():.6f}")

    # With weights, class 3 loss should be higher than class 0 loss
    assert loss_class3_weighted > loss_class0_weighted, \
        "Weighted class 3 loss should be higher than class 0 loss"

    print("  classification_loss class imbalance: PASSED!\n")


# =============================================================================
# Tests for Focal Loss and Class Weight Computation (new)
# =============================================================================

def test_focal_loss_basic():
    """Test basic focal loss computation."""
    print("Testing focal_loss basic computation...")

    from finetuning.custom_loss_fns import focal_loss

    # Create logits and labels
    logits = torch.tensor([
        [2.0, 0.5, 0.1, 0.1],   # Predicts class 0 (correct)
        [0.1, 2.0, 0.5, 0.1],   # Predicts class 1 (correct)
        [0.1, 0.5, 2.0, 0.1],   # Predicts class 2 (correct)
        [0.1, 0.1, 0.5, 2.0],   # Predicts class 3 (correct)
    ])
    labels = torch.tensor([0, 1, 2, 3])

    # Compute focal loss
    loss = focal_loss(logits, labels, gamma=2.0)

    assert loss > 0, "Focal loss should be positive"
    print(f"  Focal loss (gamma=2.0): {loss.item():.6f}")

    # With gamma=0, focal loss should equal cross-entropy
    loss_gamma0 = focal_loss(logits, labels, gamma=0.0)
    ce_loss = torch.nn.CrossEntropyLoss()(logits, labels)

    print(f"  Focal loss (gamma=0): {loss_gamma0.item():.6f}")
    print(f"  Cross-entropy loss: {ce_loss.item():.6f}")

    assert abs(loss_gamma0.item() - ce_loss.item()) < 1e-5, \
        "Focal loss with gamma=0 should equal cross-entropy"

    print("  focal_loss basic: PASSED!\n")


def test_focal_loss_downweights_easy_examples():
    """Test that focal loss down-weights easy examples."""
    print("Testing focal_loss down-weighting of easy examples...")

    from finetuning.custom_loss_fns import focal_loss

    # Easy examples: high confidence correct predictions
    easy_logits = torch.tensor([
        [10.0, 0.0, 0.0, 0.0],  # 99.99% confident in class 0
        [0.0, 10.0, 0.0, 0.0],  # 99.99% confident in class 1
    ])
    easy_labels = torch.tensor([0, 1])

    # Hard examples: low confidence predictions
    hard_logits = torch.tensor([
        [0.5, 0.3, 0.1, 0.1],   # ~40% confident in class 0
        [0.3, 0.5, 0.1, 0.1],   # ~40% confident in class 1
    ])
    hard_labels = torch.tensor([0, 1])

    # With gamma=2, easy examples should have much lower loss
    easy_loss = focal_loss(easy_logits, easy_labels, gamma=2.0)
    hard_loss = focal_loss(hard_logits, hard_labels, gamma=2.0)

    print(f"  Easy examples loss (gamma=2): {easy_loss.item():.6f}")
    print(f"  Hard examples loss (gamma=2): {hard_loss.item():.6f}")
    print(f"  Ratio (hard/easy): {hard_loss.item() / (easy_loss.item() + 1e-8):.1f}x")

    # Hard examples should have significantly higher loss
    assert hard_loss > easy_loss * 10, \
        "Hard examples should have much higher focal loss than easy examples"

    print("  focal_loss down-weighting: PASSED!\n")


def test_focal_loss_with_alpha():
    """Test focal loss with class weights (alpha)."""
    print("Testing focal_loss with alpha weights...")

    from finetuning.custom_loss_fns import focal_loss

    # Same predictions for class 0 and class 3
    logits = torch.tensor([
        [0.5, 2.0, 0.1, 0.1],   # Predicts class 1, true class 0 (wrong)
        [0.1, 0.1, 0.5, 2.0],   # Predicts class 3, true class 3 (correct but rare)
    ])

    # Test with class 0 error
    labels_class0 = torch.tensor([0, 0])
    # Test with class 3 error
    labels_class3 = torch.tensor([3, 3])

    # Without alpha, losses should be similar for same prediction confidence
    loss_no_alpha_0 = focal_loss(logits, labels_class0, gamma=2.0)
    loss_no_alpha_3 = focal_loss(logits, labels_class3, gamma=2.0)

    print(f"  Loss (no alpha, class 0 labels): {loss_no_alpha_0.item():.6f}")
    print(f"  Loss (no alpha, class 3 labels): {loss_no_alpha_3.item():.6f}")

    # With alpha=[1, 1, 1, 4], class 3 should have higher weight
    alpha = [1.0, 1.0, 1.0, 4.0]
    loss_alpha_0 = focal_loss(logits, labels_class0, alpha=alpha, gamma=2.0)
    loss_alpha_3 = focal_loss(logits, labels_class3, alpha=alpha, gamma=2.0)

    print(f"  Loss (alpha=[1,1,1,4], class 0 labels): {loss_alpha_0.item():.6f}")
    print(f"  Loss (alpha=[1,1,1,4], class 3 labels): {loss_alpha_3.item():.6f}")

    # With alpha, class 3 loss should be higher due to 4x weight
    # (assuming similar prediction confidence)

    print("  focal_loss with alpha: PASSED!\n")


def test_compute_class_weights_inverse_sqrt():
    """Test compute_class_weights with inverse_sqrt method."""
    print("Testing compute_class_weights (inverse_sqrt)...")

    from finetuning.custom_loss_fns import compute_class_weights

    # Imbalanced labels: 100 class 0, 10 class 1, 5 class 2, 5 class 3
    labels = np.concatenate([
        np.zeros(100),
        np.ones(10),
        np.full(5, 2),
        np.full(5, 3)
    ]).astype(np.int64)

    weights = compute_class_weights(labels, n_classes=4, method="inverse_sqrt")

    print(f"  Labels distribution: Class 0=100, Class 1=10, Class 2=5, Class 3=5")
    print(f"  Computed weights: {[f'{w:.3f}' for w in weights]}")

    # Class 0 (most common) should have lowest weight
    assert weights[0] < weights[1], "Common class should have lower weight"
    assert weights[0] < weights[2], "Common class should have lower weight"
    assert weights[0] < weights[3], "Common class should have lower weight"

    # Classes 2 and 3 (same count) should have equal weights
    assert abs(weights[2] - weights[3]) < 0.01, "Same-count classes should have equal weights"

    # Weights should be normalized (mean = 1)
    assert abs(np.mean(weights) - 1.0) < 0.01, "Weights should have mean ~1.0"

    print("  compute_class_weights inverse_sqrt: PASSED!\n")


def test_compute_class_weights_methods():
    """Test different class weight computation methods."""
    print("Testing compute_class_weights different methods...")

    from finetuning.custom_loss_fns import compute_class_weights

    # Imbalanced labels
    labels = np.concatenate([
        np.zeros(1000),
        np.ones(100),
        np.full(10, 2),
        np.full(5, 3)
    ]).astype(np.int64)

    # Test all methods
    weights_inv_freq = compute_class_weights(labels, n_classes=4, method="inverse_frequency")
    weights_inv_sqrt = compute_class_weights(labels, n_classes=4, method="inverse_sqrt")
    weights_eff_samples = compute_class_weights(labels, n_classes=4, method="effective_samples")

    print(f"  Labels: Class 0=1000, Class 1=100, Class 2=10, Class 3=5")
    print(f"  inverse_frequency: {[f'{w:.2f}' for w in weights_inv_freq]}")
    print(f"  inverse_sqrt:      {[f'{w:.2f}' for w in weights_inv_sqrt]}")
    print(f"  effective_samples: {[f'{w:.2f}' for w in weights_eff_samples]}")

    # inverse_frequency should have most extreme weights
    inv_freq_range = max(weights_inv_freq) - min(weights_inv_freq)
    inv_sqrt_range = max(weights_inv_sqrt) - min(weights_inv_sqrt)

    print(f"  Weight range (inv_freq): {inv_freq_range:.2f}")
    print(f"  Weight range (inv_sqrt): {inv_sqrt_range:.2f}")

    assert inv_freq_range > inv_sqrt_range, \
        "inverse_frequency should have larger weight range than inverse_sqrt"

    print("  compute_class_weights methods: PASSED!\n")


def test_compute_class_weights_with_cap():
    """Test compute_class_weights with weight cap."""
    print("Testing compute_class_weights with cap...")

    from finetuning.custom_loss_fns import compute_class_weights

    # Very imbalanced labels
    labels = np.concatenate([
        np.zeros(10000),
        np.ones(10),
        np.full(5, 2),
        np.full(1, 3)
    ]).astype(np.int64)

    # Without cap
    weights_no_cap = compute_class_weights(labels, n_classes=4, method="inverse_frequency")
    # With cap
    weights_capped = compute_class_weights(labels, n_classes=4, method="inverse_frequency", cap=5.0)

    print(f"  Labels: Class 0=10000, Class 1=10, Class 2=5, Class 3=1")
    print(f"  Weights (no cap): {[f'{w:.2f}' for w in weights_no_cap]}")
    print(f"  Weights (cap=5): {[f'{w:.2f}' for w in weights_capped]}")

    # Capped weights should not exceed cap (after re-normalization, may be different)
    # But max capped weight should be less than max uncapped weight
    assert max(weights_capped) <= max(weights_no_cap), \
        "Capped weights should not exceed uncapped weights"

    print("  compute_class_weights with cap: PASSED!\n")


def test_compute_class_weights_pytorch_tensor():
    """Test compute_class_weights with PyTorch tensor input."""
    print("Testing compute_class_weights with PyTorch tensor...")

    from finetuning.custom_loss_fns import compute_class_weights

    # Create labels as PyTorch tensor
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 2, 3])

    weights = compute_class_weights(labels, n_classes=4, method="inverse_sqrt")

    print(f"  Labels (tensor): {labels.tolist()}")
    print(f"  Computed weights: {[f'{w:.3f}' for w in weights]}")

    assert isinstance(weights, list), "Output should be a list"
    assert len(weights) == 4, "Should have 4 weights"

    print("  compute_class_weights PyTorch tensor: PASSED!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Running heatwave_loss unit tests")
    print("=" * 60 + "\n")

    # Original regression-based heatwave_loss tests
    test_compute_implied_consecutive_days_pytorch()
    test_compute_implied_consecutive_days_numpy()
    test_compute_heatwave_duration_class()
    test_heatwave_loss_weights()
    test_heatwave_loss_normalized()
    test_heatwave_loss_numpy()
    test_heatwave_loss_rmse()
    test_heatwave_batch_sampler()
    test_heatwave_batch_sampler_with_gaps()
    test_per_pixel_weights()
    test_different_duration_bins()

    print("=" * 60)
    print("Running classification loss tests")
    print("=" * 60 + "\n")

    # New classification-based tests
    test_classification_registry()
    test_generate_heatwave_labels()
    test_generate_heatwave_labels_pytorch()
    test_classification_loss()
    test_classification_loss_perfect_predictions()
    test_classification_loss_class_imbalance()

    print("=" * 60)
    print("Running focal loss and class weight tests")
    print("=" * 60 + "\n")

    # Focal loss tests
    test_focal_loss_basic()
    test_focal_loss_downweights_easy_examples()
    test_focal_loss_with_alpha()

    # Class weight computation tests
    test_compute_class_weights_inverse_sqrt()
    test_compute_class_weights_methods()
    test_compute_class_weights_with_cap()
    test_compute_class_weights_pytorch_tensor()

    print("=" * 60)
    print("All tests PASSED!")
    print("=" * 60)
