"""
Custom loss functions for weather forecast post-processing.

This module provides unified loss functions that work for both:
- Training (PyTorch tensors, normalized Kelvin inputs)
- Evaluation (NumPy arrays, Celsius inputs)

The key functions handle temperature-based weighting and mortality-based
transformations for bias correction of 2m temperature forecasts.

Classification Loss Functions:
- heatwave_loss: Classifies heatwave duration (0-day, 1-day, 5-day, 9-day)
  Uses weighted cross-entropy loss instead of MSE.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None  # Placeholder for when torch is not available


# =============================================================================
# Classification Loss Functions Registry
# =============================================================================
# These loss functions require special data loading (all lead times concatenated
# per sample) and use ClassifierMLP instead of SimpleMLP.

CLASSIFICATION_LOSS_FNS = {"heatwave_loss"}

# Map from loss function name to number of classes
N_CLASSES = {
    "heatwave_loss": 4,  # 0-day, 1-day, 5-day, 9-day
}

# Default class weights for each classification loss (prioritize longer events)
DEFAULT_CLASS_WEIGHTS = {
    "heatwave_loss": [1.0, 2.0, 3.0, 4.0],
}

# Default focal loss gamma values for each classification loss
DEFAULT_FOCAL_GAMMA = {
    "heatwave_loss": 2.0,
}


def _is_torch_tensor(x):
    """Check if x is a PyTorch tensor."""
    return TORCH_AVAILABLE and isinstance(x, torch.Tensor)


def _get_ops(x):
    """
    Return the appropriate operations module (torch or numpy) based on input type.

    Returns:
        tuple: (ops module, power function, ones_like function)
    """
    if _is_torch_tensor(x):
        return torch, torch.pow, torch.ones_like
    else:
        return np, np.power, np.ones_like


def mortality_dose_response(temp_c):
    """
    Calculate mortality rate relative to reference temperature.

    Based on a rough approximation from Figure 4c of Carleton et al. 2022.
    Reference temperature is 22C where mortality effect is assumed to be zero.

    Args:
        temp_c: Temperature in Celsius (torch.Tensor or np.ndarray)

    Returns:
        Relative mortality rate (same type as input)
    """
    ops, pow_fn, _ = _get_ops(temp_c)
    delta = temp_c - 22.0
    mortality_estimate = 0.04 * pow_fn(delta, 2) - 0.00125 * pow_fn(delta, 3)
    return mortality_estimate


def mortality_weighted_loss(preds, targets, is_normalized, std_out=None, mean_out=None,
                           return_rmse=False):
    """
    Mortality-weighted loss/metric for temperature forecasts.

    Converts temperature predictions to mortality space using a dose-response
    curve from Carleton et al. 2022, then computes MSE (or RMSE) on the
    mortality estimates. This weights errors at extreme temperatures more
    heavily since they have larger health impacts.

    Args:
        preds: Predictions (torch.Tensor or np.ndarray)
        targets: Ground truth targets (same type as preds)
        is_normalized: bool - True if inputs are normalized Kelvin values,
                       False if inputs are already in Celsius
        std_out: Standard deviation for denormalization (required if is_normalized=True)
        mean_out: Mean for denormalization (required if is_normalized=True)
        return_rmse: If True, return RMSE; if False, return MSE (default)

    Returns:
        Loss value (torch.Tensor if inputs are tensors, float otherwise)

    Raises:
        ValueError: If is_normalized=True but std_out or mean_out not provided
    """
    ops, pow_fn, _ = _get_ops(preds)

    if is_normalized:
        if std_out is None or mean_out is None:
            raise ValueError("std_out and mean_out required when is_normalized=True")
        # Denormalize
        preds = preds * std_out + mean_out
        targets = targets * std_out + mean_out
        # Convert Kelvin to Celsius
        preds_c = preds - 273.15
        targets_c = targets - 273.15
    else:
        # Already in Celsius
        preds_c = preds
        targets_c = targets

    # Convert to mortality space
    mortality_preds = mortality_dose_response(preds_c)
    mortality_targets = mortality_dose_response(targets_c)

    # Compute errors in mortality space
    errors = mortality_targets - mortality_preds
    squared_errors = errors ** 2
    mse = squared_errors.mean()

    if return_rmse:
        if _is_torch_tensor(mse):
            return ops.sqrt(mse)
        else:
            return float(np.sqrt(mse))

    return mse


def extreme_heat_loss(preds, targets, is_normalized, std_out=None, mean_out=None,
                     return_rmse=False):
    """
    Extreme heat weighted loss/metric for temperature forecasts.

    Penalizes errors at extreme temperatures more heavily since they are
    more damaging. Applies multiplicative weights based on target temperature:
    - T <= 25C: weight = 1.0 (baseline)
    - 25C < T <= 30C: weight = 6.0
    - T > 30C: weight = 11.0

    Args:
        preds: Predictions (torch.Tensor or np.ndarray)
        targets: Ground truth targets (same type as preds)
        is_normalized: bool - True if inputs are normalized Kelvin values,
                       False if inputs are already in Celsius
        std_out: Standard deviation for denormalization (required if is_normalized=True)
        mean_out: Mean for denormalization (required if is_normalized=True)
        return_rmse: If True, return RMSE; if False, return MSE (default)

    Returns:
        Loss value (torch.Tensor if inputs are tensors, float otherwise)

    Raises:
        ValueError: If is_normalized=True but std_out or mean_out not provided
    """
    ops, _, ones_like = _get_ops(preds)

    if is_normalized:
        if std_out is None or mean_out is None:
            raise ValueError("std_out and mean_out required when is_normalized=True")
        # Denormalize
        preds = preds * std_out + mean_out
        targets = targets * std_out + mean_out
        # Convert Kelvin to Celsius
        preds_c = preds - 273.15
        targets_c = targets - 273.15
    else:
        # Already in Celsius
        preds_c = preds
        targets_c = targets

    errors = targets_c - preds_c
    squared_errors = errors ** 2

    # Create weights based on target temperature
    weights = ones_like(errors)

    if _is_torch_tensor(weights):
        # PyTorch path
        weights = weights + ((targets_c > 25) & (targets_c <= 30)).float() * 5  # 6x total
        weights = weights + (targets_c > 30).float() * 10  # 11x total
    else:
        # NumPy path
        weights = weights + ((targets_c > 25) & (targets_c <= 30)).astype(float) * 5
        weights = weights + (targets_c > 30).astype(float) * 10

    # Compute weighted MSE
    weighted_mse = (weights * squared_errors).mean()

    if return_rmse:
        if _is_torch_tensor(weighted_mse):
            return ops.sqrt(weighted_mse)
        else:
            return float(np.sqrt(weighted_mse))

    return weighted_mse


def joint_temp_wind_loss(preds, targets, is_normalized, std_out=None, mean_out=None,
                         return_rmse=False, n_output_vars=2, n_lat=None, n_lon=None):
    """
    Joint loss for simultaneous temperature and wind speed correction.

    Separates the flattened prediction/target vectors into per-variable components,
    computes MSE for each variable independently, then returns a weighted combination.
    Both variables are denormalized to physical units before computing the loss so that
    the weighting is meaningful across different physical scales.

    The flattened feature ordering must be [variable, latitude, longitude], which is
    the ordering produced by load_forecasts() after the dimension fix. Variable 0 is
    expected to be 2m_temperature (Kelvin) and variable 1 is 10m_wind_speed (m/s).

    Args:
        preds: Predictions [batch_size, n_output_vars * n_lat * n_lon]
               (torch.Tensor or np.ndarray)
        targets: Ground truth [batch_size, n_output_vars * n_lat * n_lon]
                 (same type as preds)
        is_normalized: bool - True if inputs are normalized, False if in physical units
        std_out: Standard deviation for denormalization [n_output_vars * n_lat * n_lon]
                 (required if is_normalized=True)
        mean_out: Mean for denormalization [n_output_vars * n_lat * n_lon]
                  (required if is_normalized=True)
        return_rmse: If True, return RMSE instead of MSE
        n_output_vars: Number of output variables (default 2)
        n_lat: Number of latitude points (required)
        n_lon: Number of longitude points (required)

    Returns:
        Weighted loss value (torch.Tensor if inputs are tensors, float otherwise)
    """
    ops, _, _ = _get_ops(preds)

    if n_lat is None or n_lon is None:
        raise ValueError("n_lat and n_lon are required for joint_temp_wind_loss")

    if is_normalized:
        if std_out is None or mean_out is None:
            raise ValueError("std_out and mean_out required when is_normalized=True")
        preds = preds * std_out + mean_out
        targets = targets * std_out + mean_out

    # Reshape to (batch, n_output_vars, n_lat, n_lon)
    batch_size = preds.shape[0]
    if _is_torch_tensor(preds):
        preds_4d = preds.view(batch_size, n_output_vars, n_lat, n_lon)
        targets_4d = targets.view(batch_size, n_output_vars, n_lat, n_lon)
    else:
        preds_4d = preds.reshape(batch_size, n_output_vars, n_lat, n_lon)
        targets_4d = targets.reshape(batch_size, n_output_vars, n_lat, n_lon)

    # Per-variable errors: variable 0 = temperature, variable 1 = wind speed
    # Shape of each: [batch_size, n_lat, n_lon]
    temp_errors = preds_4d[:, 0] - targets_4d[:, 0]
    wind_errors = preds_4d[:, 1] - targets_4d[:, 1]

    # Per-pixel same-sign check: raw weight = 2 if same sign, 1 if different
    same_sign = (temp_errors * wind_errors) > 0
    if _is_torch_tensor(same_sign):
        pixel_weights = 1.0 + same_sign.float()
    else:
        pixel_weights = 1.0 + same_sign.astype(float)

    # Normalize weights to sum to 1 across all pixels in the batch
    pixel_weights = pixel_weights / pixel_weights.sum()

    # Weighted MSE per variable (weights sum to 1, so result is comparable to plain MSE)
    temp_se = temp_errors ** 2
    wind_se = wind_errors ** 2
    weighted_temp_mse = (pixel_weights * temp_se).sum()
    weighted_wind_mse = (pixel_weights * wind_se).sum()

    # Equal weighting between the two variables
    weighted_loss = (weighted_temp_mse + weighted_wind_mse) / 2.0

    if return_rmse:
        if _is_torch_tensor(weighted_loss):
            return ops.sqrt(weighted_loss)
        else:
            return float(np.sqrt(weighted_loss))

    return weighted_loss


def quantile_loss(preds, targets, quantile=0.95):
    """
    Quantile loss (pinball loss) for asymmetric error penalization.

    Useful for optimizing specific quantiles of the error distribution.
    For quantile > 0.5, under-predictions are penalized more heavily.
    For quantile < 0.5, over-predictions are penalized more heavily.

    Note: This function works with normalized values and does not require
    denormalization since the quantile loss is scale-invariant.

    Args:
        preds: Predictions (torch.Tensor or np.ndarray)
        targets: Ground truth targets (same type as preds)
        quantile: Target quantile (default 0.95)

    Returns:
        Loss value (torch.Tensor if inputs are tensors, float otherwise)
    """
    ops, _, _ = _get_ops(preds)

    errors = targets - preds

    if _is_torch_tensor(errors):
        loss = ops.max((quantile - 1) * errors, quantile * errors).mean()
    else:
        loss = np.maximum((quantile - 1) * errors, quantile * errors).mean()
        loss = float(loss)

    return loss


def compute_implied_consecutive_days(above_threshold, lead_time_days):
    """
    Compute implied consecutive heat wave days based on which lead times are above threshold.
    Counts consecutive days from the first lead time until the streak breaks.
    Computed independently for each spatial pixel.

    Args:
        above_threshold: Boolean tensor/array [n_timestamps, n_lead_times, n_spatial]
        lead_time_days: List of lead time values in days (e.g., [1, 5, 9])

    Returns:
        Tensor/array of implied consecutive days [n_timestamps, n_spatial]
    """
    _, n_lead_times, _ = above_threshold.shape

    if _is_torch_tensor(above_threshold):
        # Convert to float for cumprod (True->1.0, False->0.0)
        above_float = above_threshold.float()

        # Cumulative product along lead_time dimension
        # This creates a mask where streak_mask[t, lt, s] = 1 if all lead times 0..lt are above threshold
        streak_mask = above_float.cumprod(dim=1)  # [n_ts, n_lt, n_spatial]

        # Create lead time tensor and broadcast
        lead_time_tensor = torch.tensor(lead_time_days, dtype=torch.float32, device=above_threshold.device)
        lead_time_tensor = lead_time_tensor.view(1, n_lead_times, 1)  # [1, n_lt, 1]

        # Multiply streak_mask by lead_time_days and take max along lead_time dim
        # This gives the maximum lead time day where the streak is still active
        implied_days = (streak_mask * lead_time_tensor).max(dim=1).values  # [n_ts, n_spatial]

    else:
        # NumPy implementation
        above_float = above_threshold.astype(np.float32)

        # Cumulative product along lead_time dimension
        streak_mask = np.cumprod(above_float, axis=1)

        # Create lead time array and broadcast
        lead_time_array = np.array(lead_time_days, dtype=np.float32).reshape(1, n_lead_times, 1)

        # Multiply and take max
        implied_days = (streak_mask * lead_time_array).max(axis=1)  # [n_ts, n_spatial]

    return implied_days


def compute_heatwave_duration_class(implied_days, duration_bins):
    """
    Convert implied consecutive heatwave days to duration class indices.

    Args:
        implied_days: Tensor/array of implied consecutive days [n_timestamps, n_spatial]
        duration_bins: List of bin boundaries (upper bounds, exclusive).
                      E.g., [1, 3, 6] means: class 0: 0 days, class 1: 1-2 days,
                      class 2: 3-5 days, class 3: 6+ days

    Returns:
        Class indices [n_timestamps, n_spatial] with values in [0, len(duration_bins)]
    """
    if _is_torch_tensor(implied_days):
        classes = torch.zeros_like(implied_days, dtype=torch.long)
        for i, boundary in enumerate(duration_bins):
            classes = classes + (implied_days >= boundary).long()
    else:
        classes = np.zeros_like(implied_days, dtype=np.int64)
        for i, boundary in enumerate(duration_bins):
            classes = classes + (implied_days >= boundary).astype(np.int64)

    return classes


# =============================================================================
# Classification Functions (for heatwave_loss and future classification losses)
# =============================================================================

def generate_heatwave_labels(observations, lead_time_days, threshold_celsius=35.0,
                             duration_bins=None):
    """
    Generate heatwave duration class labels from observed temperatures.

    Uses consecutive lead times above threshold logic:
    - If 1-day, 5-day, 9-day all above threshold = 9-day heatwave (class 3)
    - If 1-day, 5-day above but 9-day below = 5-day heatwave (class 2)
    - If only 1-day above = 1-day heatwave (class 1)
    - If none above = no heatwave (class 0)

    Args:
        observations: Temperature observations [n_timestamps, n_lead_times, n_spatial]
                      Should be in Celsius (not normalized Kelvin)
        lead_time_days: List of lead times in days (e.g., [1, 5, 9])
        threshold_celsius: Temperature threshold for heatwave (default 35.0)
        duration_bins: Bin boundaries for duration classes (default [1, 3, 6])

    Returns:
        labels: Class indices [n_timestamps, n_spatial] with values 0, 1, 2, or 3
            Class 0: No heatwave (0 days above threshold)
            Class 1: Short heatwave (1-2 days)
            Class 2: Medium heatwave (3-5 days)
            Class 3: Long heatwave (6+ days)
    """
    if duration_bins is None:
        duration_bins = [1, 3, 6]

    # Check which observations are above threshold
    above_threshold = observations > threshold_celsius  # [n_ts, n_lt, n_spatial]

    # Compute implied consecutive days using existing helper
    implied_days = compute_implied_consecutive_days(above_threshold, lead_time_days)

    # Convert to duration classes using existing helper
    labels = compute_heatwave_duration_class(implied_days, duration_bins)

    return labels  # [n_timestamps, n_spatial]


def classification_loss(logits, labels, class_weights=None, device=None):
    """
    Weighted cross-entropy loss for classification tasks.

    Generic function usable for any classification loss (heatwave, future losses, etc.)

    Args:
        logits: Model outputs [batch_size, n_classes] (PyTorch tensor)
        labels: Ground truth class indices [batch_size] (long tensor)
        class_weights: Per-class weights [n_classes], e.g., [1, 2, 3, 4]
                       Can be a list, numpy array, or torch tensor
        device: Torch device for weight tensor (inferred from logits if None)

    Returns:
        Weighted cross-entropy loss (scalar torch.Tensor)

    Raises:
        RuntimeError: If PyTorch is not available
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for classification_loss")

    if device is None and _is_torch_tensor(logits):
        device = logits.device

    if class_weights is not None:
        if _is_torch_tensor(class_weights):
            weights = class_weights.to(device)
        else:
            weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    return criterion(logits, labels)


def focal_loss(logits, labels, alpha=None, gamma=2.0, device=None):
    """
    Focal Loss for multi-class classification.

    Focal loss down-weights easy examples and focuses training on hard examples.
    This is particularly useful for class-imbalanced datasets where the model
    can achieve high accuracy by just predicting the majority class.

    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Where:
    - p_t is the predicted probability for the true class
    - alpha_t is the class weight for the true class
    - gamma is the focusing parameter (higher = more focus on hard examples)

    With gamma=2:
    - A sample predicted with 90% confidence: weight = (1-0.9)^2 = 0.01
    - A sample predicted with 50% confidence: weight = (1-0.5)^2 = 0.25
    - Hard examples get 25x more weight than easy ones

    Args:
        logits: Model outputs [batch_size, n_classes] (raw logits, not softmax)
        labels: Ground truth class indices [batch_size] (long tensor)
        alpha: Per-class weights [n_classes] for class balancing.
               If None, uses uniform weights (1.0 for all classes).
               Can be a list, numpy array, or torch tensor.
        gamma: Focusing parameter (default 2.0). Higher gamma = more focus on
               hard examples. gamma=0 is equivalent to standard cross-entropy.
        device: Torch device (inferred from logits if None)

    Returns:
        Focal loss (scalar torch.Tensor)

    Raises:
        RuntimeError: If PyTorch is not available
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for focal_loss")

    if device is None and _is_torch_tensor(logits):
        device = logits.device

    batch_size = logits.shape[0]

    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=1)

    # Get the probability for the true class: p_t
    p_t = probs[torch.arange(batch_size, device=device), labels]

    # Compute focal weight: (1 - p_t)^gamma
    # This down-weights easy examples (high p_t) and up-weights hard examples (low p_t)
    focal_weight = (1 - p_t) ** gamma

    # Compute cross-entropy loss per sample: -log(p_t)
    ce_loss = -torch.log(p_t + 1e-8)

    # Apply alpha (class-specific weights) if provided
    if alpha is not None:
        if not _is_torch_tensor(alpha):
            alpha = torch.tensor(alpha, dtype=torch.float32, device=device)
        else:
            alpha = alpha.to(device)
        alpha_t = alpha[labels]
        focal_loss_val = alpha_t * focal_weight * ce_loss
    else:
        focal_loss_val = focal_weight * ce_loss

    return focal_loss_val.mean()


def compute_class_weights(labels, n_classes, method="inverse_sqrt", smoothing=0.0, cap=None):
    """
    Compute class weights from label distribution.

    Useful for handling class imbalance by computing weights based on the
    actual class distribution in the training data.

    Args:
        labels: Array of class labels [n_samples] (numpy array or torch tensor)
        n_classes: Number of classes
        method: Weight computation method:
            - "inverse_frequency": weight_i = n_samples / (n_classes * count_i)
              Most aggressive - rare classes get very high weights
            - "inverse_sqrt": weight_i = sqrt(n_samples / count_i)
              Moderate - less extreme than pure inverse frequency
            - "effective_samples": Based on "Class-Balanced Loss Based on
              Effective Number of Samples" (Cui et al., 2019)
        smoothing: Add smoothing to counts to prevent extreme weights for
                   very rare classes (default 0.0, no smoothing)
        cap: Maximum weight cap to prevent extreme weights (default None = no cap)

    Returns:
        class_weights: List of weights [n_classes], normalized to have mean=1.0

    Example:
        >>> labels = np.array([0, 0, 0, 0, 1, 2])  # Imbalanced
        >>> weights = compute_class_weights(labels, 3, method="inverse_sqrt")
        >>> # Class 0 (4 samples) gets lower weight than Class 1,2 (1 sample each)
    """
    if _is_torch_tensor(labels):
        labels = labels.cpu().numpy()

    # Count samples per class
    counts = np.zeros(n_classes, dtype=np.float64)
    unique, cnts = np.unique(labels, return_counts=True)
    for cls, cnt in zip(unique, cnts):
        counts[int(cls)] = cnt

    # Add smoothing to prevent division by zero and extreme weights
    counts = counts + smoothing

    # Handle classes with zero samples
    counts = np.maximum(counts, 1.0)

    n_samples = len(labels)

    if method == "inverse_frequency":
        # Standard inverse frequency weighting
        # weight_i = n_samples / (n_classes * count_i)
        weights = n_samples / (n_classes * counts)
    elif method == "inverse_sqrt":
        # Square root to reduce extreme weights
        # weight_i = sqrt(n_samples / count_i)
        weights = np.sqrt(n_samples / counts)
    elif method == "effective_samples":
        # From "Class-Balanced Loss Based on Effective Number of Samples"
        # Effective number E_n = (1 - beta^n) / (1 - beta)
        # Weight = 1 / E_n
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / (effective_num + 1e-8)
    else:
        raise ValueError(f"Unknown method: {method}. "
                         f"Choose from: inverse_frequency, inverse_sqrt, effective_samples")

    # Normalize weights to have mean = 1
    weights = weights / weights.mean()

    # Apply cap if specified
    if cap is not None:
        weights = np.clip(weights, 0, cap)
        # Re-normalize after capping
        weights = weights / weights.mean()

    return weights.tolist()


# Register label generators for classification losses
# This allows run_subregion_experiment to get the right label generator for each loss
LABEL_GENERATORS = {
    "heatwave_loss": generate_heatwave_labels,
    # Future: "drought_loss": generate_drought_labels, etc.
}


def heatwave_loss(preds, targets, lead_time_indices, n_lead_times,
                  is_normalized, std_out=None, mean_out=None,
                  threshold_celsius=35.0,
                  lead_time_days=None,
                  duration_bins=None,
                  weight_base=1.0, weight_per_class=1.0,
                  return_rmse=False):
    """
    Heat wave weighted MSE loss with duration-class-based weights.

    Computes MSE between predictions and targets, weighted by heatwave duration class.
    Heatwave duration is determined by how many consecutive lead times have temperatures
    above the threshold. Longer heatwaves get higher weights, penalizing errors during
    sustained dangerous heat events more heavily.

    Default duration bins (4 classes):
        - Class 0: No heatwave (0 days above threshold) - weight = weight_base
        - Class 1: Short heatwave (1-2 days) - weight = weight_base + 1 * weight_per_class
        - Class 2: Medium heatwave (3-5 days) - weight = weight_base + 2 * weight_per_class
        - Class 3: Long heatwave (6+ days) - weight = weight_base + 3 * weight_per_class

    Args:
        preds: Model predictions (temperature corrections) [batch_size, n_spatial]
        targets: Ground truth temperatures [batch_size, n_spatial]
        lead_time_indices: Lead time index for each sample [batch_size]
        n_lead_times: Number of distinct lead times
        is_normalized: Whether inputs are normalized Kelvin
        std_out, mean_out: Statistics for denormalization
        threshold_celsius: Temperature threshold for heat wave (default 35°C)
        lead_time_days: List of lead time values in days (e.g., [1, 3, 5, 7, 9])
        duration_bins: List of bin boundaries for duration classes.
                      Default: [1, 3, 6] meaning classes for 0, 1-2, 3-5, 6+ days
        weight_base: Base weight for class 0 (no heatwave) (default 1.0)
        weight_per_class: Additional weight per class index (default 1.0)
        return_rmse: If True, return RMSE instead of MSE

    Returns:
        Loss value (torch.Tensor if inputs are tensors, float otherwise)

    Raises:
        ValueError: If is_normalized=True but std_out or mean_out not provided
        ValueError: If batch_size is not divisible by n_lead_times
    """
    if duration_bins is None:
        duration_bins = [1, 3, 6]  # 4 classes: 0 days, 1-2 days, 3-5 days, 6+ days

    n_classes = len(duration_bins) + 1
    ops, _, _ = _get_ops(preds)

    batch_size = preds.shape[0]
    n_spatial = preds.shape[1]

    if batch_size % n_lead_times != 0:
        raise ValueError(f"batch_size ({batch_size}) must be divisible by n_lead_times ({n_lead_times})")

    n_timestamps = batch_size // n_lead_times

    # Denormalize if needed
    if is_normalized:
        if std_out is None or mean_out is None:
            raise ValueError("std_out and mean_out required when is_normalized=True")
        preds_denorm = preds * std_out + mean_out
        targets_denorm = targets * std_out + mean_out
        # Convert Kelvin to Celsius
        preds_c = preds_denorm - 273.15
        targets_c = targets_denorm - 273.15
    else:
        preds_c = preds
        targets_c = targets

    # Compute squared errors
    errors = targets_c - preds_c
    squared_errors = errors ** 2

    # Reshape targets to [n_timestamps, n_lead_times, n_spatial] for duration class computation
    if _is_torch_tensor(targets_c):
        targets_reshaped = targets_c.view(n_timestamps, n_lead_times, n_spatial)
        squared_errors_reshaped = squared_errors.view(n_timestamps, n_lead_times, n_spatial)
    else:
        targets_reshaped = targets_c.reshape(n_timestamps, n_lead_times, n_spatial)
        squared_errors_reshaped = squared_errors.reshape(n_timestamps, n_lead_times, n_spatial)

    # Compute per-pixel threshold check: [n_timestamps, n_lead_times, n_spatial]
    above_threshold = targets_reshaped > threshold_celsius

    # Compute implied consecutive days per (timestamp, pixel): [n_timestamps, n_spatial]
    if lead_time_days is None:
        lead_time_days = list(range(1, n_lead_times + 1))

    implied_days = compute_implied_consecutive_days(above_threshold, lead_time_days)

    # Convert to duration classes: [n_timestamps, n_spatial]
    duration_classes = compute_heatwave_duration_class(implied_days, duration_bins)

    # Create class weights array/tensor
    class_weights_list = [weight_base + i * weight_per_class for i in range(n_classes)]

    # Map duration classes to weights: [n_timestamps, n_spatial]
    if _is_torch_tensor(duration_classes):
        class_weights_tensor = torch.tensor(class_weights_list, dtype=preds.dtype, device=preds.device)
        weights_per_pixel = class_weights_tensor[duration_classes]  # Index into weights by class
    else:
        class_weights_array = np.array(class_weights_list)
        weights_per_pixel = class_weights_array[duration_classes]

    # Expand weights to [n_timestamps, n_lead_times, n_spatial]
    if _is_torch_tensor(weights_per_pixel):
        weights = weights_per_pixel.unsqueeze(1).expand(-1, n_lead_times, -1)
    else:
        weights = np.expand_dims(weights_per_pixel, axis=1)
        weights = np.broadcast_to(weights, (n_timestamps, n_lead_times, n_spatial))

    # Compute weighted MSE
    weighted_squared_errors = weights * squared_errors_reshaped
    weighted_mse = weighted_squared_errors.mean()

    if return_rmse:
        if _is_torch_tensor(weighted_mse):
            return ops.sqrt(weighted_mse)
        else:
            return float(np.sqrt(weighted_mse))

    return weighted_mse


class HeatWaveBatchSampler:
    """
    Batch sampler that ensures each batch contains complete timestamp groups.
    Required for heat wave loss since we need all lead times for each timestamp together.
    Skips incomplete timestamp groups (where some lead times were removed due to NaN).

    Each timestamp group consists of n_lead_times consecutive samples in the original
    data ordering. This sampler identifies complete groups and yields batches that
    contain only complete groups.

    Args:
        n_samples: Total number of samples in dataset
        n_lead_times: Number of lead times per timestamp
        lead_time_indices: Array of lead time indices for each sample [n_samples]
        batch_size_timestamps: Number of timestamps per batch (actual batch size = this * n_lead_times)
        shuffle: Whether to shuffle timestamps between epochs (default True)
    """

    def __init__(self, n_samples, n_lead_times, lead_time_indices, batch_size_timestamps, shuffle=True):
        self.n_lead_times = n_lead_times
        self.batch_size_timestamps = batch_size_timestamps
        self.shuffle = shuffle

        # Identify complete timestamp groups
        # A complete group has n_lead_times consecutive samples with indices [0, 1, ..., n_lead_times-1]
        self.valid_timestamp_starts = []

        expected_pattern = list(range(n_lead_times))
        i = 0
        while i <= n_samples - n_lead_times:
            # Check if samples i to i+n_lead_times-1 form a complete group
            group_indices = lead_time_indices[i:i + n_lead_times].tolist()
            if group_indices == expected_pattern:
                self.valid_timestamp_starts.append(i)
                i += n_lead_times
            else:
                # Skip to next potential group start
                i += 1

        self.n_valid_timestamps = len(self.valid_timestamp_starts)

        if self.n_valid_timestamps == 0:
            raise ValueError("No complete timestamp groups found in dataset")

    def __iter__(self):
        # Get timestamp indices (indices into valid_timestamp_starts)
        timestamp_order = list(range(self.n_valid_timestamps))

        if self.shuffle:
            import random
            random.shuffle(timestamp_order)

        # Yield batches of complete timestamp groups
        for i in range(0, self.n_valid_timestamps, self.batch_size_timestamps):
            batch_timestamp_indices = timestamp_order[i:i + self.batch_size_timestamps]

            # Convert to sample indices
            batch_samples = []
            for ts_idx in batch_timestamp_indices:
                start_idx = self.valid_timestamp_starts[ts_idx]
                batch_samples.extend(range(start_idx, start_idx + self.n_lead_times))

            yield batch_samples

    def __len__(self):
        return (self.n_valid_timestamps + self.batch_size_timestamps - 1) // self.batch_size_timestamps
