"""
Custom loss functions for weather forecast post-processing.

This module provides unified loss functions that work for both:
- Training (PyTorch tensors, normalized Kelvin inputs)
- Evaluation (NumPy arrays, Celsius inputs)

The key functions handle temperature-based weighting and mortality-based
transformations for bias correction of 2m temperature forecasts.
"""

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


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
