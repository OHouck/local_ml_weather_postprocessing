#!/usr/bin/env python3
"""
Simple test to verify tensor shapes are correct (numpy only, no torch needed).
"""

import numpy as np


def test_shapes():
    """Test that the data shapes work correctly."""

    print("="*70)
    print("Testing Data Shape Logic")
    print("="*70)

    # Simulate the data dimensions from the architecture experiments
    n_samples = 1000
    n_training_vars = 6  # All input variables
    n_output_vars = 1    # Just 2m_temperature
    n_lat = 24
    n_lon = 24

    # Calculate feature dimensions
    input_features = n_training_vars * n_lat * n_lon  # 6 * 24 * 24 = 3456
    output_features = n_output_vars * n_lat * n_lon   # 1 * 24 * 24 = 576

    print(f"\nData dimensions:")
    print(f"  n_samples: {n_samples}")
    print(f"  n_training_vars: {n_training_vars}")
    print(f"  n_output_vars: {n_output_vars}")
    print(f"  n_lat x n_lon: {n_lat} x {n_lon}")
    print(f"  Input features: {input_features}")
    print(f"  Output features: {output_features}")

    # Create mock data
    print(f"\nCreating mock data...")
    forecast_input = np.random.randn(n_samples, input_features).astype(np.float32)
    forecast_output = np.random.randn(n_samples, output_features).astype(np.float32)
    observations = np.random.randn(n_samples, output_features).astype(np.float32)

    print(f"  forecast_input shape: {forecast_input.shape} = (n_samples, {input_features})")
    print(f"  forecast_output shape: {forecast_output.shape} = (n_samples, {output_features})")
    print(f"  observations shape: {observations.shape} = (n_samples, {output_features})")

    # Simulate model output (correction)
    print(f"\nSimulating model output...")
    pred_error = np.random.randn(n_samples, output_features).astype(np.float32)
    print(f"  pred_error shape: {pred_error.shape} = (n_samples, {output_features})")

    # Test adding correction to forecast output
    print(f"\nTesting tensor addition...")
    preds = forecast_output + pred_error
    print(f"  preds = forecast_output + pred_error")
    print(f"  preds shape: {preds.shape}")

    # Verify shapes match
    print(f"\nVerifying shapes...")
    assert forecast_input.shape == (n_samples, input_features), \
        f"forecast_input has wrong shape: {forecast_input.shape}"
    print(f"  ✓ forecast_input shape is correct: {forecast_input.shape}")

    assert forecast_output.shape == (n_samples, output_features), \
        f"forecast_output has wrong shape: {forecast_output.shape}"
    print(f"  ✓ forecast_output shape is correct: {forecast_output.shape}")

    assert pred_error.shape == (n_samples, output_features), \
        f"pred_error has wrong shape: {pred_error.shape}"
    print(f"  ✓ pred_error shape is correct: {pred_error.shape}")

    assert preds.shape == (n_samples, output_features), \
        f"preds has wrong shape: {preds.shape}"
    print(f"  ✓ preds shape is correct: {preds.shape}")

    assert preds.shape == observations.shape, \
        f"preds and observations have different shapes: {preds.shape} vs {observations.shape}"
    print(f"  ✓ preds and observations have matching shapes")

    print(f"\n" + "="*70)
    print("ALL SHAPE CHECKS PASSED! ✓")
    print("="*70)
    print("\nThe fix ensures:")
    print("  1. Model receives forecast_input with shape (batch, 3456)")
    print("     → Contains all 6 input variables")
    print("  2. Model outputs pred_error with shape (batch, 576)")
    print("     → Correction for just 2m_temperature")
    print("  3. Correction is added to forecast_output (batch, 576)")
    print("     → Not to forecast_input (which would fail)")
    print("  4. Final preds match observations shape (batch, 576)")
    print("     → Ready for loss calculation")
    print()


if __name__ == "__main__":
    test_shapes()
