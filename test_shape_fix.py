#!/usr/bin/env python3
"""
Test script to verify the tensor shape fix for the finetuning training loop.

This creates mock data with the same shapes as the real data and verifies that:
1. The dataloader unpacks correctly
2. The model can process the inputs
3. The correction can be added to forecast outputs
4. All tensor shapes match as expected
"""

import torch
import numpy as np
import sys
sys.path.append('finetuning')

from finetune import create_dataloader, SimpleMLP, UNet


def test_dataloader_and_shapes():
    """Test that dataloader and tensor operations work with correct shapes."""

    print("="*70)
    print("Testing Tensor Shape Fix")
    print("="*70)

    # Simulate the data dimensions from the architecture experiments
    n_samples = 1000
    n_training_vars = 6  # All input variables
    n_output_vars = 1    # Just 2m_temperature
    n_lat = 24
    n_lon = 24
    n_lead_times = 3
    batch_size = 128

    # Calculate feature dimensions
    input_features = n_training_vars * n_lat * n_lon  # 6 * 24 * 24 = 3456
    output_features = n_output_vars * n_lat * n_lon   # 1 * 24 * 24 = 576

    print(f"\nData dimensions:")
    print(f"  n_samples: {n_samples}")
    print(f"  n_training_vars: {n_training_vars}")
    print(f"  n_output_vars: {n_output_vars}")
    print(f"  n_lat x n_lon: {n_lat} x {n_lon}")
    print(f"  n_lead_times: {n_lead_times}")
    print(f"  Input features: {input_features}")
    print(f"  Output features: {output_features}")

    # Create mock data
    print(f"\nCreating mock data...")
    forecast_input = np.random.randn(n_samples, input_features).astype(np.float32)
    forecast_output = np.random.randn(n_samples, output_features).astype(np.float32)
    observations = np.random.randn(n_samples, output_features).astype(np.float32)
    lead_time_indices = np.random.randint(0, n_lead_times, n_samples)
    day_of_year_features = np.random.randn(n_samples, 2).astype(np.float32)

    print(f"  forecast_input shape: {forecast_input.shape}")
    print(f"  forecast_output shape: {forecast_output.shape}")
    print(f"  observations shape: {observations.shape}")
    print(f"  lead_time_indices shape: {lead_time_indices.shape}")
    print(f"  day_of_year_features shape: {day_of_year_features.shape}")

    # Create dataloader
    print(f"\nCreating dataloader...")
    dataloader = create_dataloader(
        forecast_input, forecast_output, observations,
        lead_time_indices, day_of_year_features, batch_size
    )
    print(f"  Dataloader created with batch_size={batch_size}")

    # Test unpacking one batch
    print(f"\nTesting batch unpacking...")
    for fc_input_batch, fc_output_batch, y_batch, lead_time_batch, doy_batch in dataloader:
        print(f"  fc_input_batch shape: {fc_input_batch.shape}")
        print(f"  fc_output_batch shape: {fc_output_batch.shape}")
        print(f"  y_batch shape: {y_batch.shape}")
        print(f"  lead_time_batch shape: {lead_time_batch.shape}")
        print(f"  doy_batch shape: {doy_batch.shape}")

        # Verify shapes
        assert fc_input_batch.shape[1] == input_features, \
            f"Expected fc_input_batch to have {input_features} features, got {fc_input_batch.shape[1]}"
        assert fc_output_batch.shape[1] == output_features, \
            f"Expected fc_output_batch to have {output_features} features, got {fc_output_batch.shape[1]}"
        assert y_batch.shape[1] == output_features, \
            f"Expected y_batch to have {output_features} features, got {y_batch.shape[1]}"

        print(f"  ✓ All batch shapes are correct!")
        break

    # Test MLP model
    print(f"\nTesting SimpleMLP model...")
    mlp_model = SimpleMLP(
        input_dim=input_features,
        hidden_dim=1024,
        output_dim=output_features,
        num_hidden_layers=4,
        n_lead_times=n_lead_times,
        dropout_rate=0.2
    )
    print(f"  Model created: {mlp_model.__class__.__name__}")
    print(f"    Input dim: {input_features}")
    print(f"    Output dim: {output_features}")

    # Test forward pass
    for fc_input_batch, fc_output_batch, y_batch, lead_time_batch, doy_batch in dataloader:
        pred_error = mlp_model(fc_input_batch, lead_time_batch, doy_batch)
        print(f"  pred_error shape: {pred_error.shape}")

        # Verify output shape
        assert pred_error.shape == fc_output_batch.shape, \
            f"Expected pred_error shape {fc_output_batch.shape}, got {pred_error.shape}"

        # Test adding correction to forecast output
        preds = fc_output_batch + pred_error
        print(f"  preds shape (after adding correction): {preds.shape}")

        # Verify final prediction shape
        assert preds.shape == y_batch.shape, \
            f"Expected preds shape {y_batch.shape}, got {preds.shape}"

        print(f"  ✓ MLP model forward pass successful!")
        print(f"  ✓ Tensor addition successful!")
        break

    # Test UNet model
    print(f"\nTesting UNet model...")
    unet_model = UNet(
        input_dim=input_features,
        hidden_dim=32,
        output_dim=output_features,
        n_lat=n_lat,
        n_lon=n_lon,
        n_lead_times=n_lead_times,
        dropout_rate=0.1
    )
    print(f"  Model created: {unet_model.__class__.__name__}")
    print(f"    Input dim: {input_features}")
    print(f"    Output dim: {output_features}")

    # Test forward pass
    for fc_input_batch, fc_output_batch, y_batch, lead_time_batch, doy_batch in dataloader:
        pred_error = unet_model(fc_input_batch, lead_time_batch, doy_batch)
        print(f"  pred_error shape: {pred_error.shape}")

        # Verify output shape
        assert pred_error.shape == fc_output_batch.shape, \
            f"Expected pred_error shape {fc_output_batch.shape}, got {pred_error.shape}"

        # Test adding correction to forecast output
        preds = fc_output_batch + pred_error
        print(f"  preds shape (after adding correction): {preds.shape}")

        # Verify final prediction shape
        assert preds.shape == y_batch.shape, \
            f"Expected preds shape {y_batch.shape}, got {preds.shape}"

        print(f"  ✓ UNet model forward pass successful!")
        print(f"  ✓ Tensor addition successful!")
        break

    print(f"\n" + "="*70)
    print("ALL TESTS PASSED! ✓")
    print("="*70)
    print("\nThe tensor shape fix is working correctly:")
    print("  • Dataloader unpacks 5 items correctly")
    print("  • Model receives forecast inputs (6 vars × 24×24 = 3456 features)")
    print("  • Model outputs correction (1 var × 24×24 = 576 features)")
    print("  • Correction is added to forecast outputs (576 features)")
    print("  • Final prediction matches observation shape (576 features)")
    print()


if __name__ == "__main__":
    test_dataloader_and_shapes()
