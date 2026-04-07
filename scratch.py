"""Scratchpad for quick model/ensemble experiments.

Use this file for temporary benchmarking and small synthetic checks.
It is intentionally lightweight and should not become production code.
"""

import numpy as np

from finetuning.finetune import combine_snapshot_predictions


def demo_snapshot_weighting():
    """Compare uniform vs inverse-loss snapshot averaging on toy predictions."""
    snapshot_predictions = [
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([[0.8, 2.2], [3.1, 3.9]]),
        np.array([[1.2, 1.9], [2.9, 4.1]]),
    ]
    snapshot_val_losses = [0.12, 0.08, 0.20]

    uniform = combine_snapshot_predictions(snapshot_predictions, mode='uniform')
    weighted = combine_snapshot_predictions(
        snapshot_predictions,
        snapshot_val_losses=snapshot_val_losses,
        mode='inverse_loss',
    )

    print("Uniform mean:\n", uniform)
    print("Inverse-loss weighted mean:\n", weighted)


if __name__ == "__main__":
    demo_snapshot_weighting()