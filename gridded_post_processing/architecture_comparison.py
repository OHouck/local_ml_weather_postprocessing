"""
Section 4 (part 2): compare gridded model specifications on the evaluation patches.

Trains every model variant of the architecture comparison (figure 5) for Pangu
on the evaluation patches, a reproducible 5% sample of the land patches, for
2m temperature and 10m wind speed:

    mlp                    single MLP with early stopping
    mlp_snapshot3          MLP snapshot ensemble of 3 runs (the production model)
    mlp_blockk3_snapshot1  block leave-time-out ensemble
    mlp_snapshot3_perlt    MLP snapshot ensemble of 3 runs per lead time
    unet                   single U-Net with early stopping

Outputs are written with the same naming as the production run, so the
production model's evaluation patches are shared with section 5 rather than
trained twice. Each variant uses its tuned hyperparameters from
tuned_hyperparameters.json (see hyperparameter_search.py).

Run on its own with:
    uv run python gridded_post_processing/architecture_comparison.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (EVALUATION_PATCH_FRACTION, MODEL_VARIANTS, PATCH_SAMPLE_SEED,  # noqa: E402
                    VARIABLES, sample_continent_patches, setup_directories)
from gridded_post_processing.run_post_processing import post_process_and_save  # noqa: E402
from gridded_post_processing.training import select_device  # noqa: E402


def run():
    """
    Train every model variant on every evaluation patch.

    Inputs:
        None.

    Returns:
        None. Writes one zarr file per variant, variable and patch.
    """
    directories = setup_directories()
    device = select_device()
    evaluation_patches = sample_continent_patches(directories, EVALUATION_PATCH_FRACTION,
                                                  PATCH_SAMPLE_SEED, split="eval")
    print(f"Architecture comparison on {len(evaluation_patches)} evaluation patches")
    for model_suffix in MODEL_VARIANTS:
        for variable in VARIABLES:
            for continent, patch_number, patch_coordinates in evaluation_patches:
                post_process_and_save(directories, "pangu", variable, continent, patch_number,
                                      patch_coordinates, model_suffix, device)


if __name__ == "__main__":
    run()
