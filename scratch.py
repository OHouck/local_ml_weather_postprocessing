"""Scratchpad for quick model/ensemble experiments.

Use this file for temporary benchmarking and small synthetic checks.
It is intentionally lightweight and should not become production code.
"""

import numpy as np


patch_files = ["/Users/ohouck/globus/forecast_data/processed/africa_patches.npy",
               "/Users/ohouck/globus/forecast_data/processed/asia_patches.npy",
               "/Users/ohouck/globus/forecast_data/processed/europe_patches.npy",
               "/Users/ohouck/globus/forecast_data/processed/north_america_patches.npy",
               "/Users/ohouck/globus/forecast_data/processed/south_america_patches.npy",
               "/Users/ohouck/globus/forecast_data/processed/oceania_patches.npy"]


total_patches = 0
for patch_file in patch_files:
    patches = np.load(patch_file)
    n_patches = len(patches)
    total_patches += n_patches
print(f"Total number of patches across all continents: {total_patches}")
