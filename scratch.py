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

africa_patches = np.load("/Users/ohouck/globus/forecast_data/processed/africa_patches.npy")
asia_patches = np.load("/Users/ohouck/globus/forecast_data/processed/asia_patches.npy")

print(f"Africa patches shape: {africa_patches.shape}"
      f"\nAsia patches shape: {asia_patches.shape}")

print(africa_patches)

