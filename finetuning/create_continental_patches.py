#!/usr/bin/env python3
"""
Script to create global 6x6 degree land patches divided by continent.

This script:
1. Divides the world into 6x6 degree grid cells
2. Keeps grid cells where >50% is land
3. Drops grid cells over Antarctica
4. Divides grid cells by continents and saves as separate .npy files
   (africa_patches.npy, asia_patches.npy, europe_patches.npy, etc.)

Author: AI Assistant
Date: 2025-11-17
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from helper_funcs import setup_directories
from clean_and_sample_climate_zones import create_global_land_patches


def main():
    """Create global land patches divided by continent."""

    # Setup directories
    dirs = setup_directories()

    # Create global land patches
    # Parameters:
    #   patch_size_deg: Size of each patch in degrees (default: 6)
    #   land_threshold: Minimum fraction of land (default: 0.5 = 50%)

    continent_patches = create_global_land_patches(
        dirs,
        patch_size_deg=6,
        land_threshold=0.5
    )

    # Print summary
    print("\n" + "="*70)
    print("Summary of created patches:")
    print("="*70)
    total = 0
    for continent, patches in continent_patches.items():
        n = len(patches)
        if n > 0:
            print(f"{continent:15s}: {n:4d} patches")
            total += n
    print(f"{'Total':15s}: {total:4d} patches")
    print("="*70)

    # Display saved file locations
    print(f"\nPatch files saved to: {dirs['processed']}/")
    print("\nFiles created:")
    for continent, patches in continent_patches.items():
        if len(patches) > 0:
            print(f"  - {continent}_patches.npy")

    print("\n" + "="*70)
    print("To load patches in another script:")
    print("="*70)
    print("import numpy as np")
    print(f"patches = np.load('{dirs['processed']}/africa_patches.npy', allow_pickle=True)")
    print("# Each patch is a tuple: (lat_slice, lon_slice)")
    print("# where lat_slice and lon_slice are numpy arrays of coordinates")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
