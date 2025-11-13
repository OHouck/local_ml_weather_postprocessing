#!/usr/bin/env python3
"""
Test script for the map_global_improvements function.

This script generates global improvement maps for post-processed weather forecasts.
It creates 3 maps (one for each lead time: 24h, 120h, 216h) showing RMSE percent
improvement across all processed patches.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from helper_funcs import setup_directories
from finetuning.figures_finetuning import map_global_improvements

def main():
    # Setup directories
    dirs = setup_directories()

    # Example 1: Map improvements for PANGU 10m_wind_speed
    # This will create 3 maps (24h, 120h, 216h lead times)
    print("=" * 80)
    print("Creating global improvement maps for PANGU 10m_wind_speed...")
    print("=" * 80)
    map_global_improvements(
        dirs=dirs,
        model="pangu",
        variable="10m_wind_speed",
        zone_types=["tropical", "arid", "temperate", "flat", "hilly", "mountainous"]
    )

    # Example 2: Map improvements for IFS 2m_temperature
    print("\n" + "=" * 80)
    print("Creating global improvement maps for IFS 2m_temperature...")
    print("=" * 80)
    map_global_improvements(
        dirs=dirs,
        model="ifs",
        variable="2m_temperature",
        zone_types=["tropical", "arid", "temperate", "flat", "hilly", "mountainous"]
    )

    # Example 3: Map only climate zones for PANGU
    print("\n" + "=" * 80)
    print("Creating climate zones only maps for PANGU 10m_wind_speed...")
    print("=" * 80)
    map_global_improvements(
        dirs=dirs,
        model="pangu",
        variable="10m_wind_speed",
        zone_types=["tropical", "arid", "temperate"]
    )

    print("\n" + "=" * 80)
    print("All maps created successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()
