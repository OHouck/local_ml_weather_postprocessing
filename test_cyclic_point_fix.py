#!/usr/bin/env python3
"""
Test script to verify add_cyclic_point fix for 0° meridian smudging.
Creates a simple test case with data near the prime meridian.
"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point

# Create test data with discontinuity at 0° meridian
lons = np.arange(-180, 180, 1.0)
lats = np.arange(-90, 91, 1.0)
lon_2d, lat_2d = np.meshgrid(lons, lats)

# Create test pattern: alternating values near 0° meridian
# This will show smudging if not handled correctly
data = np.sin(np.deg2rad(lon_2d) * 10) * np.cos(np.deg2rad(lat_2d) * 5)

# Create figure with two subplots
fig = plt.figure(figsize=(16, 6))

# Subplot 1: WITHOUT cyclic point (shows smudging)
ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
ax1.coastlines()
ax1.set_title('WITHOUT cyclic point\n(expect smudging at 0° meridian)', fontsize=12)

mesh1 = ax1.pcolormesh(
    lon_2d, lat_2d, data,
    transform=ccrs.PlateCarree(),
    cmap='RdBu_r',
    shading='nearest'
)
plt.colorbar(mesh1, ax=ax1, orientation='horizontal', pad=0.05, shrink=0.8)

# Add vertical line at 0° meridian to highlight the problem area
ax1.axvline(x=0, color='yellow', linewidth=2, linestyle='--', alpha=0.7)

# Subplot 2: WITH cyclic point (no smudging)
ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
ax2.coastlines()
ax2.set_title('WITH cyclic point\n(smooth at 0° meridian)', fontsize=12)

# Add cyclic point
data_cyclic, lons_cyclic = add_cyclic_point(data, coord=lons)
lon_2d_cyclic, lat_2d_cyclic = np.meshgrid(lons_cyclic, lats)

mesh2 = ax2.pcolormesh(
    lon_2d_cyclic, lat_2d_cyclic, data_cyclic,
    transform=ccrs.PlateCarree(),
    cmap='RdBu_r',
    shading='nearest'
)
plt.colorbar(mesh2, ax=ax2, orientation='horizontal', pad=0.05, shrink=0.8)

# Add vertical line at 0° meridian
ax2.axvline(x=0, color='yellow', linewidth=2, linestyle='--', alpha=0.7)

plt.suptitle('Cyclic Point Fix Test - Look at 0° Meridian (Yellow Line)',
             fontsize=14, fontweight='bold')
plt.tight_layout()

# Save figure
output_path = '/home/user/ai_weather_ag/test_cyclic_point_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✅ Test figure saved to: {output_path}")
print(f"   Compare the two plots at the 0° meridian (yellow line)")
print(f"   Left plot should show smudging/artifacts")
print(f"   Right plot should be smooth and continuous")

# Print info
print(f"\nData shape without cyclic point: {data.shape}")
print(f"Data shape with cyclic point: {data_cyclic.shape}")
print(f"Longitude array length without cyclic point: {len(lons)}")
print(f"Longitude array length with cyclic point: {len(lons_cyclic)}")
print(f"Added longitude value: {lons_cyclic[-1]}")
