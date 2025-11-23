#!/usr/bin/env python3
"""
Test script to diagnose pixel-level smearing in pcolormesh vs imshow.

This script creates a simple checkerboard pattern and renders it using
different methods to identify which approach gives crisp, pixel-perfect output.
"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.colors import TwoSlopeNorm

# Create a simple test pattern - checkerboard with alternating values
# This makes smearing very obvious
lats = np.arange(-10, 10, 0.25)  # 80 points
lons = np.arange(60, 80, 0.25)    # 80 points

# Create checkerboard pattern
lon_grid, lat_grid = np.meshgrid(lons, lats)
data = np.zeros_like(lon_grid)

# Create sharp transitions - every other pixel alternates between -100 and +100
for i in range(len(lats)):
    for j in range(len(lons)):
        if (i + j) % 4 == 0:
            data[i, j] = 100.0  # Strong positive
        elif (i + j) % 4 == 2:
            data[i, j] = -100.0  # Strong negative
        else:
            data[i, j] = np.nan  # Missing data

print(f"Test data shape: {data.shape}")
print(f"Lat range: {lats.min():.2f} to {lats.max():.2f}")
print(f"Lon range: {lons.min():.2f} to {lons.max():.2f}")
print(f"Valid pixels: {np.count_nonzero(~np.isnan(data))}")

# Setup colormap
norm = TwoSlopeNorm(vmin=-100, vcenter=0, vmax=100)
cmap = plt.cm.RdBu

# Test different rendering methods
fig = plt.figure(figsize=(20, 15))

# Method 1: pcolormesh with shading='auto' (original)
ax1 = fig.add_subplot(2, 3, 1, projection=ccrs.PlateCarree())
ax1.coastlines()
mesh1 = ax1.pcolormesh(lons, lats, data, transform=ccrs.PlateCarree(),
                        cmap=cmap, norm=norm, shading='auto', rasterized=True)
ax1.set_title("Method 1: pcolormesh shading='auto', rasterized=True\n(BASELINE - Expected to smear)",
              fontsize=10, fontweight='bold')
ax1.set_extent([lons.min(), lons.max(), lats.min(), lats.max()])

# Method 2: pcolormesh with shading='nearest' (current attempt)
ax2 = fig.add_subplot(2, 3, 2, projection=ccrs.PlateCarree())
ax2.coastlines()
mesh2 = ax2.pcolormesh(lons, lats, data, transform=ccrs.PlateCarree(),
                        cmap=cmap, norm=norm, shading='nearest', rasterized=True)
ax2.set_title("Method 2: pcolormesh shading='nearest', rasterized=True\n(CURRENT - May still smear)",
              fontsize=10, fontweight='bold')
ax2.set_extent([lons.min(), lons.max(), lats.min(), lats.max()])

# Method 3: pcolormesh with shading='nearest', no rasterization
ax3 = fig.add_subplot(2, 3, 3, projection=ccrs.PlateCarree())
ax3.coastlines()
mesh3 = ax3.pcolormesh(lons, lats, data, transform=ccrs.PlateCarree(),
                        cmap=cmap, norm=norm, shading='nearest', rasterized=False)
ax3.set_title("Method 3: pcolormesh shading='nearest', rasterized=False\n(Test: does rasterization cause smearing?)",
              fontsize=10, fontweight='bold')
ax3.set_extent([lons.min(), lons.max(), lats.min(), lats.max()])

# Method 4: imshow with interpolation='nearest'
ax4 = fig.add_subplot(2, 3, 4, projection=ccrs.PlateCarree())
ax4.coastlines()
# imshow requires extent = [xmin, xmax, ymin, ymax]
# and origin='lower' to match pcolormesh orientation
im4 = ax4.imshow(data, transform=ccrs.PlateCarree(),
                 extent=[lons.min(), lons.max(), lats.min(), lats.max()],
                 origin='lower', cmap=cmap, norm=norm,
                 interpolation='nearest', rasterized=True)
ax4.set_title("Method 4: imshow interpolation='nearest', rasterized=True\n(RECOMMENDED - Pixel-perfect)",
              fontsize=10, fontweight='bold')
ax4.set_extent([lons.min(), lons.max(), lats.min(), lats.max()])

# Method 5: imshow with interpolation='none'
ax5 = fig.add_subplot(2, 3, 5, projection=ccrs.PlateCarree())
ax5.coastlines()
im5 = ax5.imshow(data, transform=ccrs.PlateCarree(),
                 extent=[lons.min(), lons.max(), lats.min(), lats.max()],
                 origin='lower', cmap=cmap, norm=norm,
                 interpolation='none', rasterized=True)
ax5.set_title("Method 5: imshow interpolation='none', rasterized=True\n(ALTERNATIVE - Also pixel-perfect)",
              fontsize=10, fontweight='bold')
ax5.set_extent([lons.min(), lons.max(), lats.min(), lats.max()])

# Method 6: pcolormesh with cell edges explicitly defined
# For shading='flat', need n+1 coordinates for n pixels
# Create cell edge coordinates by adding half-steps
lat_step = lats[1] - lats[0]
lon_step = lons[1] - lons[0]
lats_edges = np.concatenate([[lats[0] - lat_step/2],
                              lats[:-1] + lat_step/2,
                              [lats[-1] + lat_step/2]])
lons_edges = np.concatenate([[lons[0] - lon_step/2],
                              lons[:-1] + lon_step/2,
                              [lons[-1] + lon_step/2]])

ax6 = fig.add_subplot(2, 3, 6, projection=ccrs.PlateCarree())
ax6.coastlines()
mesh6 = ax6.pcolormesh(lons_edges, lats_edges, data,
                        transform=ccrs.PlateCarree(),
                        cmap=cmap, norm=norm, shading='flat', rasterized=True)
ax6.set_title("Method 6: pcolormesh with cell edges, shading='flat'\n(Test: explicit edge coordinates)",
              fontsize=10, fontweight='bold')
ax6.set_extent([lons.min(), lons.max(), lats.min(), lats.max()])

# Add colorbar
fig.colorbar(mesh1, ax=fig.get_axes(), orientation='horizontal',
             label='Improvement (%)', pad=0.05, shrink=0.8)

plt.suptitle('Pixel-Level Rendering Comparison: Identifying Smearing Artifacts\n' +
             'Look for sharp boundaries between red/blue pixels. Blurry transitions = smearing.',
             fontsize=14, fontweight='bold')

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig('/home/user/ai_weather_ag/test_pixel_rendering.png', dpi=150, bbox_inches='tight')
print("\nTest figure saved to: /home/user/ai_weather_ag/test_pixel_rendering.png")

# Also create a zoomed-in version to see details
fig2 = plt.figure(figsize=(20, 6))

# Zoom to small region to see individual pixels clearly
zoom_lat_min, zoom_lat_max = -2, 2
zoom_lon_min, zoom_lon_max = 68, 72

for idx, (method_name, ax_func) in enumerate([
    ("pcolormesh nearest (current)",
     lambda ax: ax.pcolormesh(lons, lats, data, transform=ccrs.PlateCarree(),
                              cmap=cmap, norm=norm, shading='nearest', rasterized=True)),
    ("imshow nearest (recommended)",
     lambda ax: ax.imshow(data, transform=ccrs.PlateCarree(),
                          extent=[lons.min(), lons.max(), lats.min(), lats.max()],
                          origin='lower', cmap=cmap, norm=norm,
                          interpolation='nearest', rasterized=True)),
    ("imshow none (alternative)",
     lambda ax: ax.imshow(data, transform=ccrs.PlateCarree(),
                          extent=[lons.min(), lons.max(), lats.min(), lats.max()],
                          origin='lower', cmap=cmap, norm=norm,
                          interpolation='none', rasterized=True)),
], 1):
    ax = fig2.add_subplot(1, 3, idx, projection=ccrs.PlateCarree())
    ax.coastlines()
    ax_func(ax)
    ax.set_extent([zoom_lon_min, zoom_lon_max, zoom_lat_min, zoom_lat_max])
    ax.set_title(f'{method_name}\n(zoomed to see individual pixels)',
                 fontsize=11, fontweight='bold')
    ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)

plt.suptitle('ZOOMED VIEW: Pixel-Level Detail Comparison', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('/home/user/ai_weather_ag/test_pixel_rendering_zoomed.png', dpi=200, bbox_inches='tight')
print("Zoomed test figure saved to: /home/user/ai_weather_ag/test_pixel_rendering_zoomed.png")

print("\n" + "="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)
print("\nPlease examine the output images:")
print("1. test_pixel_rendering.png - Overview of 6 different methods")
print("2. test_pixel_rendering_zoomed.png - Zoomed detail comparison")
print("\nExpected findings:")
print("- pcolormesh methods may show blurry transitions between pixels")
print("- imshow methods should show SHARP boundaries (pixel-perfect)")
print("\nIf imshow gives crisp output, the fix is to replace pcolormesh")
print("with imshow in the actual code.")
print("="*70)
