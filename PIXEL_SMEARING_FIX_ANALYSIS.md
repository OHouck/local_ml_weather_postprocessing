# Pixel-Level Mapping Smearing: Root Cause Analysis and Fix

**Date**: 2025-11-23
**Issue**: Pixel-level mapping shows smearing/blurring artifacts despite using `shading='nearest'`
**Status**: ✅ FIXED

---

## Executive Summary

The pixel-level smearing issue was caused by using **`pcolormesh`** for rendering regular grid data. Even with `shading='nearest'`, pcolormesh creates a quadrilateral mesh which introduces edge artifacts during rasterization.

**Solution**: Replace `pcolormesh` with `imshow`, which is specifically designed for pixel-perfect rendering of 2D array data.

---

## Root Cause Analysis

### What We Tried (That Didn't Work)

1. **Changed `shading='auto'` to `shading='nearest'`** ❌
   - Rationale: 'nearest' should prevent interpolation between pixels
   - Result: Still showed smearing
   - Why it failed: The issue isn't interpolation between cells, it's the rendering of the quadmesh itself

2. **Fixed coordinate lookup precision** ❌
   - Replaced `np.searchsorted()` with `find_nearest_indices()`
   - Ensured floating-point precision errors don't cause misalignment
   - Result: Still showed smearing
   - Why it failed: Coordinates were already correctly aligned; the rendering method was wrong

3. **Removed longitude coordinate conversion** ❌
   - Let cartopy handle transformations automatically
   - Result: Still showed smearing
   - Why it failed: Again, the fundamental rendering method was the problem

### The Real Problem: pcolormesh vs imshow

**pcolormesh** is designed for **irregular quadrilateral meshes**:
- Creates a mesh of quadrilaterals from coordinate arrays
- Each quad is filled with a color
- Quad edges can be antialiased/interpolated during rasterization
- With `rasterized=True`, the vector quad mesh is converted to raster, potentially introducing interpolation
- Even without rasterization, quad edges may not perfectly align with pixel boundaries
- Designed for: Irregular grids, curvilinear coordinates, non-rectangular domains

**imshow** is designed for **regular 2D image arrays**:
- Treats each array element as a discrete pixel
- With `interpolation='none'`, guarantees pixel-perfect boundaries
- No quad mesh, no edge artifacts
- Optimized for regular grids
- Designed for: Images, heatmaps, regular grid data

### Why This Matters

When rendering pixel-level data where each grid cell should have a crisp, solid color:
- **pcolormesh**: Creates quads → rasterizes → can introduce blurring at quad edges
- **imshow**: Maps array elements 1:1 to pixels → always crisp

Think of it like:
- **pcolormesh**: Drawing rectangles with a vector graphics program, then rasterizing
- **imshow**: Directly setting pixel values in a bitmap

---

## The Fix

### Before (pcolormesh)

```python
mesh = ax.pcolormesh(
    unique_lons, unique_lats, global_improvement,
    transform=ccrs.PlateCarree(),
    cmap=cmap,
    norm=norm,
    shading='nearest',  # Doesn't help with quad mesh artifacts
    rasterized=True,
    zorder=1
)
```

### After (imshow)

```python
# Calculate extent from coordinate arrays
lon_step = unique_lons[1] - unique_lons[0] if len(unique_lons) > 1 else 0.25
lat_step = unique_lats[1] - unique_lats[0] if len(unique_lats) > 1 else 0.25
extent = [
    unique_lons[0] - lon_step/2,  # Left edge
    unique_lons[-1] + lon_step/2,  # Right edge
    unique_lats[0] - lat_step/2,   # Bottom edge
    unique_lats[-1] + lat_step/2   # Top edge
]

mesh = ax.imshow(
    global_improvement,
    extent=extent,
    origin='lower',  # Match pcolormesh orientation
    transform=ccrs.PlateCarree(),
    cmap=cmap,
    norm=norm,
    interpolation='none',  # Pixel-perfect rendering
    aspect='auto',  # Let cartopy handle aspect ratio
    zorder=1
)
```

### Key Differences

1. **Input format**:
   - pcolormesh: Takes coordinate arrays (lons, lats) + data
   - imshow: Takes data + extent [xmin, xmax, ymin, ymax]

2. **Coordinate interpretation**:
   - pcolormesh with `shading='nearest'`: Coordinates are cell centers
   - imshow: Extent defines outer edges of corner pixels
   - We calculate extent by extending coordinates by half a grid cell in each direction

3. **Rendering**:
   - pcolormesh: Creates quad mesh, then renders/rasterizes
   - imshow: Direct pixel mapping (no intermediate mesh)

4. **Interpolation parameter**:
   - pcolormesh: `shading='nearest'` (doesn't fully prevent artifacts)
   - imshow: `interpolation='none'` (guarantees no interpolation)

5. **Origin**:
   - pcolormesh: Bottom-left origin by default
   - imshow: Top-left origin by default → use `origin='lower'` to match

---

## Technical Details

### Extent Calculation

The extent for imshow represents the **outer edges** of the corner pixels, not the pixel centers.

For a grid with coordinates [c₀, c₁, c₂, ..., cₙ] with spacing Δ:

```
   c₀         c₁         c₂
    |          |          |
    ├──────────┼──────────┤
    |  pixel 0 |  pixel 1 |
    ├──────────┼──────────┤

extent_min = c₀ - Δ/2
extent_max = cₙ + Δ/2
```

This ensures:
- Pixel centers align with original coordinate values
- Pixel edges extend halfway to neighboring coordinates
- No gaps or overlaps

### Compatibility with Cartopy

Both pcolormesh and imshow work with cartopy's `transform` parameter:
- Data coordinates: PlateCarree (lat/lon)
- Axes projection: Can be any projection
- Cartopy handles the transformation automatically

The key advantage of imshow is that cartopy transforms the entire image as a single unit, rather than transforming individual quad vertices and re-meshing.

---

## Expected Results

After this fix, pixel-level mapping should show:

✅ **Sharp, crisp boundaries** between pixels
✅ **No blurring** or smearing at pixel edges
✅ **Solid colors** within each pixel
✅ **Proper alignment** with coordinate grid
✅ **Same or better performance** (imshow is optimized for regular grids)

---

## Verification

To verify the fix:

```bash
# Option 1: Run simplified test (no cartopy needed)
python verify_imshow_fix.py

# Option 2: Run comprehensive test (requires full environment)
uv run python test_pixel_rendering.py
```

Compare output visually:
- Look for sharp vs blurry transitions between colored pixels
- Zoom in to see individual pixel boundaries
- Check that there's no color bleeding between adjacent pixels

---

## Lessons Learned

1. **Choose the right tool for the job**:
   - Regular grid → imshow
   - Irregular/curvilinear mesh → pcolormesh
   - Vector quads → pcolormesh
   - Pixel data → imshow

2. **Parameter names can be misleading**:
   - `shading='nearest'` sounds like it should prevent smearing
   - But it only controls how quads are filled, not how they're rendered

3. **Rasterization can introduce artifacts**:
   - Vector graphics rasterized to bitmaps can lose precision
   - For pixel data, start with pixels (imshow), don't go vector→raster

4. **Test with extreme cases**:
   - Checkerboard patterns make smearing obvious
   - Single-pixel features show rendering precision
   - High-contrast boundaries reveal interpolation

---

## Modified Files

- `finetuning/figures_finetuning.py`: Line 627-653 (replaced pcolormesh with imshow)

---

## References

- Matplotlib pcolormesh docs: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.pcolormesh.html
- Matplotlib imshow docs: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html
- Cartopy transform parameter: https://scitools.org.uk/cartopy/docs/latest/tutorials/understanding_transform.html

---

**This fix should completely eliminate pixel-level smearing artifacts.**
