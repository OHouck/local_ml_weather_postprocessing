"""
Section 1 (part 2): divide the world's land into the 6x6 degree patches the
gridded post-processing is trained on.

The globe's 0.25 degree grid is tiled into non-overlapping 24 x 24 cell
patches. A patch is kept if at least half of it is land (by the GPM IMERG
land-sea mask), it is not south of 60S, and it is not over Greenland. Each
kept patch is assigned to a continent by the position of its centre.

Writes processed/{continent}_patches.npy for each continent: an array of
shape (n_patches, 2, 24) holding the latitudes and longitudes of every patch.
A patch matching no continent box is left out.

Run on its own with:
    uv run python data_preparation/create_land_patches.py
"""

import gzip
import os
import shutil
import sys
import urllib.request

import numpy as np
import rioxarray  # noqa: F401  (adds the .rio accessor used below)
import xarray as xr
from rasterio.enums import Resampling
from rasterio.transform import Affine

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import CONTINENTS, setup_directories  # noqa: E402

LAND_SEA_MASK_URL = "https://gpm.nasa.gov/sites/default/files/downloads/IMERG_land_sea_mask.nc.gz"
PATCH_SIZE_DEGREES = 6
GRID_SPACING_DEGREES = 0.25
MINIMUM_LAND_FRACTION = 0.5


def regrid_to_quarter_degree(field):
    """
    Resample a latitude-longitude field onto the global 0.25 degree grid by
    nearest neighbour.

    The target grid has longitudes 0, 0.25, ..., 359.75 and latitudes -90,
    -89.75, ..., 90. (The reprojection is anchored with its northern edge a
    quarter cell above 90N, which is how the published patch lists were made.)

    Inputs:
        field (xarray.DataArray): dims ('lat', 'lon') in degrees.

    Returns:
        xarray.DataArray with dims ('lat', 'lon') on the target grid, latitude
        ascending.
    """
    field = field.rio.write_crs("EPSG:4326", inplace=True)
    field = field.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    target_longitudes = np.arange(0, 360, GRID_SPACING_DEGREES)
    target_latitudes_descending = np.arange(-90, 90.25, GRID_SPACING_DEGREES)[::-1]
    transform = Affine(GRID_SPACING_DEGREES, 0, 0 - GRID_SPACING_DEGREES / 2,
                       0, -GRID_SPACING_DEGREES, 90.25 + GRID_SPACING_DEGREES / 2)
    regridded = field.rio.reproject(
        dst_crs=field.rio.crs, transform=transform,
        shape=(target_latitudes_descending.size, target_longitudes.size),
        resampling=Resampling.nearest).rename({"x": "lon", "y": "lat"})
    return regridded.assign_coords(lon=target_longitudes,
                                   lat=target_latitudes_descending).sortby("lat")


def continent_of(center_latitude, center_longitude):
    """
    Name the continent a patch belongs to, from simplified boxes around each continent.

    Boxes are checked in order, so where they overlap the first match wins.

    Inputs:
        center_latitude (float): degrees north.
        center_longitude (float): degrees east, 0..360.

    Returns:
        str: 'africa', 'asia', 'europe', 'north_america', 'south_america',
        'oceania', 'greenland' or 'unknown'. (Patches south of 60S are
        skipped before this is called.)
    """
    longitude = center_longitude if center_longitude <= 180 else center_longitude - 360
    latitude = center_latitude
    boxes = [  # (name, south, north, west, east)
        ("greenland", 60, 84, -75, -10),
        ("oceania", -47, 10, 110, 180),
        ("south_america", -56, 18, -92, -34),   # includes Central America
        ("north_america", 18, 83, -170, -52),
        ("africa", -35, 37, -18, 52),
        ("europe", 35, 71, -25, 60),
        ("asia", -10, 80, 25, 180),
    ]
    for name, south, north, west, east in boxes:
        if south <= latitude <= north and west <= longitude <= east:
            return name
    return "unknown"


def run():
    """
    Build and save the land patch lists of every continent.

    Inputs:
        None.

    Returns:
        None. Writes processed/{continent}_patches.npy.
    """
    directories = setup_directories()

    # Download the IMERG land-sea mask (percent water) once
    mask_path = os.path.join(directories["raw"], "IMERG_land_sea_mask.nc")
    if not os.path.exists(mask_path):
        print(f"  Downloading {LAND_SEA_MASK_URL}")
        with urllib.request.urlopen(LAND_SEA_MASK_URL, timeout=120) as response, \
                open(mask_path, "wb") as mask_file:
            shutil.copyfileobj(gzip.GzipFile(fileobj=response), mask_file)

    # Land is under 20% water; resample to the 0.25 degree grid (missing = ocean)
    water_percent = xr.open_dataset(mask_path)["landseamask"]
    is_land = regrid_to_quarter_degree(xr.where(water_percent < 20, 1, 0))
    is_land = xr.where(is_land < 0, 0, is_land)
    latitudes, longitudes = is_land.lat.values, is_land.lon.values

    # Tile the globe and keep the patches that are mostly land
    cells_per_patch = int(PATCH_SIZE_DEGREES / GRID_SPACING_DEGREES)
    patches_by_continent = {continent: [] for continent in CONTINENTS}
    for row in range(0, len(latitudes) - cells_per_patch + 1, cells_per_patch):
        patch_latitudes = latitudes[row:row + cells_per_patch]
        if np.mean(patch_latitudes) < -60:
            continue
        for column in range(0, len(longitudes) - cells_per_patch + 1, cells_per_patch):
            patch_longitudes = longitudes[column:column + cells_per_patch]
            land_fraction = float(is_land.sel(lat=patch_latitudes, lon=patch_longitudes).mean())
            if land_fraction < MINIMUM_LAND_FRACTION:
                continue
            continent = continent_of(np.mean(patch_latitudes), np.mean(patch_longitudes))
            if continent in patches_by_continent:
                patches_by_continent[continent].append((patch_latitudes, patch_longitudes))

    for continent, patches in patches_by_continent.items():
        np.save(os.path.join(directories["processed"], f"{continent}_patches.npy"), patches)
        print(f"  {continent}: {len(patches)} patches")


if __name__ == "__main__":
    run()
