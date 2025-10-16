# Author: Ozma Houck 
# Date Created: 6/09/2025

import os
import socket
import numpy as np
import xarray as xr
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import random

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from helper_funcs import setup_directories

# Map the new region strings to Koppen‐Geiger codes:
CLIMATE_ZONE_MAP = {
    'tropical':  1,
    'arid':       2,
    'temperate':  3,
    'cold':       4,
    'polar':      5,
}

def regrid_to_025(
    da: xr.DataArray,
    resolution: float = 0.25,
    resampling: Resampling = Resampling.nearest,
    lon_min: float = 0,
    lon_max: float = 360,
    lat_min: float = -90,
    lat_max: float = 90.25,
) -> xr.DataArray:
    """
    Reproject any EPSG:4326 DataArray onto a strict 0.25° global grid:
      lon = [0, 0.25, …, 359.75]
      lat = [-90, -89.75, …, 90] 

    Parameters
    ----------
    da
        must have dims ('lat','lon') and real-valued coords (any resolution/offset).
    resolution
        target cell size in degrees.
    resampling
        how to aggregate the original pixels.
    lon_min, lon_max
        bounds of the longitude domain (deg east).
    lat_min, lat_max
        bounds of the latitude domain (deg north).

    Returns
    -------
    regridded : xr.DataArray
        dims ('lat','lon') on the new grid, with exactly the above coords.
    """
    # 1) ensure georeferencing
    da = da.rio.write_crs("EPSG:4326", inplace=True)
    da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat")

    # 2) define the STRICT target coords
    dst_lons = np.arange(lon_min, lon_max, resolution)
    # we'll create descending lat for the reprojection engine,
    # then flip back to ascending.
    lat_asc = np.arange(lat_min, lat_max, resolution)
    dst_lats_desc = lat_asc[::-1]

    # 3) compute the Affine transform so that PIXEL CENTERS land exactly on our coords
    #    solve for west,north such that:
    #      lon_center[0] = west + 0.5*res  ->  west = lon_min - 0.5*res
    #      lat_center[0] = north - 0.5*res ->  north = lat_max + 0.5*res
    west = lon_min - resolution / 2
    north = lat_max + resolution / 2
    transform = Affine(resolution, 0, west, 0, -resolution, north)

    # 4) run the reproject with explicit shape + transform
    height = dst_lats_desc.size
    width = dst_lons.size

    reproj = da.rio.reproject(
        dst_crs=da.rio.crs,
        transform=transform,
        shape=(height, width),
        resampling=resampling,
    ).rename({"x": "lon", "y": "lat"})

    # 5) assign our strict coords and sort latitude ascending
    out = reproj.assign_coords(lon=dst_lons, lat=dst_lats_desc).sortby("lat")
    return out

def regrid_and_save_climate_zones(dirs):
    # 1) open the raw 0.1° mask
    land_mask_path = os.path.join(dirs["raw"], "IMERG_land_sea_mask.nc")
    da = xr.open_dataset(land_mask_path)["landseamask"]
    # 3)  make it strictly binary (land vs sea)
    #    e.g. if original mask < 20 is land:
    da = xr.where(da < 20, 1, 0)

    # 2) regrid to 0.25°
    sea_mask = regrid_to_025(da, resolution=0.25)
    sea_mask.name = "landseamask"
    sea_mask.attrs = da.attrs  # carry over any metadata

    # ==========================================================================
    # open the Koppen-Geiger climate zones data
    # ==========================================================================

    koppen_path = os.path.join(dirs["raw"], "koppen_geiger_nc", "1991_2020",
                                 "koppen_geiger_0p1.nc")
    da = xr.open_dataset(koppen_path, engine = "netcdf4")['kg_class']
    # lon from -180 to 180, to 0 to 360 with 0 at the center (around UK) to
    # match other data
    da['lon'] = da['lon'] + 180
    da = da.assign_coords(lon=(((da.lon + 180) % 360))).sortby("lon") 
    climate_zones = regrid_to_025(da, resolution=0.25)

    unique_classes = climate_zones.values.flatten()
    unique_classes = np.unique(unique_classes)

    # using legend in the Koppen-Geiger documentation:
    tropical = [1,2,3]
    arid = [4,5,6,7]
    temperate = [8,9,10,11,12,13,14,15,16]
    cold = [17,18,19,20,21,22,23, 24,25,26,27,28]
    polar = [29,30]

    # create new climate zones array with 5 classes based on borader Koppen-Geiger class
    climate_zones_simplified = climate_zones.copy()
    climate_zones_simplified = xr.full_like(climate_zones, np.nan)

    # Apply the sea mask to the climate zones data
    climate_zones_simplified = climate_zones_simplified.where(sea_mask == 1)
    climate_zones_simplified = xr.where(climate_zones.isin(tropical), 1, climate_zones_simplified)
    climate_zones_simplified = xr.where(climate_zones.isin(arid), 2, climate_zones_simplified)
    climate_zones_simplified = xr.where(climate_zones.isin(temperate), 3, climate_zones_simplified)
    climate_zones_simplified = xr.where(climate_zones.isin(cold), 4, climate_zones_simplified)
    climate_zones_simplified = xr.where(climate_zones.isin(polar), 5, climate_zones_simplified)

    # rename lat and lon to latitude and longitude
    climate_zones_simplified = climate_zones_simplified.rename({"lat": "latitude", "lon": "longitude"})

    # save masked climate zones
    climate_zones_simplified.name = "climate_zones"
    climate_zones_simplified.attrs = climate_zones.attrs  # carry over any metadata

    climate_zones_path = os.path.join(dirs["processed"], "climate_zones_0p25.nc")
    climate_zones_simplified.to_netcdf(climate_zones_path)

def get_patch_shape(subregion):
    """
    Given args.subregion like '2x2', return number of gridpoints in lat and lon
    """
    deg_lat, deg_lon = map(int, subregion.split('x'))
    nlat = int(deg_lat / 0.25)
    nlon = int(deg_lon / 0.25)
    return nlat, nlon

def sample_climate_zone_patches(
    cz_da: xr.DataArray,
    zone: int,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    nlat: int,
    nlon: int,
    N: int,
    threshold: float = 0.75
):
    """
    Return a list of N (lat_slice, lon_slice) each of shape (nlat,nlon),
    drawn at random (with replacement) from cz_da restricted to
    lat_vals×lon_vals, such that ≥threshold fraction = zone.
    """
    # restrict to your region grid
    cz = cz_da.sel(latitude=lat_vals, longitude=lon_vals)
    lats = cz.latitude.values
    lons = cz.longitude.values
    H, W = len(lats), len(lons)

    patches = []
    for _ in range(N):
        while True:
            i = random.randint(0, H - nlat)
            j = random.randint(0, W - nlon)
            block = cz.isel(latitude=slice(i, i+nlat),
                            longitude=slice(j, j+nlon))
            frac = (block.values == zone).sum() / float(nlat * nlon)
            if frac >= threshold:
                patches.append((lats[i:i+nlat], lons[j:j+nlon]))
                break
    return patches

def create_climate_zone_patches(dirs):
    """
    Create patches of climate zones for bootstrapping.
    """
    # load the regridded climate zones data
    # this is the 0.25° resolution data
    climate_zones_path = os.path.join(dirs["processed"], "climate_zones_0p25.nc")
    climate_zones = xr.open_dataset(climate_zones_path, engine = "netcdf4")

    # determines degrees of different patches
    subregion = "2x2"

    lat_values = climate_zones.latitude.values
    lon_values = climate_zones.longitude.values

    nlat_patch, nlon_patch = get_patch_shape(subregion)

    # sample and create patches
    climate_zone_list = ["tropical", "arid", "temperate", "cold", "polar"]
    for zone in climate_zone_list:
        zone_code = CLIMATE_ZONE_MAP[zone]
        print(f"Sampling {zone} patches...")

        patch_path = os.path.join(dirs["processed"], f"climate_zone_patches_{zone}_{subregion}.npy")
        # check if the patches already exist
        if os.path.exists(patch_path):
            print(f"Patch file {patch_path} already exists. Skipping sampling.")
            continue

        # Sample N patches for this zone
        N = 50
        patches = sample_climate_zone_patches(
            climate_zones.climate_zones,
            zone_code,
            lat_values,
            lon_values,
            nlat_patch,
            nlon_patch,
            N
        )
        # save patches to disk
        np.save(patch_path, patches)

def main():

    dirs = setup_directories()

    # regrid koppen-geiger climate zones to 0.25°
    regrid_and_save_climate_zones(dirs)

    # sample and save climate zone patches for bootstrapping
    # create_climate_zone_patches(dirs) # uncomment if need to resample patches


    # load and plot the data
    climate_zones_path = os.path.join(dirs["processed"], "climate_zones_0p25.nc")
    climate_zones_simplified = xr.open_dataset(climate_zones_path, engine="netcdf4")["climate_zones"]

    # Better colors for climate zones
    colors = ['#228B22', '#FFFF00', '#90EE90', '#6495ED', '#ADD8E6']  # green, yellow, light green, light blue, pale blue
    cmap = mcolors.ListedColormap(colors)

    # Set up discrete colormap boundaries
    bounds = np.arange(0.5, 6.5, 1)  # 0.5, 1.5, 2.5, 3.5, 4.5, 5.5
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Load data
    climate_zones_path = os.path.join(dirs["processed"], "climate_zones_0p25.nc")
    climate_zones_simplified = xr.open_dataset(climate_zones_path, engine="netcdf4")["climate_zones"]

    # shift longitude to 0-360 range starting from 0 to be mapped easier
    climate_zones_simplified = climate_zones_simplified.assign_coords(
        longitude=(((climate_zones_simplified.longitude + 180) % 360))).sortby("longitude") 

    # now create version of climate zones with patches
    tropical_patch_path = os.path.join(dirs["processed"], f"climate_zone_patches_tropical_2x2.npy")
    temperate_patch_path = os.path.join(dirs["processed"], f"climate_zone_patches_temperate_2x2.npy")
    arid_patch_path = os.path.join(dirs["processed"], f"climate_zone_patches_arid_2x2.npy")

    tropical_patches = np.load(tropical_patch_path, allow_pickle=True)
    temperate_patches = np.load(temperate_patch_path, allow_pickle=True)
    arid_patches = np.load(arid_patch_path, allow_pickle=True)

    subregion_size = 6
    buffer = subregion_size // 2

    # Define bounding boxes to create rectangles to plot
    india_bounds = {"lat0": 22 - buffer, "lat1": 22 + buffer, "lon0": 77 - buffer + 180, "lon1": 77 + buffer + 180}
    usa_south_bounds = {"lat0": 35 - buffer, "lat1": 35 + buffer, "lon0": -100 - buffer + 180, "lon1": -100 + buffer + 180}
    amazon_bounds = {"lat0": -5 - buffer, "lat1": -5 + buffer, "lon0": -65 - buffer + 180, "lon1": -65 + buffer + 180}
    british_columbia_bounds = {"lat0": 53 - buffer + 0.25, "lat1": 53 + buffer, "lon0": -125 - buffer + 180, "lon1": -125 + buffer + 180}
    ethiopia_bounds = {"lat0": 9 - buffer, "lat1": 9 + buffer, "lon0": 39 - buffer + 180, "lon1": 39 + buffer + 180}


    manual_regions = [
        {"name": "India", "bounds": india_bounds},
        {"name": "USA South", "bounds": usa_south_bounds}, 
        {"name": "Amazon", "bounds": amazon_bounds},
        {"name": "British Columbia", "bounds": british_columbia_bounds},
        {"name": "Ethiopia", "bounds": ethiopia_bounds},
    ]
        

    # Plot
    plt.figure(figsize=(12, 6))
    im = climate_zones_simplified.plot(cmap=cmap, norm=norm, add_colorbar=False)

    # Function to add patch rectangles
    def add_patch_rectangles(patches, color, alpha=0.7):
        for patch in patches:
            lats = patch[0]  # First row is latitudes
            lons = patch[1]  # Second row is longitudes
            
            # Get bounds of the patch
            min_lat, max_lat = lats.min(), lats.max()
            min_lon, max_lon = lons.min(), lons.max()

            min_lon = (min_lon + 180) % 360  # Ensure longitude is in plotting range
            max_lon = (max_lon + 180) % 360  # Ensure longitude is in plotting range
            
            # Calculate width and height
            width = max_lon - min_lon
            height = max_lat - min_lat
            
            # Add rectangle
            rect = Rectangle((min_lon, min_lat), width, height, 
                            facecolor=color, alpha=alpha, edgecolor='black', linewidth=0.5)
            plt.gca().add_patch(rect)

    # Function to add manual region rectangles
    def add_manual_rectangles(manual_regions, color='red', alpha=0.2, label="Manual Regions"):
        for i, region in enumerate(manual_regions):
            bounds = region["bounds"]
            min_lat, max_lat = bounds["lat0"], bounds["lat1"]
            min_lon, max_lon = bounds["lon0"], bounds["lon1"]
            
            # Calculate width and height for 10x10 rectangle
            width = max_lon - min_lon
            height = max_lat - min_lat
            
            # Add large 10x10 rectangle (more transparent)
            rect = Rectangle((min_lon, min_lat), width, height, 
                            facecolor=color, alpha=alpha, edgecolor='black', linewidth=1.0,
                            label=label if i == 0 else "")
            plt.gca().add_patch(rect)
            
            # Calculate center 2x2 rectangle
            # center_lat = (min_lat + max_lat) / 2
            # center_lon = (min_lon + max_lon) / 2
            
            # # 2x2 degree rectangle centered on the region center (no longer using)
            # small_width = 2.0
            # small_height = 2.0
            # small_min_lon = center_lon - small_width / 2
            # small_min_lat = center_lat - small_height / 2
            
            # # Add small 2x2 rectangle (almost opaque)
            # small_rect = Rectangle((small_min_lon, small_min_lat), small_width, small_height,
            #                      facecolor=color, alpha=0.9, edgecolor='black', linewidth=1.5,
            #                      label="2x2 Center Regions" if i == 0 else "")
            # plt.gca().add_patch(small_rect)

    # Add patches for each climate zone
    add_patch_rectangles(tropical_patches, colors[0])  # Green
    add_patch_rectangles(arid_patches, colors[1])      # Gold
    add_patch_rectangles(temperate_patches, colors[2]) # Blue
    add_manual_rectangles(manual_regions, color='red', alpha=0.5, label="Manual Regions")


    # Add custom colorbar with proper labels
    cbar = plt.colorbar(im, ticks=[1, 2, 3, 4, 5], shrink=0.8)
    cbar.ax.set_yticklabels(['Tropical', 'Arid', 'Temperate', 'Cold', 'Polar'])
    cbar.set_label('Climate Zone')

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98))

    plt.title("Climate Zones (0.25°) with Sampling Patches")
    plt.tight_layout()

    fig_path = os.path.join(dirs["fig"], "climate_zones_with_patches.png")
    plt.savefig(fig_path, dpi=300)


if __name__ == "__main__":
    main()
