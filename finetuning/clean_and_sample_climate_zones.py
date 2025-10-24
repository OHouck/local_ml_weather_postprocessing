# Author: Ozma Houck 
# Date Created: 6/09/2025
# Modified: Added topographical zone sampling based on stdor

import os
import socket
import numpy as np
import xarray as xr
import pandas as p
import rioxarray
from rasterio.enums import Resampling
from rasterio.transform import Affine
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import random
from typing import Dict

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

# Map topographic zones
TOPO_ZONE_MAP = {
    'flat': 1,
    'hilly': 2,
    'mountainous': 3,
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

def apply_sea_mask(da: xr.DataArray, dirs: Dict) -> xr.DataArray:

    # open the raw 0.1° mask
    sea_mask_path= os.path.join(dirs["raw"], "IMERG_land_sea_mask.nc")
    sea_mask= xr.open_dataset(sea_mask_path)["landseamask"]

    # make strict binary of land vs sea <20 is land in original mask
    sea_mask = xr.where(sea_mask < 20, 1, 0)

    # regrid to 0.25°
    sea_mask_p25 = regrid_to_025(sea_mask, resolution=0.25)
    sea_mask_p25.name = "landseamask"
    sea_mask_p25.attrs = sea_mask.attrs  # carry over any metadata

    # Replace nodata values with 0 (ocean)
    sea_mask_p25 = xr.where(sea_mask_p25 < 0, 0, sea_mask_p25)

    # Align coordinates - rename to match da's coordinate names
    if 'latitude' in da.dims and 'lat' in sea_mask_p25.dims:
        sea_mask_p25 = sea_mask_p25.rename({'lat': 'latitude', 'lon': 'longitude'})
    
    # Align the grids - this ensures coordinates match exactly
    sea_mask_p25, da_aligned = xr.align(sea_mask_p25, da, join='inner')

    # mask the data array
    da_masked = da.where(sea_mask_p25 == 1)

    return da_masked

def mask_antartica(da: xr.DataArray) -> xr.DataArray:
    """
    Mask out Antarctica from data array.
    
    Antarctica is defined as latitudes < -60°
    Parameters
    ----------
    da : xr.DataArray
        Input data array with latitude and longitude coordinates
        
    Returns
    -------
    masked_da : xr.DataArray
        Data array with Antarctica masked (set to NaN)
    """
    # Get the coordinate names (could be 'lat'/'lon' or 'latitude'/'longitude')
    if 'latitude' in da.dims:
        lat_name = 'latitude'
        lon_name = 'longitude'
    elif 'lat' in da.dims:
        lat_name = 'lat'
        lon_name = 'lon'
    else:
        raise ValueError("Could not find latitude/longitude dimensions in data array")
    
    # Create a copy to avoid modifying the original
    masked_da = da.copy()
    
    # Mask Antarctica (all latitudes south of -60°)
    antarctica_mask = masked_da[lat_name] < -60
    masked_da = masked_da.where(~antarctica_mask)
    
    # Add metadata
    if not hasattr(masked_da, 'attrs'):
        masked_da.attrs = {}
    masked_da.attrs['polar_regions_masked'] = 'Antarctica (<-60°) masked'
    
    return masked_da

def bin_topography(sdor_da: xr.DataArray, dirs: Dict) -> xr.DataArray:
    """
    Bin the standard deviation of orography into 3 topographic zones,
    excluding Antarctica.
    
    - flat (bottom 25%): zone = 1
    - hilly (middle 50%): zone = 2  
    - mountainous (top 25%): zone = 3
    
    Parameters
    ----------
    sdor_da : xr.DataArray
        Standard deviation of orography data
    dirs : Dict
        Directory paths
        
    Returns
    -------
    topo_zones : xr.DataArray
        Topographic zones (1, 2, or 3) with same shape as input
    """
    # Apply sea mask first
    sdor_masked = apply_sea_mask(sdor_da, dirs)
    
    # Apply polar region mask (Antarctica)
    sdor_masked = mask_antartica(sdor_masked)
    
    # Get valid (non-NaN) values for computing percentiles
    valid_values = sdor_masked.values[~np.isnan(sdor_masked.values)]
    
    # Calculate 25th and 75th percentiles
    p25 = np.percentile(valid_values, 25)
    p75 = np.percentile(valid_values, 75)
    
    print(f"Topography percentiles (excluding polar regions) - 25th: {p25:.2f}, 75th: {p75:.2f}")
    
    # Create binned topography zones
    topo_zones = xr.full_like(sdor_masked, np.nan)
    
    # Assign zones: 1 = flat, 2 = hilly, 3 = mountainous
    topo_zones = xr.where(sdor_masked <= p25, 1, topo_zones)
    topo_zones = xr.where((sdor_masked > p25) & (sdor_masked <= p75), 2, topo_zones)
    topo_zones = xr.where(sdor_masked > p75, 3, topo_zones)
    
    # Set attributes
    topo_zones.name = "topo_zones"
    topo_zones.attrs['long_name'] = 'Topographic zones based on stdor (excluding polar regions)'
    topo_zones.attrs['description'] = 'Zone 1: flat (bottom 25%), Zone 2: hilly (middle 50%), Zone 3: mountainous (top 25%). Antarctica masked.'
    topo_zones.attrs['p25'] = p25
    topo_zones.attrs['p75'] = p75
    
    return topo_zones

def regrid_and_save_climate_zones(dirs):

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
    climate_zones_simplified = apply_sea_mask(climate_zones_simplified, dirs)
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

def sample_zone_patches(
    da: xr.DataArray,
    zone: int,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    nlat: int,
    nlon: int,
    N: int,
    threshold: float = 0.75,
    zone_type: str = 'climate',
    exclude_masked: bool = True
):
    """
    Return a list of N (lat_slice, lon_slice) each of shape (nlat,nlon),
    drawn at random (with replacement) from da restricted to
    lat_vals×lon_vals, such that ≥threshold fraction = zone.
    
    Parameters
    ----------
    da : xr.DataArray
        Data array containing zone classifications
    zone : int
        Zone number to sample
    lat_vals : np.ndarray
        Latitude values to sample from
    lon_vals : np.ndarray
        Longitude values to sample from
    nlat : int
        Number of latitude points in each patch
    nlon : int
        Number of longitude points in each patch
    N : int
        Number of patches to sample
    threshold : float
        Minimum fraction of patch that must be in the target zone
    zone_type : str
        Either 'climate' or 'topographic' to specify which zone type
    exclude_masked : bool
        If True, reject patches that contain any NaN values (masked regions)
        
    Returns
    -------
    patches : list
        List of N tuples (lat_slice, lon_slice)
    """
    # restrict to your region grid
    da = da.sel(latitude=lat_vals, longitude=lon_vals)
    lats = da.latitude.values
    lons = da.longitude.values
    H, W = len(lats), len(lons)

    patches = []
    attempts = 0
    max_attempts = N * 1000  # Prevent infinite loops
    
    for _ in range(N):
        while attempts < max_attempts:
            attempts += 1
            i = random.randint(0, H - nlat)
            j = random.randint(0, W - nlon)
            
            lat_slice = lats[i:i+nlat]
            lon_slice = lons[j:j+nlon]
            patch = da.sel(latitude=lat_slice, longitude=lon_slice)
            patch_values = patch.values
            
            # If exclude_masked is True, skip patches with any NaN values
            if exclude_masked and np.any(np.isnan(patch_values)):
                continue
            
            # Check if at least threshold fraction equals zone
            # For patches without NaN, use regular mean; otherwise use nanmean
            if exclude_masked:
                valid = (patch_values == zone)
                fraction = np.mean(valid)
            else:
                valid = (patch_values == zone)
                fraction = np.nanmean(valid)
            
            if fraction >= threshold:
                patches.append((lat_slice, lon_slice))
                break
        
        if attempts >= max_attempts:
            print(f"Warning: Could only sample {len(patches)} patches for {zone_type} zone {zone} (requested {N})")
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
        patches = sample_zone_patches(
            climate_zones.climate_zones,
            zone_code,
            lat_values,
            lon_values,
            nlat_patch,
            nlon_patch,
            N,
            zone_type='climate'
        )
        # save patches to disk
        np.save(patch_path, patches)

def create_topographic_zone_patches(topo_zones: xr.DataArray, dirs: Dict):
    """
    Create patches of topographic zones for bootstrapping.
    
    Parameters
    ----------
    topo_zones : xr.DataArray
        Binned topographic zones (1=flat, 2=hilly, 3=mountainous)
    dirs : Dict
        Directory paths
    """

    subregion = "2x2"
    lat_values = topo_zones.latitude.values
    lon_values = topo_zones.longitude.values

    nlat_patch, nlon_patch = get_patch_shape(subregion)
    topo_zone_list = ["flat", "hilly", "mountainous"]
    
    for zone_name in topo_zone_list:
        zone_code = TOPO_ZONE_MAP[zone_name]
        print(f"Sampling {zone_name} topographic patches...")
        
        patch_path = os.path.join(dirs["processed"], f"topo_zone_patches_{zone_name}_{subregion}.npy")
        
        # check if the patches already exist
        if os.path.exists(patch_path):
            print(f"Patch file {patch_path} already exists. Skipping sampling.")
            continue
        
        # Sample N patches for this zone
        N = 50
        patches = sample_zone_patches(
            topo_zones,
            zone_code,
            lat_values,
            lon_values,
            nlat_patch,
            nlon_patch,
            N,
            threshold=0.75,
            zone_type='topographic',
            exclude_masked=True  # Explicitly exclude patches with NaN values
        )
        
        # save patches to disk
        np.save(patch_path, patches)
        print(f"Saved {len(patches)} patches to {patch_path}")

def main():

    dirs = setup_directories()


    # Get ERA5 Standard Deviation of Orthography from static variables 
    # Downloaded to initialize aurora model
    era5_static_path = os.path.join(dirs["raw"], "era5_static.nc")
    sdor = xr.open_dataset(era5_static_path, engine="netcdf4")["sdor"]
    print("Original sdor shape:", sdor.shape)

    # Bin topography into 3 zones
    topo_zones = bin_topography(sdor, dirs)
    print("Topographic zones shape:", topo_zones.shape)
    print("Unique zones:", np.unique(topo_zones.values[~np.isnan(topo_zones.values)]))
    
    # Save topographic zones
    topo_zones_path = os.path.join(dirs["processed"], "topo_zones_0p25.nc")
    topo_zones.to_netcdf(topo_zones_path)
    print(f"Saved topographic zones to {topo_zones_path}")

    # regrid koppen-geiger climate zones to 0.25°
    regrid_and_save_climate_zones(dirs)

    # sample and save climate zone patches for bootstrapping
    # create_climate_zone_patches(dirs) # uncomment if need to resample patches
    
    # sample and save topographic zone patches for bootstrapping
    # create_topographic_zone_patches(topo_zones, dirs) # uncomment to sample topo patches


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
                            facecolor=color, alpha=alpha, edgecolor='black', linewidth=1)
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

    # ==========================================================================
    # Plot topographic zones with patches
    # ==========================================================================
    
    # Load topographic zones
    topo_zones_path = os.path.join(dirs["processed"], "topo_zones_0p25.nc")
    topo_zones_plot = xr.open_dataarray(topo_zones_path, engine="netcdf4")
    
    # Shift longitude to 0-360 range starting from 0 for easier mapping
    topo_zones_plot = topo_zones_plot.assign_coords(
        longitude=(((topo_zones_plot.longitude + 180) % 360))).sortby("longitude")
    
    # Define colors for topographic zones: flat=green, hilly=yellow, mountainous=brown
    topo_colors = ['#2E7D32', '#FFD54F', '#6D4C41']  # dark green, yellow, brown
    topo_cmap = mcolors.ListedColormap(topo_colors)
    
    # Set up discrete colormap boundaries for 3 zones
    topo_bounds = np.arange(0.5, 4.5, 1)  # 0.5, 1.5, 2.5, 3.5
    topo_norm = mcolors.BoundaryNorm(topo_bounds, topo_cmap.N)
    
    # Load topographic patches if they exist
    flat_patch_path = os.path.join(dirs["processed"], "topo_zone_patches_flat_2x2.npy")
    hilly_patch_path = os.path.join(dirs["processed"], "topo_zone_patches_hilly_2x2.npy")
    mountainous_patch_path = os.path.join(dirs["processed"], "topo_zone_patches_mountainous_2x2.npy")
    
    patches_exist = (os.path.exists(flat_patch_path) and 
                     os.path.exists(hilly_patch_path) and 
                     os.path.exists(mountainous_patch_path))
    
    if patches_exist:
        flat_patches = np.load(flat_patch_path, allow_pickle=True)
        hilly_patches = np.load(hilly_patch_path, allow_pickle=True)
        mountainous_patches = np.load(mountainous_patch_path, allow_pickle=True)
        print(f"Loaded {len(flat_patches)} flat, {len(hilly_patches)} hilly, {len(mountainous_patches)} mountainous patches")
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    im_topo = topo_zones_plot.plot(cmap=topo_cmap, norm=topo_norm, add_colorbar=False)
    
    # Add patch rectangles if they exist
    if patches_exist:
        add_patch_rectangles(flat_patches, topo_colors[0], alpha=0.6)        # Green
        add_patch_rectangles(hilly_patches, topo_colors[1], alpha=0.6)       # Yellow
        add_patch_rectangles(mountainous_patches, topo_colors[2], alpha=0.6) # Brown
    
    # Add custom colorbar with proper labels
    cbar_topo = plt.colorbar(im_topo, ticks=[1, 2, 3], shrink=0.8)
    cbar_topo.ax.set_yticklabels(['Flat (≤25th pct)', 'Hilly (25-75th pct)', 'Mountainous (≥75th pct)'])
    cbar_topo.set_label('Topographic Zone')
    
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    
    if patches_exist:
        plt.title("Topographic Zones (0.25°) with Sampling Patches")
    else:
        plt.title("Topographic Zones (0.25°)")
    
    plt.tight_layout()
    
    fig_path_topo = os.path.join(dirs["fig"], "topo_zones_with_patches.png")
    plt.savefig(fig_path_topo, dpi=300)
    print(f"Saved topographic zones plot to {fig_path_topo}")


if __name__ == "__main__":
    main()
