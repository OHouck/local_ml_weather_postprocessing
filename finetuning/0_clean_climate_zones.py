# Author: Ozma Houck 
# Date Created: 6/09/2025

import os
import socket
import numpy as np
import xarray as xr
import pandas as pd
from rasterio.enums import Resampling
from rasterio.transform import Affine
from matplotlib import pyplot as plt


def setup_directories():
    # Determine root directory based on environment.
    nodename = socket.gethostname()
    if nodename == "oMac.local":  # local laptop
        root = os.path.expanduser(
            "~/OneDrive - The University of Chicago/ai_weather_ag/data"
        )
    else:
        raise Exception("Unknown environment, please specify the root directory")

    dirs = {
        "root": root,
        "raw": os.path.join(root, "raw"),
        "processed": os.path.join(root, "processed"),
        "fig": os.path.join(root, "../figures/finetuning"),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs

def regrid_to_025(
    da: xr.DataArray,
    resolution: float = 0.25,
    resampling: Resampling = Resampling.nearest,
    lon_min: float = 0,
    lon_max: float = 360,
    lat_min: float = -90,
    lat_max: float = 90,
) -> xr.DataArray:
    """
    Reproject any EPSG:4326 DataArray onto a strict 0.25° global grid:
      lon = [0, 0.25, …, 359.75]
      lat = [-90, -89.75, …, 89.75]

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


def main():
    dirs = setup_directories()

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
    da = da.assign_coords(lon=(((da.lon + 180) % 360))) .sortby("lon") 
    climate_zones = regrid_to_025(da, resolution=0.25)

    # print unique Koppen-Geiger classes
    unique_classes = climate_zones.values.flatten()
    unique_classes = np.unique(unique_classes)

    # using legend in the Koppen-Geiger documentation:
    tropical = [1,2,3]
    arid = [4,5,6,7]
    temperate = [8,9,10,11,12,13,14,15,16]
    cold = [17,18,19,20,21,22,23, 24,25,26,27,28]
    polar = [29,30]
    # create new cliamte zones array with 5 classes based on first digit of Koppen-Geiger class
    climate_zones_simplified = climate_zones.copy()
    climate_zones_simplified = xr.where(climate_zones.isin(tropical), 1, climate_zones_simplified)
    climate_zones_simplified = xr.where(climate_zones.isin(arid), 2, climate_zones_simplified)
    climate_zones_simplified = xr.where(climate_zones.isin(temperate), 3, climate_zones_simplified)
    climate_zones_simplified = xr.where(climate_zones.isin(cold), 4, climate_zones_simplified)
    climate_zones_simplified = xr.where(climate_zones.isin(polar), 5, climate_zones_simplified)

    # Apply the sea mask to the climate zones data
    climate_zones_simplified = climate_zones_simplified.where(sea_mask == 1)

    # save masked climate zones
    climate_zones_simplified.name = "climate_zones"
    climate_zones_simplified.attrs = climate_zones.attrs  # carry over any metadata
    climate_zones_path = os.path.join(dirs["processed"], "climate_zones_0p25.nc")
    climate_zones_simplified.to_netcdf(climate_zones_path)

    # # plot mask and climate zones
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # sea_mask.plot(cmap='viridis', add_colorbar=True)
    # plt.title("Sea Mask (0.25°)")
    # plt.subplot(1, 2, 2)
    # climate_zones_simplified.plot(cmap='Set3', add_colorbar=True)
    # plt.title("Climate Zones (0.25°)")
    # plt.tight_layout()
    # fig_path = os.path.join(dirs["fig"], "climate_zones_and_sea_mask.png")
    # plt.show()


if __name__ == "__main__":
    main()
