# Author: Ozma Houck
# Date: Created 6/11/2025

import xarray as xr
import numpy as np
import random
import os
import socket
import glob

# Map the new region strings to Koppen‐Geiger codes:
CLIMATE_ZONE_MAP = {
    'tropical':  1,
    'arid':       2,
    'temperate':  3,
    'cold':       4,
    'polar':      5,
}

# Purpose: save patches of of climate zones to be used for bootstrapping
def setup_directories():
    # Determine root directory based on environment.
    nodename = socket.gethostname()
    if nodename == "oMac.local":  # local laptop
        root = os.path.expanduser(
            "~/OneDrive - The University of Chicago/ai_weather_ag/data"
        )
    else:
        raise Exception("Unknown environment, please specify the root directory")

    # file_list = sorted(glob.glob("/Volumes/wd_external_hd/weatherbench/train_global/**/*pangu*.nc", recursive=True))

    dirs = {
        "root": root,
        "raw": os.path.join(root, "raw"),
        "processed": os.path.join(root, "processed"),
        "fig": os.path.join(root, "../figures/finetuning"),
        "external": os.path.join("Volumes" ,"wd_external_hd", "weatherbench")
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs

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

def main():
    dirs = setup_directories()

    climate_zones_path = os.path.join(dirs["processed"], "climate_zones_0p25.nc")
    climate_zones = xr.open_dataset(climate_zones_path, engine = "netcdf4")

    # determines degrees of different patches
    subregion = "1x1"

    lat_values = climate_zones.latitude.values
    lon_values = climate_zones.longitude.values

    nlat_patch, nlon_patch = get_patch_shape(subregion)

    climate_zone_list = ["tropical", "arid", "temperate", "cold", "polar"]
    for zone in climate_zone_list:
        zone_code = CLIMATE_ZONE_MAP[zone]
        print(f"Sampling {zone} patches...")

        # Sample N patches for this zone
        N = 100 
        patches = sample_climate_zone_patches(
            climate_zones.climate_zones,
            zone_code,
            lat_values,
            lon_values,
            nlat_patch,
            nlon_patch,
            N
        )
        pangu_fc_file_paths = sorted(glob.glob(os.path.join(dirs["external"], "train_global", "*pangu*forecast*.nc"), recursive=True))
        pangu_obs_file_paths = sorted(glob.glob(os.path.join(dirs["external"], "train_global", "*pangu*obs*.nc"), recursive=True))

        for idx, (lat_values, lon_values) in enumerate(patches, start = 1):

            # Save patches of forecast data
            fc_out_path = os.path.join(
                dirs["processed"],
                'cleaned_weatherbench_downloads',
                f"train_{zone}",
                "pangu"
                f"pangu_train_forecast_data_{zone}_{subregion}_patch_{idx}.nc"
            )

            # delete the file if it already exists
            if os.path.exists(fc_out_path):
                os.remove(fc_out_path)

            for fc_file_path in pangu_fc_file_paths:
                fc= xr.open_dataset(fc_file_path)
                fc= fc.sel(latitude=lat_values, longitude=lon_values)
                fc.to_netcdf(fc_out_path, mode='a', format='NETCDF4')


            # Save patches of observation data
            obs_out_path = os.path.join(
                dirs["processed"],
                'cleaned_weatherbench_downloads',
                f"train_{zone}",
                "pangu"
                f"pangu_train_obs_data_{zone}_{subregion}_patch_{idx}.nc"
            )

            # delete the file if it already exists
            if os.path.exists(obs_out_path):
                os.remove(obs_out_path)

            for obs_file_path in pangu_obs_file_paths:
                obs = xr.open_dataset(obs_file_path)
                obs = obs.sel(latitude=lat_values, longitude=lon_values)
                obs.to_netcdf(obs_out_path, mode='a', format='NETCDF4')




if __name__ == "__main__":
    main()