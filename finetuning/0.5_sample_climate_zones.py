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
    subregion = "2x2"

    lat_values = climate_zones.latitude.values
    lon_values = climate_zones.longitude.values

    nlat_patch, nlon_patch = get_patch_shape(subregion)

    # climate_zone_list = ["tropical", "arid", "temperate", "cold", "polar"]
    climate_zone_list = ["arid"]
    for zone in climate_zone_list:
        zone_code = CLIMATE_ZONE_MAP[zone]
        print(f"Sampling {zone} patches...")

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
        patch_path = os.path.join(dirs["processed"], f"climate_zone_patches_{zone}_{subregion}.npy")
        np.save(patch_path, patches)

        train_base_dir = "/Volumes/wd_external_hd/weatherbench/train_global"
        pangu_fc_train_file_paths = sorted(glob.glob(os.path.join(train_base_dir, "**", "pangu_train_forecast_data*.nc"), recursive=True))
        pangu_obs_train_file_paths = sorted(glob.glob(os.path.join(train_base_dir,"**", "pangu_train_obs_data*.nc"), recursive=True))

        test_base_dir = "/Volumes/wd_external_hd/weatherbench/test_global"
        pangu_fc_test_file_paths = sorted(glob.glob(os.path.join(test_base_dir, "**", "pangu_test_forecast_data*.nc"), recursive=True))
        pangu_obs_test_file_paths = sorted(glob.glob(os.path.join(test_base_dir,"**", "pangu_test_obs_data*.nc"), recursive=True))

        train_out_folder = os.path.join(
            dirs["processed"],
            'cleaned_weatherbench_downloads',
            f"train_{zone}",
            "pangu")
        # make folder if it does not exist
        os.makedirs(train_out_folder, exist_ok=True)

        test_out_folder = os.path.join(
            dirs["processed"],
            'cleaned_weatherbench_downloads',
            f"test_{zone}",
            "pangu")

        # read the patches back in
        patches = np.load(patch_path, allow_pickle=True)

        # count patches already saved for this zone and subregion
        num_patches_saved = len(glob.glob(os.path.join(dirs["processed"], 'cleaned_weatherbench_downloads', f"train_{zone}", "pangu", f"pangu_train_forecast_data_{zone}_{subregion}_patch_*.nc")))
        starting_idx = num_patches_saved + 1

        for idx, (lat_coords, lon_coords) in enumerate(patches, start = starting_idx):


            lat_min = lat_coords.min()
            lat_max = lat_coords.max()
            lon_min = lon_coords.min()
            lon_max = lon_coords.max()

            lat_values  = np.arange(lat_min, lat_max + 0.25, 0.25)
            lon_values = np.arange(lon_min, lon_max + 0.25, 0.25)

            # define a little function that selects your patch
            def preprocess_patch(ds):
                return ds.sel(latitude=lat_values, longitude=lon_values).sortby('latitude')

            # Save patches of forecast data
            fc_train_patch_path = os.path.join(train_out_folder, f"pangu_train_forecast_data_{zone}_{subregion}_patch_{idx}.nc")

            # open, slice, and concat in one go
            fc_train_patch = xr.open_mfdataset(
                pangu_fc_train_file_paths,
                preprocess=preprocess_patch,
                concat_dim="time",
                combine="nested",       # or "by_coords" if you know each time is strictly increasing
                engine="netcdf4"
            )
            # now write the single, stitched‐together file
            fc_train_patch.to_netcdf(fc_train_patch_path)

            fc_test_patch_path = os.path.join(test_out_folder, f"pangu_test_forecast_data_{zone}_{subregion}_patch_{idx}.nc")
            fc_test_patch = xr.open_mfdataset(
                pangu_fc_test_file_paths,
                preprocess=preprocess_patch,
                concat_dim="time",
                combine="nested",       # or "by_coords" if you know each time is strictly increasing
                engine="netcdf4"

            )
            fc_test_patch.to_netcdf(fc_test_patch_path)


            obs_train_patch_path = os.path.join(train_out_folder, f"pangu_train_obs_data_{zone}_{subregion}_patch_{idx}.nc")
            obs_train_patch = xr.open_mfdataset(
                pangu_obs_train_file_paths,
                preprocess=preprocess_patch,
                concat_dim="time",
                combine="nested",       # or "by_coords" if you know each time is strictly increasing
                engine="netcdf4"
            )
            obs_train_patch.to_netcdf(obs_train_patch_path)

            obs_test_patch_path = os.path.join(test_out_folder, f"pangu_test_obs_data_{zone}_{subregion}_patch_{idx}.nc")
            obs_test_patch = xr.open_mfdataset(
                pangu_obs_test_file_paths,
                preprocess=preprocess_patch,
                concat_dim="time",
                combine="nested",       # or "by_coords" if you know each time is strictly increasing
                engine="netcdf4"
            )
            obs_test_patch.to_netcdf(obs_test_patch_path)



if __name__ == "__main__":
    main()