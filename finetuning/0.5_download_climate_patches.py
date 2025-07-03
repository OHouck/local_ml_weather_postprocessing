# Author: Ozma Houck
# Date: Created 6/11/2025

import xarray as xr
import numpy as np
import os
import socket
import glob

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
        "external": os.path.join("Volumes" ,"wd_external_hd", "weatherbench")
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def main():
    dirs = setup_directories()

    # determines degrees of different patches
    subregion = "2x2"

    # climate_zone_list = ["tropical", "arid", "temperate"]
    climate_zone_list = ["tropical"]
    for zone in climate_zone_list:

        # read in the patches
        patch_path = os.path.join(dirs["processed"], f"climate_zone_patches_{zone}_{subregion}.npy")
        patches = np.load(patch_path, allow_pickle=True)

        assert len(patches) == 50   

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


        # count patches already saved for this zone and subregion
        train_patch_files = glob.glob(os.path.join(dirs["processed"], 'cleaned_weatherbench_downloads', f"train_{zone}", "pangu", f"pangu_train_forecast_data_{zone}_{subregion}_patch_*.nc"))
        test_patch_files = glob.glob(os.path.join(dirs["processed"], 'cleaned_weatherbench_downloads', f"test_{zone}", "pangu", f"pangu_test_forecast_data_{zone}_{subregion}_patch_*.nc"))

        # save the index of all saved patches and process the unsaved ones
        train_patch_nums = sorted(list([int(os.path.basename(f).split('_')[-1].split('.')[0]) for f in train_patch_files]))
        test_patch_nums = sorted(list([int(os.path.basename(f).split('_')[-1].split('.')[0]) for f in test_patch_files]))

        # unsaved patches are those that are not in both train and test sets   
        unsaved_patch_nums = sorted(list(set(range(1, 50 + 1)) - (set(train_patch_nums) & set(test_patch_nums))))
        print(f"unsaved patches for {zone} {subregion}: {unsaved_patch_nums}")

        for patch_num in unsaved_patch_nums:

            patch = patches[patch_num-1]
            lat_min = patch[0,].min()
            lat_max = patch[0,].max()
            lon_min = patch[1,].min()
            lon_max = patch[1,].max()

            lat_values  = np.arange(lat_min, lat_max + 0.25, 0.25)
            lon_values = np.arange(lon_min, lon_max + 0.25, 0.25)

            # define a little function that selects your patch
            def preprocess_patch(ds):
                return ds.sel(latitude=lat_values, longitude=lon_values).sortby('latitude')

            # Save patches of forecast data
            fc_train_patch_path = os.path.join(train_out_folder, f"pangu_train_forecast_data_{zone}_{subregion}_patch_{patch_num}.nc")

            # open, slice, and concat in one go
            fc_train_patch = xr.open_mfdataset(
                pangu_fc_train_file_paths,
                preprocess=preprocess_patch,
                concat_dim="time",
                combine="nested",       # or "by_coords" if you know each time is strictly increasing
                engine="netcdf4",
                decode_timedelta = False
            )
            # now write the single, stitched‐together file
            fc_train_patch.to_netcdf(fc_train_patch_path)

            fc_test_patch_path = os.path.join(test_out_folder, f"pangu_test_forecast_data_{zone}_{subregion}_patch_{patch_num}.nc")
            fc_test_patch = xr.open_mfdataset(
                pangu_fc_test_file_paths,
                preprocess=preprocess_patch,
                concat_dim="time",
                combine="nested",       # or "by_coords" if you know each time is strictly increasing
                engine="netcdf4",
                decode_timedelta = False

            )
            fc_test_patch.to_netcdf(fc_test_patch_path)
            obs_train_patch_path = os.path.join(train_out_folder, f"pangu_train_obs_data_{zone}_{subregion}_patch_{patch_num}.nc")
            
            obs_train_patch = xr.open_mfdataset(
                pangu_obs_train_file_paths,
                preprocess=preprocess_patch,
                concat_dim="time",
                combine="nested",       # or "by_coords" if you know each time is strictly increasing
                engine="netcdf4",
                decode_timedelta = False
            )
            obs_train_patch.to_netcdf(obs_train_patch_path)

            obs_test_patch_path = os.path.join(test_out_folder, f"pangu_test_obs_data_{zone}_{subregion}_patch_{patch_num}.nc")
            obs_test_patch = xr.open_mfdataset(
                pangu_obs_test_file_paths,
                preprocess=preprocess_patch,
                concat_dim="time",
                combine="nested",       # or "by_coords" if you know each time is strictly increasing
                engine="netcdf4",
                decode_timedelta = False
            )
            obs_test_patch.to_netcdf(obs_test_patch_path)



if __name__ == "__main__":
    main()