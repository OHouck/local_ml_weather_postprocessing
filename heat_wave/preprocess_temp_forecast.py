"""
heat_wave/preprocess_temp_forecast.py

Prepare the AIFS heat-wave forecast subset and its matching ERA5 ground truth for
post-processing.

The source file (heat_wave_training_subset.nc) stores 24-hour 2m_temperature AIFS
forecasts over a 6x6 degree PNW box for summer days 2000-2025. Its `time` coordinate
is the forecast *initialization* time (00:00 UTC, stamped from the source filename)
and the lead-time dimension was dropped when the subset was built.

The project's post-processing pipeline (finetuning/post_process.post_process_forecasts
-> prepare_forecasts_and_targets._arrays_from_inmemory_datasets) instead expects:
  * forecast_ds: dims (time, prediction_timedelta, latitude, longitude) where `time`
    is the forecast *valid* time, and
  * ground_truth_ds: ERA5 indexed by that same valid `time`.

So this script (1) shifts the forecast `time` by +24h to the valid time, (2) re-adds a
prediction_timedelta=[24h] dimension, (3) pulls matching ERA5 2m_temperature from the
public ARCO-ERA5 store on the same grid/valid times, and (4) caches both datasets to
the local processed directory so training never has to re-download.

Run with: uv run python heat_wave/preprocess_temp_forecast.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent.parent))
from helper_funcs import setup_directories

# Source forecast subset (AIFS 24h 2m_temperature, PNW summer days 2000-2025).
FORECAST_NC = (
    "/Users/ohouck/Library/CloudStorage/OneDrive-TheUniversityofChicago/"
    "heat_postprocessing/heat_wave_training_subset.nc"
)

# Public ARCO-ERA5 store (same source as prepare_forecasts_and_targets.download_target_data).
ERA5_ZARR = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

LEAD_TIME_HOURS = 24
VARIABLE = "2m_temperature"


def cache_paths():
    """Return the output cache file paths for the cleaned forecast and ERA5 truth.

    Inputs: none (uses helper_funcs.setup_directories for the processed root).

    Returns:
        tuple[str, str]: (forecast_path, era5_path) under <processed>/heat_wave/.
    """
    dirs = setup_directories()
    cache_dir = os.path.join(dirs["processed"], "heat_wave")
    os.makedirs(cache_dir, exist_ok=True)
    forecast_path = os.path.join(cache_dir, "forecast_pnw_2mt_24h.nc")
    era5_path = os.path.join(cache_dir, "era5_pnw_2mt_truth.nc")
    return forecast_path, era5_path


def build_forecast_dataset(nc_path):
    """Load the AIFS subset and convert it to the pipeline's forecast convention.

    Shifts `time` from init time to valid time (init + LEAD_TIME_HOURS) and re-adds a
    prediction_timedelta dimension holding the single 24h lead time.

    Inputs:
        nc_path (str): path to heat_wave_training_subset.nc.

    Returns:
        xr.Dataset: dims (time, prediction_timedelta, latitude, longitude) with
            `time` as the forecast valid time and the 2m_temperature variable.
    """
    ds = xr.open_dataset(nc_path)
    lead = np.timedelta64(LEAD_TIME_HOURS, "h")
    # init time -> valid time
    ds = ds.assign_coords(time=ds.time + lead)
    # re-add the (single) lead-time dimension the pipeline selects on
    ds = ds.expand_dims(prediction_timedelta=[lead])
    return ds.load()


def fetch_era5_truth(forecast_ds):
    """Pull ERA5 2m_temperature matching the forecast grid and valid times.

    Inputs:
        forecast_ds (xr.Dataset): cleaned forecast dataset (valid-time indexed) whose
            latitude/longitude/time coordinates define the target selection.

    Returns:
        xr.Dataset: ERA5 2m_temperature with dims (time, latitude, longitude) aligned
            to forecast_ds.
    """
    era5 = xr.open_zarr(ERA5_ZARR, consolidated=True, storage_options={"token": "anon"})
    era5 = era5[[VARIABLE]]
    # ERA5 is a global 0.25 deg grid that aligns with the forecast box; nearest snaps
    # any floating-point coordinate mismatch.
    era5 = era5.sel(
        latitude=forecast_ds.latitude,
        longitude=forecast_ds.longitude,
        method="nearest",
    )
    era5 = era5.sel(time=forecast_ds.time.values)
    return era5.load()


def main():
    """Build and cache the cleaned forecast and matching ERA5 ground-truth datasets."""
    forecast_path, era5_path = cache_paths()

    if os.path.exists(forecast_path) and os.path.exists(era5_path):
        print(f"Cache already present:\n  {forecast_path}\n  {era5_path}\nNothing to do.")
        return

    print(f"Loading forecast subset: {FORECAST_NC}")
    forecast_ds = build_forecast_dataset(FORECAST_NC)
    print(
        f"  forecast dims={dict(forecast_ds.sizes)}, "
        f"valid time range {forecast_ds.time.values.min()} .. {forecast_ds.time.values.max()}"
    )

    print(f"Fetching matching ERA5 from {ERA5_ZARR} ...")
    era5_ds = fetch_era5_truth(forecast_ds)
    print(f"  era5 dims={dict(era5_ds.sizes)}")

    forecast_ds.to_netcdf(forecast_path)
    era5_ds.to_netcdf(era5_path)
    print(f"Wrote:\n  {forecast_path}\n  {era5_path}")

    # Sanity check: forecast - truth errors should be physically small (single-digit K),
    # which confirms the +24h valid-time shift direction is correct. Compare on raw
    # numpy values so a sub-grid float mismatch in coords can't silently NaN the diff.
    fc_vals = forecast_ds[VARIABLE].squeeze("prediction_timedelta").values
    err = fc_vals - era5_ds[VARIABLE].values
    rmse = float(np.sqrt(np.nanmean(err**2)))
    print(f"Sanity check: raw forecast vs ERA5 RMSE = {rmse:.3f} K")


if __name__ == "__main__":
    main()
