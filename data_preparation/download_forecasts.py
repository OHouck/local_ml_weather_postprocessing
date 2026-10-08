"""
Section 1 (part 1): download the gridded Pangu-Weather and IFS HRES forecasts.

Both come from WeatherBench2 on the 0.25 degree grid:
    Pangu  gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr
    IFS    gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr

Only what the analysis uses is kept: forecasts initialised at 00 UTC, at
lead times of 1, 5 and 9 days, for 2m temperature and the 10m u and v wind.
WeatherBench2 indexes forecasts by initialisation time; they are re-indexed
here by VALID time (initialisation time + lead time), so that every lead time
of a given day sits beside the same verifying observation.

Writes raw/{model}_{year}.zarr for 2018-2022, with dimensions (valid_time,
prediction_timedelta, latitude, longitude). Years already on disk are skipped.

Run on its own with:
    uv run python data_preparation/download_forecasts.py
"""

import os
import shutil
import sys

import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (ALL_YEARS, DOWNLOADED_VARIABLES, FORECAST_MODELS,  # noqa: E402
                    LEAD_TIMES_HOURS, setup_directories)

FORECAST_URLS = {
    "pangu": "gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr",
    "ifs": "gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr",
}


def forecasts_by_valid_time(forecast_model):
    """
    Open a WeatherBench2 forecast archive and re-index it by valid time.

    Inputs:
        forecast_model (str): 'pangu' or 'ifs'.

    Returns:
        xarray.Dataset (lazy) with dims (valid_time, prediction_timedelta,
        latitude, longitude), holding the 00 UTC initialisations at the
        analysis lead times. A valid time whose initialisation is outside the
        archive is missing (NaN) at that lead time.
    """
    archive = xr.open_zarr(FORECAST_URLS[forecast_model], storage_options={"token": "anon"},
                           decode_timedelta=True)
    archive = archive[DOWNLOADED_VARIABLES].sel(
        prediction_timedelta=[np.timedelta64(lead_time, "h") for lead_time in LEAD_TIMES_HOURS])
    archive = archive.sel(time=archive.time.dt.hour == 0)

    # Valid time = initialisation time + lead time, one lead time at a time
    by_lead_time = []
    for lead_time in archive.prediction_timedelta.values:
        at_lead = archive.sel(prediction_timedelta=lead_time).rename({"time": "valid_time"})
        by_lead_time.append(at_lead.assign_coords(valid_time=at_lead.valid_time + lead_time))
    combined = xr.concat(by_lead_time, dim="prediction_timedelta", join="outer")
    combined = combined.sortby("valid_time")
    return combined.transpose("valid_time", "prediction_timedelta", "latitude", "longitude")


def run():
    """
    Download every forecast model and year that is not already on disk.

    Inputs:
        None.

    Returns:
        None. Writes raw/{model}_{year}.zarr.
    """
    directories = setup_directories()
    for forecast_model in FORECAST_MODELS:
        all_forecasts = None
        for year in ALL_YEARS:
            output_path = os.path.join(directories["raw"], f"{forecast_model}_{year}.zarr")
            if os.path.exists(output_path):
                continue
            if all_forecasts is None:
                all_forecasts = forecasts_by_valid_time(forecast_model)
            year_forecasts = all_forecasts.sel(valid_time=slice(f"{year}-01-01", f"{year}-12-31"))

            # One global field per chunk, so that reading one valid time and
            # lead time (as the station preparation does) decompresses only it
            year_forecasts = year_forecasts.chunk({"valid_time": 1, "prediction_timedelta": 1,
                                                   "latitude": -1, "longitude": -1})

            # Write beside the target and rename when done, so a partial
            # download is never mistaken for a finished one
            print(f"Downloading {forecast_model} {year} to {output_path}")
            partial_path = f"{output_path}.partial"
            shutil.rmtree(partial_path, ignore_errors=True)
            with ProgressBar():
                year_forecasts.drop_encoding().to_zarr(partial_path, mode="w")
            os.rename(partial_path, output_path)


if __name__ == "__main__":
    run()
