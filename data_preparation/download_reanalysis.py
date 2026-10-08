"""
Section 2: download the gridded truth: ERA5 reanalysis and the IFS HRES analysis.

    ERA5     ARCO-ERA5 (gs://gcp-public-data-arco-era5), the truth for Pangu,
             which was trained on ERA5
    HRES t0  WeatherBench2's IFS HRES analysis (the forecast at lead time 0),
             the truth for IFS, following the WeatherBench2 convention of
             verifying a forecast against its own analysis

Only 00 UTC fields of 2m temperature and the 10m u and v wind are kept,
since every forecast in the analysis verifies at 00 UTC. Writes
raw/era5_{year}.zarr and raw/hres_t0_{year}.zarr for 2018-2022, with
dimensions (time, latitude, longitude).

Also downloads raw/era5_static.nc, the ERA5 surface geopotential, from which
the station preparation takes each grid cell's elevation. This one file comes
from the Copernicus Climate Data Store and needs a CDS account (an API key in
~/.cdsapirc).

Files already on disk are skipped.

Run on its own with:
    uv run python data_preparation/download_reanalysis.py
"""

import os
import shutil
import sys

import xarray as xr
from dask.diagnostics import ProgressBar

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import ALL_YEARS, DOWNLOADED_VARIABLES, setup_directories  # noqa: E402

TRUTH_URLS = {
    "era5": "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
    "hres_t0": "gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr",
}


def run():
    """
    Download every truth source and year that is not already on disk.

    Inputs:
        None.

    Returns:
        None. Writes raw/{era5,hres_t0}_{year}.zarr and raw/era5_static.nc.
    """
    directories = setup_directories()
    for truth_source, url in TRUTH_URLS.items():
        archive = None
        for year in ALL_YEARS:
            output_path = os.path.join(directories["raw"], f"{truth_source}_{year}.zarr")
            if os.path.exists(output_path):
                continue
            # Opened without dask (chunks=None): ARCO-ERA5 holds hundreds of
            # variables in 1.3 million hourly chunks, and building dask arrays
            # for all of them takes tens of GB of memory before any selection
            if archive is None:
                archive = xr.open_zarr(url, storage_options={"token": "anon"},
                                       chunks=None)[DOWNLOADED_VARIABLES]

            # The 00 UTC fields of the year, one global field per chunk
            year_fields = archive.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
            year_fields = year_fields.sel(time=year_fields.time.dt.hour == 0)
            year_fields = year_fields.chunk({"time": 1, "latitude": -1, "longitude": -1})

            # Write beside the target and rename when done, so a partial
            # download is never mistaken for a finished one
            print(f"Downloading {truth_source} {year} to {output_path}")
            partial_path = f"{output_path}.partial"
            shutil.rmtree(partial_path, ignore_errors=True)
            with ProgressBar():
                year_fields.drop_encoding().to_zarr(partial_path, mode="w")
            os.rename(partial_path, output_path)

    # ERA5 surface geopotential (any date will do: it does not change)
    static_path = os.path.join(directories["raw"], "era5_static.nc")
    if not os.path.exists(static_path):
        import cdsapi
        print(f"Downloading the ERA5 surface geopotential to {static_path}")
        cdsapi.Client().retrieve("reanalysis-era5-single-levels", {
            "product_type": ["reanalysis"], "variable": ["geopotential"],
            "year": ["2023"], "month": ["01"], "day": ["01"], "time": ["00:00"],
            "data_format": "netcdf", "download_format": "unarchived"}, static_path)


if __name__ == "__main__":
    run()
