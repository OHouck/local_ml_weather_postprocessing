"""
heat_wave_training_subset.py

Build a regional training subset of AIFS forecasts for a heat-wave study near
Portland, Oregon. For each forecast file initialized in June/July/August of
2010-2025, this script:

  1. Opens the source forecast zarr.
  2. Filters to 2m_temperature at 24-hour lead time
     over a 6x6 degree box centered on (45 N, -122 E).
  3. Writes a small per-file subset zarr to /net/scratch/ohouck.

After every per-file subset is written, the script concatenates all of them
along the forecast initialization time into a single netCDF file (also under
/net/scratch/ohouck) and prints the resulting file size.

Intended to be run on the cluster where /net/monsoon/reforecast/aifs is mounted.
"""

import os
import re
import shutil
from pathlib import Path

import numpy as np
import xarray as xr

# Register numcodecs v2 codecs (Blosc, Zstd, ...) as zarr v3-compatible so
# `xr.open_zarr` can read AIFS stores that were written with numcodecs.Blosc.
# No-op on older numcodecs that don't ship a zarr3 module. If you instead pin
# zarr<3 in this env you can delete this import.
try:
    import numcodecs.zarr3  # noqa: F401
except ImportError:
    pass


SOURCE_DIR = Path("/net/monsoon/reforecast/aifs")
SCRATCH_DIR = Path("/net/scratch/ohouck")
SUBSET_DIR = SCRATCH_DIR / "heat_wave_subsets"
COMBINED_PATH = SCRATCH_DIR / "heat_wave_training_subset.nc"

CENTER_LAT = 45.0
CENTER_LON = -122.0       # degrees east; -122 == 238 in 0-360 convention
HALF_BOX_DEG = 3.0        # 6x6 degree box around the center point

YEAR_MIN = 2010
YEAR_MAX = 2010 # change back to 2025 after testing
SUMMER_MONTHS = {6, 7, 8}

VARIABLES = ["2m_temperature"]
LEAD_TIME_HOURS = 24

# Source filenames look like "2025-07-13T00.zarr".
FILENAME_RE = re.compile(r"^(\d{4})-(\d{2})-\d{2}T\d{2}\.zarr$")


def list_forecast_files(source_dir):
    """Return the AIFS forecast zarr paths that fall within the target window.

    Inputs:
        source_dir (Path): directory containing AIFS forecast .zarr stores.

    Returns:
        list[Path]: forecast paths for init dates in YEAR_MIN-YEAR_MAX whose
            month is June, July, or August, sorted chronologically by filename.
    """
    matched = []
    for entry in sorted(source_dir.iterdir()):
        m = FILENAME_RE.match(entry.name)
        if not m:
            continue
        year, month = int(m.group(1)), int(m.group(2))
        if YEAR_MIN <= year <= YEAR_MAX and month in SUMMER_MONTHS:
            matched.append(entry)
    return matched


def subset_one_forecast(src_path, dst_path):
    """Open one AIFS forecast and write a variable/lead-time/region subset.

    Filters to VARIABLES, a 6x6 degree box around (CENTER_LAT, CENTER_LON), and
    the 24-hour lead time (selected from prediction_timedelta for the sub-daily
    variable and from prediction_timedelta_daily for the daily-averaged one).

    Inputs:
        src_path (Path): path to the source forecast .zarr.
        dst_path (Path): path where the per-file subset .zarr will be written.

    Returns:
        Path: dst_path on successful write.
    """
    # Disable time decoding on open: the `prediction_timedelta_daily` coord
    # has units "days since <init-date>" which xarray's time decoder chokes on
    # even with cftime installed. We'll set `time` ourselves from the filename
    # after subsetting variables (which drops the daily coord).
    ds = xr.open_zarr(
        src_path,
        consolidated=True,
        decode_times=False,
        decode_timedelta=True,
    )

    # AIFS longitudes are stored in [0, 360); convert the center accordingly.
    lon_center_0_360 = CENTER_LON % 360
    lon_min = lon_center_0_360 - HALF_BOX_DEG
    lon_max = lon_center_0_360 + HALF_BOX_DEG
    lat_min = CENTER_LAT - HALF_BOX_DEG
    lat_max = CENTER_LAT + HALF_BOX_DEG

    # latitude runs 90 -> -90, so slice from the higher value to the lower one.
    ds = ds[VARIABLES].sel(
        latitude=slice(lat_max, lat_min),
        longitude=slice(lon_min, lon_max),
    )

    # 2m_temperature is indexed by prediction_timedelta
    lead = np.timedelta64(LEAD_TIME_HOURS, "h")
    if "prediction_timedelta" in ds["2m_temperature"].dims:
        ds["2m_temperature"] = ds["2m_temperature"].sel(prediction_timedelta=lead)

    # Drop now-degenerate lead-time coords so per-file outputs concat cleanly.
    drop_coords = [
        c for c in ("prediction_timedelta", "prediction_timedelta_daily")
        if c in ds.coords
    ]
    if drop_coords:
        ds = ds.drop_vars(drop_coords)

    # Stamp `time` from the filename (e.g. "2010-08-19T00" -> 2010-08-19T00).
    # We opened with decode_times=False so the raw `time` is just a numeric
    # offset; the init date is in the filename itself.
    init_iso = src_path.stem.replace("T", "T") + ":00:00"  # "2010-08-19T00:00:00"
    ds = ds.assign_coords(time=("time", np.array([init_iso], dtype="datetime64[ns]")))

    ds = ds.load()  # materialize before write so the source store can close
    if dst_path.exists():
        shutil.rmtree(dst_path)
    ds.to_zarr(dst_path, mode="w", consolidated=True)
    ds.close()
    return dst_path


def combine_subsets(subset_paths, combined_path):
    """Concatenate per-file subsets along forecast init time, write one netCDF.

    Inputs:
        subset_paths (list[Path]): per-file subset .zarr stores to combine.
        combined_path (Path): destination netCDF path.

    Returns:
        Path: combined_path.
    """
    datasets = [
        xr.open_zarr(p, consolidated=True, decode_timedelta=True)
        for p in subset_paths
    ]
    combined = xr.concat(datasets, dim="time")
    combined = combined.sortby("time")
    if combined_path.exists():
        combined_path.unlink()
    combined.load().to_netcdf(combined_path)
    for d in datasets:
        d.close()
    return combined_path


def human_size(num_bytes):
    """Format a byte count as a human-readable string.

    Inputs:
        num_bytes (int): size in bytes.

    Returns:
        str: a string such as '12.34 MB'.
    """
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0


def main():
    """Drive the per-file subset step and the final concat/export step."""
    SUBSET_DIR.mkdir(parents=True, exist_ok=True)

    files = list_forecast_files(SOURCE_DIR)
    print(f"Found {len(files)} forecast files to subset in {SOURCE_DIR}.")

    subset_paths = []
    for i, src in enumerate(files, 1):
        dst = SUBSET_DIR / f"{src.stem}_subset.zarr"
        # if dst.exists():
        #     print(f"[{i}/{len(files)}] skipping (already subset): {dst.name}")
        #     subset_paths.append(dst)
        #     continue
        print(f"[{i}/{len(files)}] subsetting {src.name} -> {dst.name}")
        try:
            subset_one_forecast(src, dst)
            subset_paths.append(dst)
        except Exception as e:
            print(f"  WARNING: failed to subset {src.name}: {e}")

    if not subset_paths:
        print("No subsets were produced; nothing to combine.")
        return

    print(f"\nCombining {len(subset_paths)} subsets into {COMBINED_PATH}")
    combine_subsets(subset_paths, COMBINED_PATH)

    total_bytes = os.path.getsize(COMBINED_PATH)
    print(f"Combined file: {COMBINED_PATH}")
    print(f"Total file size: {human_size(total_bytes)}")


if __name__ == "__main__":
    main()
