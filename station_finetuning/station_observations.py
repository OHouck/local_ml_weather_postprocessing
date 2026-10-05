"""Discovery, download and parsing of HadISD surface station observations.

The post-processing models in this folder are trained against real station
observations rather than ERA5, so this module supplies that ground truth.

We use HadISD (Met Office Hadley Centre; Dunn et al. 2012, 2016), a selected and
quality-controlled subset of NOAA's Integrated Surface Database. It is free,
global, and its automated QC (duplicate months, frequent values, spikes, streaks,
climatological outliers, neighbour checks, ...) is published and citable, which
raw ISD-Lite is not. HadISD is not homogenised, and it shares ISD's station
identifiers ('{usaf}-{wban}'), so everything downstream is keyed exactly as it
was under ISD-Lite. v3.4.3.2025f is the final release: ISD itself stopped
updating on 29 August 2025.

Each station is published as one netCDF covering its whole record (2-5 MB
compressed). Only the 00 UTC temperature and wind speed are ever used, so on
download each file is reduced to those two series and the netCDF is discarded;
the cache is ~0.3 MB per station rather than ~5 MB.

One practical constraint shapes everything downstream: Pangu and IFS forecasts
in this project are initialised at 00 UTC, so every verification time is also
00 UTC. Many national synoptic stations only report at 03 and 12 UTC and
therefore never coincide with a verification time. In practice this restricts
the usable sample to near-hourly reporting stations, which are mostly airports.
That attrition is itself a finding about observing infrastructure, so the
coverage statistics are retained rather than silently dropped.
"""

import gzip
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import xarray as xr

# Stations reporting on fewer than this fraction of 00 UTC training days are
# dropped at training time and reported as unusable when a dataset is prepared.
MINIMUM_TRAINING_COVERAGE = 0.70

HADISD_VERSION = "3.4.3.2025f"
HADISD_BASE_URL = "https://www.metoffice.gov.uk/hadobs/hadisd/v343_2025f"
HADISD_STATION_LIST_URL = (
    f"{HADISD_BASE_URL}/files/hadisd_station_info_v343_2025f.txt")
HADISD_STATION_URL_TEMPLATE = (
    f"{HADISD_BASE_URL}/data/hadisd.{HADISD_VERSION}_19310101-20250829_"
    "{station_id}.nc.gz")

# HadISD has no names, countries or record dates in its station list, so those
# are joined in from ISD's own metadata table on the shared identifier.
ISD_HISTORY_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"

# Values removed by HadISD's QC are kept in the file but overwritten with this
# sentinel; plain missing values are -1e30 and xarray already decodes them to
# NaN. Anything below this threshold is therefore a QC rejection.
HADISD_FLAGGED_THRESHOLD = -1e29

SELECTED_STATION_COLUMNS = ["usaf", "wban", "station_name", "country",
                            "latitude", "longitude", "elevation_metres",
                            "region"]


def _download_once(url, path):
    """Download url to path unless the file already exists.

    Inputs:
        url (str): file to fetch.
        path (str): local destination; its directory is created if missing.

    Returns:
        str: path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        print(f"downloading {url}", flush=True)
        urllib.request.urlretrieve(url, path)
    return path


def read_station_metadata(metadata_path):
    """Read a station metadata csv without losing zero-padded identifiers.

    'usaf' and 'wban' are zero-padded identifiers, not numbers: pandas would
    read '010010' as the integer 10010, which then fails to match the ids
    stored alongside the arrays. Every reader of these csvs goes through here
    so the coercion cannot creep back in at one call site.

    Inputs:
        metadata_path (str): csv written by prepare_station_dataset.py, or any
            results csv carrying the same identifier columns.

    Returns:
        pd.DataFrame: the csv's rows, with 'usaf' and 'wban' as strings.
    """
    return pd.read_csv(metadata_path, dtype={"usaf": str, "wban": str})


def _read_active_station_table(work_directory, first_required_day,
                               last_required_day):
    """Read the HadISD station list and keep the stations usable over the study period.

    Every HadISD station is a candidate. ISD's record dates are used to drop
    stations known to start after or stop before the study period, so that
    training and test years come from the same instrument; a HadISD station
    with no isd-history row (a handful of composites) is kept, and left to the
    00 UTC coverage screen. Rows with no usable position are dropped.

    Inputs:
        work_directory (str): directory the station pipeline caches into; the
            HadISD station list and isd-history.csv are downloaded here once.
        first_required_day (int): earliest day the station must already be
            reporting, as YYYYMMDD, e.g. 20180101.
        last_required_day (int): latest day the station must still be
            reporting, as YYYYMMDD, e.g. 20221231.

    Returns:
        pd.DataFrame: the surviving rows, with 'usaf', 'wban', 'station_name',
            'country', 'latitude', 'longitude' (0..360) and 'elevation_metres'.
    """
    station_list_path = _download_once(
        HADISD_STATION_LIST_URL,
        os.path.join(work_directory, os.path.basename(HADISD_STATION_LIST_URL)))
    isd_history_path = _download_once(
        ISD_HISTORY_URL, os.path.join(work_directory, "isd-history.csv"))

    # The list is fixed-width: 'usaf-wban', latitude, longitude, elevation,
    # then a free-text name that is ignored in favour of isd-history's.
    stations = pd.read_csv(
        station_list_path, sep=r"\s+", header=None, usecols=[0, 1, 2, 3],
        names=["station_id", "latitude", "longitude", "elevation_metres"],
        dtype={"station_id": str})
    stations["usaf"] = stations["station_id"].str[:6]
    stations["wban"] = stations["station_id"].str[7:]

    # dtype=str preserves the leading zeros of the identifiers.
    history = pd.read_csv(isd_history_path, dtype=str).rename(columns={
        "USAF": "usaf", "WBAN": "wban", "STATION NAME": "station_name",
        "CTRY": "country"})
    history["begin_day"] = pd.to_numeric(history["BEGIN"], errors="coerce")
    history["end_day"] = pd.to_numeric(history["END"], errors="coerce")
    stations = stations.merge(
        history[["usaf", "wban", "station_name", "country", "begin_day",
                 "end_day"]].drop_duplicates(["usaf", "wban"]),
        on=["usaf", "wban"], how="left")

    latitude, longitude = stations["latitude"], stations["longitude"]
    has_position = (latitude.notna() & longitude.notna()
                    & ~((latitude == 0.0) & (longitude == 0.0)))
    stations["longitude"] = longitude % 360.0
    # NaN dates (no isd-history row) compare False on both sides, so they pass.
    outside_period = ((stations["begin_day"] > first_required_day)
                      | (stations["end_day"] < last_required_day))
    return stations[has_position & ~outside_period].copy()


def find_stations_in_boxes(work_directory, region_boxes, first_required_day,
                           last_required_day):
    """Select HadISD stations that sit inside given boxes and were active throughout.

    Inputs:
        work_directory (str): station pipeline cache directory; see
            _read_active_station_table.
        region_boxes (dict): maps a region name (str) to a tuple
            (min_latitude, max_latitude, min_longitude, max_longitude) in
            degrees. Longitudes may be given in either -180..180 or 0..360;
            they are compared in 0..360.
        first_required_day (int): earliest day the station must already be
            reporting, as YYYYMMDD, e.g. 20180101.
        last_required_day (int): latest day the station must still be
            reporting, as YYYYMMDD, e.g. 20221231.

    Returns:
        pd.DataFrame: one row per selected station with the columns in
            SELECTED_STATION_COLUMNS. Station identifiers are kept as
            zero-padded strings because the URL scheme requires them.
    """
    station_table = _read_active_station_table(
        work_directory, first_required_day, last_required_day)

    selected_frames = []
    for region_name, box in region_boxes.items():
        min_latitude, max_latitude, min_longitude, max_longitude = box
        inside_box = (
            (station_table["latitude"] >= min_latitude)
            & (station_table["latitude"] <= max_latitude)
            & (station_table["longitude"] >= min_longitude % 360.0)
            & (station_table["longitude"] <= max_longitude % 360.0))

        matching = station_table[inside_box].copy()
        matching["region"] = region_name
        selected_frames.append(matching)

    if not selected_frames:
        return pd.DataFrame(columns=SELECTED_STATION_COLUMNS)

    return pd.concat(selected_frames,
                     ignore_index=True)[SELECTED_STATION_COLUMNS]


def find_stations_globally(work_directory, first_required_day,
                           last_required_day, latitude_band=None,
                           maximum_stations=None, random_seed=58):
    """Select every active HadISD station on Earth, optionally banded and capped.

    Inputs:
        work_directory (str): station pipeline cache directory; see
            _read_active_station_table.
        first_required_day (int): earliest day the station must already be
            reporting, as YYYYMMDD.
        last_required_day (int): latest day the station must still be
            reporting, as YYYYMMDD.
        latitude_band (tuple or None): (min_latitude, max_latitude) in degrees
            to restrict the draw to, e.g. (-23.5, 23.5) for the tropics. None
            takes the whole globe.
        maximum_stations (int or None): cap on the number returned. When the
            selection is larger, a reproducible random subset is taken rather
            than the first N, which would be ordered by USAF identifier and so
            geographically clustered. None keeps everything.
        random_seed (int): seed for that subsample, so a capped run repeats.

    Returns:
        pd.DataFrame: one row per selected station with the columns in
            SELECTED_STATION_COLUMNS. 'region' is 'global', or 'band' when a
            latitude_band was given.
    """
    station_table = _read_active_station_table(
        work_directory, first_required_day, last_required_day)

    region_name = "global"
    if latitude_band is not None:
        minimum_latitude, maximum_latitude = latitude_band
        station_table = station_table[
            (station_table["latitude"] >= minimum_latitude)
            & (station_table["latitude"] <= maximum_latitude)].copy()
        region_name = "band"
    station_table["region"] = region_name

    if maximum_stations is not None and len(station_table) > maximum_stations:
        station_table = station_table.sample(n=maximum_stations,
                                             random_state=random_seed)

    return station_table.reset_index(drop=True)[SELECTED_STATION_COLUMNS]


def _cache_path(destination_directory, station_id):
    """Return the path of one station's cached 00 UTC series.

    Inputs:
        destination_directory (str): the HadISD cache directory.
        station_id (str): '{usaf}-{wban}'.

    Returns:
        str: '<destination_directory>/<station_id>.csv.gz'.
    """
    return os.path.join(destination_directory, f"{station_id}.csv.gz")


def _reduce_to_00utc(netcdf_path, output_path):
    """Write one HadISD file's 00 UTC temperature and wind speed to a small csv.

    QC-rejected values are set to NaN, so the cache only ever holds data that
    passed HadISD's checks.

    Inputs:
        netcdf_path (str): an uncompressed HadISD station netCDF.
        output_path (str): csv.gz to write, with columns 'date',
            'temperature_celsius' and 'wind_speed_ms', one row per 00 UTC
            report over the station's whole record.

    Returns:
        None.
    """
    # Only the 00 UTC rows are decoded: the full hourly record runs to
    # hundreds of thousands of rows per station.
    with xr.open_dataset(netcdf_path) as dataset:
        at_00utc = np.flatnonzero(dataset["time"].dt.hour.values == 0)
        hourly = (dataset[["temperatures", "windspeeds"]].isel(time=at_00utc)
                  .to_dataframe()[["temperatures", "windspeeds"]])

    hourly = hourly.where(hourly > HADISD_FLAGGED_THRESHOLD)
    hourly.index = hourly.index.normalize().rename("date")
    hourly = hourly[~hourly.index.duplicated()].rename(columns={
        "temperatures": "temperature_celsius", "windspeeds": "wind_speed_ms"})
    hourly.to_csv(output_path)


def download_station_files(station_frame, destination_directory,
                           parallel_downloads=6):
    """Fetch HadISD files for the given stations and cache their 00 UTC series.

    Downloads run in a thread pool because they are latency-bound. Reading the
    netCDF happens on the calling thread, one file at a time, because the HDF5
    library underneath is not thread-safe and crashes the interpreter when two
    files are opened concurrently. Stations already cached are skipped, so a
    re-run only fetches what is missing; per-station failures are reported
    rather than raised.

    Inputs:
        station_frame (pd.DataFrame): must contain 'usaf' and 'wban' as
            zero-padded strings.
        destination_directory (str): directory for the cached csv.gz files; it
            is created if missing.
        parallel_downloads (int): number of concurrent HTTP requests.

    Returns:
        int: number of stations whose cache exists after the call.
    """
    os.makedirs(destination_directory, exist_ok=True)
    station_ids = (station_frame["usaf"] + "-" + station_frame["wban"]).tolist()
    wanted = [station_id for station_id in station_ids
              if not os.path.exists(_cache_path(destination_directory, station_id))]

    def fetch_one_station(station_id):
        """Return the station id and its decompressed netCDF bytes, or None on failure."""
        url = HADISD_STATION_URL_TEMPLATE.format(station_id=station_id)
        try:
            with urllib.request.urlopen(url, timeout=300) as response:
                return station_id, gzip.decompress(response.read())
        except (OSError, EOFError):
            return station_id, None

    failures = []
    if wanted:
        print(f"downloading {len(wanted)} HadISD stations", flush=True)
    with ThreadPoolExecutor(max_workers=parallel_downloads) as pool:
        for count, (station_id, payload) in enumerate(
                pool.map(fetch_one_station, wanted), start=1):
            if payload is None:
                failures.append(station_id)
                continue
            netcdf_path = os.path.join(destination_directory, f"{station_id}.nc")
            with open(netcdf_path, "wb") as netcdf_file:
                netcdf_file.write(payload)
            try:
                _reduce_to_00utc(netcdf_path,
                                 _cache_path(destination_directory, station_id))
            except (OSError, ValueError, KeyError):
                failures.append(station_id)
            finally:
                os.remove(netcdf_path)
            if count % 250 == 0:
                print(f"  {count} of {len(wanted)}", flush=True)

    if failures:
        print(f"{len(failures)} stations failed to download or parse, e.g. "
              f"{failures[:5]}", flush=True)
    return len(station_ids) - len(failures)


def load_station_observations(station_frame, source_directory,
                              verification_times):
    """Read cached 00 UTC temperature and wind speed onto a time axis.

    Values are converted to the units the forecasts use: kelvin for
    temperature and m/s for wind speed.

    Inputs:
        station_frame (pd.DataFrame): stations to load, with 'usaf' and 'wban'.
        source_directory (str): directory holding the cached csv.gz files
            written by download_station_files.
        verification_times (pd.DatetimeIndex): the 00 UTC times the forecast
            arrays are indexed by. Observations are aligned onto exactly this
            axis, with NaN where a station did not report or failed QC.

    Returns:
        np.ndarray: shape (n_times, 2, n_stations), float32. The middle axis is
            [2m_temperature in K, 10m_wind_speed in m/s]. Missing values are NaN.
    """
    observations = np.full(
        (len(verification_times), 2, len(station_frame)), np.nan, dtype=np.float32)

    station_ids = station_frame["usaf"] + "-" + station_frame["wban"]
    for station_position, station_id in enumerate(station_ids):
        path = _cache_path(source_directory, station_id)
        if not os.path.exists(path):
            continue
        series = (pd.read_csv(path, index_col="date", parse_dates=True)
                  .reindex(verification_times))
        observations[:, 0, station_position] = (
            series["temperature_celsius"].to_numpy(float) + 273.15)
        observations[:, 1, station_position] = (
            series["wind_speed_ms"].to_numpy(float))

    return observations


def training_coverage(observations, verification_times, training_years):
    """Fraction of 00 UTC training days on which each station reported temperature.

    The one definition used both to screen candidates and to filter stations at
    training time, so a station that passes one passes the other.

    Inputs:
        observations (np.ndarray): (n_times, 2, n_stations) output of
            load_station_observations.
        verification_times (pd.DatetimeIndex): the time axis of observations.
        training_years (list[int]): calendar years that count as training.

    Returns:
        np.ndarray: (n_stations,) coverage fractions in [0, 1].
    """
    in_training = np.isin(verification_times.year, training_years)
    return np.isfinite(observations[in_training, 0]).mean(axis=0)
