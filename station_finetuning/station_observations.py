"""Discovery, download and parsing of NOAA ISD-Lite surface station observations.

The post-processing models in this folder are trained against real station
observations rather than ERA5, so this module supplies that ground truth.

We use ISD-Lite (https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/) because it
is free, global, and already reduced to one fixed-width row per station-hour,
which avoids parsing the full ISD format.

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
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

ISD_HISTORY_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
ISD_LITE_URL_TEMPLATE = (
    "https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/{year}/{station_id}-{year}.gz")

# ISD-Lite encodes missing values as -9999 and stores temperature and wind speed
# scaled by ten (i.e. in tenths of a degree Celsius and tenths of m/s).
ISD_MISSING_SENTINEL = -9999
ISD_SCALE_FACTOR = 10.0


def find_stations_in_boxes(isd_history_path, region_boxes, first_required_day,
                           last_required_day):
    """Select ISD stations that sit inside given boxes and were active throughout.

    Inputs:
        isd_history_path (str): path to a local copy of isd-history.csv, the
            NOAA station metadata table.
        region_boxes (dict): maps a region name (str) to a tuple
            (min_latitude, max_latitude, min_longitude, max_longitude) in
            degrees. Longitudes may be given in either -180..180 or 0..360;
            they are compared in 0..360.
        first_required_day (int): earliest day the station must already be
            reporting, as YYYYMMDD, e.g. 20180101.
        last_required_day (int): latest day the station must still be
            reporting, as YYYYMMDD, e.g. 20221231.

    Returns:
        pd.DataFrame: one row per selected station with columns 'usaf', 'wban',
            'station_name', 'country', 'latitude', 'longitude' (0..360),
            'elevation_metres' and 'region'. Station identifiers are kept as
            zero-padded strings because the URL scheme requires them.
    """
    # dtype=str preserves leading zeros in the USAF identifier, which are part
    # of the filename on the NOAA server.
    station_table = pd.read_csv(isd_history_path, dtype=str)
    station_table["latitude"] = station_table["LAT"].astype(float)
    station_table["longitude"] = station_table["LON"].astype(float) % 360.0
    station_table["elevation_metres"] = station_table["ELEV(M)"].astype(float)
    station_table["begin_day"] = pd.to_numeric(station_table["BEGIN"])
    station_table["end_day"] = pd.to_numeric(station_table["END"])

    # Require a record that spans the whole study period, so that training and
    # test years are drawn from the same physical instrument.
    active_throughout = (
        (station_table["begin_day"] <= first_required_day)
        & (station_table["end_day"] >= last_required_day))

    selected_frames = []
    for region_name, box in region_boxes.items():
        min_latitude, max_latitude, min_longitude, max_longitude = box
        inside_box = (
            (station_table["latitude"] >= min_latitude)
            & (station_table["latitude"] <= max_latitude)
            & (station_table["longitude"] >= min_longitude % 360.0)
            & (station_table["longitude"] <= max_longitude % 360.0))

        matching = station_table[inside_box & active_throughout].copy()
        matching["region"] = region_name
        selected_frames.append(matching)

    if not selected_frames:
        return pd.DataFrame()

    combined = pd.concat(selected_frames, ignore_index=True)
    combined = combined.rename(columns={"USAF": "usaf", "WBAN": "wban",
                                        "STATION NAME": "station_name",
                                        "CTRY": "country"})
    return combined[["usaf", "wban", "station_name", "country", "latitude",
                     "longitude", "elevation_metres", "region"]]


def download_station_files(station_frame, years, destination_directory,
                           parallel_downloads=16):
    """Fetch ISD-Lite yearly files for the given stations, skipping what exists.

    Downloads are latency-bound rather than bandwidth-bound, so they run in a
    thread pool. Not every station reported in every year, and NOAA answers
    those requests with a 404, so per-file failures are expected and are
    recorded rather than raised. Anything that arrives but is not valid gzip is
    deleted, so a later read cannot trip over a half-written file.

    Inputs:
        station_frame (pd.DataFrame): must contain 'usaf' and 'wban' as
            zero-padded strings.
        years (list[int]): calendar years to download.
        destination_directory (str): directory to write the .gz files into; it
            is created if missing.
        parallel_downloads (int): number of concurrent HTTP requests.

    Returns:
        dict: maps each year (int) to the number of valid files present for
            that year after the download completes.
    """
    os.makedirs(destination_directory, exist_ok=True)

    wanted_targets = []
    for _, station in station_frame.iterrows():
        for year in years:
            station_id = f"{station['usaf']}-{station['wban']}"
            output_path = os.path.join(destination_directory,
                                       f"{station_id}-{year}.gz")
            wanted_targets.append((station_id, year, output_path))

    def fetch_one_station_year(target):
        """Download a single station-year file and report whether it is valid."""
        station_id, year, output_path = target

        # Skip anything already downloaded, so re-running is cheap.
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return year, True

        url = ISD_LITE_URL_TEMPLATE.format(year=year, station_id=station_id)
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = response.read()
        except (urllib.error.URLError, OSError):
            # A missing station-year is normal, not an error worth raising.
            return year, False

        with open(output_path, "wb") as output_file:
            output_file.write(payload)

        # Confirm the body really is gzip before counting it as usable.
        try:
            with gzip.open(output_path, "rb") as handle:
                handle.read(1)
        except OSError:
            os.remove(output_path)
            return year, False
        return year, True

    files_per_year = {year: 0 for year in years}
    with ThreadPoolExecutor(max_workers=parallel_downloads) as pool:
        for year, was_downloaded in pool.map(fetch_one_station_year, wanted_targets):
            if was_downloaded:
                files_per_year[year] += 1

    return files_per_year


def load_station_observations(station_frame, years, source_directory,
                              verification_times):
    """Read 00 UTC temperature and wind speed for every station onto a time axis.

    Only the 00 UTC row of each day is kept, because that is the only hour at
    which the forecasts in this project verify. Values are converted to the
    units the forecasts use: kelvin for temperature and m/s for wind speed.

    Inputs:
        station_frame (pd.DataFrame): stations to load, with 'usaf' and 'wban'.
        years (list[int]): calendar years to read.
        source_directory (str): directory holding the downloaded .gz files.
        verification_times (pd.DatetimeIndex): the 00 UTC times the forecast
            arrays are indexed by. Observations are aligned onto exactly this
            axis, with NaN where a station did not report.

    Returns:
        np.ndarray: shape (n_times, 2, n_stations), float32. The middle axis is
            [2m_temperature in K, 10m_wind_speed in m/s]. Missing values are NaN.
    """
    observations = np.full(
        (len(verification_times), 2, len(station_frame)), np.nan, dtype=np.float32)

    for station_position, (_, station) in enumerate(station_frame.iterrows()):
        yearly_frames = []
        for year in years:
            file_path = os.path.join(
                source_directory, f"{station['usaf']}-{station['wban']}-{year}.gz")
            if not os.path.exists(file_path) or os.path.getsize(file_path) < 100:
                continue

            # ISD-Lite rows are whitespace separated: year, month, day, hour,
            # air temperature, dew point, sea level pressure, wind direction,
            # wind speed, sky cover, and two precipitation accumulations. Only
            # the six columns we need are parsed, by pandas rather than a Python
            # loop, because a full year of hourly records is ~8,700 rows per
            # station and there are thousands of station-years.
            hourly = pd.read_csv(
                file_path, sep=r"\s+", header=None,
                usecols=[0, 1, 2, 3, 4, 8],
                names=["year", "month", "day", "hour",
                       "air_temperature_celsius", "wind_speed_ms"])

            # Keep only the 00 UTC record, the single hour at which the
            # forecasts in this project verify.
            hourly = hourly[hourly["hour"] == 0]
            if hourly.empty:
                continue

            hourly = hourly.assign(
                date=pd.to_datetime(hourly[["year", "month", "day"]]),
                air_temperature_celsius=hourly["air_temperature_celsius"]
                .replace(ISD_MISSING_SENTINEL, np.nan) / ISD_SCALE_FACTOR,
                wind_speed_ms=hourly["wind_speed_ms"]
                .replace(ISD_MISSING_SENTINEL, np.nan) / ISD_SCALE_FACTOR)
            yearly_frames.append(
                hourly[["date", "air_temperature_celsius", "wind_speed_ms"]])

        if not yearly_frames:
            continue

        # Reindex onto the forecast time axis in one vectorised step. Assigning
        # row by row would let pandas coerce NaN to NaT in mixed-dtype rows.
        station_frame_observations = (
            pd.concat(yearly_frames)
            .drop_duplicates(subset="date")
            .set_index("date")
            .reindex(verification_times))

        observations[:, 0, station_position] = (
            station_frame_observations["air_temperature_celsius"].to_numpy(float)
            + 273.15)
        observations[:, 1, station_position] = (
            station_frame_observations["wind_speed_ms"].to_numpy(float))

    return observations
