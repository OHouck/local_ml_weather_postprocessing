"""
Section 3 (part 1): choose the weather stations and download their observations.

Observations come from HadISD v3.4.3.2025f (Met Office Hadley Centre; Dunn et
al. 2012, 2016), a quality-controlled subset of NOAA's Integrated Surface
Database. Each station is one netCDF file covering its whole record; on
download it is reduced to its 00 UTC temperature and wind speed (the only
values the analysis uses, since every forecast verifies at 00 UTC), with
values rejected by HadISD's quality control set to missing, and cached as
station_finetuning/hadisd/{usaf}-{wban}.csv.gz.

The station sample has two parts, recorded in the manifest's
'sample_stratum' column:
    'balanced'        an equal number of high-income and low-and-middle-income
                      stations worldwide (1,000 each)
    'regional_boost'  every other usable station in Africa and in South
                      America outside Argentina, where the balanced draw
                      leaves most stations out because Asia dominates the
                      low-and-middle-income pool

A station is usable when it reported a 00 UTC temperature on at least 70% of
the training days (2018-2021). Because the forecasts verify at 00 UTC, this
mostly keeps near-hourly reporters, i.e. airports.

Writes station_finetuning/station_sample_manifest.csv.

Run on its own with:
    uv run python data_preparation/select_stations.py
"""

import gzip
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (MINIMUM_TRAINING_COVERAGE, TEST_YEAR, TRAINING_YEARS,  # noqa: E402
                    setup_directories)
from income_groups import load_income_group_countries, match_points_to_countries  # noqa: E402

# A station's record must span the whole study period, as YYYYMMDD
FIRST_REQUIRED_DAY = TRAINING_YEARS[0] * 10000 + 101
LAST_REQUIRED_DAY = TEST_YEAR * 10000 + 1231

STATIONS_PER_INCOME_GROUP = 1000
# Candidates screened per station wanted, because coverage is only known after download
SCREENING_OVERSAMPLE = 2.0
SAMPLE_SEED = 58
BOOST_CONTINENTS = ["africa", "south_america"]
BOOST_EXCLUDED_COUNTRIES = ["ARG"]
PARALLEL_DOWNLOADS = 6

HADISD_BASE_URL = "https://www.metoffice.gov.uk/hadobs/hadisd/v343_2025f"
HADISD_STATION_LIST_URL = f"{HADISD_BASE_URL}/files/hadisd_station_info_v343_2025f.txt"
HADISD_STATION_URL = (f"{HADISD_BASE_URL}/data/hadisd.3.4.3.2025f_19310101-20250829_"
                      "{station_id}.nc.gz")
# HadISD's station list has no names, countries or record dates; those come
# from ISD's own station table, joined on the shared identifier
ISD_HISTORY_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
# HadISD overwrites values rejected by quality control with -2e30
HADISD_REJECTED_THRESHOLD = -1e29

# Natural Earth's continent names -> the continent names used in this project
CONTINENT_NAMES = {"Africa": "africa", "Asia": "asia", "Europe": "europe",
                   "North America": "north_america", "South America": "south_america",
                   "Oceania": "oceania"}


def find_active_stations(station_directory):
    """
    List every HadISD station that reported throughout the study period.

    Stations known (from ISD's record dates) to start after 2018 or stop
    before the end of 2022 are dropped; stations with no ISD record dates are
    kept and left to the coverage screen.

    Inputs:
        station_directory (str): where the station lists are cached.

    Returns:
        pandas.DataFrame with 'usaf', 'wban', 'station_name', 'country',
        'latitude', 'longitude' (0..360), 'elevation_metres' and 'region'.
    """
    # Download the HadISD station list and ISD's station table once
    station_list_path = os.path.join(station_directory, os.path.basename(HADISD_STATION_LIST_URL))
    isd_history_path = os.path.join(station_directory, "isd-history.csv")
    for url, path in [(HADISD_STATION_LIST_URL, station_list_path),
                      (ISD_HISTORY_URL, isd_history_path)]:
        if not os.path.exists(path):
            print(f"  Downloading {url}")
            urllib.request.urlretrieve(url, path)

    # The HadISD list is fixed-width: 'usaf-wban', latitude, longitude, elevation, name
    stations = pd.read_csv(station_list_path, sep=r"\s+", header=None, usecols=[0, 1, 2, 3],
                           names=["station_id", "latitude", "longitude", "elevation_metres"],
                           dtype={"station_id": str})
    stations["usaf"] = stations["station_id"].str[:6]
    stations["wban"] = stations["station_id"].str[7:]

    # Join names, countries and record dates from ISD's station table
    history = pd.read_csv(isd_history_path, dtype=str).rename(columns={
        "USAF": "usaf", "WBAN": "wban", "STATION NAME": "station_name", "CTRY": "country"})
    history["begin_day"] = pd.to_numeric(history["BEGIN"], errors="coerce")
    history["end_day"] = pd.to_numeric(history["END"], errors="coerce")
    stations = stations.merge(history[["usaf", "wban", "station_name", "country", "begin_day",
                                       "end_day"]].drop_duplicates(["usaf", "wban"]),
                              on=["usaf", "wban"], how="left")

    # Drop stations with no position and stations not active over the whole
    # study period (missing record dates compare False, so they are kept)
    has_position = (stations["latitude"].notna() & stations["longitude"].notna()
                    & ~((stations["latitude"] == 0.0) & (stations["longitude"] == 0.0)))
    stations["longitude"] = stations["longitude"] % 360.0
    outside_period = ((stations["begin_day"] > FIRST_REQUIRED_DAY)
                      | (stations["end_day"] < LAST_REQUIRED_DAY))
    stations = stations[has_position & ~outside_period].copy()
    stations["region"] = "global"
    return stations.reset_index(drop=True)[["usaf", "wban", "station_name", "country", "latitude",
                                            "longitude", "elevation_metres", "region"]]


def download_station_observations(stations, observation_directory):
    """
    Download HadISD files and cache each station's 00 UTC temperature and wind speed.

    Downloads run in parallel threads; each file is then reduced on the main
    thread, one at a time, because the HDF5 library underneath netCDF is not
    thread-safe. Stations already cached are skipped, and failures are
    reported rather than raised.

    Inputs:
        stations (pandas.DataFrame): with 'usaf' and 'wban'.
        observation_directory (str): where the csv.gz files are cached.

    Returns:
        int, the number of these stations with cached observations afterwards.
    """
    os.makedirs(observation_directory, exist_ok=True)
    station_ids = (stations["usaf"] + "-" + stations["wban"]).tolist()
    missing_ids = [station_id for station_id in station_ids if not os.path.exists(
        os.path.join(observation_directory, f"{station_id}.csv.gz"))]

    def fetch(station_id):
        """
        Download one station's compressed netCDF.

        Inputs:
            station_id (str): '{usaf}-{wban}'.

        Returns:
            tuple (station_id, decompressed bytes or None if the download failed).
        """
        try:
            url = HADISD_STATION_URL.format(station_id=station_id)
            with urllib.request.urlopen(url, timeout=300) as response:
                return station_id, gzip.decompress(response.read())
        except (OSError, EOFError):
            return station_id, None

    failures = []
    if missing_ids:
        print(f"  Downloading {len(missing_ids)} HadISD stations")
    with ThreadPoolExecutor(max_workers=PARALLEL_DOWNLOADS) as pool:
        for station_id, netcdf_bytes in pool.map(fetch, missing_ids):
            if netcdf_bytes is None:
                failures.append(station_id)
                continue
            netcdf_path = os.path.join(observation_directory, f"{station_id}.nc")
            with open(netcdf_path, "wb") as netcdf_file:
                netcdf_file.write(netcdf_bytes)
            try:
                # Keep the 00 UTC reports, with quality-control rejections set to missing
                with xr.open_dataset(netcdf_path) as dataset:
                    midnight_reports = np.flatnonzero(dataset["time"].dt.hour.values == 0)
                    reports = (dataset[["temperatures", "windspeeds"]]
                               .isel(time=midnight_reports).to_dataframe()
                               [["temperatures", "windspeeds"]])
                reports = reports.where(reports > HADISD_REJECTED_THRESHOLD)
                reports.index = reports.index.normalize().rename("date")
                reports = reports[~reports.index.duplicated()].rename(columns={
                    "temperatures": "temperature_celsius", "windspeeds": "wind_speed_ms"})
                reports.to_csv(os.path.join(observation_directory, f"{station_id}.csv.gz"))
            except (OSError, ValueError, KeyError):
                failures.append(station_id)
            finally:
                os.remove(netcdf_path)
    if failures:
        print(f"  {len(failures)} stations failed to download or parse, e.g. {failures[:5]}")
    return len(station_ids) - len(failures)


def load_station_observations(stations, observation_directory, verification_times):
    """
    Read the cached 00 UTC observations onto a common time axis.

    Inputs:
        stations (pandas.DataFrame): with 'usaf' and 'wban'.
        observation_directory (str): where download_station_observations() cached them.
        verification_times (pandas.DatetimeIndex): the 00 UTC times to align onto.

    Returns:
        numpy.ndarray of shape (n_times, 2, n_stations), float32: 2m
        temperature in K and 10m wind speed in m/s, NaN where the station did
        not report or its value failed quality control.
    """
    observations = np.full((len(verification_times), 2, len(stations)), np.nan,
                           dtype=np.float32)
    for station_position, station_id in enumerate(stations["usaf"] + "-" + stations["wban"]):
        path = os.path.join(observation_directory, f"{station_id}.csv.gz")
        if not os.path.exists(path):
            continue
        reports = pd.read_csv(path, index_col="date", parse_dates=True).reindex(verification_times)
        observations[:, 0, station_position] = (reports["temperature_celsius"].to_numpy(float)
                                                + 273.15)
        observations[:, 1, station_position] = reports["wind_speed_ms"].to_numpy(float)
    return observations


def select_income_balanced_sample(stations, stations_per_income_group):
    """
    Draw an equal number of stations at random from each income group.

    Within a group the draw is uniform, so continents keep their natural
    proportions (HadISD has no high-income stations in Africa at all, so
    continents could not be balanced anyway).

    Inputs:
        stations (pandas.DataFrame): candidates with an 'income_group' column.
        stations_per_income_group (int): stations to take from each group; a
            smaller group contributes all of its stations.

    Returns:
        pandas.DataFrame of the drawn stations, by income group, each group in
        its original order and keeping its original row labels. Stations with
        no income group are left out.
    """
    drawn_groups = []
    for _, group in stations[stations["income_group"].notna()].groupby("income_group"):
        if len(group) > stations_per_income_group:
            group = group.sample(n=stations_per_income_group, random_state=SAMPLE_SEED)
        drawn_groups.append(group.sort_index())
    return pd.concat(drawn_groups)


def run():
    """
    Choose the station sample, download its observations, and write the manifest.

    Inputs:
        None.

    Returns:
        None. Writes station_finetuning/station_sample_manifest.csv.
    """
    directories = setup_directories()
    station_directory = directories["station_output"]
    observation_directory = os.path.join(station_directory, "hadisd")

    # Every active station, labelled with its income group, continent and country
    candidates = find_active_stations(station_directory)
    countries = load_income_group_countries(directories)
    countries["continent"] = countries["continent"].map(CONTINENT_NAMES)
    country_attributes = match_points_to_countries(candidates["latitude"], candidates["longitude"],
                                                   countries, ["income_group", "continent", "iso3"])
    for column in ["income_group", "continent", "iso3"]:
        candidates[column] = country_attributes[column].to_numpy()
    candidates["in_boost_region"] = (candidates["continent"].isin(BOOST_CONTINENTS)
                                     & ~candidates["iso3"].isin(BOOST_EXCLUDED_COUNTRIES)
                                     & candidates["income_group"].notna())
    print(f"{len(candidates)} active stations, "
          f"{candidates['income_group'].notna().sum()} with an income group")

    # Screen an oversampled pool (plus every boost-region station) for coverage
    screened_per_group = int(round(STATIONS_PER_INCOME_GROUP * SCREENING_OVERSAMPLE))
    screening_pool = pd.concat([select_income_balanced_sample(candidates, screened_per_group),
                                candidates[candidates["in_boost_region"]]])
    screening_pool = screening_pool.drop_duplicates(subset=["usaf", "wban"]).reset_index(drop=True)
    print(f"Screening {len(screening_pool)} candidates")
    download_station_observations(screening_pool, observation_directory)
    training_days = pd.date_range(f"{TRAINING_YEARS[0]}-01-01", f"{TRAINING_YEARS[-1]}-12-31",
                                  freq="D")
    observations = load_station_observations(screening_pool, observation_directory, training_days)
    screening_pool["training_coverage"] = np.isfinite(observations[:, 0]).mean(axis=0)
    usable = screening_pool[screening_pool["training_coverage"] >= MINIMUM_TRAINING_COVERAGE]
    print(f"{len(usable)} of {len(screening_pool)} clear the "
          f"{MINIMUM_TRAINING_COVERAGE:.0%} 00 UTC coverage bar")

    # The final sample: the balanced draw, then the rest of the boost regions
    balanced = select_income_balanced_sample(usable, STATIONS_PER_INCOME_GROUP)
    boost = usable[usable["in_boost_region"]]
    selected = pd.concat([balanced.assign(sample_stratum="balanced"),
                          boost.assign(sample_stratum="regional_boost")])
    selected = selected.drop_duplicates(subset=["usaf", "wban"]).reset_index(drop=True)
    selected = selected.drop(columns="in_boost_region")

    # A few offshore stations match no continent; they still count towards the sample
    selected["continent"] = selected["continent"].fillna("unassigned")

    manifest_path = os.path.join(station_directory, "station_sample_manifest.csv")
    selected.to_csv(manifest_path, index=False)
    print(f"Wrote {manifest_path}: {len(selected)} stations")
    print(pd.crosstab(selected["continent"], [selected["sample_stratum"],
                                              selected["income_group"]], margins=True))


if __name__ == "__main__":
    run()
