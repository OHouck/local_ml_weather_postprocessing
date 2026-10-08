"""
Section 3 (part 2): build the station datasets the station post-processing trains on.

For every station in the manifest written by select_stations.py, and for each
forecast model (Pangu and IFS), this extracts:
    1. a 5x5 grid-cell neighbourhood of the raw forecast (2m temperature and
       10m u and v wind) around the station, at every 00 UTC valid time and
       lead time: the network inputs
    2. ERA5 at the station's grid cell: the gridded training target the
       station target is compared against
    3. the station's own 00 UTC observations (from the HadISD cache)
    4. a 30-year (1990-2019) ERA5 day-of-year climatology at the station's
       grid cell (WeatherBench2's, cached locally on first use), which the
       anomaly correlation is computed against
    5. the ERA5 surface elevation of the station's grid cell, used to adjust
       temperature from the grid cell's height to the station's

Every lookup uses the same grid cell for a station: its nearest 0.25 degree
cell, with ties rounded up (see snap_to_grid_cell).

Writes into processed/station_finetuning/:
    station_dataset_stations.csv              station metadata, one row per station
    station_dataset_{model}.npz               the small arrays (times, observations,
                                              ERA5, climatology, elevations)
    station_dataset_{model}_neighbourhoods.npy  the forecast neighbourhoods, shape
                                              (station, time, lead, variable, 5, 5),
                                              read one station at a time in training

Run on its own with:
    uv run python data_preparation/prepare_station_data.py
"""

import os
import sys

import numpy as np
import pandas as pd
import xarray as xr
import zarr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (ALL_YEARS, DOWNLOADED_VARIABLES, FORECAST_MODELS,  # noqa: E402
                    LEAD_TIMES_HOURS, NEIGHBOURHOOD_HALF_WIDTH, TRAINING_YEARS, VARIABLES,
                    setup_directories)
from data_preparation.select_stations import load_station_observations  # noqa: E402

GRID_SPACING_DEGREES = 0.25
STANDARD_GRAVITY = 9.80665

# WeatherBench2's ERA5 climatology: 1990-2019, by hour and day of year,
# smoothed with a 61-day running window
WEATHERBENCH_CLIMATOLOGY_URL = ("gs://weatherbench2/datasets/era5-hourly-climatology/"
                                "1990-2019_6h_1440x721.zarr")


def snap_to_grid_cell(latitudes, longitudes):
    """
    Coordinates of each station's nearest grid cell.

    Every archive is read at the cell this returns. Ties (a station exactly
    halfway between cells) always round up, so the cell does not depend on
    whether an archive stores latitude north-to-south or south-to-north.

    Inputs:
        latitudes, longitudes (numpy.ndarray): station coordinates in degrees.

    Returns:
        tuple (cell_latitudes, cell_longitudes), longitudes in 0..360.
    """
    cell_latitudes = np.floor(np.asarray(latitudes, float) / GRID_SPACING_DEGREES + 0.5)
    cell_longitudes = np.floor(np.asarray(longitudes, float) / GRID_SPACING_DEGREES + 0.5)
    return (cell_latitudes * GRID_SPACING_DEGREES,
            (cell_longitudes * GRID_SPACING_DEGREES) % 360.0)


def grid_indices(archive_path, latitudes, longitudes):
    """
    Row and column of each station's grid cell in a zarr archive.

    Inputs:
        archive_path (str): a zarr store with 'latitude' and 'longitude' arrays.
        latitudes, longitudes (numpy.ndarray): station coordinates in degrees.

    Returns:
        tuple (latitude_indices, longitude_indices, number_of_latitudes,
        number_of_longitudes).
    """
    archive = zarr.open(archive_path, mode="r")
    latitude_grid = np.asarray(archive["latitude"][:])
    longitude_grid = np.asarray(archive["longitude"][:])
    cell_latitudes, cell_longitudes = snap_to_grid_cell(latitudes, longitudes)
    latitude_indices = np.abs(latitude_grid[None, :] - cell_latitudes[:, None]).argmin(axis=1)
    longitude_indices = np.abs(longitude_grid[None, :] - cell_longitudes[:, None]).argmin(axis=1)
    return latitude_indices, longitude_indices, len(latitude_grid), len(longitude_grid)


def extract_forecast_neighbourhoods(raw_directory, forecast_model, latitudes, longitudes,
                                    output_path):
    """
    Write the 5x5 forecast neighbourhood around every station to a .npy file.

    Each global field is read once and every station's window gathered from
    it, so the cost is set by the archive size, not the number of stations.
    The array is written station-first into a memory-mapped file, so each
    station's block is contiguous on disk and only one year is held in memory.

    Inputs:
        raw_directory (str): directory holding '{forecast_model}_{year}.zarr'.
        forecast_model (str): 'pangu' or 'ifs'.
        latitudes, longitudes (numpy.ndarray): station coordinates.
        output_path (str): .npy file to write.

    Returns:
        pandas.DatetimeIndex of the 00 UTC valid times along the time axis.
    """
    archive_paths = {year: os.path.join(raw_directory, f"{forecast_model}_{year}.zarr")
                     for year in ALL_YEARS}
    window_size = 2 * NEIGHBOURHOOD_HALF_WIDTH + 1

    # Every station's window of rows and columns; latitude stops at the
    # poles, longitude wraps around
    latitude_indices, longitude_indices, number_of_latitudes, number_of_longitudes = grid_indices(
        archive_paths[ALL_YEARS[0]], latitudes, longitudes)
    offsets = np.arange(-NEIGHBOURHOOD_HALF_WIDTH, NEIGHBOURHOOD_HALF_WIDTH + 1)
    window_rows = np.clip(latitude_indices[:, None] + offsets[None, :], 0, number_of_latitudes - 1)
    window_columns = (longitude_indices[:, None] + offsets[None, :]) % number_of_longitudes

    # Find the 00 UTC valid times and the wanted lead times in each year's archive
    year_positions = {}
    for year in ALL_YEARS:
        decoded = xr.open_zarr(archive_paths[year], decode_timedelta=True)
        valid_times = decoded.valid_time.to_index()
        archive_lead_hours = (decoded.prediction_timedelta.values
                              / np.timedelta64(1, "h")).astype(int)
        year_positions[year] = (
            np.where(valid_times.hour == 0)[0].tolist(),
            [int(np.where(archive_lead_hours == lead_time)[0][0])
             for lead_time in LEAD_TIMES_HOURS],
            valid_times[valid_times.hour == 0])
    verification_times = pd.DatetimeIndex(np.concatenate(
        [year_positions[year][2].values for year in ALL_YEARS]))

    neighbourhoods = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.float32,
        shape=(len(latitudes), len(verification_times), len(LEAD_TIMES_HOURS),
               len(DOWNLOADED_VARIABLES), window_size, window_size))
    first_time = 0
    for year in ALL_YEARS:
        time_positions, lead_positions, _ = year_positions[year]
        archive = zarr.open(archive_paths[year], mode="r")

        # Gather one year time-first, then write it station-first in one go
        year_values = np.empty((len(time_positions), len(LEAD_TIMES_HOURS),
                                len(DOWNLOADED_VARIABLES), len(latitudes), window_size,
                                window_size), dtype=np.float32)
        for variable_position, variable in enumerate(DOWNLOADED_VARIABLES):
            for output_position, time_position in enumerate(time_positions):
                fields = archive[variable][time_position, lead_positions]
                for lead_position in range(len(lead_positions)):
                    year_values[output_position, lead_position, variable_position] = (
                        fields[lead_position][window_rows[:, :, None], window_columns[:, None, :]])
        neighbourhoods[:, first_time:first_time + len(time_positions)] = (
            year_values.transpose(3, 0, 1, 2, 4, 5))
        first_time += len(time_positions)
        print(f"  {forecast_model} forecast neighbourhoods extracted for {year}", flush=True)
    neighbourhoods.flush()
    return verification_times


def extract_era5_at_stations(raw_directory, verification_times, latitudes, longitudes):
    """
    Read ERA5 2m temperature and 10m wind speed at each station's grid cell.

    Inputs:
        raw_directory (str): directory holding 'era5_{year}.zarr'.
        verification_times (pandas.DatetimeIndex): the 00 UTC times to read.
        latitudes, longitudes (numpy.ndarray): station coordinates.

    Returns:
        numpy.ndarray of shape (n_times, 2, n_stations), float32: temperature
        in K and wind speed (from the u and v components) in m/s.
    """
    archive_template = os.path.join(raw_directory, "era5_{year}.zarr")
    latitude_indices, longitude_indices, _, _ = grid_indices(
        archive_template.format(year=ALL_YEARS[0]), latitudes, longitudes)
    era5_values = np.full((len(verification_times), 2, len(latitudes)), np.nan, dtype=np.float32)
    time_position = {time: position for position, time in enumerate(verification_times)}

    for year in ALL_YEARS:
        archive = zarr.open(archive_template.format(year=year), mode="r")
        archive_times = xr.open_zarr(archive_template.format(year=year)).time.to_index()
        for archive_position, archive_time in enumerate(archive_times):
            output_position = time_position.get(archive_time)
            if output_position is None:
                continue  # not a 00 UTC verification time
            era5_values[output_position, 0] = (
                archive["2m_temperature"][archive_position][latitude_indices, longitude_indices])
            eastward = archive["10m_u_component_of_wind"][archive_position][latitude_indices,
                                                                             longitude_indices]
            northward = archive["10m_v_component_of_wind"][archive_position][latitude_indices,
                                                                              longitude_indices]
            era5_values[output_position, 1] = np.sqrt(eastward ** 2 + northward ** 2)
        print(f"  ERA5 extracted at stations for {year}", flush=True)
    return era5_values


def load_climatology_at_stations(raw_directory, latitudes, longitudes):
    """
    Read the 00 UTC ERA5 day-of-year climatology at each station's grid cell.

    The remote WeatherBench2 store is chunked so that even one hour costs
    about 9 GB of reads, so its 00 UTC slice is saved locally the first time
    (to raw/era5_climatology_1990-2019_00utc.zarr) and read from there.

    Inputs:
        raw_directory (str): the raw data directory.
        latitudes, longitudes (numpy.ndarray): station coordinates.

    Returns:
        tuple (temperature, wind_speed) climatologies, each of shape
        (366, n_stations); row d is day of year d + 1.
    """
    climatology_path = os.path.join(raw_directory, "era5_climatology_1990-2019_00utc.zarr")
    if not os.path.exists(climatology_path):
        print(f"  Downloading the WeatherBench2 climatology to {climatology_path}")
        climatology = xr.open_zarr(WEATHERBENCH_CLIMATOLOGY_URL, storage_options={"token": "anon"})
        midnight = climatology[VARIABLES].sel(hour=0, drop=True).drop_encoding()
        # Write beside the target and rename when done, so a partial download
        # is never mistaken for a finished one
        midnight.chunk({"dayofyear": 61}).to_zarr(f"{climatology_path}.partial", mode="w")
        os.rename(f"{climatology_path}.partial", climatology_path)

    cell_latitudes, cell_longitudes = snap_to_grid_cell(latitudes, longitudes)
    at_stations = xr.open_zarr(climatology_path).sel(
        latitude=xr.DataArray(cell_latitudes, dims="station"),
        longitude=xr.DataArray(cell_longitudes, dims="station"), method="nearest").load()
    return tuple(at_stations[variable].transpose("dayofyear", "station").values.astype(np.float64)
                 for variable in VARIABLES)


def run():
    """
    Build the station dataset of every forecast model.

    Inputs:
        None.

    Returns:
        None. Writes the files listed in the module description.
    """
    directories = setup_directories()
    station_directory = directories["station_output"]
    stations = pd.read_csv(os.path.join(station_directory, "station_sample_manifest.csv"),
                           dtype={"usaf": str, "wban": str})
    latitudes, longitudes = stations["latitude"].to_numpy(), stations["longitude"].to_numpy()
    print(f"Preparing station datasets for {len(stations)} stations")

    # Grid-cell elevation from the ERA5 surface geopotential
    static_fields = xr.open_dataset(os.path.join(directories["raw"], "era5_static.nc"),
                                    engine="netcdf4")
    cell_latitudes, cell_longitudes = snap_to_grid_cell(latitudes, longitudes)
    stations["grid_elevation_metres"] = np.asarray(
        (static_fields["z"].isel(valid_time=0) / STANDARD_GRAVITY).sel(
            latitude=xr.DataArray(cell_latitudes, dims="station"),
            longitude=xr.DataArray(cell_longitudes, dims="station"), method="nearest").values,
        dtype=float)
    temperature_climatology, wind_climatology = load_climatology_at_stations(
        directories["raw"], latitudes, longitudes)

    previous_verification_times = None
    for forecast_model in FORECAST_MODELS:
        verification_times = extract_forecast_neighbourhoods(
            directories["raw"], forecast_model, latitudes, longitudes,
            os.path.join(station_directory, f"station_dataset_{forecast_model}_neighbourhoods.npy"))

        # Observations and ERA5 do not depend on the forecast model, so they
        # are only read again if this model has different valid times
        if previous_verification_times is None or not verification_times.equals(
                previous_verification_times):
            observations = load_station_observations(
                stations, os.path.join(station_directory, "hadisd"), verification_times)
            era5_values = extract_era5_at_stations(directories["raw"], verification_times,
                                                   latitudes, longitudes)
            previous_verification_times = verification_times

        # Share of training days on which each station reported a temperature,
        # on the first model's (Pangu's) valid times. The models' valid times
        # can differ by a day at the start of the record, and one coverage
        # must serve both, since the station table is shared.
        if forecast_model == FORECAST_MODELS[0]:
            in_training_years = np.isin(verification_times.year, TRAINING_YEARS)
            stations["training_coverage"] = np.isfinite(
                observations[in_training_years, 0]).mean(axis=0)
        np.savez_compressed(
            os.path.join(station_directory, f"station_dataset_{forecast_model}.npz"),
            verification_times=verification_times.values, era5_values=era5_values,
            station_observations=observations, temperature_climatology=temperature_climatology,
            wind_climatology=wind_climatology,
            grid_elevations=stations["grid_elevation_metres"].to_numpy(),
            station_usaf=stations["usaf"].to_numpy().astype(str),
            station_wban=stations["wban"].to_numpy().astype(str))

    stations.to_csv(os.path.join(station_directory, "station_dataset_stations.csv"), index=False)


if __name__ == "__main__":
    run()
