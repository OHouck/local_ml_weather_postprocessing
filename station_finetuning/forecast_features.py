"""Extraction of forecast predictors and reference fields at station locations.

Three things are pulled out of the global archives for each station:

1. A 5x5 grid-cell neighbourhood of the raw Pangu forecast, which is what the
   post-processing network sees as input. A window sweep (see README.md) found
   that roughly one degree is the useful extent; widening it past a few degrees
   trades away local resolution and costs skill.
2. ERA5 at the nearest grid cell, which serves as the comparison ground truth.
   Training against ERA5 instead of station observations is the specification
   the main paper uses, and reproducing it here is what isolates the effect of
   the training target.
3. A smoothed day-of-year climatology, needed to form the anomalies that
   anomaly correlation is computed from.

Forecast arrays are indexed by VALID time, not initialisation time. A forecast
at lead L valid at time T was initialised at T - L. This was verified against
the raw archives: the ground-truth field stored beside a lead-24h forecast at
time T matches ERA5 at T, not at T + 24h.
"""

import numpy as np
import pandas as pd
import xarray as xr
import zarr

# Standard gravity, used to convert ERA5 surface geopotential to an elevation.
STANDARD_GRAVITY = 9.80665

# Half-width of the predictor window, in grid cells. 2 gives a 5x5 box, which at
# the 0.25 degree archive resolution spans about 1.25 degrees.
NEIGHBOURHOOD_HALF_WIDTH = 2

# Width of the centred day-of-year smoothing window used for the climatology.
CLIMATOLOGY_SMOOTHING_DAYS = 31


def resolve_nearest_grid_indices(archive_path, station_latitudes, station_longitudes):
    """Map station coordinates to their nearest cell on an archive's grid.

    Every archive in this project shares the same 0.25 degree latitude/longitude
    grid, but the lookup is done against the file actually being read so that a
    differently gridded source would still resolve correctly.

    Inputs:
        archive_path (str): path to a zarr store holding 'latitude' and
            'longitude' coordinate arrays.
        station_latitudes (np.ndarray): station latitudes in degrees.
        station_longitudes (np.ndarray): station longitudes in 0..360.

    Returns:
        tuple: (latitude_indices, longitude_indices, latitude_grid,
            longitude_grid). The first two are integer arrays, one entry per
            station; the last two are the archive's coordinate arrays, returned
            so callers can size window offsets without reopening the store.
    """
    archive = zarr.open(archive_path, mode="r")
    latitude_grid = np.asarray(archive["latitude"][:])
    longitude_grid = np.asarray(archive["longitude"][:])

    # argmin over the absolute difference is exact for a regular grid and avoids
    # a per-station xarray .sel call, which is far slower for hundreds of points.
    latitude_indices = np.array(
        [np.abs(latitude_grid - value).argmin() for value in station_latitudes])
    longitude_indices = np.array(
        [np.abs(longitude_grid - value).argmin() for value in station_longitudes])
    return latitude_indices, longitude_indices, latitude_grid, longitude_grid


def extract_forecast_neighbourhoods(forecast_zarr_template, years, lead_times_hours,
                                    variable_names, station_latitudes,
                                    station_longitudes):
    """Pull a square neighbourhood of forecast values around each station.

    Reads one global slab per (valid time, lead time, variable) and gathers the
    window for every station from that slab, so each slab is decompressed once
    rather than once per station.

    Inputs:
        forecast_zarr_template (str): path template containing '{year}', e.g.
            '/path/pangu_{year}.zarr'.
        years (list[int]): calendar years to read.
        lead_times_hours (list[int]): forecast lead times to keep, e.g.
            [24, 120, 216]. Must exist in the archive's prediction_timedelta.
        variable_names (list[str]): archive variable names to read, e.g.
            ['2m_temperature', '10m_u_component_of_wind',
             '10m_v_component_of_wind'].
        station_latitudes (np.ndarray): station latitudes on the archive grid.
        station_longitudes (np.ndarray): station longitudes on the archive grid,
            in 0..360.

    Returns:
        tuple: (verification_times, neighbourhood_values) where
            verification_times is a pd.DatetimeIndex of 00 UTC valid times and
            neighbourhood_values is a float32 array of shape
            (n_times, n_leads, n_variables, n_stations, window, window).
    """
    window_size = 2 * NEIGHBOURHOOD_HALF_WIDTH + 1

    # Resolve each station to its nearest grid cell once, using the first year's
    # coordinate arrays. The grid is identical across years.
    (latitude_indices, longitude_indices, latitude_grid,
     longitude_grid) = resolve_nearest_grid_indices(
        forecast_zarr_template.format(year=years[0]),
        station_latitudes, station_longitudes)

    # Pre-compute the row and column indices of every station's window. Latitude
    # is clipped at the poles; longitude wraps around the date line.
    offsets = np.arange(-NEIGHBOURHOOD_HALF_WIDTH, NEIGHBOURHOOD_HALF_WIDTH + 1)
    window_rows = np.clip(latitude_indices[:, None] + offsets[None, :],
                          0, len(latitude_grid) - 1)
    window_columns = (longitude_indices[:, None] + offsets[None, :]) % len(longitude_grid)

    times_per_year, values_per_year = [], []
    for year in years:
        archive = zarr.open(forecast_zarr_template.format(year=year), mode="r")
        # Open with xarray purely to CF-decode the time and lead coordinates;
        # reading the raw integers would misinterpret "hours since ..." as
        # nanoseconds.
        decoded = xr.open_zarr(forecast_zarr_template.format(year=year),
                               decode_timedelta=True)
        valid_times = decoded.valid_time.to_index()
        archive_lead_hours = (
            decoded.prediction_timedelta.values / np.timedelta64(1, "h")).astype(int)
        lead_positions = [int(np.where(archive_lead_hours == lead)[0][0])
                          for lead in lead_times_hours]
        midnight_positions = np.where(valid_times.hour == 0)[0]

        year_values = np.full(
            (len(midnight_positions), len(lead_times_hours), len(variable_names),
             len(station_latitudes), window_size, window_size),
            np.nan, dtype=np.float32)

        for variable_position, variable_name in enumerate(variable_names):
            variable_array = archive[variable_name]
            for output_position, time_position in enumerate(midnight_positions):
                # Read every wanted lead in one call. Requesting them one at a
                # time can decompress the same chunk repeatedly when a chunk
                # spans the lead axis.
                slabs_for_all_leads = variable_array[time_position, lead_positions]
                for lead_position in range(len(lead_positions)):
                    global_slab = slabs_for_all_leads[lead_position]
                    # Advanced indexing gathers all stations' windows at once.
                    year_values[output_position, lead_position, variable_position] = (
                        global_slab[window_rows[:, :, None], window_columns[:, None, :]])

        times_per_year.append(valid_times[midnight_positions])
        values_per_year.append(year_values)
        print(f"  extracted forecast neighbourhoods for {year}", flush=True)

    verification_times = pd.DatetimeIndex(
        np.concatenate([times.values for times in times_per_year]))
    neighbourhood_values = np.concatenate(values_per_year, axis=0)
    return verification_times, neighbourhood_values


def extract_era5_at_stations(era5_zarr_template, years, verification_times,
                             station_latitudes, station_longitudes):
    """Read ERA5 temperature and wind speed at each station's nearest grid cell.

    ERA5 plays two roles here: it is the alternative training target (the one
    the main paper uses) and it supplies the climatology used to form anomalies.
    Wind speed is derived from the u and v components at native resolution.

    Inputs:
        era5_zarr_template (str): path template containing '{year}'.
        years (list[int]): calendar years to read.
        verification_times (pd.DatetimeIndex): the 00 UTC times to align onto,
            as returned by extract_forecast_neighbourhoods.
        station_latitudes (np.ndarray): station latitudes on the archive grid.
        station_longitudes (np.ndarray): station longitudes in 0..360.

    Returns:
        np.ndarray: shape (n_times, 2, n_stations), float32, where the middle
            axis is [2m_temperature in K, 10m_wind_speed in m/s].
    """
    latitude_indices, longitude_indices, _, _ = resolve_nearest_grid_indices(
        era5_zarr_template.format(year=years[0]),
        station_latitudes, station_longitudes)

    era5_values = np.full(
        (len(verification_times), 2, len(station_latitudes)), np.nan, dtype=np.float32)

    # Map each verification time to its row in the ERA5 archive for that year.
    time_position_lookup = {time: position
                            for position, time in enumerate(verification_times)}

    for year in years:
        archive = zarr.open(era5_zarr_template.format(year=year), mode="r")
        archive_times = xr.open_zarr(era5_zarr_template.format(year=year)).time.to_index()
        temperature_array = archive["2m_temperature"]
        eastward_wind_array = archive["10m_u_component_of_wind"]
        northward_wind_array = archive["10m_v_component_of_wind"]

        for archive_position, archive_time in enumerate(archive_times):
            output_position = time_position_lookup.get(archive_time)
            if output_position is None:
                continue  # not a 00 UTC verification time

            era5_values[output_position, 0] = (
                temperature_array[archive_position][latitude_indices, longitude_indices])
            eastward = eastward_wind_array[archive_position][
                latitude_indices, longitude_indices]
            northward = northward_wind_array[archive_position][
                latitude_indices, longitude_indices]
            era5_values[output_position, 1] = np.sqrt(eastward ** 2 + northward ** 2)

        print(f"  extracted ERA5 station values for {year}", flush=True)

    return era5_values


def build_day_of_year_climatology(era5_values, verification_times, climatology_years):
    """Build a smoothed day-of-year climatology at each station.

    Anomaly correlation requires removing the seasonal cycle, otherwise stations
    with a large, trivially predictable annual swing score artificially well.
    Following Linsenmeier & Shrader, the climatology comes from ERA5 at the
    station's nearest grid cell and is applied to both forecast and observation.

    Inputs:
        era5_values (np.ndarray): shape (n_times, 2, n_stations), as returned by
            extract_era5_at_stations.
        verification_times (pd.DatetimeIndex): times matching axis 0.
        climatology_years (list[int]): years to average over. These should be
            the training years only, so the test year does not inform the
            climatology.

    Returns:
        dict: maps variable position (0 for temperature, 1 for wind speed) to a
            float array of shape (366, n_stations) giving the smoothed
            climatological value for each day-of-year.
    """
    in_climatology_period = np.isin(verification_times.year, climatology_years)
    day_of_year_index = verification_times.dayofyear.values - 1
    number_of_stations = era5_values.shape[2]

    climatology_by_variable = {}
    for variable_position in range(2):
        # Raw per-day means first: with four years of data each day-of-year has
        # only a handful of samples, hence the smoothing that follows.
        daily_means = np.full((366, number_of_stations), np.nan)
        for day in range(366):
            matching_days = in_climatology_period & (day_of_year_index == day)
            if matching_days.any():
                daily_means[day] = np.nanmean(
                    era5_values[matching_days, variable_position], axis=0)

        # Smooth with a centred window that wraps around the year boundary, so
        # late December and early January are treated as adjacent.
        half_window = CLIMATOLOGY_SMOOTHING_DAYS // 2
        wrapped = np.concatenate(
            [daily_means[-half_window:], daily_means, daily_means[:half_window]], axis=0)
        smoothing_kernel = np.ones(CLIMATOLOGY_SMOOTHING_DAYS) / CLIMATOLOGY_SMOOTHING_DAYS
        climatology_by_variable[variable_position] = np.apply_along_axis(
            lambda column: np.convolve(column, smoothing_kernel, mode="valid"),
            0, wrapped)

    return climatology_by_variable


def read_station_grid_elevation(era5_static_path, station_latitudes,
                                station_longitudes):
    """Look up the model's surface elevation at each station's grid cell.

    The grid cell average elevation usually differs from the station's own
    elevation, which is the main reason a coarse forecast is biased at a point.
    Trotta et al. (2025) correct this with a dry adiabatic lapse rate before
    verifying temperature; the same correction is applied in train_station_models.

    Inputs:
        era5_static_path (str): path to era5_static.nc, holding surface
            geopotential 'z'.
        station_latitudes (np.ndarray): station latitudes in degrees.
        station_longitudes (np.ndarray): station longitudes in 0..360.

    Returns:
        np.ndarray: grid cell elevation in metres, one value per station.
    """
    static_fields = xr.open_dataset(era5_static_path, engine="netcdf4")
    elevation_field = static_fields["z"].isel(valid_time=0) / STANDARD_GRAVITY

    # Select all stations in one vectorised call. Looping with .sel per station
    # costs hundreds of separate xarray lookups for no benefit.
    latitude_selector = xr.DataArray(np.asarray(station_latitudes), dims="station")
    longitude_selector = xr.DataArray(np.asarray(station_longitudes), dims="station")
    grid_elevations = elevation_field.sel(
        latitude=latitude_selector, longitude=longitude_selector,
        method="nearest").values

    return np.asarray(grid_elevations, dtype=float)
