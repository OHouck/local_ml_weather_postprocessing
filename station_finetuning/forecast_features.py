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
3. A 30-year (1990-2019) ERA5 day-of-year climatology at 00 UTC, needed to
   form the anomalies that anomaly correlation is computed from. It is the
   precomputed WeatherBench2 climatology, following Linsenmeier & Shrader's
   use of a 30-year ERA5 normal at each station's nearest grid cell.

Forecast arrays are indexed by VALID time, not initialisation time. A forecast
at lead L valid at time T was initialised at T - L. This was verified against
the raw archives: the ground-truth field stored beside a lead-24h forecast at
time T matches ERA5 at T, not at T + 24h.
"""

import os

import numpy as np
import pandas as pd
import xarray as xr
import zarr

# Standard gravity, used to convert ERA5 surface geopotential to an elevation.
STANDARD_GRAVITY = 9.80665

# Every archive in this project, including the climatology and the ERA5 static
# fields, shares this regular latitude/longitude grid.
GRID_SPACING_DEGREES = 0.25

# Half-width of the predictor window, in grid cells. 2 gives a 5x5 box, which at
# the 0.25 degree archive resolution spans about 1.25 degrees.
NEIGHBOURHOOD_HALF_WIDTH = 2

# WeatherBench2's precomputed ERA5 climatology: 1990-2019, dimensions
# hour (0/6/12/18) x dayofyear (1-366) on the 0.25 degree grid, smoothed by
# WeatherBench2 with a 61-day running window.
WEATHERBENCH_CLIMATOLOGY_URL = ("gs://weatherbench2/datasets/"
                                "era5-hourly-climatology/1990-2019_6h_1440x721.zarr")
# In the order of the variable axis of the station arrays. The climatology's
# wind speed is the mean of speed, not the speed of the mean wind components.
CLIMATOLOGY_VARIABLES = ["2m_temperature", "10m_wind_speed"]


def neighbourhood_array_path(dataset_path):
    """Return the .npy sidecar holding a dataset's forecast neighbourhoods.

    The neighbourhood array is the only large piece of the dataset, so it lives
    beside the npz rather than inside it: an npz member must be decompressed
    whole, while a plain .npy can be memory-mapped and read one station at a
    time. This module writes that file, so it owns the naming.

    Inputs:
        dataset_path (str): path to the npz written by
            prepare_station_dataset.py.

    Returns:
        str: the sibling path ending '_neighbourhoods.npy'.
    """
    return dataset_path.replace(".npz", "") + "_neighbourhoods.npy"


def load_neighbourhood_array(dataset_path):
    """Memory-map a dataset's forecast neighbourhood array.

    Inputs:
        dataset_path (str): path to the npz written by
            prepare_station_dataset.py.

    Returns:
        np.memmap: shape (n_stations, n_times, n_leads, n_variables, window,
            window), read-only.

    Raises:
        FileNotFoundError: if the sidecar is missing, which is what an older
            dataset looks like. Those stored the array inside the npz and in a
            time-leading layout, so they cannot be read here and must be
            rebuilt with prepare_station_dataset.py.
    """
    array_path = neighbourhood_array_path(dataset_path)
    if not os.path.exists(array_path):
        raise FileNotFoundError(
            f"{array_path} not found. Datasets built before the station-axis "
            f"change stored the neighbourhoods inside the npz in a different "
            f"layout; rebuild with prepare_station_dataset.py")
    return np.load(array_path, mmap_mode="r")


def snap_to_grid_cell(station_latitudes, station_longitudes):
    """Return the coordinates of each station's nearest grid cell.

    Every lookup in the pipeline goes through this one rule, so the forecast,
    ERA5, elevation and climatology are all read at the same cell. Resolving
    the cell separately in each archive does not guarantee that: a station
    exactly halfway between two cells is a tie, and argmin or xarray's
    'nearest' break ties by array order, which differs between archives that
    store latitude north-to-south and south-to-north. Here ties always round
    up in coordinate value, and longitude wraps so 359.9 maps to 0.

    Inputs:
        station_latitudes (np.ndarray): station latitudes in degrees.
        station_longitudes (np.ndarray): station longitudes in degrees, either
            convention.

    Returns:
        tuple: (cell_latitudes, cell_longitudes) as float arrays, longitudes
            in 0..360.
    """
    def round_half_up(values):
        scaled = np.asarray(values, dtype=float) / GRID_SPACING_DEGREES
        return np.floor(scaled + 0.5) * GRID_SPACING_DEGREES

    return (round_half_up(station_latitudes),
            round_half_up(station_longitudes) % 360.0)


def resolve_nearest_grid_indices(archive_path, station_latitudes, station_longitudes):
    """Map station coordinates to their cell on an archive's grid.

    Stations are first snapped to their cell with snap_to_grid_cell, so the
    index found here points at the same cell in every archive whatever order
    it stores its coordinates in.

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
    cell_latitudes, cell_longitudes = snap_to_grid_cell(station_latitudes,
                                                        station_longitudes)

    # The snapped coordinates sit exactly on the grid, so argmin finds them
    # with no ties to break.
    latitude_indices = np.abs(latitude_grid[None, :]
                              - cell_latitudes[:, None]).argmin(axis=1)
    longitude_indices = np.abs(longitude_grid[None, :]
                               - cell_longitudes[:, None]).argmin(axis=1)
    return latitude_indices, longitude_indices, latitude_grid, longitude_grid


def extract_forecast_neighbourhoods(forecast_zarr_template, years, lead_times_hours,
                                    variable_names, station_latitudes,
                                    station_longitudes, output_path):
    """Pull a square neighbourhood of forecast values around each station.

    Reads one global slab per (valid time, lead time, variable) and gathers the
    window for every station from that slab, so each slab is decompressed once
    rather than once per station. Cost is therefore set by the archive, not by
    how many stations are requested: 400 stations and 9,000 stations read the
    same bytes.

    What does scale with the station count is the output, at about 1.6 MB per
    station over a five-year study period. Two things keep that manageable. The
    station axis leads, so each station's block is contiguous and
    train_station_models.py can memory-map the file and touch only the station
    it is fitting. And when output_path is given the array is written straight
    to that memory-mapped file, so peak resident memory is one year's buffer
    rather than the whole array plus a concatenated copy of it.

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
        output_path (str): .npy file to write the array into as a memory map.
            Use neighbourhood_array_path() to name it.

    Returns:
        tuple: (verification_times, neighbourhood_values) where
            verification_times is a pd.DatetimeIndex of 00 UTC valid times and
            neighbourhood_values is a float32 array of shape
            (n_stations, n_times, n_leads, n_variables, window, window).
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

    # First pass reads only the time and lead coordinates, which is cheap, so
    # that the full output can be sized before any field data is touched.
    year_plans, times_per_year = [], []
    for year in years:
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
        # Kept as plain ints to match lead_positions: zarr 3 routes a numpy
        # integer alongside a list to basic indexing, which rejects the list.
        midnight_positions = np.where(valid_times.hour == 0)[0].tolist()
        year_plans.append((year, midnight_positions, lead_positions))
        times_per_year.append(valid_times[midnight_positions])

    verification_times = pd.DatetimeIndex(
        np.concatenate([times.values for times in times_per_year]))
    output_shape = (len(station_latitudes), len(verification_times),
                    len(lead_times_hours), len(variable_names),
                    window_size, window_size)

    neighbourhood_values = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.float32, shape=output_shape)

    first_time_position = 0
    for year, midnight_positions, lead_positions in year_plans:
        archive = zarr.open(forecast_zarr_template.format(year=year), mode="r")

        # Gather into a time-leading buffer for one year, then transpose it into
        # the station-leading output in a single bulk write. Writing each slab
        # straight to the output would scatter one small write per station
        # across the whole file, which is far slower against a memory map.
        # Every (time, lead, variable) cell is written by the loop below, so
        # there is nothing to pre-fill; np.full would memset 3 GB per year.
        year_values = np.empty(
            (len(midnight_positions), len(lead_times_hours), len(variable_names),
             len(station_latitudes), window_size, window_size),
            dtype=np.float32)

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

        last_time_position = first_time_position + len(midnight_positions)
        neighbourhood_values[:, first_time_position:last_time_position] = (
            year_values.transpose(3, 0, 1, 2, 4, 5))
        first_time_position = last_time_position
        del year_values
        print(f"  extracted forecast neighbourhoods for {year}", flush=True)

    neighbourhood_values.flush()
    return verification_times, neighbourhood_values


def extract_era5_at_stations(era5_zarr_template, years, verification_times,
                             station_latitudes, station_longitudes):
    """Read ERA5 temperature and wind speed at each station's nearest grid cell.

    ERA5 here is the alternative training target (the one the main paper
    uses). Wind speed is derived from the u and v components at native resolution.

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


def download_weatherbench_climatology(output_path):
    """Cache the 00 UTC slice of the WeatherBench2 ERA5 climatology locally.

    The remote store is chunked three hours by three days of the full globe, so
    even one hour costs about 9 GB of reads. Saving just the 00 UTC fields of
    the two variables (about 1.5 GB) once means every later station lookup,
    for any station set, is local.

    Inputs:
        output_path (str): local zarr to write, e.g.
            '<raw>/era5_climatology_1990-2019_00utc.zarr'. Left untouched if
            it already exists.

    Returns:
        str: output_path.
    """
    if os.path.exists(output_path):
        return output_path
    print(f"downloading WeatherBench2 climatology to {output_path}", flush=True)
    climatology = xr.open_zarr(WEATHERBENCH_CLIMATOLOGY_URL,
                               storage_options={"token": "anon"})
    # Hour 0 only: every verification time in this analysis is 00 UTC. The
    # remote store's zarr-v2 chunking and compressor cannot be written by the
    # local zarr-v3 writer, so the local store takes default encoding.
    at_hour = climatology[CLIMATOLOGY_VARIABLES].sel(
        hour=0, drop=True).drop_encoding()
    # Write beside the target and rename at the end, so an interrupted
    # download is never mistaken for a finished one by the check above.
    # 61-day chunks just split the 366 days into six even blocks.
    partial_path = f"{output_path}.partial"
    at_hour.chunk({"dayofyear": 61}).to_zarr(partial_path, mode="w")
    os.rename(partial_path, output_path)
    return output_path


def load_station_climatology(climatology_path, station_latitudes,
                             station_longitudes):
    """Read the day-of-year climatology at each station's grid cell.

    Anomaly correlation requires removing the seasonal cycle, otherwise stations
    with a large, trivially predictable annual swing score artificially well.
    Following Linsenmeier & Shrader, the climatology is a 30-year ERA5 normal
    at the station's nearest grid cell, applied to both forecast and
    observation. The cell comes from snap_to_grid_cell, the same rule every
    other lookup uses.

    Inputs:
        climatology_path (str): local zarr from download_weatherbench_climatology.
        station_latitudes (np.ndarray): station latitudes in degrees.
        station_longitudes (np.ndarray): station longitudes in degrees.

    Returns:
        tuple: (temperature, wind_speed) climatologies, each a float array of
            shape (366, n_stations); row d is day-of-year d + 1, matching the
            0-based day-of-year index used elsewhere.
    """
    cell_latitudes, cell_longitudes = snap_to_grid_cell(station_latitudes,
                                                        station_longitudes)
    at_stations = xr.open_zarr(climatology_path).sel(
        latitude=xr.DataArray(cell_latitudes, dims="station"),
        longitude=xr.DataArray(cell_longitudes, dims="station"),
        method="nearest").load()
    return tuple(at_stations[variable].transpose("dayofyear", "station")
                 .values.astype(np.float64)
                 for variable in CLIMATOLOGY_VARIABLES)


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

    # Select all stations in one vectorised call, at the same cells every other
    # lookup uses. The snapped coordinates are exact grid points, so 'nearest'
    # only absorbs floating-point differences in the stored coordinates.
    cell_latitudes, cell_longitudes = snap_to_grid_cell(station_latitudes,
                                                        station_longitudes)
    grid_elevations = elevation_field.sel(
        latitude=xr.DataArray(cell_latitudes, dims="station"),
        longitude=xr.DataArray(cell_longitudes, dims="station"),
        method="nearest").values

    return np.asarray(grid_elevations, dtype=float)
