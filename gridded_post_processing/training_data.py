"""
Load one 6x6 degree patch of forecasts and their gridded truth as training arrays.

Forecasts are read from raw/{model}_{year}.zarr (indexed by valid time, with
one entry per lead time) and the truth from raw/{era5|hres_t0}_{year}.zarr,
both written by the data_preparation scripts. Only 00 UTC valid times are
used.

Every (valid day, lead time) pair is one sample. A sample's forecast and truth
are the patch flattened to a vector of n_latitudes x n_longitudes values.
"""

import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import GRIDDED_TRUTH_SOURCE, LEAD_TIMES_HOURS  # noqa: E402

# Archive variables needed to build each corrected variable
ARCHIVE_VARIABLES = {
    "2m_temperature": ["2m_temperature"],
    "10m_wind_speed": ["10m_u_component_of_wind", "10m_v_component_of_wind"],
}


def open_yearly_archives(raw_directory, source, years, variable, latitudes, longitudes):
    """
    Open the yearly zarr files of one data source, cut out a patch, and join them.

    Inputs:
        raw_directory (str): directory holding '{source}_{year}.zarr'.
        source (str): 'pangu', 'ifs', 'era5' or 'hres_t0'.
        years (list of int): years to read.
        variable (str): the corrected variable; only the archive variables it
            is built from are read.
        latitudes, longitudes (numpy.ndarray): grid cells of the patch.

    Returns:
        xarray.Dataset with a 'time' dimension (the valid time), sorted by time
        and latitude, with each time appearing once. Data is not yet loaded.
    """
    yearly_datasets = []
    for year in years:
        dataset = xr.open_zarr(os.path.join(raw_directory, f"{source}_{year}.zarr"), chunks="auto",
                               decode_timedelta=True)
        if "valid_time" in dataset.dims:
            dataset = dataset.rename({"valid_time": "time"})
        dataset = dataset[ARCHIVE_VARIABLES[variable]]
        yearly_datasets.append(dataset.sel(latitude=latitudes, longitude=longitudes)
                               .sortby("latitude"))
    combined = xr.concat(yearly_datasets, dim="time", combine_attrs="override").sortby("time")

    # Neighbouring years can share a time step; keep its first occurrence
    _, first_occurrences = np.unique(combined.time.values, return_index=True)
    return combined.isel(time=sorted(first_occurrences))


def load_patch_data(directories, forecast_model, variable, latitudes, longitudes, start_date,
                    end_date):
    """
    Load a patch's forecasts and truth between two dates as flat training arrays.

    Wind speed is computed from the u and v wind components. Samples with any
    missing value in the forecast or the truth are dropped.

    Inputs:
        directories (dict): output of common.setup_directories().
        forecast_model (str): 'pangu' or 'ifs'.
        variable (str): '2m_temperature' or '10m_wind_speed'.
        latitudes, longitudes (numpy.ndarray): grid cells of the patch.
        start_date, end_date (str): first and last valid day, 'YYYY-MM-DD'.

    Returns:
        dict with
            'forecast'             (n_samples, n_cells) raw forecast
            'truth'                (n_samples, n_cells) gridded truth
            'lead_time_index'      (n_samples,) position of the lead time in
                                   common.LEAD_TIMES_HOURS
            'day_of_year_features' (n_samples, 2) sine and cosine of the day of year
            'valid_times'          (n_samples,) valid time of each sample
            'latitudes', 'longitudes' the patch grid (latitude ascending)
    """
    requested_days = pd.date_range(start=start_date, end=end_date, freq="D").to_numpy()
    years = list(range(int(start_date[:4]), int(end_date[:4]) + 1))
    lead_times = [np.timedelta64(lead_time, "h") for lead_time in LEAD_TIMES_HOURS]

    forecast_dataset = open_yearly_archives(directories["raw"], forecast_model, years, variable,
                                            latitudes, longitudes)
    truth_dataset = open_yearly_archives(directories["raw"], GRIDDED_TRUTH_SOURCE[forecast_model],
                                         years, variable, latitudes, longitudes)

    # Keep the requested lead times and the 00 UTC days present in both
    common_times = np.intersect1d(forecast_dataset.time.values, truth_dataset.time.values)
    common_times = np.intersect1d(common_times, requested_days)
    forecast_dataset = forecast_dataset.sel(prediction_timedelta=lead_times,
                                            time=common_times).load()
    truth_dataset = truth_dataset.sel(time=common_times).load()

    # Build wind speed from its components
    if variable == "10m_wind_speed":
        for dataset in (forecast_dataset, truth_dataset):
            dataset["10m_wind_speed"] = np.sqrt(dataset["10m_u_component_of_wind"] ** 2
                                                + dataset["10m_v_component_of_wind"] ** 2)

    # One sample per (valid day, lead time), lead time varying fastest; each
    # day's truth is repeated for every lead time
    number_of_lead_times = len(lead_times)
    forecast = (forecast_dataset[variable]
                .transpose("time", "prediction_timedelta", "latitude", "longitude").values
                .reshape(len(common_times) * number_of_lead_times, -1))
    truth = np.repeat(truth_dataset[variable].transpose("time", "latitude", "longitude").values
                      .reshape(len(common_times), -1), number_of_lead_times, axis=0)
    lead_time_index = np.tile(np.arange(number_of_lead_times), len(common_times))
    valid_times = np.repeat(common_times, number_of_lead_times)

    # Sine and cosine of the day of year, so the seasonal cycle has no jump at new year
    day_of_year_angle = 2 * np.pi * pd.DatetimeIndex(valid_times).dayofyear.to_numpy() / 365.0
    day_of_year_features = np.stack([np.sin(day_of_year_angle), np.cos(day_of_year_angle)],
                                    axis=1)

    # Drop samples with any missing value
    complete = ~(np.isnan(forecast).any(axis=1) | np.isnan(truth).any(axis=1))
    return {
        "forecast": forecast[complete],
        "truth": truth[complete],
        "lead_time_index": lead_time_index[complete],
        "day_of_year_features": day_of_year_features[complete],
        "valid_times": valid_times[complete],
        "latitudes": forecast_dataset.latitude.values,
        "longitudes": forecast_dataset.longitude.values,
    }
