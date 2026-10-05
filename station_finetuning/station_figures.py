"""Paper figures for the station-level post-processing analysis.

Every figure is built from the sampled run (run_station_sample.sh), plus the
main paper's gridded outputs sampled at the same stations for the IFS
numbers:

    Fig 1  one figure per variable: a map of raw Pangu ACC at every station
           (1 day lead) with the high-income minus LMIC ACC gap at 1, 5 and
           9 days in a narrow panel right of the map, both verified against
           station observations. 2m
           temperature is the main-text figure; 10m wind speed and the IFS
           versions go in the appendix
    Fig 2  maps of the per-station % change in ACC from station-trained
           post-processing, 1 day lead, one figure per variable (2m
           temperature main text, 10m wind speed appendix)
    Fig 3  the ACC gap between high-income and LMIC stations for raw Pangu,
           ERA5-trained and station-trained post-processing, at 1, 5 and 9
           days ('_with_mos' adds a linear MOS bar)
    Fig 4  the % change in RMSE split into bias, information error and noise
           error (Bonavita & Geer), for networks trained on the ERA5 target
           and on the station target, both verified against stations
           ('_with_mos' adds a linear MOS column)

See SOURCE_STATUS below for which inputs are current and which are outdated.

Run with::

    uv run python station_finetuning/station_figures.py
    uv run python station_finetuning/station_figures.py --figures fig1 fig2
"""

import argparse
import glob
import os
import sys

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# The shared project helpers live one directory up and supply the machine
# specific data paths that the rest of the repository uses.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helper_funcs import CONTINENTS, setup_directories  # noqa: E402

# Income group names, colours, the paper's figure directory and its matplotlib
# defaults all come from the paper's figure module (importing it applies those
# rcParams), so the station figures and the gridded figures agree.
from finetuning.erl_figures import (INCOME_GROUP_COLORS,  # noqa: E402
                                    INCOME_GROUP_ORDER, LEAD_TIMES,
                                    _erl_output_dir, _model_kwargs)
from finetuning.figures_finetuning import filter_patch_zarr_files  # noqa: E402

from forecast_features import snap_to_grid_cell  # noqa: E402
from station_observations import read_station_metadata  # noqa: E402
from verification import (DRY_ADIABATIC_LAPSE_RATE,  # noqa: E402
                          MINIMUM_SAMPLES_TO_SCORE, decompose_forecast_error)

# ---------------------------------------------------------------------------
# DATA STATUS (checked 2026-09-22)
# ---------------------------------------------------------------------------
# Every verification row carries a 'source' tag, and each figure's footnote is
# built from the sources it draws on. None means current.
SOURCE_STATUS = {
    # station_results_sample.csv from run_station_sample.sh. Pangu only;
    # networks trained on the station target and on the ERA5 target, every
    # method scored against station observations.
    "station_pipeline": None,
    # The main paper's gridded outputs in finetuning_output/{pangu,ifs}, sampled
    # at each station's nearest pixel. The figures use them for the IFS
    # numbers.
    "gridded_paper": ("OUTDATED: IFS values come from the main paper's "
                      "gridded outputs sampled at station pixels; re-run "
                      "before submission."),
}

GRIDDED_CACHE_NAME = "station_gridded_verification.csv"

# In the order of the variable axis of the station arrays.
VARIABLE_NAMES = ["2m_temperature", "10m_wind_speed"]
VARIABLE_LABELS = {"2m_temperature": "2m temperature",
                   "10m_wind_speed": "10m wind speed"}
VARIABLE_TITLES = {"2m_temperature": "2m Temperature",
                   "10m_wind_speed": "10m Wind Speed"}
VARIABLE_SHORT_NAMES = {"2m_temperature": "t2m", "10m_wind_speed": "wind"}
LEAD_LABELS = {24: "1 day", 120: "5 day", 216: "9 day"}
MODEL_LABELS = {"pangu": "Pangu-Weather", "ifs": "IFS HRES"}
INCOME_SHORT_LABELS = {"Low and middle income": "LMIC", "High income": "HI"}
STATION_KEYS = ["usaf", "wban"]

# Where each model's raw-forecast score against stations is read from. The
# raw IFS forecast has not been through the station pipeline, so it comes
# from the gridded outputs.
RAW_STATION_SOURCE = {"pangu": "station_pipeline", "ifs": "gridded_paper"}

# The station pipeline's Pangu rows, verified against stations: the data for
# Figs 2-4. Its two post-processing targets, method name -> panel label.
STATION_PIPELINE_PANGU = ("station_pipeline", "pangu", "station")
STATION_TRAINED_METHOD = "mlp_station_tuned"
TRAINING_TARGETS = {"mlp_era5_tuned": "Trained on ERA5",
                    "mlp_station_tuned": "Trained on stations"}
# The linear MOS reference (station observation regressed on the forecast, per
# lead), drawn alongside the networks in the '_with_mos' versions of Figs 3-4.
LINEAR_MOS = {"linear_mos": "Linear MOS"}

# RMSE^2 = bias^2 + information_error^2 + noise_error^2, so these three
# columns split any change in RMSE exactly (see _rmse_change_components).
ERROR_COMPONENT_COLORS = {"bias": "#9e9e9e",
                          "information_error": "#7b3294",
                          "noise_error": "#e08214"}
ERROR_COMPONENT_LABELS = {"bias": "Bias",
                          "information_error": "Information error",
                          "noise_error": "Noise error"}

# Maroon for the income-gap bar charts (Figs 1 and 3).
MAROON = "#800000"
GAP_AXIS_LABEL = "ACC gap (high income − LMIC)"
LIGHT_MAROON = "#d9a3a3"

# Station maps show marker shape by income group, colour by the mapped value.
INCOME_MARKERS = {"High income": "o", "Low and middle income": "^"}
MAP_EXTENT = [-180, 180, -60, 85]


def _to_plot_longitude(longitudes):
    """Wrap longitudes from the archives' 0..360 convention into -180..180.

    Inputs:
        longitudes (float or np.ndarray or pd.Series): longitudes in degrees,
            in either convention.

    Returns:
        The same type, with every value in -180..180.
    """
    return ((longitudes + 180.0) % 360.0) - 180.0


def _draw_basemap(axis):
    """Draw the shared land/ocean background with unlabelled gridlines.

    Inputs:
        axis (cartopy GeoAxes): a PlateCarree map axis to draw on.

    Returns:
        None. The features are added to axis in place, and the extent is set
            to MAP_EXTENT (Antarctica has no stations).
    """
    axis.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    axis.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
    axis.add_feature(cfeature.LAND, facecolor="#eeeeee", zorder=0)
    axis.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="#555555",
                     zorder=1)
    axis.add_feature(cfeature.BORDERS, linewidth=0.25, edgecolor="#999999",
                     zorder=1)
    axis.gridlines(draw_labels=False, linewidth=0.3, color="#bbbbbb",
                   linestyle="--", alpha=0.6)


# ---------------------------------------------------------------------------
# Data for the paper figures
# ---------------------------------------------------------------------------

def load_sample_chunks(work_directory, chunk_prefix="station_sample"):
    """Read every continent chunk of the sampled run into station-axis arrays.

    run_station_sample.sh deletes each chunk's forecast neighbourhoods after
    training, but keeps the npz holding the station observations and the ERA5
    climatology. Those are all that is needed to verify another forecast at the
    same stations.

    Inputs:
        work_directory (str): '<processed>/station_finetuning'.
        chunk_prefix (str): the output prefix the chunks were prepared with;
            chunks are '<chunk_prefix>_<continent>.npz' plus a matching
            '_stations.csv'.

    Returns:
        dict with keys:
            'stations' (pd.DataFrame): every chunk's metadata csv, concatenated
                in station-axis order, plus 'plot_longitude'.
            'verification_times' (pd.DatetimeIndex): valid times, shared by
                every chunk.
            'observations' (np.ndarray): (time, variable, station) station
                observations, variables in VARIABLE_NAMES order.
            'climatology' (np.ndarray): (day of year, variable, station) ERA5
                climatology the station pipeline computes anomalies against.

    Raises:
        FileNotFoundError: if no chunk is found.
        ValueError: if chunks disagree on their verification times.
    """
    metadata_paths = sorted(glob.glob(
        os.path.join(work_directory, f"{chunk_prefix}_*_stations.csv")))
    if not metadata_paths:
        raise FileNotFoundError(
            f"no '{chunk_prefix}_*_stations.csv' in {work_directory}; run "
            f"run_station_sample.sh first")

    station_frames, observations, climatologies = [], [], []
    verification_times = None
    for metadata_path in metadata_paths:
        dataset = np.load(metadata_path.replace("_stations.csv", ".npz"))
        chunk_times = pd.DatetimeIndex(dataset["verification_times"])
        if verification_times is None:
            verification_times = chunk_times
        elif not verification_times.equals(chunk_times):
            raise ValueError(f"{metadata_path} has different verification "
                             f"times from the other chunks")
        station_frames.append(read_station_metadata(metadata_path))
        observations.append(dataset["station_observations"])
        climatologies.append(np.stack([dataset["temperature_climatology"],
                                       dataset["wind_climatology"]], axis=1))

    stations = pd.concat(station_frames, ignore_index=True)
    stations["plot_longitude"] = _to_plot_longitude(stations["longitude"])
    return {"stations": stations,
            "verification_times": verification_times,
            "observations": np.concatenate(observations, axis=2),
            "climatology": np.concatenate(climatologies, axis=2)}


def load_station_pipeline_results(results_path, stations):
    """Read the combined station-pipeline results and tag where they came from.

    Inputs:
        results_path (str): station_results_sample.csv written by
            'run_station_sample.sh combine'.
        stations (pd.DataFrame): station metadata from load_sample_chunks,
            used to attach longitude, which the results csv does not carry.

    Returns:
        pd.DataFrame: one row per station, variable, method and lead, with the
            verification metrics plus 'longitude', 'plot_longitude' and the
            tags source='station_pipeline', forecast_model='pangu',
            truth='station'.
    """
    results = read_station_metadata(results_path).merge(
        stations[STATION_KEYS + ["longitude", "plot_longitude"]],
        on=STATION_KEYS, how="left")
    return results.assign(source="station_pipeline", forecast_model="pangu",
                          truth="station")


def _nearest_pixels(grid_latitudes, grid_longitudes, station_latitudes,
                    station_longitudes):
    """Find each station's grid cell on a patch grid.

    Stations are snapped with forecast_features.snap_to_grid_cell, the rule
    the station pipeline uses, so a gridded patch is read at the same cell as
    the station's forecast, ERA5 and climatology.

    Inputs:
        grid_latitudes (np.ndarray): the patch's latitude coordinate.
        grid_longitudes (np.ndarray): the patch's longitude coordinate, 0..360.
        station_latitudes (np.ndarray): station latitudes.
        station_longitudes (np.ndarray): station longitudes, 0..360.

    Returns:
        tuple: (latitude index, longitude index, inside), where the indices
            are per-station positions on the patch axes and inside (bool) is
            True for stations whose cell is one of the patch's pixels.
    """
    cell_latitudes, cell_longitudes = snap_to_grid_cell(station_latitudes,
                                                        station_longitudes)
    latitude_distance = np.abs(cell_latitudes[:, None]
                               - grid_latitudes[None, :])
    longitude_distance = np.abs(_to_plot_longitude(
        cell_longitudes[:, None] - grid_longitudes[None, :]))
    # The snapped cells are exact grid points; the tolerance only absorbs the
    # float32 storage of the patch coordinates.
    inside = ((latitude_distance.min(axis=1) < 1e-3)
              & (longitude_distance.min(axis=1) < 1e-3))
    return (latitude_distance.argmin(axis=1), longitude_distance.argmin(axis=1),
            inside)


def _score_gridded_patch(patch, variable_position, forecast_model, chunks,
                         station_positions, latitude_index, longitude_index):
    """Verify one gridded patch's raw and corrected forecasts at its stations.

    Every station is scored twice on an identical sample of days: against the
    patch's own reanalysis truth (ERA5 for Pangu, the IFS t=0 analysis for
    IFS) and against the station observation. Temperature verified against a
    station gets the same lapse-rate elevation adjustment as in the station
    pipeline, which moves bias and RMSE but never ACC.

    Inputs:
        patch (xr.Dataset): one gridded output zarr from the main paper.
        variable_position (int): index of the variable in VARIABLE_NAMES.
        forecast_model (str): 'pangu' or 'ifs'.
        chunks (dict): output of load_sample_chunks.
        station_positions (np.ndarray): station-axis positions inside patch.
        latitude_index (np.ndarray): each of those stations' pixel row.
        longitude_index (np.ndarray): each of those stations' pixel column.

    Returns:
        list[dict]: one metrics row per station, lead, truth and method
            ('raw' or 'mlp_reanalysis_target').
    """
    variable = VARIABLE_NAMES[variable_position]
    kinds = ("original", "corrected", "ground_truth")

    # Gridded outputs are indexed by valid time, like the station arrays; keep
    # only the times the station record covers.
    time_positions = chunks["verification_times"].get_indexer(patch.time.values)
    in_station_record = time_positions >= 0
    time_positions = time_positions[in_station_record]
    day_of_year = chunks["verification_times"].dayofyear.values[time_positions] - 1

    pixels = patch[[f"{variable}_{kind}_lt{lead}h" for lead in LEAD_TIMES
                    for kind in kinds]].isel(
        time=in_station_record,
        latitude=xr.DataArray(latitude_index, dims="station"),
        longitude=xr.DataArray(longitude_index, dims="station")).load()

    rows = []
    for column, station_position in enumerate(station_positions):
        station = chunks["stations"].iloc[station_position]
        station_truth = chunks["observations"][time_positions, variable_position,
                                               station_position]
        climatology = chunks["climatology"][day_of_year, variable_position,
                                            station_position]
        elevation_offset = 0.0
        if variable == "2m_temperature":
            elevation_offset = DRY_ADIABATIC_LAPSE_RATE * (
                station["grid_elevation_metres"] - station["elevation_metres"])

        for lead in LEAD_TIMES:
            raw, corrected, reanalysis = (
                pixels[f"{variable}_{kind}_lt{lead}h"].values[:, column]
                for kind in kinds)
            usable = (np.isfinite(raw) & np.isfinite(corrected)
                      & np.isfinite(reanalysis) & np.isfinite(station_truth))
            if usable.sum() < MINIMUM_SAMPLES_TO_SCORE:
                continue
            for truth_name, truth_values, offset in (
                    ("reanalysis", reanalysis, 0.0),
                    ("station", station_truth, elevation_offset)):
                for method_name, forecast in (("raw", raw),
                                              ("mlp_reanalysis_target",
                                               corrected)):
                    metrics = decompose_forecast_error(
                        forecast[usable] + offset, truth_values[usable],
                        climatology[usable])
                    if metrics is None:
                        continue
                    metrics.update(
                        usaf=station["usaf"], wban=station["wban"],
                        income_group=station["income_group"],
                        latitude=station["latitude"],
                        longitude=station["longitude"],
                        plot_longitude=station["plot_longitude"],
                        variable=variable, method=method_name,
                        lead_hours=lead, n_test=int(usable.sum()),
                        source="gridded_paper", forecast_model=forecast_model,
                        truth=truth_name)
                    rows.append(metrics)
    return rows


def extract_gridded_verification(directories, chunks, station_ids):
    """Verify the main paper's gridded Pangu and IFS outputs at the stations.

    The patches are the same files the paper's figures read (the configuration
    in erl_figures._model_kwargs). Their status is SOURCE_STATUS['gridded_paper'];
    re-run with --refresh_gridded once they are regenerated.

    Each station takes the nearest pixel of the first patch that covers it.
    Stations on no land patch (small islands, offshore platforms) are left out.

    Inputs:
        directories (dict): from helper_funcs.setup_directories().
        chunks (dict): output of load_sample_chunks.
        station_ids (pd.MultiIndex): (usaf, wban) of the stations to score,
            normally those the station pipeline trained on, so every figure
            shares one station set.

    Returns:
        pd.DataFrame: rows in the same schema as the station-pipeline results,
            tagged source='gridded_paper'.
    """
    stations = chunks["stations"]
    wanted = pd.MultiIndex.from_frame(stations[STATION_KEYS]).isin(station_ids)
    station_latitudes = stations["latitude"].to_numpy()
    station_longitudes = stations["longitude"].to_numpy()
    output_root = os.path.join(directories["processed"], "finetuning_output")

    rows = []
    for forecast_model in MODEL_LABELS:
        for variable_position, variable in enumerate(VARIABLE_NAMES):
            unassigned = wanted.copy()
            patch_paths = sorted(
                path for continent in CONTINENTS
                for path in filter_patch_zarr_files(
                    os.path.join(output_root, forecast_model, continent),
                    variable, **_model_kwargs()))
            print(f"  {forecast_model} {variable}: scanning "
                  f"{len(patch_paths)} patches", flush=True)
            for patch_path in patch_paths:
                if not unassigned.any():
                    break
                patch = xr.open_zarr(patch_path)
                latitude_index, longitude_index, inside = _nearest_pixels(
                    patch.latitude.values, patch.longitude.values,
                    station_latitudes, station_longitudes)
                inside &= unassigned
                if not inside.any():
                    continue
                unassigned &= ~inside
                rows.extend(_score_gridded_patch(
                    patch, variable_position, forecast_model, chunks,
                    np.flatnonzero(inside), latitude_index[inside],
                    longitude_index[inside]))
            print(f"    {int((wanted & ~unassigned).sum())} of "
                  f"{int(wanted.sum())} stations on a patch", flush=True)
    return pd.DataFrame(rows)


def build_verification_table(directories, results_path=None,
                             refresh_gridded=False, include_gridded=True):
    """Assemble every verification row the paper figures draw from.

    Inputs:
        directories (dict): from helper_funcs.setup_directories().
        results_path (str or None): combined station-pipeline results. Defaults
            to '<processed>/station_finetuning/station_results_sample.csv'.
        refresh_gridded (bool): re-extract the gridded verification even when
            its cache csv exists. Needed after the gridded models are re-run.
        include_gridded (bool): add the gridded rows. False skips reading (or
            extracting) them, for figures that use the station pipeline only.

    Returns:
        pd.DataFrame: station-pipeline rows (CURRENT) and, when requested,
            gridded rows (OUTDATED) in one table, told apart by 'source'.
    """
    work_directory = os.path.join(directories["processed"], "station_finetuning")
    results_path = results_path or os.path.join(work_directory,
                                                "station_results_sample.csv")
    chunks = load_sample_chunks(work_directory)
    pipeline_results = load_station_pipeline_results(results_path,
                                                     chunks["stations"])
    if not include_gridded:
        return pipeline_results

    cache_path = os.path.join(work_directory, GRIDDED_CACHE_NAME)
    if os.path.exists(cache_path) and not refresh_gridded:
        gridded_results = read_station_metadata(cache_path)
    else:
        print("extracting gridded outputs at station pixels", flush=True)
        gridded_results = extract_gridded_verification(
            directories, chunks,
            pd.MultiIndex.from_frame(pipeline_results[STATION_KEYS]))
        gridded_results.to_csv(cache_path, index=False)
        print(f"  cached to {cache_path}", flush=True)

    # Keep only the station-variable-leads the pipeline scored. The gridded
    # scoring has a looser sample minimum, so without this a few stations the
    # pipeline skipped for too few training days would enter the gridded
    # figures only. Applied after the cache so the cache stays a superset.
    scored = pipeline_results.loc[pipeline_results["method"] == "raw",
                                  STATION_KEYS + ["variable", "lead_hours"]]
    gridded_results = gridded_results.merge(scored, how="inner")

    return pd.concat([pipeline_results, gridded_results], ignore_index=True)


# ---------------------------------------------------------------------------
# Shared figure helpers
# ---------------------------------------------------------------------------

def _select(verification, variable, source, forecast_model, truth, method):
    """Return the verification rows for one variable and one data spec.

    Inputs:
        verification (pd.DataFrame): output of build_verification_table.
        variable (str): e.g. '2m_temperature'.
        source, forecast_model, truth, method (str): which rows to take, e.g.
            ('gridded_paper', 'ifs', 'station', 'raw').

    Returns:
        pd.DataFrame: the matching rows that have an income group.
    """
    return verification[(verification["variable"] == variable)
                        & (verification["source"] == source)
                        & (verification["forecast_model"] == forecast_model)
                        & (verification["truth"] == truth)
                        & (verification["method"] == method)
                        & verification["income_group"].notna()]


def _restrict_to_shared_stations(frames):
    """Keep only the station-leads present in every frame.

    Comparing two frames is only meaningful on one sample: without this a
    station missing from one of them would shift one bar's mean but not the
    other's.

    Inputs:
        frames (list[pd.DataFrame]): verification rows to intersect.

    Returns:
        list[pd.DataFrame]: the same frames, filtered to the station-leads
            they all share.
    """
    keys = [pd.MultiIndex.from_frame(frame[STATION_KEYS + ["lead_hours"]])
            for frame in frames]
    shared = keys[0]
    for key in keys[1:]:
        shared = shared.intersection(key)
    return [frame[key.isin(shared)] for frame, key in zip(frames, keys)]


def _pair_before_after(baseline, corrected, columns):
    """Line up each station's rows before and after post-processing.

    Inputs:
        baseline (pd.DataFrame): verification rows before post-processing.
        corrected (pd.DataFrame): rows after post-processing, same truth.
        columns (list[str]): metric columns to carry from both frames.

    Returns:
        pd.DataFrame: one row per station and lead, with the station keys,
            'lead_hours', the baseline's 'income_group', 'latitude' and
            'plot_longitude', and each of columns suffixed '_before' and
            '_after'.
    """
    keys = STATION_KEYS + ["lead_hours"]
    attributes = ["income_group", "latitude", "plot_longitude"]
    return baseline[keys + attributes + columns].merge(
        corrected[keys + columns], on=keys, suffixes=("_before", "_after"))


def _post_processed_acc(verification, variable, method, lead_hours=None):
    """Pair each station's raw and post-processed Pangu ACC.

    The before/after comparison behind Figs 2 and 3, verified against
    station observations.

    Inputs:
        verification (pd.DataFrame): output of build_verification_table.
        variable (str): e.g. '2m_temperature'.
        method (str): the post-processing method, a key of TRAINING_TARGETS
            or of LINEAR_MOS.
        lead_hours (int or None): lead time, one of LEAD_TIMES; None keeps
            every lead.

    Returns:
        pd.DataFrame: the rows of _pair_before_after with 'acc_before' (raw)
            and 'acc_after' (post-processed).
    """
    at_lead = (verification if lead_hours is None else
               verification[verification["lead_hours"] == lead_hours])
    return _pair_before_after(
        _select(at_lead, variable, *STATION_PIPELINE_PANGU, "raw"),
        _select(at_lead, variable, *STATION_PIPELINE_PANGU, method),
        ["acc"])


def _group_mean_and_interval(frame, value_columns, interval_column):
    """Average over stations by income group and lead, with a 95% interval.

    The interval is on the group MEAN (1.96 standard errors across stations),
    which is the inference a claim like "LMIC skill is lower" needs, not the
    spread across stations.

    Inputs:
        frame (pd.DataFrame): per-station rows with 'income_group' and
            'lead_hours'.
        value_columns (list[str]): columns to average.
        interval_column (str): column to put the interval on.

    Returns:
        pd.DataFrame: indexed by (income_group, lead_hours) and reindexed to
            every group and lead, with the averaged columns plus 'interval'.
    """
    grouped = frame.groupby(["income_group", "lead_hours"])
    summary = grouped[value_columns].mean()
    summary["interval"] = 1.96 * grouped[interval_column].sem()
    return summary.reindex(pd.MultiIndex.from_product(
        [INCOME_GROUP_ORDER, LEAD_TIMES]))


def _status_footnote(sources):
    """Build a figure's data-status footnote from the sources it draws on.

    Inputs:
        sources (iterable[str]): the 'source' tags the figure's data came from.

    Returns:
        str: the SOURCE_STATUS notes of those sources; empty when all are
            current.
    """
    return " ".join(SOURCE_STATUS[source] for source in dict.fromkeys(sources)
                    if SOURCE_STATUS[source])


def _station_count(frame):
    """Number of distinct stations in a frame of verification rows.

    Inputs:
        frame (pd.DataFrame): rows carrying the STATION_KEYS columns.

    Returns:
        int: the count of unique (usaf, wban) pairs.
    """
    return len(frame[STATION_KEYS].drop_duplicates())


def _map_stations(axis, frame, column, colormap, norm, label, extend,
                  colorbar_axis=None):
    """Draw a basemap, the stations coloured by one column, and a colorbar.

    Inputs:
        axis (cartopy GeoAxes): the map panel.
        frame (pd.DataFrame): one row per station with 'latitude',
            'plot_longitude', 'income_group' and column.
        column (str): the value to colour stations by.
        colormap (str): matplotlib colormap name.
        norm (matplotlib Normalize): maps values to colours.
        label (str): colorbar label.
        extend (str): colorbar extension, e.g. 'min' or 'both'.
        colorbar_axis (matplotlib Axes or None): where to draw the colorbar;
            None steals space from axis.

    Returns:
        None. Everything is drawn on axis and its figure.
    """
    _draw_basemap(axis)
    for income_group, marker in INCOME_MARKERS.items():
        in_group = frame[frame["income_group"] == income_group]
        scatter = axis.scatter(in_group["plot_longitude"], in_group["latitude"],
                               c=in_group[column], cmap=colormap, norm=norm,
                               marker=marker, s=24, edgecolors="#333333",
                               linewidths=0.3, transform=ccrs.PlateCarree(),
                               zorder=4)
    placement = (dict(cax=colorbar_axis) if colorbar_axis is not None
                 else dict(ax=axis, shrink=0.75, pad=0.02))
    colorbar = axis.figure.colorbar(scatter, extend=extend, **placement)
    colorbar.set_label(label)


def _label_income_bars(axis, bar_positions_by_group):
    """Label bars with their income group and group them under lead times.

    Inputs:
        axis (matplotlib Axes): the bar panel.
        bar_positions_by_group (dict): income group -> x positions of its bars,
            one per lead time in LEAD_TIMES.

    Returns:
        None.
    """
    groups = [group for group, group_positions in bar_positions_by_group.items()
              for _ in group_positions]
    axis.set_xticks(np.concatenate(list(bar_positions_by_group.values())))
    axis.set_xticklabels([INCOME_SHORT_LABELS[group] for group in groups],
                         fontsize=9)
    for tick, group in zip(axis.get_xticklabels(), groups):
        tick.set_color(INCOME_GROUP_COLORS[group])

    lead_centres = np.mean(list(bar_positions_by_group.values()), axis=0)
    below_axis = transforms.blended_transform_factory(axis.transData,
                                                      axis.transAxes)
    for centre, lead in zip(lead_centres, LEAD_TIMES):
        axis.text(centre, -0.08, LEAD_LABELS[lead], transform=below_axis,
                  ha="center", va="top", fontsize=11)


def _save(figure, save_path, sources, note=""):
    """Add the footnote, write the figure, close it and report where it went.

    Inputs:
        figure (matplotlib Figure): the finished figure.
        save_path (str): destination file.
        sources (iterable[str]): the 'source' tags the figure drew on; their
            SOURCE_STATUS notes are appended to the footnote.
        note (str): figure-specific footnote text, written first.

    Returns:
        None.
    """
    footnote = " ".join(text for text in (note, _status_footnote(sources))
                        if text)
    if footnote:
        figure.text(0.01, -0.01, footnote, fontsize=9, style="italic",
                    va="top", wrap=True)
    figure.savefig(save_path, bbox_inches="tight")
    plt.close(figure)
    print(f"  Saved: {save_path}", flush=True)


# ---------------------------------------------------------------------------
# Figure 1 -- raw forecast ACC and forecast inequality against stations
# ---------------------------------------------------------------------------

def _income_gap(frame, column="acc"):
    """High-income minus LMIC mean ACC by lead, with a 95% interval.

    Positive means the LMIC forecast is worse. The interval treats the two
    groups as independent samples of stations: 1.96 * sqrt(SE_HI^2 +
    SE_LMIC^2), with each SE the standard deviation across stations over
    sqrt(station count).

    Inputs:
        frame (pd.DataFrame): per-station rows with 'income_group',
            'lead_hours' and column.
        column (str): the ACC column to compare, default 'acc'.

    Returns:
        pd.DataFrame: indexed by lead_hours (every lead in LEAD_TIMES) with
            columns 'gap' and 'interval'.
    """
    summary = _group_mean_and_interval(frame, [column], column)
    high_income = summary.loc["High income"]
    low_middle = summary.loc["Low and middle income"]
    return pd.DataFrame({
        "gap": high_income[column] - low_middle[column],
        # The per-group intervals are 1.96 * SE, so they add in quadrature.
        "interval": np.sqrt(high_income["interval"] ** 2
                            + low_middle["interval"] ** 2),
    })


def _draw_gap_bars(axis, gaps, styles, title):
    """Draw side-by-side income-gap bars at every lead on one panel.

    Shared by Figs 1 and 3 so the two read side by side.

    Inputs:
        axis (matplotlib Axes): the panel.
        gaps (list[pd.DataFrame]): one _income_gap frame per bar series.
        styles (list[dict]): matplotlib bar keywords per series, optionally
            including 'label' for the legend.
        title (str): panel title.

    Returns:
        list[np.ndarray]: per series, the x position of its bar at each lead.
    """
    bar_width = min(0.36, 0.8 / len(gaps))
    lead_positions = np.arange(len(LEAD_TIMES))
    series_positions = []
    for position, (gap, style) in enumerate(zip(gaps, styles)):
        positions = (lead_positions
                     + (position - (len(gaps) - 1) / 2) * bar_width)
        axis.bar(positions, gap["gap"], width=bar_width, yerr=gap["interval"],
                 linewidth=0.8, zorder=3,
                 error_kw=dict(elinewidth=0.8, capsize=3, ecolor="#333333"),
                 **style)
        series_positions.append(positions)
    axis.axhline(0, color="#333333", linewidth=0.8, zorder=2)
    axis.set_title(title, fontsize=13)
    axis.set_xticks(lead_positions)
    axis.set_xticklabels([LEAD_LABELS[lead] for lead in LEAD_TIMES])
    return series_positions


def plot_acc_and_inequality(verification, forecast_model, variable, save_path,
                            lead_hours=24, side_by_side=True):
    """Fig 1: map of raw-forecast ACC beside the ACC income gap, one variable.

    Map: each station's raw-forecast ACC against its observations at one
    lead, marker shape showing the income group. The colour scale is fixed at
    0 to 1 so every version of the figure reads against the others. Bars: the
    high-income minus LMIC gap in mean ACC at every lead, verified against
    the same station observations.

    Inputs:
        verification (pd.DataFrame): output of build_verification_table.
        forecast_model (str): 'pangu' or 'ifs' (appendix).
        variable (str): '2m_temperature' (main text for Pangu) or
            '10m_wind_speed' (appendix).
        save_path (str): file to write the figure to.
        lead_hours (int): lead time to map, one of LEAD_TIMES.
        side_by_side (bool): True (default) puts a narrow bar panel to the
            right of the map in one row; False puts the bars below the map.

    Returns:
        None. Writes the figure to save_path.
    """
    source = RAW_STATION_SOURCE[forecast_model]
    raw = _select(verification, variable, source, forecast_model, "station",
                  "raw")
    at_lead = raw[raw["lead_hours"] == lead_hours]
    if side_by_side:
        # wspace leaves room for the map's colorbar and the bars' y label.
        figure = plt.figure(figsize=(11, 3.2))
        grid = figure.add_gridspec(1, 2, width_ratios=[1, 0.38], wspace=0.7)
        map_cell, bar_cell = grid[0, 0], grid[0, 1]
        # The narrow bar panel needs a two-line title and a short y label.
        gap_title = ("ACC Skill Gap Between High and\n"
                     "Low and Middle Income Countries")
        gap_label = "ACC gap (HI − LMIC)"
    else:
        figure = plt.figure(figsize=(12, 9))
        grid = figure.add_gridspec(2, 1, height_ratios=[4.3, 3.2],
                                   hspace=0.22)
        map_cell, bar_cell = grid[0, 0], grid[1, 0]
        gap_title = ("ACC Skill Gap Between High and Low and Middle Income "
                     "Countries")
        gap_label = GAP_AXIS_LABEL

    map_axis = figure.add_subplot(map_cell, projection=ccrs.PlateCarree())
    _map_stations(map_axis, at_lead, "acc", "viridis",
                  Normalize(vmin=0.0, vmax=1.0), "ACC", "min",
                  # An inset tracks the map's drawn box, so the colorbar
                  # matches the map's height whatever its aspect ratio.
                  colorbar_axis=map_axis.inset_axes([1.02, 0, 0.02, 1]))
    map_axis.set_title(f"{MODEL_LABELS[forecast_model]} "
                       f"{VARIABLE_TITLES[variable]}, "
                       f"{LEAD_LABELS[lead_hours]} lead ACC error",
                       fontsize=13)

    bar_axis = figure.add_subplot(bar_cell)
    _draw_gap_bars(bar_axis, [_income_gap(raw)],
                   [dict(color=MAROON, edgecolor=MAROON)], gap_title)
    bar_axis.set_ylabel(gap_label)

    _save(figure, save_path, [source])


# ---------------------------------------------------------------------------
# Figure 2 -- where station-trained post-processing changes ACC
# ---------------------------------------------------------------------------

def map_acc_change(verification, variable, save_path, lead_hours=24):
    """Fig 2: map of each station's % change in ACC from post-processing.

    The change is station-trained Pangu against raw Pangu, both verified
    against the station's observations, as a percentage of the raw ACC.
    Blue means the forecast improved. The colour scale is clipped at the
    90th percentile of the absolute change, because a station with raw ACC
    near zero (common for wind) can post a change of several hundred percent.

    Inputs:
        verification (pd.DataFrame): output of build_verification_table.
        variable (str): '2m_temperature' (main text) or '10m_wind_speed'
            (appendix).
        save_path (str): file to write the figure to.
        lead_hours (int): lead time to map, one of LEAD_TIMES.

    Returns:
        None. Writes the figure to save_path.
    """
    figure, axis = plt.subplots(figsize=(13, 5.5),
                                subplot_kw=dict(projection=ccrs.PlateCarree()))
    paired = _post_processed_acc(verification, variable,
                                 STATION_TRAINED_METHOD, lead_hours)
    paired["percent_change"] = (100 * (paired["acc_after"]
                                       - paired["acc_before"])
                                / paired["acc_before"])

    limit = float(np.nanpercentile(np.abs(paired["percent_change"]), 90))
    _map_stations(axis, paired, "percent_change", "RdBu",
                  TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
                  "% change in ACC", "both")
    axis.set_title(f"{VARIABLE_TITLES[variable]} ACC change after "
                   f"Post-Processing Pangu-Weather", fontsize=13)

    _save(figure, save_path, [STATION_PIPELINE_PANGU[0]])


# ---------------------------------------------------------------------------
# Figure 3 -- post-processing and the income gap
# ---------------------------------------------------------------------------

def _gap_closed_label(raw_gap, after_gap, after_interval):
    """Describe how much of the raw income gap post-processing closes.

    Inputs:
        raw_gap (float): the raw gap.
        after_gap, after_interval (float): the gap after post-processing and
            its 95% half-width.

    Returns:
        str: 'reversed' when the post-processed gap's interval lies entirely
            below zero, otherwise e.g. '24%'. Callers handle the no-raw-gap
            case, where a share is meaningless.
    """
    if after_gap + after_interval < 0:
        return "reversed"
    return f"{100 * (1 - after_gap / raw_gap):.0f}%"


def plot_gap_before_after(verification, save_path, include_mos=False):
    """Fig 3: the ACC income gap before and after post-processing, by lead.

    One panel per variable. At each lead the bars are the high-income minus
    LMIC gap in mean ACC for raw Pangu, optionally Pangu corrected by linear
    MOS, Pangu post-processed by the network trained on ERA5 (the main
    paper's method), and by the network trained on station observations, all
    verified against station observations on the same stations. Above each
    post-processed bar is the share of the raw gap it closes ('reversed' when
    its interval lies entirely below zero); a lead whose raw gap interval
    includes zero is labelled 'no raw gap'.

    Inputs:
        verification (pd.DataFrame): output of build_verification_table.
        save_path (str): file to write the figure to.
        include_mos (bool): add a linear MOS bar after the raw bar.

    Returns:
        None. Writes the figure to save_path.
    """
    # Post-processing method -> bar style, in drawing order after the raw bar.
    method_bars = {
        "mlp_era5_tuned": dict(color="white", edgecolor=MAROON, hatch="///",
                               label="Post-processed (ERA5-trained)"),
        STATION_TRAINED_METHOD: dict(color=MAROON, edgecolor=MAROON,
                                     label="Post-processed (station-trained)")}
    if include_mos:
        method_bars = {method: dict(color="white", edgecolor=MAROON,
                                    hatch="...", label=label)
                       for method, label in LINEAR_MOS.items()} | method_bars
    bars = ([dict(color=LIGHT_MAROON, edgecolor=MAROON, label="Raw")]
            + list(method_bars.values()))
    figure, axes = plt.subplots(1, 2, figsize=(9 + 1.25 * len(bars), 5.5),
                                sharey=True)

    for axis, variable in zip(axes, VARIABLE_NAMES):
        paired = _restrict_to_shared_stations([
            _post_processed_acc(verification, variable, method)
            for method in method_bars])
        # Every frame carries the same raw ACC on the shared stations.
        raw = _income_gap(paired[0], "acc_before")
        post_processed = [_income_gap(frame, "acc_after") for frame in paired]
        positions = _draw_gap_bars(
            axis, [raw] + post_processed, bars,
            f"{VARIABLE_LABELS[variable]} "
            f"({_station_count(paired[0])} stations)")
        lead_centres = np.mean(positions, axis=0)

        for lead_index, (raw_gap, raw_interval) in enumerate(
                zip(raw["gap"], raw["interval"])):
            if raw_gap - raw_interval <= 0:
                axis.annotate("no raw gap",
                              (lead_centres[lead_index],
                               max(raw_gap + raw_interval, 0)),
                              xytext=(0, 6), textcoords="offset points",
                              ha="center", fontsize=10, color="#333333")
                continue
            for gap, bar_positions in zip(post_processed, positions[1:]):
                after_gap, after_interval = gap.iloc[lead_index]
                axis.annotate(
                    _gap_closed_label(raw_gap, after_gap, after_interval),
                    (bar_positions[lead_index],
                     max(after_gap + after_interval, 0)),
                    xytext=(0, 4), textcoords="offset points", ha="center",
                    fontsize=8, color="#333333")
        axis.margins(y=0.12)  # headroom for the annotations
    axes[0].set_ylabel(GAP_AXIS_LABEL)

    figure.suptitle("Pangu-Weather forecast inequality before and after "
                    "post-processing", fontsize=14, weight="bold")
    figure.legend(*axes[0].get_legend_handles_labels(), ncol=len(bars),
                  loc="upper center", bbox_to_anchor=(0.5, 0.95),
                  frameon=False)
    figure.tight_layout(rect=[0, 0, 1, 0.93])
    _save(figure, save_path, [STATION_PIPELINE_PANGU[0]],
          "Verified against station observations. Labels: share of the raw "
          "gap closed. Whiskers: 95% CI of the difference in station means.")


# ---------------------------------------------------------------------------
# Figure 4 -- training on stations versus training on ERA5
# ---------------------------------------------------------------------------

def _rmse_change_components(baseline, corrected):
    """Split each station's % change in RMSE exactly into its components.

    Because RMSE^2 = bias^2 + IE^2 + NE^2,

        dRMSE / RMSE_before
            = sum over components of d(component^2)
              / ((RMSE_before + RMSE_after) * RMSE_before)

    so the three component terms sum to the % change in RMSE with no residual.

    Inputs:
        baseline (pd.DataFrame): verification rows before post-processing.
        corrected (pd.DataFrame): rows after post-processing, same truth.

    Returns:
        pd.DataFrame: the paired rows from _pair_before_after, plus one column
            per error component and 'total' (the % change in RMSE; negative
            means the forecast improved), all in percent of the raw RMSE.
    """
    paired = _pair_before_after(baseline, corrected,
                                list(ERROR_COMPONENT_COLORS) + ["rmse"])
    scale = 100 / ((paired["rmse_before"] + paired["rmse_after"])
                   * paired["rmse_before"])
    for component in ERROR_COMPONENT_COLORS:
        paired[component] = scale * (paired[f"{component}_after"] ** 2
                                     - paired[f"{component}_before"] ** 2)
    paired["total"] = 100 * (paired["rmse_after"] / paired["rmse_before"] - 1)
    return paired


def plot_rmse_decomposition(verification, save_path, include_mos=False):
    """Fig 4: the decomposed % change in RMSE, ERA5 target versus station target.

    Rows are variables and columns are the network's training target
    (optionally preceded by a linear MOS column); every panel is verified
    against station observations. Each bar is the station
    mean % change in RMSE for one income group and lead, stacked into bias,
    information error and noise error (positive parts above zero, negative
    parts below), and the black diamond is the total with its 95% CI. A
    post-processor that only damps the forecast shows up as a noise-error
    reduction offset by an information-error increase; one that adds
    information reduces both.

    Inputs:
        verification (pd.DataFrame): output of build_verification_table.
        save_path (str): file to write the figure to.
        include_mos (bool): add a linear MOS column before the networks.

    Returns:
        None. Writes the figure to save_path.
    """
    columns = dict(TRAINING_TARGETS)
    note = ("Pangu-Weather post-processed with the same network, trained on "
            "ERA5 (left) or on station observations (right); all panels "
            "verified against station observations.")
    if include_mos:
        columns = LINEAR_MOS | columns
        note = ("Pangu-Weather corrected by linear MOS (left) or the same "
                "network trained on ERA5 (centre) or on station observations "
                "(right); all panels verified against station observations.")
    figure, axes = plt.subplots(2, len(columns),
                                figsize=(5.5 * len(columns), 10), sharey="row")
    bar_width = 0.34
    lead_positions = np.arange(len(LEAD_TIMES))
    bar_positions_by_group = {
        group: lead_positions + (position - 0.5) * bar_width * 1.1
        for position, group in enumerate(INCOME_GROUP_ORDER)}

    for row, variable in enumerate(VARIABLE_NAMES):
        baseline = _select(verification, variable, *STATION_PIPELINE_PANGU,
                           "raw")
        for column, (method, target_label) in enumerate(columns.items()):
            axis = axes[row, column]
            changes = _rmse_change_components(
                baseline, _select(verification, variable,
                                  *STATION_PIPELINE_PANGU, method))
            summary = _group_mean_and_interval(
                changes, list(ERROR_COMPONENT_COLORS) + ["total"], "total")

            for income_group, positions in bar_positions_by_group.items():
                group_summary = summary.loc[income_group]
                above = np.zeros(len(LEAD_TIMES))
                below = np.zeros(len(LEAD_TIMES))
                for component, color in ERROR_COMPONENT_COLORS.items():
                    values = group_summary[component].to_numpy()
                    axis.bar(positions, values, width=bar_width,
                             bottom=np.where(values >= 0, above, below),
                             color=color, edgecolor="white", linewidth=0.5,
                             zorder=3)
                    above += np.nan_to_num(np.clip(values, 0, None))
                    below += np.nan_to_num(np.clip(values, None, 0))
                axis.errorbar(positions, group_summary["total"],
                              yerr=group_summary["interval"], fmt="D",
                              color="black", markersize=4, capsize=3,
                              elinewidth=0.9, zorder=4)

            axis.axhline(0, color="#333333", linewidth=0.8, zorder=2)
            _label_income_bars(axis, bar_positions_by_group)
            axis.set_title(f"{target_label}: {VARIABLE_LABELS[variable]}\n"
                           f"({_station_count(changes)} stations)",
                           fontsize=12)
            if column == 0:
                axis.set_ylabel("% change in RMSE")

    legend_handles = [Patch(facecolor=color, label=ERROR_COMPONENT_LABELS[name])
                      for name, color in ERROR_COMPONENT_COLORS.items()]
    legend_handles.append(Line2D([0], [0], marker="D", color="black",
                                 linestyle="", label="Total (95% CI)"))
    figure.legend(handles=legend_handles, ncol=4, loc="upper center",
                  bbox_to_anchor=(0.5, 1.03), frameon=False)
    figure.tight_layout(h_pad=3)
    _save(figure, save_path, [STATION_PIPELINE_PANGU[0]], note)


def main():
    """Build the station paper figures from the command line.

    Inputs: read from the command line; run with --help for the options.

    Returns:
        None. Writes the requested figures to the ERL figure directory.
    """
    figure_names = ["fig1", "fig2", "fig3", "fig4"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", nargs="+", default=figure_names,
                        choices=figure_names,
                        help="which figures to draw (default: all); fig1 "
                             "includes its IFS appendix versions")
    parser.add_argument("--results", default=None,
                        help="combined station-pipeline results; defaults to "
                             "station_results_sample.csv")
    parser.add_argument("--refresh_gridded", action="store_true",
                        help="re-extract the gridded outputs at the stations "
                             "instead of reading the cache")
    parser.add_argument("--lead", type=int, default=24, choices=LEAD_TIMES,
                        help="lead time in hours for the maps in figures 1 "
                             "and 2")
    arguments = parser.parse_args()

    directories = setup_directories()
    output_directory = _erl_output_dir(directories)
    # Only fig 1's IFS appendix versions use the gridded rows.
    verification = build_verification_table(
        directories, arguments.results, arguments.refresh_gridded,
        include_gridded="fig1" in arguments.figures)
    lead = arguments.lead

    def output_path(name):
        return os.path.join(output_directory, f"station_{name}.png")

    for variable, short in VARIABLE_SHORT_NAMES.items():
        # 2m temperature is the main-text version; wind goes in the appendix.
        appendix = "" if variable == "2m_temperature" else "appendix_"
        if "fig1" in arguments.figures:
            # IFS versions always go in the appendix.
            for model, prefix in (("pangu", appendix), ("ifs", "appendix_")):
                plot_acc_and_inequality(
                    verification, model, variable,
                    output_path(f"fig1_{prefix}acc_inequality_{model}_{short}"),
                    lead)
        if "fig2" in arguments.figures:
            map_acc_change(
                verification, variable,
                output_path(f"fig2_{appendix}acc_change_map_{short}_lt{lead}h"),
                lead)
    # Figs 3 and 4 are also drawn with a linear MOS reference, as '_with_mos'.
    for include_mos, suffix in ((False, ""), (True, "_with_mos")):
        if "fig3" in arguments.figures:
            plot_gap_before_after(
                verification, output_path(f"fig3_gap_before_after{suffix}"),
                include_mos)
        if "fig4" in arguments.figures:
            plot_rmse_decomposition(
                verification, output_path(f"fig4_rmse_decomposition{suffix}"),
                include_mos)


if __name__ == "__main__":
    main()
