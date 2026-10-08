"""
Section 9: figures for the station-level post-processing.

Reads the station results written by
station_post_processing/run_station_post_processing.py
(station_results_pangu.csv and station_results_ifs.csv: one row per station,
variable, method and lead time, every forecast verified against the station's
own observations) and saves the station figures of the paper to the figure
directory (erc_figures/):

    Fig 1  map of each station's raw forecast ACC at a 1 day lead, above the
           high-income minus LMIC ACC gap at 1, 5 and 9 days. One figure per
           forecast model and variable; Pangu 2m temperature is the main-text
           version, the others go in the appendix.
    Fig 2  map of each station's change in ACC from station-trained
           post-processing of Pangu, 1 day lead, one figure per variable.
    Fig 3  the % change in RMSE split into bias, information error and noise
           error (Bonavita & Geer 2026), with the % change in ACC alongside,
           for the ERA5-trained and the station-trained network ('_with_mos'
           adds a linear MOS column).
    Fig 4  the ACC income gap for raw Pangu and Pangu post-processed by the
           network trained on ERA5 and by the network trained on stations
           ('_with_mos' adds a linear MOS bar).

Run on its own with:
    uv run python figures/station_figures.py
"""

import os
import sys

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import FORECAST_MODELS, LEAD_TIMES_HOURS, VARIABLES, setup_directories  # noqa: E402
from figures.gridded_figures import (LEAD_TIME_LABELS, PUBLICATION_STYLE,  # noqa: E402
                                     VARIABLE_LABELS, VARIABLE_TITLES)
from income_groups import INCOME_GROUP_COLORS, INCOME_GROUP_ORDER  # noqa: E402

mpl.rcParams.update(PUBLICATION_STYLE)

VARIABLE_SHORT_NAMES = {"2m_temperature": "t2m", "10m_wind_speed": "wind"}
MODEL_LABELS = {"pangu": "Pangu-Weather", "ifs": "IFS HRES"}
INCOME_SHORT_LABELS = {"Low and middle income": "LMIC", "High income": "HI"}
STATION_KEYS = ["usaf", "wban"]

# The post-processing methods in the results files, with their labels
STATION_TRAINED = "network_trained_on_stations"
ERA5_TRAINED = "network_trained_on_era5"
LINEAR_MOS = "linear_mos"

# RMSE^2 = bias^2 + information_error^2 + noise_error^2, so these three
# components split any change in RMSE exactly
ERROR_COMPONENT_COLORS = {"bias": "#9e9e9e", "information_error": "#7b3294",
                          "noise_error": "#e08214"}
ERROR_COMPONENT_LABELS = {"bias": "Bias", "information_error": "Information error",
                          "noise_error": "Noise error"}
ACC_CHANGE_COLOR = "#d7191c"

MAROON = "#800000"
LIGHT_MAROON = "#d9a3a3"
GAP_AXIS_LABEL = "ACC gap (high income − LMIC)"

# On the station maps, marker shape shows the income group and colour the value
INCOME_MARKERS = {"High income": "o", "Low and middle income": "^"}
MAP_EXTENT = [-180, 180, -60, 85]


def load_station_results(directories):
    """
    Read the station results of every forecast model, grouped for the figures.

    Inputs:
        directories (dict): output of common.setup_directories().

    Returns:
        dict mapping (forecast_model, variable, method) -> pandas.DataFrame of
        that combination's rows (one per station and lead time), restricted to
        stations with an income group and carrying a 'plot_longitude' column
        in -180..180.
    """
    tables = []
    for forecast_model in FORECAST_MODELS:
        results_path = os.path.join(directories["station_output"],
                                    f"station_results_{forecast_model}.csv")
        tables.append(pd.read_csv(results_path, dtype={"usaf": str, "wban": str}))
    results = pd.concat(tables, ignore_index=True)
    results = results[results["income_group"].notna()].copy()
    results["plot_longitude"] = ((results["longitude"] + 180.0) % 360.0) - 180.0
    return {key: rows for key, rows in results.groupby(["forecast_model", "variable", "method"])}


def group_mean_and_interval(frame, value_columns, interval_column):
    """
    Average over stations by income group and lead time, with a 95% interval.

    The interval is on the group mean (1.96 standard errors across stations),
    not the spread across stations.

    Inputs:
        frame (pandas.DataFrame): one row per station and lead, with
            'income_group' and 'lead_hours'.
        value_columns (list of str): columns to average.
        interval_column (str): column to compute the interval of.

    Returns:
        pandas.DataFrame indexed by (income_group, lead_hours), covering every
        group and lead, with the averaged columns plus 'interval'.
    """
    grouped = frame.groupby(["income_group", "lead_hours"])
    summary = grouped[value_columns].mean()
    summary["interval"] = 1.96 * grouped[interval_column].sem()
    return summary.reindex(pd.MultiIndex.from_product([INCOME_GROUP_ORDER, LEAD_TIMES_HOURS]))


def income_gap(frame, column):
    """
    High-income minus LMIC mean of a column at each lead time, with a 95% interval.

    The two groups are independent samples of stations, so their intervals
    add in quadrature. A positive gap means LMIC forecasts are worse.

    Inputs:
        frame (pandas.DataFrame): one row per station and lead, with
            'income_group', 'lead_hours' and the column.
        column (str): the ACC column to compare.

    Returns:
        pandas.DataFrame indexed by lead_hours with columns 'gap' and 'interval'.
    """
    summary = group_mean_and_interval(frame, [column], column)
    high_income = summary.loc["High income"]
    low_and_middle_income = summary.loc["Low and middle income"]
    return pd.DataFrame({
        "gap": high_income[column] - low_and_middle_income[column],
        "interval": np.sqrt(high_income["interval"] ** 2
                            + low_and_middle_income["interval"] ** 2),
    })


def draw_gap_bars(axis, gaps, bar_styles, title):
    """
    Draw side-by-side income-gap bars at every lead time (figures 1 and 4).

    Inputs:
        axis (matplotlib Axes): the panel.
        gaps (list of pandas.DataFrame): one income_gap() table per bar series.
        bar_styles (list of dict): matplotlib bar keywords for each series,
            optionally with a 'label' for the legend.
        title (str): panel title.

    Returns:
        list of numpy.ndarray: for each series, the x position of its bar at
        each lead time.
    """
    bar_width = min(0.36, 0.8 / len(gaps))
    lead_positions = np.arange(len(LEAD_TIMES_HOURS))
    series_positions = []
    for series_index, (gap, bar_style) in enumerate(zip(gaps, bar_styles)):
        bar_positions = lead_positions + (series_index - (len(gaps) - 1) / 2) * bar_width
        axis.bar(bar_positions, gap["gap"], width=bar_width, yerr=gap["interval"],
                 linewidth=0.8, zorder=3,
                 error_kw=dict(elinewidth=0.8, capsize=3, ecolor="#333333"), **bar_style)
        series_positions.append(bar_positions)
    axis.axhline(0, color="#333333", linewidth=0.8, zorder=2)
    axis.set_title(title, fontsize=13)
    axis.set_xticks(lead_positions)
    axis.set_xticklabels([LEAD_TIME_LABELS[lead_time] for lead_time in LEAD_TIMES_HOURS])
    return series_positions


def map_stations(axis, stations, column, colormap, color_scale, colorbar_label, colorbar_extend,
                 colorbar_axis=None):
    """
    Draw a world map with every station coloured by one column, and a colourbar.

    Inputs:
        axis (cartopy GeoAxes): the map panel.
        stations (pandas.DataFrame): one row per station with 'latitude',
            'plot_longitude', 'income_group' and the column.
        column (str): the value to colour stations by.
        colormap (str): matplotlib colormap name.
        color_scale (matplotlib Normalize): maps values to colours.
        colorbar_label (str), colorbar_extend (str): colourbar label and extension.
        colorbar_axis (matplotlib Axes or None): where to draw the colourbar;
            None takes space from the map.

    Returns:
        None.
    """
    axis.set_extent(MAP_EXTENT, crs=ccrs.PlateCarree())
    axis.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
    axis.add_feature(cfeature.LAND, facecolor="#eeeeee", zorder=0)
    axis.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="#555555", zorder=1)
    axis.add_feature(cfeature.BORDERS, linewidth=0.25, edgecolor="#999999", zorder=1)
    axis.gridlines(draw_labels=False, linewidth=0.3, color="#bbbbbb", linestyle="--", alpha=0.6)

    for income_group, marker in INCOME_MARKERS.items():
        group_stations = stations[stations["income_group"] == income_group]
        points = axis.scatter(group_stations["plot_longitude"], group_stations["latitude"],
                              c=group_stations[column], cmap=colormap, norm=color_scale,
                              marker=marker, s=24, edgecolors="#333333", linewidths=0.3,
                              transform=ccrs.PlateCarree(), zorder=4)
    if colorbar_axis is not None:
        colorbar = axis.figure.colorbar(points, extend=colorbar_extend, cax=colorbar_axis)
    else:
        colorbar = axis.figure.colorbar(points, extend=colorbar_extend, ax=axis, shrink=0.75,
                                        pad=0.02)
    colorbar.set_label(colorbar_label)


def plot_acc_and_inequality(directories, results, forecast_model, variable, output_name,
                            lead_time=24):
    """
    Figure 1: map of raw forecast ACC at each station, above the ACC income gap.

    The colour scale is fixed at 0 to 1 so every version of the figure can be
    read against the others.

    Inputs:
        directories (dict): output of common.setup_directories().
        results (dict): output of load_station_results().
        forecast_model (str): 'pangu' or 'ifs'.
        variable (str): '2m_temperature' or '10m_wind_speed'.
        output_name (str): filename to save in the figure directory.
        lead_time (int): lead time of the map, in hours.

    Returns:
        None. Saves the figure.
    """
    raw = results[forecast_model, variable, "raw"]

    # A large map with a short bar panel below it
    figure = plt.figure(figsize=(11, 7.5))
    grid = figure.add_gridspec(2, 1, height_ratios=[1, 0.55], hspace=0.3)

    # The colourbar sits in an inset so it matches the map's drawn height
    map_axis = figure.add_subplot(grid[0], projection=ccrs.PlateCarree())
    map_stations(map_axis, raw[raw["lead_hours"] == lead_time], "acc", "viridis",
                 Normalize(vmin=0.0, vmax=1.0), "ACC", "min",
                 colorbar_axis=map_axis.inset_axes([1.02, 0, 0.02, 1]))
    map_axis.set_title(f"{MODEL_LABELS[forecast_model]} {VARIABLE_TITLES[variable]}, "
                       f"{LEAD_TIME_LABELS[lead_time]} lead ACC error", fontsize=13)

    bar_axis = figure.add_subplot(grid[1])
    draw_gap_bars(bar_axis, [income_gap(raw, "acc")], [dict(color=MAROON, edgecolor=MAROON)],
                  "ACC Skill Gap Between High and Low and Middle Income Countries")
    bar_axis.set_ylabel(GAP_AXIS_LABEL)

    output_path = os.path.join(directories["figures"], output_name)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    print(f"  Saved: {output_path}")


def map_acc_change(directories, results, variable, output_name, lead_time=24):
    """
    Figure 2: map of each station's change in ACC from station-trained post-processing.

    The change is station-trained Pangu ACC minus raw Pangu ACC; blue means
    the forecast improved. The colour scale is symmetric about zero and
    clipped at the 90th percentile of the absolute change so a few large
    changes do not wash out the rest.

    Inputs:
        directories (dict): output of common.setup_directories().
        results (dict): output of load_station_results().
        variable (str): '2m_temperature' or '10m_wind_speed'.
        output_name (str): filename to save in the figure directory.
        lead_time (int): lead time of the map, in hours.

    Returns:
        None. Saves the figure.
    """
    # Pair each station's raw and post-processed ACC
    keys = STATION_KEYS + ["lead_hours"]
    raw = results["pangu", variable, "raw"]
    corrected = results["pangu", variable, STATION_TRAINED]
    raw = raw[raw["lead_hours"] == lead_time]
    corrected = corrected[corrected["lead_hours"] == lead_time]
    paired = raw[keys + ["income_group", "latitude", "plot_longitude", "acc"]].merge(
        corrected[keys + ["acc"]], on=keys, suffixes=("_before", "_after"))
    paired["acc_change"] = paired["acc_after"] - paired["acc_before"]

    figure, axis = plt.subplots(figsize=(13, 5.5),
                                subplot_kw=dict(projection=ccrs.PlateCarree()))
    color_limit = float(np.nanpercentile(np.abs(paired["acc_change"]), 90))
    map_stations(axis, paired, "acc_change", "RdBu",
                 TwoSlopeNorm(vmin=-color_limit, vcenter=0.0, vmax=color_limit),
                 "Change in ACC", "both")
    axis.set_title(f"{VARIABLE_TITLES[variable]} ACC change after "
                   f"Post-Processing Pangu-Weather", fontsize=13)

    output_path = os.path.join(directories["figures"], output_name)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    print(f"  Saved: {output_path}")


def plot_gap_before_after(directories, results, output_name, include_linear_mos=False):
    """
    Figure 4: the ACC income gap before and after post-processing, by lead time.

    One panel per variable. At each lead the bars are the high-income minus
    LMIC gap in mean ACC for raw Pangu, (optionally) Pangu corrected by linear
    MOS, and Pangu post-processed by the network trained on ERA5 and by the
    network trained on stations, all on the same stations. Above each
    post-processed bar is the share of the raw gap it closes ('reversed' when
    its interval lies entirely below zero); a lead whose raw gap interval
    includes zero is labelled 'no raw gap'.

    Inputs:
        directories (dict): output of common.setup_directories().
        results (dict): output of load_station_results().
        output_name (str): filename to save in the figure directory.
        include_linear_mos (bool): add a linear MOS bar after the raw bar.

    Returns:
        None. Saves the figure.
    """
    # Post-processing method -> bar style, in drawing order after the raw bar
    method_bar_styles = {}
    if include_linear_mos:
        method_bar_styles[LINEAR_MOS] = dict(color="white", edgecolor=MAROON, hatch="...",
                                             label="Linear MOS")
    method_bar_styles[ERA5_TRAINED] = dict(color="white", edgecolor=MAROON, hatch="///",
                                           label="Post-processed (ERA5-trained)")
    method_bar_styles[STATION_TRAINED] = dict(color=MAROON, edgecolor=MAROON,
                                              label="Post-processed (station-trained)")
    bar_styles = ([dict(color=LIGHT_MAROON, edgecolor=MAROON, label="Raw")]
                  + list(method_bar_styles.values()))

    figure, axes = plt.subplots(1, 2, figsize=(9 + 1.25 * len(bar_styles), 5.5), sharey=True)
    keys = STATION_KEYS + ["lead_hours"]
    for axis, variable in zip(axes, VARIABLES):
        # Pair the raw and post-processed ACC of every station, for every method
        raw = results["pangu", variable, "raw"]
        paired_by_method = [
            raw[keys + ["income_group", "acc"]].merge(
                results["pangu", variable, method][keys + ["acc"]], on=keys,
                suffixes=("_before", "_after"))
            for method in method_bar_styles]

        # Compare every method on the same stations: keep the station-leads
        # that every method scored
        shared_keys = pd.MultiIndex.from_frame(paired_by_method[0][keys])
        for paired in paired_by_method[1:]:
            shared_keys = shared_keys.intersection(pd.MultiIndex.from_frame(paired[keys]))
        paired_by_method = [paired[pd.MultiIndex.from_frame(paired[keys]).isin(shared_keys)]
                            for paired in paired_by_method]

        # Every paired table carries the same raw ACC on the shared stations
        raw_gap = income_gap(paired_by_method[0], "acc_before")
        post_processed_gaps = [income_gap(paired, "acc_after") for paired in paired_by_method]
        station_count = len(paired_by_method[0][STATION_KEYS].drop_duplicates())
        bar_positions = draw_gap_bars(axis, [raw_gap] + post_processed_gaps, bar_styles,
                                      f"{VARIABLE_LABELS[variable]} ({station_count} stations)")
        lead_centres = np.mean(bar_positions, axis=0)

        # Label each post-processed bar with the share of the raw gap it closes
        for lead_index, (gap_before, interval_before) in enumerate(
                zip(raw_gap["gap"], raw_gap["interval"])):
            if gap_before - interval_before <= 0:
                axis.annotate("no raw gap",
                              (lead_centres[lead_index], max(gap_before + interval_before, 0)),
                              xytext=(0, 6), textcoords="offset points", ha="center",
                              fontsize=10, color="#333333")
                continue
            for gap, positions in zip(post_processed_gaps, bar_positions[1:]):
                gap_after, interval_after = gap.iloc[lead_index]
                if gap_after + interval_after < 0:
                    label = "reversed"
                else:
                    label = f"{100 * (1 - gap_after / gap_before):.0f}%"
                axis.annotate(label, (positions[lead_index], max(gap_after + interval_after, 0)),
                              xytext=(0, 4), textcoords="offset points", ha="center",
                              fontsize=8, color="#333333")
        axis.margins(y=0.12)  # headroom for the labels
    axes[0].set_ylabel(GAP_AXIS_LABEL)

    figure.suptitle("Pangu-Weather forecast inequality before and after post-processing",
                    fontsize=14, weight="bold")
    figure.legend(*axes[0].get_legend_handles_labels(), ncol=len(bar_styles),
                  loc="upper center", bbox_to_anchor=(0.5, 0.95), frameon=False)
    figure.tight_layout(rect=[0, 0, 1, 0.93])
    figure.text(0.01, -0.01, "Verified against station observations. Labels: share of the "
                "raw gap closed. Whiskers: 95% CI of the difference in station means.",
                fontsize=9, style="italic", va="top", wrap=True)

    output_path = os.path.join(directories["figures"], output_name)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    print(f"  Saved: {output_path}")


def plot_rmse_decomposition(directories, results, output_name, include_linear_mos=False):
    """
    Figure 3: the % change in RMSE split into bias, information and noise error,
    with the % change in ACC alongside.

    Rows are variables and columns are post-processing methods (the network
    trained on ERA5 and on stations, optionally preceded by linear MOS); every
    panel is verified against station observations. Each bar is the station
    mean % change in RMSE for one income group and lead, stacked into its
    three components (positive parts above zero, negative parts below), and
    the black diamond is the total with its 95% interval. The red dot is the
    % change in the group's mean ACC, 100 * mean change / mean raw ACC, with a
    95% interval from the spread of the station changes (the raw mean is
    treated as fixed). The group mean is used rather than the mean of each
    station's % change because stations with a raw ACC near zero would
    dominate the latter.

    Because RMSE^2 = bias^2 + IE^2 + NE^2,

        change in RMSE / RMSE_before
            = sum over components of change in component^2
              / ((RMSE_before + RMSE_after) * RMSE_before)

    so the three parts add up to the total exactly.

    Inputs:
        directories (dict): output of common.setup_directories().
        results (dict): output of load_station_results().
        output_name (str): filename to save in the figure directory.
        include_linear_mos (bool): add a linear MOS column before the networks.

    Returns:
        None. Saves the figure.
    """
    column_labels = {ERA5_TRAINED: "Trained on ERA5", STATION_TRAINED: "Trained on stations"}
    footnote = ("Pangu-Weather post-processed with the same network, trained on ERA5 (left) "
                "or on station observations (right); all panels verified against station "
                "observations.")
    if include_linear_mos:
        column_labels = {LINEAR_MOS: "Linear MOS"} | column_labels
        footnote = ("Pangu-Weather corrected by linear MOS (left) or the same network trained "
                    "on ERA5 (centre) or on station observations (right); all panels verified "
                    "against station observations.")

    figure, axes = plt.subplots(2, len(column_labels), figsize=(5.5 * len(column_labels), 10),
                                sharey="row")
    bar_width = 0.34
    lead_positions = np.arange(len(LEAD_TIMES_HOURS))
    bar_positions_by_group = {income_group: lead_positions + (group_index - 0.5) * bar_width * 1.1
                              for group_index, income_group in enumerate(INCOME_GROUP_ORDER)}
    keys = STATION_KEYS + ["lead_hours"]
    metric_columns = list(ERROR_COMPONENT_COLORS) + ["rmse", "acc"]

    for row, variable in enumerate(VARIABLES):
        raw = results["pangu", variable, "raw"]
        for column, (method, column_label) in enumerate(column_labels.items()):
            axis = axes[row, column]

            # Each station's % change in RMSE, split into its components
            changes = raw[keys + ["income_group"] + metric_columns].merge(
                results["pangu", variable, method][keys + metric_columns], on=keys,
                suffixes=("_before", "_after"))
            scale = 100 / ((changes["rmse_before"] + changes["rmse_after"])
                           * changes["rmse_before"])
            for component in ERROR_COMPONENT_COLORS:
                changes[component] = scale * (changes[f"{component}_after"] ** 2
                                              - changes[f"{component}_before"] ** 2)
            changes["total"] = 100 * (changes["rmse_after"] / changes["rmse_before"] - 1)
            summary = group_mean_and_interval(changes, list(ERROR_COMPONENT_COLORS) + ["total"],
                                              "total")

            # % change in each group's mean ACC, with the interval of the mean change
            changes["acc_change"] = changes["acc_after"] - changes["acc_before"]
            acc_summary = group_mean_and_interval(changes, ["acc_change", "acc_before"],
                                                  "acc_change")
            acc_percent_change = 100 * acc_summary["acc_change"] / acc_summary["acc_before"]
            acc_percent_interval = 100 * acc_summary["interval"] / acc_summary["acc_before"]

            # Stack the components for each income group: positive parts
            # upwards from zero, negative parts downwards
            for income_group, positions in bar_positions_by_group.items():
                group_summary = summary.loc[income_group]
                height_above = np.zeros(len(LEAD_TIMES_HOURS))
                height_below = np.zeros(len(LEAD_TIMES_HOURS))
                for component, color in ERROR_COMPONENT_COLORS.items():
                    values = group_summary[component].to_numpy()
                    axis.bar(positions, values, width=bar_width,
                             bottom=np.where(values >= 0, height_above, height_below),
                             color=color, edgecolor="white", linewidth=0.5, zorder=3)
                    height_above += np.nan_to_num(np.clip(values, 0, None))
                    height_below += np.nan_to_num(np.clip(values, None, 0))
                # Total RMSE change and ACC change side by side over each bar
                axis.errorbar(positions - 0.15 * bar_width, group_summary["total"],
                              yerr=group_summary["interval"], fmt="D", color="black",
                              markersize=4, capsize=3, elinewidth=0.9, zorder=4)
                axis.errorbar(positions + 0.15 * bar_width, acc_percent_change.loc[income_group],
                              yerr=acc_percent_interval.loc[income_group], fmt="o",
                              color=ACC_CHANGE_COLOR, markersize=4.5, capsize=3, elinewidth=0.9,
                              zorder=4)
            axis.axhline(0, color="#333333", linewidth=0.8, zorder=2)

            # Income group under each bar, lead time under each pair of bars
            bar_groups = [income_group for income_group, positions
                          in bar_positions_by_group.items() for _ in positions]
            axis.set_xticks(np.concatenate(list(bar_positions_by_group.values())))
            axis.set_xticklabels([INCOME_SHORT_LABELS[group] for group in bar_groups], fontsize=9)
            for tick_label, income_group in zip(axis.get_xticklabels(), bar_groups):
                tick_label.set_color(INCOME_GROUP_COLORS[income_group])
            below_axis = transforms.blended_transform_factory(axis.transData, axis.transAxes)
            for lead_centre, lead_time in zip(
                    np.mean(list(bar_positions_by_group.values()), axis=0), LEAD_TIMES_HOURS):
                axis.text(lead_centre, -0.08, LEAD_TIME_LABELS[lead_time], transform=below_axis,
                          ha="center", va="top", fontsize=11)

            station_count = len(changes[STATION_KEYS].drop_duplicates())
            axis.set_title(f"{column_label}: {VARIABLE_LABELS[variable]}\n"
                           f"({station_count} stations)", fontsize=12)
            if column == 0:
                axis.set_ylabel("% change")

    legend_entries = [Patch(facecolor=color, label=ERROR_COMPONENT_LABELS[component])
                      for component, color in ERROR_COMPONENT_COLORS.items()]
    legend_entries.append(Line2D([0], [0], marker="D", color="black", linestyle="",
                                 label="RMSE (95% CI, lower is better)"))
    legend_entries.append(Line2D([0], [0], marker="o", color=ACC_CHANGE_COLOR, linestyle="",
                                 label="ACC (95% CI, higher is better)"))
    figure.legend(handles=legend_entries, ncol=5, loc="upper center", columnspacing=1.0,
                  bbox_to_anchor=(0.5, 1.03), frameon=False)
    figure.tight_layout(h_pad=3)
    figure.text(0.01, -0.01, footnote, fontsize=9, style="italic", va="top", wrap=True)

    output_path = os.path.join(directories["figures"], output_name)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    print(f"  Saved: {output_path}")


def run():
    """
    Draw every station post-processing figure into the figure directory.

    Comment out a line below to skip that figure.

    Inputs:
        None.

    Returns:
        None.
    """
    directories = setup_directories()
    print(f"Writing station figures to {directories['figures']}")
    results = load_station_results(directories)

    for variable, short_name in VARIABLE_SHORT_NAMES.items():
        # 2m temperature is the main-text version, wind speed goes in the appendix
        appendix = "" if variable == "2m_temperature" else "appendix_"

        # Figure 1: raw ACC and the income gap (IFS versions are appendix figures)
        plot_acc_and_inequality(directories, results, "pangu", variable,
                                f"station_fig1_{appendix}acc_inequality_pangu_{short_name}.png")
        plot_acc_and_inequality(directories, results, "ifs", variable,
                                f"station_fig1_appendix_acc_inequality_ifs_{short_name}.png")

        # Figure 2: change in ACC from station-trained post-processing
        map_acc_change(directories, results, variable,
                       f"station_fig2_{appendix}acc_change_map_{short_name}_lt24h.png")

    # Figures 3 and 4, each also drawn with a linear MOS reference
    plot_rmse_decomposition(directories, results, "station_fig3_rmse_decomposition.png")
    plot_rmse_decomposition(directories, results, "station_fig3_rmse_decomposition_with_mos.png",
                            include_linear_mos=True)
    plot_gap_before_after(directories, results, "station_fig4_gap_before_after.png")
    plot_gap_before_after(directories, results, "station_fig4_gap_before_after_with_mos.png",
                          include_linear_mos=True)


if __name__ == "__main__":
    run()
