"""
Section 8: figures for the gridded (ERA5 / HRES analysis) post-processing.

Reads the per-patch zarr outputs written by
gridded_post_processing/run_post_processing.py (and, for the architecture
comparison, by gridded_post_processing/architecture_comparison.py) and saves
every gridded figure of the paper to the figure directory (erc_figures/):

    erl_fig1_summary_equity.png              RMSE improvement by World Bank income group
    erl_fig2_global_maps_unified.png         Pangu improvement maps, 1 and 9 day leads
    erl_fig5_arch_comparison_temperature.png architecture comparison, 2m temperature
    erl_fig5_arch_comparison_wind.png        architecture comparison, 10m wind speed
    erl_appA1_model_compare_boxplot.png      Pangu vs IFS improvement per patch
    erl_appA2_pangu_5day_maps.png            Pangu improvement maps, 5 day lead
    erl_appA3_ifs_maps.png                   IFS improvement maps, all leads
    erl_appA8_arch_eval_regions.png          the patches used for the architecture comparison
    erl_appA9_income_group_map.png           the World Bank income groups

Every figure except the architecture comparison uses the production model, an
MLP snapshot ensemble of 3 runs trained on each 6x6 degree land patch.

Run on its own with:
    uv run python figures/gridded_figures.py
"""

import os
import sys

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (EVALUATION_PATCH_FRACTION, LEAD_TIMES_HOURS,  # noqa: E402
                    PATCH_SAMPLE_SEED, PRODUCTION_MODEL_SUFFIX, VARIABLES, gridded_output_path,
                    load_continent_patches, sample_continent_patches, setup_directories)
from income_groups import (INCOME_GROUP_COLORS, INCOME_GROUP_ORDER,  # noqa: E402
                           load_income_group_countries, match_points_to_countries)

# Publication defaults for every figure in the paper (the station figures
# apply the same ones)
PUBLICATION_STYLE = {
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "legend.title_fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
}
mpl.rcParams.update(PUBLICATION_STYLE)

# Labels shared with the station figures
LEAD_TIME_LABELS = {24: "1 day", 120: "5 day", 216: "9 day"}
VARIABLE_LABELS = {"2m_temperature": "2m temperature", "10m_wind_speed": "10m wind speed"}
VARIABLE_TITLES = {"2m_temperature": "2m Temperature", "10m_wind_speed": "10m Wind Speed"}

# The model variants compared in figure 5 (keys of common.MODEL_VARIANTS), in
# bar order, with their labels and colours
ARCHITECTURE_VARIANTS = [
    ("MLP", "mlp", "#1f77b4"),
    ("MLP Snapshot Ensemble ×3", "mlp_snapshot3", "#2ca02c"),
    ("Block LTHO Ensemble", "mlp_blockk3_snapshot1", "#9467bd"),
    ("Per-LT MLP Snapshot ×3", "mlp_snapshot3_perlt", "#d62728"),
    ("UNet", "unet", "#ff7f0e"),
]


def load_patch_outputs(directories, forecast_model, variable, model_suffix=PRODUCTION_MODEL_SUFFIX):
    """
    Open the post-processing output of every land patch.

    Inputs:
        directories (dict): output of common.setup_directories().
        forecast_model (str): 'pangu' or 'ifs'.
        variable (str): '2m_temperature' or '10m_wind_speed'.
        model_suffix (str): model variant, a key of common.MODEL_VARIANTS.

    Returns:
        list of xarray.Dataset, one per patch that has an output file, opened
        lazily. Each holds '{variable}_{original,corrected,mean_corrected,
        ground_truth}_lt{N}h' arrays of shape (time, latitude, longitude).
    """
    patch_outputs = []
    missing_patches = 0
    for continent, patch_number, _ in load_continent_patches(directories):
        output_path = gridded_output_path(directories, forecast_model, continent, patch_number,
                                          variable, model_suffix)
        if os.path.exists(output_path):
            patch_outputs.append(xr.open_zarr(output_path, consolidated=False))
        else:
            missing_patches += 1
    print(f"  {forecast_model} {variable}: {len(patch_outputs)} patches loaded, "
          f"{missing_patches} missing")
    return patch_outputs


def pixel_improvement_maps(patch_outputs, variable, lead_time):
    """
    Compute the percent RMSE improvement at every grid cell of every patch.

    The RMSE at a grid cell is taken over the test-year days.

    Inputs:
        patch_outputs (list of xarray.Dataset): output of load_patch_outputs().
        variable (str): the corrected variable.
        lead_time (int): lead time in hours.

    Returns:
        list of dict, one per patch, with keys 'latitudes', 'longitudes'
        (1D arrays), 'improvement_percent' (2D array, latitude x longitude)
        and the patch bounds 'latitude_min', 'latitude_max', 'longitude_min',
        'longitude_max'.
    """
    patch_maps = []
    for patch_output in patch_outputs:
        truth = patch_output[f"{variable}_ground_truth_lt{lead_time}h"]
        original = patch_output[f"{variable}_original_lt{lead_time}h"]
        corrected = patch_output[f"{variable}_corrected_lt{lead_time}h"]
        original_rmse = np.sqrt(((original - truth) ** 2).mean(dim="time"))
        corrected_rmse = np.sqrt(((corrected - truth) ** 2).mean(dim="time"))
        patch_maps.append({
            "latitudes": patch_output.latitude.values,
            "longitudes": patch_output.longitude.values,
            "improvement_percent": ((original_rmse - corrected_rmse) / original_rmse * 100).values,
            "latitude_min": float(patch_output.latitude.min()),
            "latitude_max": float(patch_output.latitude.max()),
            "longitude_min": float(patch_output.longitude.min()),
            "longitude_max": float(patch_output.longitude.max()),
        })
    return patch_maps


def patch_improvement_percent(patch_output, variable, lead_time, corrected_kind="corrected"):
    """
    Compute one patch's percent RMSE improvement, pooled over all days and cells.

    Inputs:
        patch_output (xarray.Dataset): one patch's post-processing output.
        variable (str): the corrected variable.
        lead_time (int): lead time in hours.
        corrected_kind (str): 'corrected' for the network, or 'mean_corrected'
            for the mean bias correction baseline.

    Returns:
        float, or None if the patch has no valid values.
    """
    truth = patch_output[f"{variable}_ground_truth_lt{lead_time}h"].values.flatten()
    original = patch_output[f"{variable}_original_lt{lead_time}h"].values.flatten()
    corrected = patch_output[f"{variable}_{corrected_kind}_lt{lead_time}h"].values.flatten()

    valid = ~(np.isnan(truth) | np.isnan(original) | np.isnan(corrected))
    if not valid.any():
        return None
    original_rmse = np.sqrt(((original[valid] - truth[valid]) ** 2).mean())
    corrected_rmse = np.sqrt(((corrected[valid] - truth[valid]) ** 2).mean())
    if original_rmse == 0:
        return 0
    return float((original_rmse - corrected_rmse) / original_rmse * 100)


def draw_improvement_map(axis, patch_maps, color_minimum, color_maximum):
    """
    Draw pixel-level improvement on a world map, with an outline around each patch.

    The colour scale diverges at zero, so a negative value (the post-processed
    forecast is worse) is always red and a positive one always blue.

    Inputs:
        axis (cartopy GeoAxes): the map axis to draw on.
        patch_maps (list of dict): output of pixel_improvement_maps().
        color_minimum, color_maximum (float): ends of the colour scale.

    Returns:
        matplotlib.colors.TwoSlopeNorm, the colour scale used, for the colourbar.
    """
    # A tiny negative minimum keeps the scale diverging at zero even when every
    # value is positive
    color_scale = TwoSlopeNorm(vmin=min(color_minimum, -1e-3), vcenter=0, vmax=color_maximum)

    axis.set_global()
    axis.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3, zorder=0)
    axis.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
    axis.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="black", zorder=2)
    axis.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.3, edgecolor="gray", zorder=2)

    # Fill in each patch's grid cells
    for patch_map in patch_maps:
        axis.pcolormesh(patch_map["longitudes"], patch_map["latitudes"],
                        patch_map["improvement_percent"], transform=ccrs.PlateCarree(),
                        cmap=plt.cm.RdBu, norm=color_scale, shading="nearest", zorder=1)

    # Outline each patch
    outline_segments = []
    for patch_map in patch_maps:
        west, east = patch_map["longitude_min"], patch_map["longitude_max"]
        south, north = patch_map["latitude_min"], patch_map["latitude_max"]
        outline_segments.extend([[(west, south), (east, south)], [(east, south), (east, north)],
                                 [(east, north), (west, north)], [(west, north), (west, south)]])
    axis.add_collection(LineCollection(outline_segments, colors="black", linewidths=0.4,
                                       alpha=0.9, transform=ccrs.PlateCarree(), zorder=2))
    return color_scale


def plot_improvement_map_grid(directories, forecast_model, lead_times, column_titles,
                              colorbar_labels, figure_size, row_spacing, colorbar_pad,
                              colorbar_shrink, show_global_mean, title, title_size,
                              title_height, output_name):
    """
    Draw a grid of improvement maps: one row per lead time, one column per variable.

    Each column has its own colour scale, running from the 1st to the 99th
    percentile of the pixel improvements in that column, and its own colourbar.

    Inputs:
        directories (dict): output of common.setup_directories().
        forecast_model (str): 'pangu' or 'ifs'.
        lead_times (list of int): lead times in hours, one per row.
        column_titles (dict): variable -> title above its column.
        colorbar_labels (dict): variable -> label of its colourbar.
        figure_size (tuple): figure width and height in inches.
        row_spacing (float): vertical space between rows (gridspec hspace).
        colorbar_pad, colorbar_shrink (float): colourbar placement and length.
        show_global_mean (bool): write the mean improvement on each map.
        title (str), title_size (int), title_height (float): the figure title.
        output_name (str): filename to save in the figure directory.

    Returns:
        None. Saves the figure.
    """
    # Compute pixel improvements and the colour scale of each column
    maps_by_variable = {}
    color_range_by_variable = {}
    for variable in VARIABLES:
        patch_outputs = load_patch_outputs(directories, forecast_model, variable)
        maps_by_variable[variable] = {lead_time: pixel_improvement_maps(patch_outputs, variable,
                                                                        lead_time)
                                      for lead_time in lead_times}
        column_values = np.concatenate([
            patch_map["improvement_percent"][~np.isnan(patch_map["improvement_percent"])]
            for lead_time in lead_times for patch_map in maps_by_variable[variable][lead_time]])
        color_range_by_variable[variable] = np.percentile(column_values, [1, 99])

    figure = plt.figure(figsize=figure_size)
    grid = gridspec.GridSpec(len(lead_times), len(VARIABLES), hspace=row_spacing, wspace=0.06)
    for column, variable in enumerate(VARIABLES):
        color_minimum, color_maximum = color_range_by_variable[variable]
        column_axes = []

        # Draw one map per lead time
        for row, lead_time in enumerate(lead_times):
            axis = figure.add_subplot(grid[row, column], projection=ccrs.PlateCarree())
            patch_maps = maps_by_variable[variable][lead_time]
            color_scale = draw_improvement_map(axis, patch_maps, color_minimum, color_maximum)
            column_axes.append(axis)
            if row == 0:
                axis.set_title(column_titles[variable], fontsize=14, weight="bold")
            if len(lead_times) > 1:
                axis.text(-0.03, 0.5, LEAD_TIME_LABELS[lead_time], transform=axis.transAxes,
                          rotation=90, va="center", fontsize=13, weight="bold")
            if show_global_mean:
                map_values = np.concatenate([patch_map["improvement_percent"].flatten()
                                             for patch_map in patch_maps])
                axis.text(0.02, 0.04, f"Global mean: {np.nanmean(map_values):.1f}%",
                          transform=axis.transAxes, fontsize=11, zorder=5,
                          bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=3))

        # One colourbar under the column, with ticks every 5% inside the range.
        # A column of one map passes the axis itself, so that matplotlib fits
        # the colourbar inside that map's grid cell.
        color_mapping = plt.cm.ScalarMappable(norm=color_scale, cmap=plt.cm.RdBu)
        colorbar_parent = column_axes[0] if len(column_axes) == 1 else column_axes
        colorbar = figure.colorbar(color_mapping, ax=colorbar_parent, orientation="horizontal",
                                   pad=colorbar_pad, shrink=colorbar_shrink, aspect=40)
        colorbar.set_label(colorbar_labels[variable], fontsize=13, weight="bold")
        colorbar.set_ticks([tick for tick in range(0, 35, 5)
                            if max(color_minimum, 0) <= tick <= color_maximum])
        colorbar.ax.tick_params(labelsize=11)

    figure.suptitle(title, fontsize=title_size, weight="bold", y=title_height)
    output_path = os.path.join(directories["figures"], output_name)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    print(f"  Saved: {output_path}")


def plot_income_group_improvement(directories):
    """
    Figure 1: Pangu RMSE improvement by World Bank income group and lead time.

    Bars are the mean percent improvement over all land grid cells in each
    income group, and whiskers one standard deviation across those cells.

    Inputs:
        directories (dict): output of common.setup_directories().

    Returns:
        None. Saves erl_fig1_summary_equity.png.
    """
    countries = load_income_group_countries(directories)
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    bar_width = 0.25
    lead_time_positions = np.arange(len(LEAD_TIMES_HOURS))

    for axis, variable in zip(axes, VARIABLES):
        # Collect the improvement at every grid cell, for every lead time
        patch_outputs = load_patch_outputs(directories, "pangu", variable)
        pixel_tables = []
        for lead_time in LEAD_TIMES_HOURS:
            for patch_map in pixel_improvement_maps(patch_outputs, variable, lead_time):
                longitude_grid, latitude_grid = np.meshgrid(patch_map["longitudes"],
                                                            patch_map["latitudes"])
                valid = ~np.isnan(patch_map["improvement_percent"])
                pixel_tables.append(pd.DataFrame({
                    "lead_time": lead_time,
                    "improvement_percent": patch_map["improvement_percent"][valid],
                    "latitude": latitude_grid[valid],
                    "longitude": longitude_grid[valid],
                }))
        pixels = pd.concat(pixel_tables, ignore_index=True)

        # Label each grid cell with its country's income group (one spatial
        # join for all cells, because each join has a large fixed cost)
        pixels["income_group"] = match_points_to_countries(
            pixels["latitude"], pixels["longitude"], countries, ["income_group"])["income_group"]
        pixels = pixels.dropna(subset=["income_group"])
        group_statistics = (pixels.groupby(["income_group", "lead_time"])["improvement_percent"]
                            .agg(["mean", "std"]))

        # One bar per income group at each lead time
        for group_index, income_group in enumerate(INCOME_GROUP_ORDER):
            statistics = group_statistics.loc[income_group].reindex(LEAD_TIMES_HOURS)
            bar_offset = (group_index - (len(INCOME_GROUP_ORDER) - 1) / 2) * bar_width
            axis.bar(lead_time_positions + bar_offset, statistics["mean"].to_numpy(),
                     width=bar_width, yerr=statistics["std"].to_numpy(),
                     color=INCOME_GROUP_COLORS[income_group], edgecolor="#333333",
                     linewidth=0.6, label=income_group, zorder=3,
                     error_kw=dict(elinewidth=0.8, capsize=3, capthick=0.8, ecolor="#333333"))

        axis.axhline(0, color="gray", linewidth=0.7, zorder=1)
        axis.set_ylabel("RMSE improvement (%)")
        axis.set_title(VARIABLE_LABELS[variable], fontsize=14, weight="bold")
        axis.set_xticks(lead_time_positions)
        axis.set_xticklabels([LEAD_TIME_LABELS[lead_time] for lead_time in LEAD_TIMES_HOURS])
        axis.set_xlabel("Lead time")
        axis.set_ylim(bottom=0)

    # Title and a shared legend above both panels
    figure.suptitle("Forecast Improvement from Post-Processing Model by Income Group",
                    fontsize=14, weight="bold", y=0.98)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.93),
                  frameon=False, fontsize=11)
    figure.text(0.98, 0.05, "whiskers = ±1 std", ha="right", va="bottom", fontsize=10,
                style="italic")
    figure.tight_layout(rect=[0, 0.08, 1, 0.91])

    output_path = os.path.join(directories["figures"], "erl_fig1_summary_equity.png")
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    print(f"  Saved: {output_path}")


def plot_architecture_comparison(directories, variable, output_name):
    """
    Figure 5: RMSE improvement of each model variant on the evaluation patches.

    Each bar is the mean improvement across the evaluation patches and the
    whiskers one standard deviation across them. The mean bias correction
    baseline is read from the plain MLP's output files. The legend gives the
    mean training time per patch.

    Inputs:
        directories (dict): output of common.setup_directories().
        variable (str): '2m_temperature' or '10m_wind_speed'.
        output_name (str): filename to save in the figure directory.

    Returns:
        None. Saves the figure.
    """
    evaluation_patches = sample_continent_patches(directories, EVALUATION_PATCH_FRACTION,
                                                  PATCH_SAMPLE_SEED, split="eval")

    # Score every variant on every evaluation patch. The mean bias baseline
    # is stored in every output file; it is read from the first variant's.
    bar_names, bar_colors = ["Mean Bias Correction"], ["#999999"]
    bar_means, bar_deviations = [], []
    mean_bias_improvements = {lead_time: [] for lead_time in LEAD_TIMES_HOURS}
    for variant_index, (variant_name, model_suffix, color) in enumerate(ARCHITECTURE_VARIANTS):
        improvements = {lead_time: [] for lead_time in LEAD_TIMES_HOURS}
        training_minutes = []
        for continent, patch_number, _ in evaluation_patches:
            output_path = gridded_output_path(directories, "pangu", continent, patch_number,
                                              variable, model_suffix)
            if not os.path.exists(output_path):
                print(f"  Missing: {output_path}")
                continue
            patch_output = xr.open_zarr(output_path)
            training_minutes.append(float(patch_output.attrs["training_time_minutes"]))
            for lead_time in LEAD_TIMES_HOURS:
                improvements[lead_time].append(
                    patch_improvement_percent(patch_output, variable, lead_time))
                if variant_index == 0:
                    mean_bias_improvements[lead_time].append(
                        patch_improvement_percent(patch_output, variable, lead_time,
                                                  "mean_corrected"))
        bar_names.append(f"{variant_name} ({round(np.mean(training_minutes), 2):.1f} min)")
        bar_colors.append(color)
        bar_means.append([np.mean(improvements[lead_time]) for lead_time in LEAD_TIMES_HOURS])
        bar_deviations.append([np.std(improvements[lead_time]) for lead_time in LEAD_TIMES_HOURS])
    bar_means.insert(0, [np.mean(mean_bias_improvements[lead_time])
                         for lead_time in LEAD_TIMES_HOURS])
    bar_deviations.insert(0, [np.std(mean_bias_improvements[lead_time])
                              for lead_time in LEAD_TIMES_HOURS])

    # Draw the bars, grouped by lead time
    figure, axis = plt.subplots(figsize=(14, 10))
    bar_width = 0.6 / len(bar_names)
    for bar_index, (name, color) in enumerate(zip(bar_names, bar_colors)):
        bar_positions = (np.arange(len(LEAD_TIMES_HOURS))
                         + (bar_index - len(bar_names) / 2 + 0.5) * bar_width)
        axis.bar(bar_positions, bar_means[bar_index], width=bar_width, color=color, alpha=0.8,
                 edgecolor="black", linewidth=0.5, label=name, zorder=3,
                 yerr=bar_deviations[bar_index],
                 error_kw=dict(ecolor="black", capsize=3, capthick=1, elinewidth=1.2, zorder=4))

    # Leave room above the tallest whisker
    whisker_tops = np.array(bar_means) + np.array(bar_deviations)
    whisker_bottoms = np.array(bar_means) - np.array(bar_deviations)
    axis.set_ylim(min(0, whisker_bottoms.min() - 0.5), whisker_tops.max() * 1.12)

    axis.set_ylabel("RMSE Improvement (%)", fontsize=20)
    axis.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    axis.set_xticks(range(len(LEAD_TIMES_HOURS)))
    axis.set_xticklabels([f"{lead_time / 24:.0f}" for lead_time in LEAD_TIMES_HOURS])
    axis.set_xlabel("Forecast Lead Time (days)", fontsize=20)
    axis.set_title(f"Architecture Comparison: RMSE Improvement — {VARIABLE_TITLES[variable]}\n"
                   f"Model: PANGU, Patch Size: 6x6 — averaged across "
                   f"{len(evaluation_patches)} global eval cells", fontsize=20, pad=15)
    axis.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", labelsize=16)
    axis.annotate("Whiskers show ±1 std across eval cells", xy=(0.99, 0.02),
                  xycoords="axes fraction", fontsize=12, color="gray", ha="right", va="bottom")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=14,
                framealpha=0.95, edgecolor="gray", columnspacing=1.0, handlelength=1.5,
                handletextpad=0.5)
    plt.tight_layout()

    output_path = os.path.join(directories["figures"], output_name)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"  Saved: {output_path}")


def plot_model_comparison_boxplot(directories):
    """
    Appendix A1: distribution of patch-level improvement, Pangu vs IFS.

    One row per variable; boxes grouped by lead time with one box per
    forecast model. The solid black line in each box is the mean and the
    dashed grey line the median.

    Inputs:
        directories (dict): output of common.setup_directories().

    Returns:
        None. Saves erl_appA1_model_compare_boxplot.png.
    """
    model_colors = {"pangu": "#1f77b4", "ifs": "#ff7f0e"}
    lead_time_labels = {24: "1 day", 120: "5 days", 216: "9 days"}
    box_width = 0.35

    figure, axes = plt.subplots(2, 1, figsize=(12, 10))
    for row, variable in enumerate(VARIABLES):
        axis = axes[row]

        # Patch-level improvement for each model and lead time
        improvements = {}
        for forecast_model in model_colors:
            patch_outputs = load_patch_outputs(directories, forecast_model, variable)
            for lead_time in LEAD_TIMES_HOURS:
                improvements[forecast_model, lead_time] = [
                    patch_improvement_percent(patch_output, variable, lead_time)
                    for patch_output in patch_outputs]

        # One group of boxes per lead time, one box per model inside the group
        box_values, box_positions, box_colors = [], [], []
        for group_index, lead_time in enumerate(LEAD_TIMES_HOURS):
            for model_index, forecast_model in enumerate(model_colors):
                box_values.append(improvements[forecast_model, lead_time])
                box_positions.append(group_index + 1 + (model_index - 0.5) * box_width)
                box_colors.append(model_colors[forecast_model])
        boxes = axis.boxplot(box_values, positions=box_positions, widths=box_width * 0.8,
                             patch_artist=True, showmeans=True, meanline=True, showfliers=True,
                             flierprops=dict(marker="o", markersize=4, alpha=0.5))

        # Style the boxes, the mean and median lines, and the whiskers
        for box, color in zip(boxes["boxes"], box_colors):
            box.set_facecolor(color)
            box.set_alpha(0.7)
        for mean_line in boxes["means"]:
            mean_line.set_color("black")
            mean_line.set_linewidth(2)
        for median_line in boxes["medians"]:
            median_line.set_color("darkgray")
            median_line.set_linewidth(1)
            median_line.set_linestyle("--")
        for line in boxes["whiskers"] + boxes["caps"]:
            line.set_color("gray")
            line.set_linewidth(1)

        axis.axhline(y=0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
        axis.set_xticks([1, 2, 3])
        axis.set_xticklabels([lead_time_labels[lead_time] for lead_time in LEAD_TIMES_HOURS],
                             fontsize=11)
        axis.set_xlabel("Lead Time", fontsize=12)
        axis.set_ylabel("RMSE Improvement (%)", fontsize=12)
        axis.grid(True, alpha=0.3, linestyle="--", axis="y")
        axis.text(-0.08, 0.5, variable.replace("_", " ").title(), transform=axis.transAxes,
                  fontsize=12, fontweight="bold", rotation=90, verticalalignment="center")
        if row == 0:
            legend_entries = [plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.7,
                                            label=forecast_model.upper())
                              for forecast_model, color in model_colors.items()]
            legend_entries.append(Line2D([0], [0], color="black", linewidth=2, label="Mean"))
            axis.legend(handles=legend_entries, loc="upper right", fontsize=10)

    # Lay out in two passes, as the published version was: the first leaves a
    # left margin for the row labels, and the empty title (the caption
    # describes the figure) keeps its top margin
    figure.tight_layout(rect=[0.05, 0, 1, 0.98])
    figure.suptitle("")
    figure.tight_layout()
    output_path = os.path.join(directories["figures"], "erl_appA1_model_compare_boxplot.png")
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    print(f"  Saved: {output_path}")


def plot_evaluation_patch_map(directories):
    """
    Appendix A8: map of every land patch, highlighting the evaluation patches
    used for the architecture comparison.

    Inputs:
        directories (dict): output of common.setup_directories().

    Returns:
        None. Saves erl_appA8_arch_eval_regions.png.
    """
    all_patches = load_continent_patches(directories)
    evaluation_patches = sample_continent_patches(directories, EVALUATION_PATCH_FRACTION,
                                                  PATCH_SAMPLE_SEED, split="eval")
    evaluation_keys = {(continent, patch_number)
                       for continent, patch_number, _ in evaluation_patches}

    figure = plt.figure(figsize=(16, 10))
    axis = figure.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    axis.set_global()
    axis.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3, zorder=0)
    axis.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
    axis.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="black", zorder=2)
    axis.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.3, edgecolor="gray", zorder=2)

    # Outline every patch; fill in the evaluation patches
    for continent, patch_number, patch_coordinates in all_patches:
        latitudes, longitudes = patch_coordinates[0], patch_coordinates[1]
        is_evaluation_patch = (continent, patch_number) in evaluation_keys
        axis.add_patch(Rectangle(
            (longitudes.min(), latitudes.min()),
            longitudes.max() - longitudes.min(), latitudes.max() - latitudes.min(),
            facecolor="#1f77b4" if is_evaluation_patch else "none",
            edgecolor="#1f77b4" if is_evaluation_patch else "#7f7f7f",
            linewidth=0.9 if is_evaluation_patch else 0.35,
            alpha=0.75 if is_evaluation_patch else 0.45,
            transform=ccrs.PlateCarree(), zorder=3 if is_evaluation_patch else 1))

    gridlines = axis.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False,
                               linewidth=0.5, alpha=0.5, linestyle="--", zorder=4)
    gridlines.top_labels = False
    gridlines.right_labels = False
    gridlines.xlabel_style = {"size": 10}
    gridlines.ylabel_style = {"size": 10}

    evaluation_percent = 100.0 * len(evaluation_patches) / len(all_patches)
    axis.set_title(f"Architecture Testing Regions (6x6 land patches)\n"
                   f"PANGU — highlighted eval subset: {len(evaluation_patches)}/"
                   f"{len(all_patches)} ({evaluation_percent:.1f}%)",
                   fontsize=16, weight="bold", pad=20)
    legend_entries = [
        Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="#7f7f7f", linewidth=0.8,
                  label="All land 6x6 patches"),
        Rectangle((0, 0), 1, 1, facecolor="#1f77b4", edgecolor="#1f77b4", alpha=0.75,
                  linewidth=0.8, label="5% eval subset (architecture testing)"),
    ]
    axis.legend(handles=legend_entries, loc="lower left", framealpha=0.95, edgecolor="gray")
    plt.tight_layout()

    output_path = os.path.join(directories["figures"], "erl_appA8_arch_eval_regions.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"  Saved: {output_path}")


def plot_income_group_map(directories):
    """
    Appendix A9: world map coloured by World Bank 2024 income group.

    Inputs:
        directories (dict): output of common.setup_directories().

    Returns:
        None. Saves erl_appA9_income_group_map.png.
    """
    countries = load_income_group_countries(directories)

    figure = plt.figure(figsize=(13, 6.5))
    axis = figure.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    axis.set_global()
    axis.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
    axis.add_feature(cfeature.LAND, facecolor="#dddddd", alpha=0.4, zorder=0)
    for income_group in INCOME_GROUP_ORDER:
        group_countries = countries[countries["income_group"] == income_group]
        axis.add_geometries(group_countries.geometry, crs=ccrs.PlateCarree(),
                            facecolor=INCOME_GROUP_COLORS[income_group],
                            edgecolor="#333333", linewidth=0.3, zorder=2)
    axis.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="black", zorder=3)

    legend_entries = [Line2D([0], [0], marker="s", linestyle="",
                             markerfacecolor=INCOME_GROUP_COLORS[income_group],
                             markeredgecolor="#333333", markersize=10, label=income_group)
                      for income_group in INCOME_GROUP_ORDER]
    axis.legend(handles=legend_entries, title="World Bank 2024 income group",
                loc="lower left", frameon=True, fontsize=10)
    axis.set_title("World Bank 2024 income group classification", fontsize=14, weight="bold")

    output_path = os.path.join(directories["figures"], "erl_appA9_income_group_map.png")
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
    print(f"  Saved: {output_path}")


def run():
    """
    Draw every gridded post-processing figure into the figure directory.

    Comment out a line below to skip that figure.

    Inputs:
        None.

    Returns:
        None.
    """
    directories = setup_directories()
    print(f"Writing gridded figures to {directories['figures']}")

    # Figure 1: improvement by income group
    plot_income_group_improvement(directories)

    # Figure 2: Pangu improvement maps at 1 and 9 days
    plot_improvement_map_grid(
        directories, "pangu", lead_times=[24, 216],
        column_titles=VARIABLE_LABELS,
        colorbar_labels={variable: f"{label} RMSE improvement (%)"
                         for variable, label in VARIABLE_LABELS.items()},
        figure_size=(16, 9), row_spacing=0.05, colorbar_pad=0.04, colorbar_shrink=0.85,
        show_global_mean=True,
        title="Pangu-Weather post-processing improvement (1 day top, 9 day bottom)",
        title_size=15, title_height=0.99, output_name="erl_fig2_global_maps_unified.png")

    # Figure 5: architecture comparison
    plot_architecture_comparison(directories, "2m_temperature",
                                 "erl_fig5_arch_comparison_temperature.png")
    plot_architecture_comparison(directories, "10m_wind_speed",
                                 "erl_fig5_arch_comparison_wind.png")

    # Appendix A1: Pangu vs IFS
    plot_model_comparison_boxplot(directories)

    # Appendix A2: Pangu improvement maps at 5 days
    plot_improvement_map_grid(
        directories, "pangu", lead_times=[120],
        column_titles={variable: variable.replace("_", " ").title() for variable in VARIABLES},
        colorbar_labels={variable: "RMSE improvement (%)" for variable in VARIABLES},
        figure_size=(16, 5), row_spacing=None, colorbar_pad=0.07, colorbar_shrink=0.75,
        show_global_mean=False, title="Pangu-Weather: 5 day lead time RMSE improvement",
        title_size=14, title_height=1.02, output_name="erl_appA2_pangu_5day_maps.png")

    # Appendix A3: IFS improvement maps at every lead time
    plot_improvement_map_grid(
        directories, "ifs", lead_times=LEAD_TIMES_HOURS,
        column_titles=VARIABLE_LABELS,
        colorbar_labels={variable: f"IFS {label} RMSE improvement (%)"
                         for variable, label in VARIABLE_LABELS.items()},
        figure_size=(16, 13), row_spacing=0.06, colorbar_pad=0.03, colorbar_shrink=0.85,
        show_global_mean=False,
        title="IFS HRES post-processing improvement (1, 5, 9 day lead times)",
        title_size=15, title_height=0.99, output_name="erl_appA3_ifs_maps.png")

    # Appendix A8 and A9: the evaluation patches and the income groups
    plot_evaluation_patch_map(directories)
    plot_income_group_map(directories)


if __name__ == "__main__":
    run()
