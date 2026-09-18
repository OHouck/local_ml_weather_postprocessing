"""
Purpose: This script uses the data and figure making code from the houck franke
post-processing paper to generate figures showing the relative accuracies of IFS
and Pangu at different benchmarks. The figures created are:

All plots saved in figures/science_policy


A single 2x3 grid of pixel-level maps comparing the accuracy of the raw Pangu
and IFS 2m temperature forecasts:

- Rows are the accuracy metric: plain RMSE and RMSE for observations > 30 C.
- Columns are the lead time: 1 day, 5 day, and 9 day.

Unlike figure 2 in the paper (which maps how much post-processing improves a single
model), these maps compare the *raw* forecasts of two models against each other:
positive values (blue) mean Pangu-Weather is more accurate than IFS for that pixel.

Run with::

    uv run python science_policy_figures.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs

sys.path.insert(0, str(Path(__file__).parent))

from helper_funcs import setup_directories
from finetuning.figures_finetuning import load_region_data
from finetuning.custom_loss_fns import mortality_dose_response, ABOVE_30_THRESHOLD_C
from finetuning.erl_figures import (
    _model_kwargs,
    _draw_improvement_map,
    LEAD_TIMES,
)

# Lead times (hours) shown as the three map columns: 1 day, 5 day, 9 day.
LEAD_TIMES_PLOT = [24, 120, 216]
LEAD_LABELS = {24: "1 day", 120: "5 day", 216: "9 day"}

# Accuracy metrics shown as the map rows, top to bottom.
METRICS_PLOT = ["rmse", "above_30_degree_loss"]

# Full description of each accuracy metric (used in the colourbar label).
METRIC_LABELS = {
    "rmse": "RMSE",
    "above_30_degree_loss": "RMSE for observations > 30°C",
    "mortality_weighted_loss": "mortality-weighted RMSE",
}

# Short label for the left-hand side of each row.
METRIC_ROW_LABELS = {
    "rmse": "RMSE",
    "above_30_degree_loss": "RMSE, obs > 30°C",
    "mortality_weighted_loss": "Mortality-weighted",
}


def _science_policy_output_dir(dirs):
    """Return (and create) the figures/science_policy output directory."""
    out_dir = os.path.join(dirs["fig"], "science_policy")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _pixel_error_metric(forecast, ground_truth, alternate_loss_fn):
    """
    Per-pixel forecast-error metric, reduced over the time dimension.

    inputs:
        forecast: xarray.DataArray (time, latitude, longitude) of the raw
            forecast in Kelvin.
        ground_truth: xarray.DataArray (time, latitude, longitude) of the
            observed values in Kelvin, on the same grid as ``forecast``.
        alternate_loss_fn: which metric to compute. One of
            'rmse' (or None) for plain RMSE, 'above_30_degree_loss' for RMSE
            restricted to observations above ABOVE_30_THRESHOLD_C, or
            'mortality_weighted_loss' for RMSE in mortality-response space.

    output:
        xarray.DataArray (latitude, longitude) of the per-pixel metric. Pixels
        with no qualifying timesteps (e.g. never above 30 C) are NaN.
    """
    if alternate_loss_fn in (None, "rmse"):
        squared_error = (forecast - ground_truth) ** 2
        return np.sqrt(squared_error.mean(dim="time"))

    if alternate_loss_fn == "above_30_degree_loss":
        ground_truth_c = ground_truth - 273.15
        squared_error = (forecast - ground_truth) ** 2
        # Only score timesteps above the threshold; mean over time skips NaNs.
        squared_error = squared_error.where(ground_truth_c > ABOVE_30_THRESHOLD_C)
        return np.sqrt(squared_error.mean(dim="time"))

    if alternate_loss_fn == "mortality_weighted_loss":
        mortality_forecast = mortality_dose_response(forecast - 273.15)
        mortality_truth = mortality_dose_response(ground_truth - 273.15)
        squared_error = (mortality_forecast - mortality_truth) ** 2
        return np.sqrt(squared_error.mean(dim="time"))

    raise ValueError(f"Unknown alternate_loss_fn: {alternate_loss_fn}")


def _compute_pixel_model_comparison(pangu_patches, ifs_patches, variable,
                                    lead_time, alternate_loss_fn):
    """
    Per-patch pixel-level accuracy comparison of Pangu vs IFS raw forecasts.

    Matches each Pangu patch to the IFS patch covering the same 6x6 box, computes
    the per-pixel error metric for each model, and expresses the difference as a
    percent improvement of Pangu over IFS:

        improvement = (metric_ifs - metric_pangu) / metric_ifs * 100

    so positive values mean Pangu is more accurate (lower error) than IFS.

    inputs:
        pangu_patches, ifs_patches: lists of patch dicts from load_region_data
            (each with 'ds' plus lat_min/lat_max/lon_min/lon_max bounds).
        variable: variable name, e.g. '2m_temperature'.
        lead_time: lead time in hours.
        alternate_loss_fn: metric name passed to _pixel_error_metric.

    output:
        list of dicts (one per matched patch) with keys 'lats', 'lons',
        'values', 'lat_min', 'lat_max', 'lon_min', 'lon_max' — the format
        expected by _draw_improvement_map.
    """
    var_suffix = f"_lt{lead_time}h"

    def _box_key(patch):
        return (round(patch["lat_min"], 2), round(patch["lat_max"], 2),
                round(patch["lon_min"], 2), round(patch["lon_max"], 2))

    ifs_by_box = {_box_key(p): p for p in ifs_patches}

    out = []
    for pangu_patch in pangu_patches:
        ifs_patch = ifs_by_box.get(_box_key(pangu_patch))
        if ifs_patch is None:
            continue

        pangu_ds = pangu_patch["ds"]
        ifs_ds = ifs_patch["ds"]

        metric_pangu = _pixel_error_metric(
            pangu_ds[f"{variable}_original{var_suffix}"],
            pangu_ds[f"{variable}_ground_truth{var_suffix}"],
            alternate_loss_fn,
        )
        metric_ifs = _pixel_error_metric(
            ifs_ds[f"{variable}_original{var_suffix}"],
            ifs_ds[f"{variable}_ground_truth{var_suffix}"],
            alternate_loss_fn,
        )
        # Align IFS onto the Pangu grid so pixels correspond even if coordinate
        # ordering differs; mismatched cells become NaN and drop out of the map.
        metric_ifs = metric_ifs.reindex(
            latitude=metric_pangu.latitude, longitude=metric_pangu.longitude)

        improvement = (metric_ifs - metric_pangu) / metric_ifs * 100

        # Replace non-finite cells (e.g. mortality-space RMSE of exactly 0 in the
        # IFS denominator) with NaN so pcolormesh renders them transparent
        # instead of corrupting the axes limits and colourbar.
        values = np.where(np.isfinite(improvement.values), improvement.values, np.nan)

        out.append({
            "lats":    pangu_ds.latitude.values,
            "lons":    pangu_ds.longitude.values,
            "values":  values,
            "lat_min": pangu_patch["lat_min"],
            "lat_max": pangu_patch["lat_max"],
            "lon_min": pangu_patch["lon_min"],
            "lon_max": pangu_patch["lon_max"],
        })
    return out


def load_pangu_ifs_data(dirs, variable):
    """
    Load the Pangu and IFS post-processing patch datasets for one variable.

    Factored out so the three metric figures can share a single load instead of
    re-reading every zarr patch once per metric.

    inputs:
        dirs: directory dict from setup_directories().
        variable: variable name, e.g. '2m_temperature'.

    output:
        tuple (pangu_apd, ifs_apd) of load_region_data dicts (either may be None
        if that model has no matching output files).
    """
    mkw = _model_kwargs()
    pangu_apd = load_region_data(
        dirs=dirs, model="pangu", variable=variable, regions=None,
        lead_times=LEAD_TIMES, sdor_da=None, **mkw,
    )
    ifs_apd = load_region_data(
        dirs=dirs, model="ifs", variable=variable, regions=None,
        lead_times=LEAD_TIMES, sdor_da=None, **mkw,
    )
    return pangu_apd, ifs_apd


def _symmetric_ticks(lim):
    """
    Return up to five symmetric integer colourbar ticks inside [-lim, lim].

    A tick placed beyond the norm range produces a degenerate colourbar
    transform (a blank bar), so the limit is floored rather than rounded up.

    inputs:
        lim: positive float, half-width of the symmetric colour scale.

    output:
        sorted list of unique integer tick values.
    """
    tick = int(np.floor(lim))
    return sorted({-tick, -tick // 2, 0, tick // 2, tick})


def modified_unified_global_maps(dirs, save_dir, pangu_apd=None, ifs_apd=None):
    """
    2x3 grid of pixel-level maps of Pangu-vs-IFS 2m temperature accuracy, based
    on the make_figure_2_unified_global_maps function in finetuning/erl_figures.py.

    Rows are the accuracy metric (plain RMSE and RMSE for observations > 30 C);
    columns are the 1 day, 5 day, and 9 day lead times. Each pixel shows the
    percent improvement of Pangu over IFS under that metric. Each row has its own
    diverging colour scale centred at zero (blue = Pangu more accurate, red = IFS
    more accurate), shared across its three lead times so the columns are
    comparable within a row.

    inputs:
        dirs: directory dict from setup_directories().
        save_dir: directory to write the PNG into.
        pangu_apd, ifs_apd: optional pre-loaded load_region_data dicts (from
            load_pangu_ifs_data). If omitted, the data is loaded here.

    output:
        None. Writes a single PNG to save_dir and returns nothing.
    """
    print("\n=== Pangu vs IFS combined 2x3 accuracy maps ===")
    variable = "2m_temperature"

    if pangu_apd is None or ifs_apd is None:
        pangu_apd, ifs_apd = load_pangu_ifs_data(dirs, variable)
    if pangu_apd is None or ifs_apd is None:
        print("  Missing Pangu or IFS data; skipping.")
        return

    fig = plt.figure(figsize=(18, 7))
    gs = gridspec.GridSpec(len(METRICS_PLOT), len(LEAD_TIMES_PLOT),
                           hspace=0.08, wspace=0.05)

    for row, metric in enumerate(METRICS_PLOT):
        # Compute all three lead-time panels for this metric, keeping each
        # panel's finite values for the shared colour scale and annotations.
        panel_data = {}
        panel_finite = {}
        for lt in LEAD_TIMES_PLOT:
            pdata = _compute_pixel_model_comparison(
                pangu_apd[lt], ifs_apd[lt], variable, lt, metric)
            panel_data[lt] = pdata
            finite = [p["values"][np.isfinite(p["values"])].flatten()
                      for p in pdata]
            finite = [f for f in finite if f.size]
            panel_finite[lt] = np.concatenate(finite) if finite else np.array([])

        all_vals = np.concatenate(list(panel_finite.values()))
        if not all_vals.size:
            print(f"  No matched patches for {metric}; skipping row.")
            continue

        # Symmetric diverging scale from the 90th percentile of the magnitude,
        # shared across the row. The pixel-level ratio has a heavy tail (pixels
        # where one model's RMSE is tiny), so a higher percentile would wash out
        # the structure; rare extreme pixels simply saturate to the end colours.
        lim = float(np.percentile(np.abs(all_vals), 90))
        vmin, vmax = -lim, lim

        row_axes = []
        norm = cmap = None
        for col, lt in enumerate(LEAD_TIMES_PLOT):
            ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())
            norm, cmap = _draw_improvement_map(ax, panel_data[lt], vmin, vmax)
            row_axes.append(ax)

            if row == 0:
                ax.set_title(LEAD_LABELS[lt], fontsize=15, weight="bold")
            if col == 0:
                ax.text(-0.04, 0.5, METRIC_ROW_LABELS[metric],
                        transform=ax.transAxes, rotation=90, va="center",
                        ha="center", fontsize=13, weight="bold")
            ax.text(0.02, 0.04,
                    f"Global mean: {np.mean(panel_finite[lt]):+.1f}%",
                    transform=ax.transAxes, fontsize=10,
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.75, pad=2), zorder=5)

        # One vertical colourbar per row, to the right of its three maps.
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=row_axes, orientation="vertical",
                            pad=0.015, fraction=0.02, aspect=18)
        cbar.set_label(f"Pangu {METRIC_LABELS[metric]}\nimprovement over IFS (%)",
                       fontsize=10, weight="bold")
        cbar.set_ticks(_symmetric_ticks(lim))
        cbar.ax.tick_params(labelsize=10)

    fig.suptitle(
        "Pangu-Weather vs IFS 2m temperature accuracy   "
        "(blue = Pangu more accurate, red = IFS more accurate)",
        fontsize=16, weight="bold", y=0.92)

    out = os.path.join(save_dir, "science_policy_pangu_vs_ifs_combined.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def main():
    """Build the combined 3x3 Pangu-vs-IFS accuracy map in figures/science_policy."""
    dirs = setup_directories()
    save_dir = _science_policy_output_dir(dirs)
    print(f"Writing science-policy figures to: {save_dir}\n")

    # Load Pangu + IFS patches once and reuse them for every metric/lead-time.
    pangu_apd, ifs_apd = load_pangu_ifs_data(dirs, "2m_temperature")
    modified_unified_global_maps(dirs, save_dir,
                                 pangu_apd=pangu_apd, ifs_apd=ifs_apd)

    print("\nDone. All figures are in:", save_dir)


if __name__ == "__main__":
    main()
