"""
Figures for the Environmental Research Letters revision of
"Tailoring machine learning weather predictions for local impacts".

Outputs are written to /Users/ohouck/globus/forecast_data/erl_figures/
on Mac, or to ``{dirs['root']}/erl_figures`` more generally.

Two figures are produced:

erl_fig1_summary_equity.png
    Single panel composite that opens the paper. A world map of 9 day Pangu
    2m temperature post-processing improvement (top), and a bar chart of
    average improvement by Köppen climate zone (bottom) annotated with the
    approximate share of global population living in each zone. The figure
    visually anchors the equity argument: the largest forecast gains land
    in tropical and arid zones, which together host roughly half the world
    population and most low and middle income countries.

erl_fig2_global_maps_unified.png
    Replacement for the existing four panel global maps figure. A 2x2 grid
    of pixel level RMSE percent improvement for Pangu 2m temperature and
    10m wind speed at 1 day and 9 day lead times, plotted on a unified
    diverging colour scale with larger font sizes for ERL's single column
    layout.

Run with::

    uv run python finetuning/erl_figures.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from matplotlib.collections import LineCollection
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from helper_funcs import setup_directories
from finetuning.figures_finetuning import (
    load_region_data,
    validate_non_overlapping_patches,
)
from finetuning.run_improvement_regression import build_pixel_level_dataframe


CLIMATE_ZONE_NAMES = {1: "Tropical", 2: "Arid", 3: "Temperate", 4: "Cold", 5: "Polar"}

# Approximate share of global land population by Köppen zone, derived from
# Beck et al. 2018 (Köppen-Geiger 1km climate maps) cross-tabulated against
# GPWv4 2020 population. These are rough order of magnitude figures used
# only for visual context in the equity figure.
CLIMATE_ZONE_POP_SHARE = {
    "Tropical":  0.40,
    "Arid":      0.20,
    "Temperate": 0.30,
    "Cold":      0.10,
    "Polar":     0.001,
}


def _erl_output_dir(dirs):
    """Return (and create) the output directory for ERL figures."""
    out_dir = os.path.join(dirs["root"], "erl_figures")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _model_kwargs():
    """Standard preferred-model configuration used throughout the paper."""
    return dict(
        nn_architecture="mlp",
        snapshot_ensemble=3,
        block_ensemble=False,
        block_holdout=1,
        subregion="6x6",
    )


def _compute_pixel_improvement(patch_data, variable, lead_time):
    """
    Compute per-patch arrays of pixel-level RMSE percent improvement.

    Parameters
    ----------
    patch_data : list of dict
        Output of ``load_region_data`` indexed at one lead time.
    variable : str
        Variable name (for example "2m_temperature").
    lead_time : int
        Lead time in hours.

    Returns
    -------
    list of dict
        Each dict contains 'lats', 'lons', 'values' (improvement percentage)
        and patch bounding box.
    """
    var_suffix = f"_lt{lead_time}h"
    out = []
    for patch in patch_data:
        ds = patch["ds"]
        gt = ds[f"{variable}_ground_truth{var_suffix}"]
        orig = ds[f"{variable}_original{var_suffix}"]
        corr = ds[f"{variable}_corrected{var_suffix}"]

        rmse_orig = np.sqrt(((orig - gt) ** 2).mean(dim="time"))
        rmse_corr = np.sqrt(((corr - gt) ** 2).mean(dim="time"))
        improvement = ((rmse_orig - rmse_corr) / rmse_orig * 100).values

        out.append({
            "lats": ds.latitude.values,
            "lons": ds.longitude.values,
            "values": improvement,
            "lat_min": patch["lat_min"],
            "lat_max": patch["lat_max"],
            "lon_min": patch["lon_min"],
            "lon_max": patch["lon_max"],
        })
    return out


def _draw_improvement_map(ax, patch_plot_data, vmin, vmax, cmap=plt.cm.RdBu,
                           draw_boxes=True):
    """
    Render a global pixel level improvement map on ``ax``.

    The colormap is centred at zero so that improvements (positive) and
    degradations (negative) are visually separated.
    """
    if vmin < 0 < vmax:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    else:
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3, zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="black", zorder=2)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.3,
                   edgecolor="gray", zorder=2)

    for patch in patch_plot_data:
        ax.pcolormesh(
            patch["lons"], patch["lats"], patch["values"],
            transform=ccrs.PlateCarree(),
            cmap=cmap, norm=norm, shading="nearest", zorder=1,
        )

    if draw_boxes:
        segs = []
        for patch in patch_plot_data:
            lon_min, lon_max = patch["lon_min"], patch["lon_max"]
            lat_min, lat_max = patch["lat_min"], patch["lat_max"]
            segs.extend([
                [(lon_min, lat_min), (lon_max, lat_min)],
                [(lon_max, lat_min), (lon_max, lat_max)],
                [(lon_max, lat_max), (lon_min, lat_max)],
                [(lon_min, lat_max), (lon_min, lat_min)],
            ])
        ax.add_collection(LineCollection(
            segs, colors="black", linewidths=0.4, alpha=0.9,
            transform=ccrs.PlateCarree(), zorder=2,
        ))

    return norm, cmap


def _zone_means(dirs, variable, lead_time, **model_kwargs):
    """
    Compute the mean RMSE percent improvement in each Köppen climate zone.

    Reuses ``build_pixel_level_dataframe`` from ``run_improvement_regression``
    so that the pixel-level improvement and climate-zone lookups stay in a
    single place.

    Returns
    -------
    dict
        Mapping from zone name (for example "Tropical") to mean improvement.
    """
    df = build_pixel_level_dataframe(
        dirs=dirs, model="pangu", variable=variable,
        lead_times=[lead_time], **model_kwargs,
    )
    if df is None or len(df) == 0:
        return {name: np.nan for name in CLIMATE_ZONE_NAMES.values()}

    out = {}
    for code, name in CLIMATE_ZONE_NAMES.items():
        mask = df["climate_zone"] == code
        out[name] = float(df.loc[mask, "improvement_pct"].mean()) if mask.any() else np.nan
    return out


def make_figure_1_summary_equity(dirs, save_dir):
    """
    Build the lead summary figure: 9 day temperature improvement map plus a
    Köppen-zone bar chart annotated with global population share.
    """
    print("\n=== Building ERL Figure 1: summary / equity ===")

    model_kwargs = _model_kwargs()
    variable = "2m_temperature"
    lead_time = 216

    print("Loading patch data for the map panel...")
    all_patch_data = load_region_data(
        dirs=dirs, model="pangu", variable=variable, regions=None,
        lead_times=[lead_time], sdor_da=None, **model_kwargs,
    )
    if all_patch_data is None or not all_patch_data.get(lead_time):
        print("  No patch data available; skipping figure 1.")
        return

    patches = all_patch_data[lead_time]
    validate_non_overlapping_patches(patches)
    patch_plot_data = _compute_pixel_improvement(patches, variable, lead_time)

    all_vals = np.concatenate([
        p["values"][~np.isnan(p["values"])].flatten() for p in patch_plot_data
    ])
    vmin = float(np.nanpercentile(all_vals, 1))
    vmax = float(np.nanpercentile(all_vals, 99))

    print("Computing zone means for the bar panel...")
    zone_means = _zone_means(dirs, variable, lead_time, **model_kwargs)

    fig = plt.figure(figsize=(13, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1.6], hspace=0.32)

    ax_map = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
    norm, cmap = _draw_improvement_map(ax_map, patch_plot_data, vmin, vmax)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_map, orientation="horizontal",
                        pad=0.07, shrink=0.65)
    cbar.set_label("9 day 2m temperature RMSE improvement (%)",
                   fontsize=14, weight="bold")
    cbar.ax.tick_params(labelsize=12)
    ax_map.set_title(
        "Where local post-processing helps most:\n"
        "tropical and topographically complex regions",
        fontsize=15, weight="bold", pad=12,
    )

    ax_bar = fig.add_subplot(gs[1])
    zone_order = ["Tropical", "Arid", "Temperate", "Cold", "Polar"]
    means = [zone_means.get(z, np.nan) for z in zone_order]
    pop_shares = [CLIMATE_ZONE_POP_SHARE[z] for z in zone_order]

    bar_colors = ["#1f6f3f", "#bc8b3a", "#3a6fbc", "#7e7e7e", "#cccccc"]
    bars = ax_bar.bar(zone_order, means, color=bar_colors, edgecolor="black")
    ax_bar.axhline(0, color="gray", linewidth=0.8)
    ax_bar.set_ylabel("Mean RMSE improvement (%)", fontsize=13)
    ax_bar.set_xlabel("Köppen climate zone", fontsize=13)
    ax_bar.tick_params(axis="both", labelsize=12)
    ax_bar.set_title(
        "Improvement is largest in zones that host most LMIC populations",
        fontsize=14, weight="bold", pad=8,
    )
    ax_bar.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, mean, share in zip(bars, means, pop_shares):
        if np.isnan(mean):
            continue
        ax_bar.annotate(
            f"{mean:.1f}%\n~{share*100:.0f}% of\nworld pop.",
            xy=(bar.get_x() + bar.get_width() / 2, mean),
            xytext=(0, 6 if mean >= 0 else -36),
            textcoords="offset points",
            ha="center", va="bottom" if mean >= 0 else "top",
            fontsize=10,
        )

    out_path = os.path.join(save_dir, "erl_fig1_summary_equity.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def make_figure_2_unified_global_maps(dirs, save_dir):
    """
    Replacement for the existing 2x2 global maps figure.

    A 2x2 grid: rows are lead times (1 day top, 9 day bottom), columns are
    variables (2m temperature left, 10m wind speed right). All four panels
    share the same diverging colour scale per variable so that magnitudes
    can be compared directly.
    """
    print("\n=== Building ERL Figure 2: unified global maps ===")
    model_kwargs = _model_kwargs()
    variables = ["2m_temperature", "10m_wind_speed"]
    lead_times = [24, 216]

    print("Loading patch data for all panels...")
    panel_data = {}
    panel_extents = {}
    for variable in variables:
        all_patch_data = load_region_data(
            dirs=dirs, model="pangu", variable=variable, regions=None,
            lead_times=lead_times, sdor_da=None, **model_kwargs,
        )
        if all_patch_data is None:
            print(f"  No data for {variable}; skipping unified figure.")
            return
        panel_data[variable] = {}
        col_vals = []
        for lt in lead_times:
            patches = all_patch_data[lt]
            plot_data = _compute_pixel_improvement(patches, variable, lt)
            panel_data[variable][lt] = plot_data
            for p in plot_data:
                col_vals.append(p["values"][~np.isnan(p["values"])].flatten())
        all_vals = np.concatenate(col_vals)
        panel_extents[variable] = (
            float(np.nanpercentile(all_vals, 1)),
            float(np.nanpercentile(all_vals, 99)),
        )

    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 2, hspace=0.05, wspace=0.06)

    var_titles = {
        "2m_temperature": "2m temperature",
        "10m_wind_speed": "10m wind speed",
    }
    lead_labels = {24: "1 day", 216: "9 day"}

    norms = {}
    for col, variable in enumerate(variables):
        vmin, vmax = panel_extents[variable]
        for row, lt in enumerate(lead_times):
            ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())
            plot_data = panel_data[variable][lt]
            norm, cmap = _draw_improvement_map(ax, plot_data, vmin, vmax)
            norms[(row, col)] = (norm, cmap)
            if row == 0:
                ax.set_title(f"{var_titles[variable]}", fontsize=14, weight="bold")
            ax.text(-0.03, 0.5, lead_labels[lt],
                    transform=ax.transAxes, rotation=90, va="center",
                    fontsize=13, weight="bold")

    norm_t, cmap_t = norms[(0, 0)]
    norm_w, cmap_w = norms[(0, 1)]

    cbar_ax_t = fig.add_axes([0.13, 0.06, 0.34, 0.02])
    sm_t = plt.cm.ScalarMappable(norm=norm_t, cmap=cmap_t)
    sm_t.set_array([])
    cbar_t = plt.colorbar(sm_t, cax=cbar_ax_t, orientation="horizontal")
    cbar_t.set_label("2m temperature RMSE improvement (%)",
                     fontsize=12, weight="bold")
    cbar_t.ax.tick_params(labelsize=11)

    cbar_ax_w = fig.add_axes([0.55, 0.06, 0.34, 0.02])
    sm_w = plt.cm.ScalarMappable(norm=norm_w, cmap=cmap_w)
    sm_w.set_array([])
    cbar_w = plt.colorbar(sm_w, cax=cbar_ax_w, orientation="horizontal")
    cbar_w.set_label("10m wind speed RMSE improvement (%)",
                     fontsize=12, weight="bold")
    cbar_w.ax.tick_params(labelsize=11)

    fig.suptitle(
        "Pangu-Weather post-processing improvement, 1 day (top) and 9 day (bottom) lead times",
        fontsize=15, weight="bold", y=0.97,
    )

    out_path = os.path.join(save_dir, "erl_fig2_global_maps_unified.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    """Build all ERL figures and save them under {root}/erl_figures."""
    dirs = setup_directories()
    save_dir = _erl_output_dir(dirs)
    print(f"Writing ERL figures to: {save_dir}")

    make_figure_1_summary_equity(dirs, save_dir)
    make_figure_2_unified_global_maps(dirs, save_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
