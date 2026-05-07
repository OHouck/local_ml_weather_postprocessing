"""
Single script that produces every figure in the ERL paper.

All outputs are written to ``{dirs['root']}/erl_figures/``.

Main text figures
-----------------
erl_fig1_summary_equity.png
    World map of 9 day Pangu 2m temperature improvement plus a Köppen-zone
    bar chart annotated with approximate global population share. Anchors the
    equity argument visually.

erl_fig2_global_maps_unified.png
    2x2 grid of pixel-level RMSE percent improvement for Pangu 2m temperature
    and 10m wind speed at 1 day and 9 day lead times on a unified diverging
    colour scale with global mean annotation on each panel.

erl_fig3_binscatter_equator_pangu.png
    Binscatter: original RMSE and improvement vs distance from equator (Pangu).

erl_fig4_binscatter_sdor_pangu.png
    Binscatter: original RMSE and improvement vs SDOR (Pangu).

erl_fig5_arch_comparison_temperature.png
erl_fig5_arch_comparison_wind.png
    Architecture and training-procedure comparison bar charts.

Appendix figures
----------------
erl_appA1_model_compare_boxplot.png
erl_appA2_pangu_5day_maps.png
erl_appA3_ifs_maps.png
erl_appA4_ifs_binscatter_equator.png
erl_appA5_ifs_binscatter_sdor.png
erl_appA6_region_size_mlp.png  (temperature + wind in one call each)
erl_appA7_region_size_unet.png
erl_appA8_arch_eval_regions.png
erl_appA9_income_group_map.png
    World map shaded by World Bank 2024 (FY25) income group, used as the
    spatial key for figure 1.

Run with::

    uv run python finetuning/erl_figures.py
"""

import os
import sys
import shutil
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from helper_funcs import setup_directories, sample_continent_patches
from finetuning.figures_finetuning import (
    load_region_data,
    validate_non_overlapping_patches,
    lead_time_compare_binscatter,
    model_compare_boxplot,
    map_global_improvements,
    map_arch_exeriment_regions,
    plot_arch_experiment_results,
    generate_subregion_comparison_plots,
)
from finetuning.run_improvement_regression import build_pixel_level_dataframe


# ---------------------------------------------------------------------------
# Publication-ready matplotlib defaults applied to every figure in this file
# ---------------------------------------------------------------------------
mpl.rcParams.update({
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
})


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Lead times used throughout the paper
LEAD_TIMES = [24, 120, 216]

TRAIN_START, TRAIN_END = "2018-01-01", "2021-12-31"
TEST_START,  TEST_END  = "2022-01-01", "2022-12-31"

# World Bank income classification (figure 1 + appendix A9).
# OGHIST.xlsx ships the full historical record; FY25 is the classification
# announced in July 2024 (based on 2023 GNI), i.e. the "2024" income groups.
WB_OGHIST_URL  = ("https://datacatalogfiles.worldbank.org/"
                  "ddh-published/0037712/DR0090754/OGHIST.xlsx")
WB_FISCAL_YEAR = "FY25"

# Natural Earth 1:50m country polygons used to map pixels -> country -> income.
NE_COUNTRIES_URL = ("https://naciscdn.org/naturalearth/50m/cultural/"
                    "ne_50m_admin_0_countries.zip")

# OGHIST one-letter codes -> aggregated income group. The paper collapses the
# three non-high-income categories so the comparison is between high-income
# and low-and-middle-income (LMIC) countries — the standard development split.
WB_INCOME_CODE_MAP = {
    "L":  "Low and middle income",
    "LM": "Low and middle income",
    "UM": "Low and middle income",
    "H":  "High income",
}

# Dict key order is the canonical bar / legend order — derive
# INCOME_GROUP_ORDER from it so the two can't drift apart.
INCOME_GROUP_COLORS = {
    "Low and middle income": "#d7191c",
    "High income":           "#2c7bb6",
}
INCOME_GROUP_ORDER = list(INCOME_GROUP_COLORS)


def _erl_output_dir(dirs):
    """Return (and create) the unified output directory for all ERL figures."""
    out_dir = os.path.join(dirs["root"], "erl_figures")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _model_kwargs():
    """Preferred-model configuration used throughout the paper."""
    return dict(
        nn_architecture="mlp",
        snapshot_ensemble=3,
        block_ensemble=False,
        block_holdout=1,
        subregion="6x6",
    )


def _cbar_ticks(vmin, vmax):
    """
    Return a list of tick values that are guaranteed inside [vmin, vmax].

    If vmin is negative, include one labelled negative value (first integer
    >= vmin via ceiling). Always include 0 plus round positive values.
    """
    ticks = []
    if vmin < 0:
        ticks.append(int(np.ceil(vmin)))
    ticks += [t for t in [0, 5, 10, 15, 20, 25, 30] if vmin <= t <= vmax]
    return ticks


# ---------------------------------------------------------------------------
# Low-level map helpers
# ---------------------------------------------------------------------------

def _compute_pixel_improvement(patch_data, variable, lead_time):
    """
    Return per-patch dicts with pixel-level RMSE percent improvement arrays.

    Parameters
    ----------
    patch_data : list of dict
        Loaded patches from ``load_region_data`` at one lead time.
    variable : str
    lead_time : int
        Lead time in hours.

    Returns
    -------
    list of dict with keys 'lats', 'lons', 'values', lat/lon bounding box.
    """
    var_suffix = f"_lt{lead_time}h"
    out = []
    for patch in patch_data:
        ds = patch["ds"]
        gt   = ds[f"{variable}_ground_truth{var_suffix}"]
        orig = ds[f"{variable}_original{var_suffix}"]
        corr = ds[f"{variable}_corrected{var_suffix}"]
        rmse_orig = np.sqrt(((orig - gt) ** 2).mean(dim="time"))
        rmse_corr = np.sqrt(((corr - gt) ** 2).mean(dim="time"))
        out.append({
            "lats":    ds.latitude.values,
            "lons":    ds.longitude.values,
            "values":  ((rmse_orig - rmse_corr) / rmse_orig * 100).values,
            "lat_min": patch["lat_min"],
            "lat_max": patch["lat_max"],
            "lon_min": patch["lon_min"],
            "lon_max": patch["lon_max"],
        })
    return out


def _draw_improvement_map(ax, patch_plot_data, vmin, vmax,
                           cmap=plt.cm.RdBu, draw_boxes=True,
                           center_at_zero=False):
    """
    Render a pixel-level improvement map on a cartopy axes.

    Returns (norm, cmap) for attaching a colourbar.
    """
    # When all values are positive, vmin >= 0 would bypass TwoSlopeNorm and
    # map the least-improved pixels to red. A tiny negative offset anchors
    # the diverging scale at 0 without affecting visible data colors.
    norm_vmin = min(vmin, -1e-3) if center_at_zero else vmin
    norm = (TwoSlopeNorm(vmin=norm_vmin, vcenter=0, vmax=vmax)
            if norm_vmin < 0 < vmax else plt.Normalize(vmin=norm_vmin, vmax=vmax))

    ax.set_global()
    ax.add_feature(cfeature.LAND,      facecolor="lightgray", alpha=0.3, zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor="white",      zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="black", zorder=2)
    ax.add_feature(cfeature.BORDERS,   linestyle=":", linewidth=0.3,
                   edgecolor="gray", zorder=2)

    for patch in patch_plot_data:
        ax.pcolormesh(patch["lons"], patch["lats"], patch["values"],
                      transform=ccrs.PlateCarree(),
                      cmap=cmap, norm=norm, shading="nearest", zorder=1)

    if draw_boxes:
        segs = []
        for patch in patch_plot_data:
            lo, hi = patch["lon_min"], patch["lon_max"]
            la, ha = patch["lat_min"], patch["lat_max"]
            segs.extend([[(lo, la), (hi, la)], [(hi, la), (hi, ha)],
                          [(hi, ha), (lo, ha)], [(lo, ha), (lo, la)]])
        ax.add_collection(LineCollection(segs, colors="black", linewidths=0.4,
                                         alpha=0.9, transform=ccrs.PlateCarree(),
                                         zorder=2))
    return norm, cmap


def _attach_cbar(fig, sm, ax_or_list, label, vmin, vmax, orientation="horizontal",
                 pad=0.04, shrink=0.85, aspect=40):
    """Attach a labelled, tick-set colourbar to ax_or_list and return it."""
    cbar = fig.colorbar(sm, ax=ax_or_list, orientation=orientation,
                        pad=pad, shrink=shrink, aspect=aspect)
    cbar.set_label(label, fontsize=13, weight="bold")
    cbar.set_ticks(_cbar_ticks(vmin, vmax))
    cbar.ax.tick_params(labelsize=11)
    return cbar


def _post_process_binscatter(fig, model_label="Post-processing model",
                              baseline_label="Mean bias correction"):
    """
    Apply publication styling to a figure returned by lead_time_compare_binscatter.

    - Removes the auto-generated suptitle
    - Renames legend entries from internal shorthand to reader-friendly text
    - Bumps axis font sizes
    """
    fig.suptitle("")          # caption handles the description

    for ax in fig.axes:
        ax.set_xlabel(ax.get_xlabel(), fontsize=13)
        ax.set_ylabel(ax.get_ylabel(), fontsize=13)
        ax.title.set_fontsize(13)
        leg = ax.get_legend()
        if leg is None:
            continue
        # Rename method-type legend handles
        for text in leg.get_texts():
            t = text.get_text()
            if "Main model" in t or "filled" in t.lower():
                text.set_text(model_label)
            elif "Mean bias" in t or "hollow" in t.lower():
                text.set_text(baseline_label)
        leg.get_title().set_fontsize(12)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 1 — improvement by World Bank income group
# ---------------------------------------------------------------------------

def _income_cache_dir(dirs):
    """Subdirectory under dirs['raw'] that caches the WB + Natural Earth files."""
    p = os.path.join(dirs["raw"], "income_groups")
    os.makedirs(p, exist_ok=True)
    return p


def _download_if_missing(url, dest, label, timeout=60):
    """Download ``url`` to ``dest`` only if ``dest`` does not exist."""
    if os.path.exists(dest):
        return
    print(f"  Downloading {label} -> {dest}")
    with urllib.request.urlopen(url, timeout=timeout) as response, \
            open(dest, "wb") as fh:
        shutil.copyfileobj(response, fh)


def _load_wb_income_classification(dirs, fiscal_year=WB_FISCAL_YEAR):
    """
    Read the World Bank historical income classification.

    Parameters
    ----------
    dirs : dict
        Output of ``setup_directories()``.
    fiscal_year : str
        Column header in the OGHIST sheet (e.g. ``"FY25"`` for the 2024
        classification announced in July 2024).

    Returns
    -------
    dict
        Maps ISO3 code -> aggregated income group name, either
        ``"Low and middle income"`` (collapses L / LM / UM) or
        ``"High income"``. Countries without a classification in the
        requested year are omitted.
    """
    cache_dir = _income_cache_dir(dirs)
    xlsx_path = os.path.join(cache_dir, "OGHIST.xlsx")
    _download_if_missing(WB_OGHIST_URL, xlsx_path, "World Bank OGHIST.xlsx")

    raw = pd.read_excel(xlsx_path, sheet_name="Country Analytical History",
                        header=None)
    fy_row = raw.iloc[4].astype(str).tolist()
    if fiscal_year not in fy_row:
        raise ValueError(
            f"Fiscal year {fiscal_year!r} not found in OGHIST.xlsx; "
            f"available: {[c for c in fy_row if c.startswith('FY')]}")
    fy_col = fy_row.index(fiscal_year)

    out = {}
    for _, row in raw.iloc[11:].iterrows():
        iso3 = str(row.iloc[0]).strip()
        code = str(row.iloc[fy_col]).strip()
        if iso3 in ("nan", "") or code not in WB_INCOME_CODE_MAP:
            continue
        out[iso3] = WB_INCOME_CODE_MAP[code]
    return out


def _load_country_geometries(dirs):
    """
    Load Natural Earth 1:50m country polygons, downloading + extracting once.

    Returns
    -------
    geopandas.GeoDataFrame
        EPSG:4326 with columns ``['iso3', 'name', 'geometry']``. Disputed and
        indeterminate-sovereignty rows are removed so pixels there will fail
        the spatial join and be dropped from analysis.
    """
    import geopandas as gpd
    cache_dir = _income_cache_dir(dirs)
    zip_path = os.path.join(cache_dir, "ne_50m_admin_0_countries.zip")
    shp_path = os.path.join(cache_dir, "ne_50m_admin_0_countries.shp")

    if not os.path.exists(shp_path):
        _download_if_missing(NE_COUNTRIES_URL, zip_path,
                             "Natural Earth admin_0 countries shapefile")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(cache_dir)

    gdf = gpd.read_file(shp_path)
    gdf = gdf[~gdf["TYPE"].isin(["Disputed", "Indeterminate"])].copy()
    gdf = gdf.rename(columns={"ADM0_A3": "iso3", "NAME": "name"})
    return gdf[["iso3", "name", "geometry"]].reset_index(drop=True)


def _load_income_geodataframe(dirs, fiscal_year=WB_FISCAL_YEAR):
    """
    Country polygons joined to their World Bank income group.

    Returns a GeoDataFrame with ``income_group`` attached. Countries without a
    classification in the requested fiscal year (e.g. some dependencies) are
    dropped — pixels there will not match any polygon and will be excluded
    from figure 1.
    """
    countries = _load_country_geometries(dirs)
    iso3_to_income = _load_wb_income_classification(dirs, fiscal_year)
    countries["income_group"] = countries["iso3"].map(iso3_to_income)
    return countries.dropna(subset=["income_group"]).reset_index(drop=True)


def _classify_pixels_by_income(lats, lons, income_gdf):
    """
    Map pixel (lat, lon) arrays to their World Bank income group.

    Parameters
    ----------
    lats, lons : 1D array-like
        Pixel coordinates. ``lons`` may be in either ``[-180, 180]`` or
        ``[0, 360]`` convention; converted internally.
    income_gdf : geopandas.GeoDataFrame
        Output of ``_load_income_geodataframe``.

    Returns
    -------
    numpy.ndarray of dtype object
        ``income_group`` string per pixel; ``None`` for pixels outside any
        classified country (ocean, Antarctica, disputed territory).
    """
    import geopandas as gpd
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    lons180 = np.where(lons > 180, lons - 360, lons)

    points = gpd.GeoDataFrame(
        {"_idx": np.arange(len(lats))},
        geometry=gpd.points_from_xy(lons180, lats),
        crs="EPSG:4326",
    )
    # Nearest-polygon match capped at one ERA5 cell (0.25°). The 1:50m
    # shapefile simplifies coastlines enough that some coastal pixels (e.g.
    # Mumbai's islands) sit a few km offshore of their country polygon; one
    # cell absorbs them without pulling open-ocean pixels onto small island
    # nations.
    joined = gpd.sjoin_nearest(
        points, income_gdf[["income_group", "geometry"]],
        how="left", max_distance=0.25,
    )
    joined = joined.drop_duplicates(subset="_idx", keep="first").sort_values("_idx")
    out = np.full(len(lats), None, dtype=object)
    matched = joined["income_group"].notna().values
    out[joined.loc[matched, "_idx"].values] = (
        joined.loc[matched, "income_group"].values)
    return out


def _build_pixel_income_dataframe(dirs, variable, lead_times, income_gdf):
    """
    Pixel-level RMSE percent improvement tagged with World Bank income group.

    Loads all global Pangu patches at the requested lead times, computes the
    per-pixel percent improvement, and assigns each pixel an income group via
    a spatial join. Pixels with no matching country are dropped.

    Returns a long DataFrame with columns ``lead_time``, ``improvement_pct``,
    ``latitude``, ``longitude``, ``income_group``, ``patch_id`` — or ``None`` 
    if no patch data is available.
    """
    mkw = _model_kwargs()
    apd = load_region_data(
        dirs=dirs, model="pangu", variable=variable, regions=None,
        lead_times=lead_times, sdor_da=None, **mkw,
    )
    if apd is None:
        return None

    rows = []
    patch_id = 0
    for lt in lead_times:
        patches = apd.get(lt) or []
        if not patches:
            continue
        for p in _compute_pixel_improvement(patches, variable, lt):
            lon_grid, lat_grid = np.meshgrid(p["lons"], p["lats"])
            vals = p["values"]
            valid = ~np.isnan(vals)
            if not valid.any():
                patch_id += 1
                continue
            rows.append(pd.DataFrame({
                "lead_time":       lt,
                "improvement_pct": vals[valid],
                "latitude":        lat_grid[valid],
                "longitude":       lon_grid[valid],
                "patch_id":        patch_id,
            }))
            patch_id += 1
    if not rows:
        return None

    df = pd.concat(rows, ignore_index=True)
    # One spatial join for the whole variable instead of one per patch — sjoin
    # has fixed per-call overhead that dominated when looping ~thousands of
    # small patches.
    df["income_group"] = _classify_pixels_by_income(
        df["latitude"].to_numpy(), df["longitude"].to_numpy(), income_gdf)
    df = df.dropna(subset=["income_group"]).reset_index(drop=True)
    return df if len(df) else None


def make_figure_1_summary_equity(dirs, save_dir):
    """
    Figure 1: post-processing RMSE percent improvement by World Bank income
    group at 1, 5, and 9 day lead times.

    Two stacked panels (2m temperature on top, 10m wind speed on bottom).
    Each panel has three groups of two bars (low-and-middle-income vs
    high-income); bar height is the pixel-level mean RMSE percent improvement
    and whiskers are one pixel-level standard deviation. Income groups follow
    the World Bank's 2024 classification (FY25, announced July 2024) with the
    L / LM / UM categories collapsed into "Low and middle income".
    """
    print("\n=== Figure 1: improvement by World Bank income group ===")
    income_gdf = _load_income_geodataframe(dirs)

    variable_specs = [
        ("2m_temperature", "2m temperature"),
        ("10m_wind_speed", "10m wind speed"),
    ]
    lead_labels = {24: "1 day", 120: "5 day", 216: "9 day"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    bar_width = 0.25
    x = np.arange(len(LEAD_TIMES))

    for ax, (variable, var_label) in zip(axes, variable_specs):
        df = _build_pixel_income_dataframe(dirs, variable, LEAD_TIMES, income_gdf)
        if df is None or df.empty:
            print(f"  No pixel data for {variable}; skipping panel.")
            ax.set_axis_off()
            continue

        # Compute pixel-level mean and std 
        stats = (df.groupby(["income_group", "lead_time"])["improvement_pct"]
               .agg(["mean", "std"]))

        for i, ig in enumerate(INCOME_GROUP_ORDER):
            grp = (stats.loc[ig].reindex(LEAD_TIMES)
                   if ig in stats.index.get_level_values(0)
                   else pd.DataFrame(index=LEAD_TIMES, columns=["mean", "std"]))
            offset = (i - (len(INCOME_GROUP_ORDER) - 1) / 2) * bar_width
            ax.bar(
                x + offset, grp["mean"].to_numpy(), width=bar_width,
                yerr=grp["std"].to_numpy(),
                color=INCOME_GROUP_COLORS[ig],
                edgecolor="#333333", linewidth=0.6,
                error_kw=dict(elinewidth=0.8, capsize=3, capthick=0.8,
                              ecolor="#333333"),
                label=ig,
                zorder=3,
            )

        ax.axhline(0, color="gray", linewidth=0.7, zorder=1)
        ax.set_ylabel("RMSE improvement (%)")
        ax.set_title(var_label, fontsize=14, weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([lead_labels[lt] for lt in LEAD_TIMES])
        ax.set_xlabel("Lead time")
        ax.set_ylim(bottom=0)

    # Add centered title and legend above the plots
    fig.suptitle(
        "Forecast Improvement from Post-Processing Model by Income Group",
        fontsize=14, weight="bold", y=0.98
    )
    
    # Create legend below the suptitle, spanning both axes
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, ncol=2, loc="upper center",
        bbox_to_anchor=(0.5, 0.93), frameon=False, fontsize=11
    )
    
    # Add whiskers annotation in bottom right corner
    fig.text(
        0.98, 0.05, "whiskers = ±1 std",
        ha="right", va="bottom", fontsize=10, style="italic"
    )
    
    fig.tight_layout(rect=[0, 0.08, 1, 0.91])
    out = os.path.join(save_dir, "erl_fig1_summary_equity.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figure 2 — unified global improvement maps
# ---------------------------------------------------------------------------

def make_figure_2_unified_global_maps(dirs, save_dir):
    """
    2 × 2 grid of pixel-level RMSE % improvement.

    Rows: 1 day (top), 9 day (bottom).
    Columns: 2m temperature (left), 10m wind speed (right).
    Unified diverging colour scale per column with global mean annotation.
    """
    print("\n=== Figure 2: unified global maps ===")
    mkw       = _model_kwargs()
    variables = ["2m_temperature", "10m_wind_speed"]
    lead_times_plot = [24, 216]

    panel_data    = {}
    panel_extents = {}
    col_axes_all  = {}

    for variable in variables:
        apd = load_region_data(
            dirs=dirs, model="pangu", variable=variable, regions=None,
            lead_times=LEAD_TIMES, sdor_da=None, **mkw,
        )
        if apd is None:
            print(f"  No data for {variable}; skipping.")
            return
        panel_data[variable] = {}
        col_vals = []
        for lt in lead_times_plot:
            pdata = _compute_pixel_improvement(apd[lt], variable, lt)
            panel_data[variable][lt] = pdata
            for p in pdata:
                col_vals.append(p["values"][~np.isnan(p["values"])].flatten())
        all_v = np.concatenate(col_vals)
        p1, p99 = np.percentile(all_v, [1, 99])
        panel_extents[variable] = (float(p1), float(p99))

    var_titles   = {"2m_temperature": "2m temperature",
                    "10m_wind_speed": "10m wind speed"}
    lead_labels  = {24: "1 day", 216: "9 day"}

    fig  = plt.figure(figsize=(16, 9))
    gs   = gridspec.GridSpec(2, 2, hspace=0.05, wspace=0.06)
    col_norms = {}

    for col, variable in enumerate(variables):
        vmin, vmax = panel_extents[variable]
        col_axes_all[variable] = []
        for row, lt in enumerate(lead_times_plot):
            ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())
            pdata = panel_data[variable][lt]
            norm, cmap = _draw_improvement_map(ax, pdata, vmin, vmax,
                                               center_at_zero=True)
            col_axes_all[variable].append(ax)
            col_norms[variable] = (norm, cmap)
            if row == 0:
                ax.set_title(var_titles[variable], fontsize=14, weight="bold")
            ax.text(-0.03, 0.5, lead_labels[lt],
                    transform=ax.transAxes, rotation=90, va="center",
                    fontsize=13, weight="bold")
            panel_vals = np.concatenate(
                [p["values"][~np.isnan(p["values"])].flatten() for p in pdata])
            ax.text(0.02, 0.04,
                    f"Global mean: {np.nanmean(panel_vals):.1f}%",
                    transform=ax.transAxes, fontsize=11,
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.75, pad=3), zorder=5)

    for variable in variables:
        norm, cmap = col_norms[variable]
        vmin, vmax = panel_extents[variable]
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        label = ("2m temperature" if variable == "2m_temperature"
                 else "10m wind speed")
        _attach_cbar(fig, sm, col_axes_all[variable],
                     f"{label} RMSE improvement (%)", max(vmin, 0), vmax,
                     pad=0.04, shrink=0.85, aspect=40)

    fig.suptitle(
        "Pangu-Weather post-processing improvement "
        "(1 day top, 9 day bottom)",
        fontsize=15, weight="bold", y=0.99)

    out = os.path.join(save_dir, "erl_fig2_global_maps_unified.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Figures 3 & 4 — binscatter (equator distance and SDOR)
# ---------------------------------------------------------------------------

def _make_binscatter(dirs, save_dir, x_metric, out_name, model="pangu"):
    """
    Call ``lead_time_compare_binscatter``, apply publication styling, save.

    Parameters
    ----------
    x_metric : "equator_distance" or "sdor"
    out_name : output filename inside save_dir
    """
    mkw = _model_kwargs()
    fig = lead_time_compare_binscatter(
        dirs=dirs, model=model, x_metric=x_metric,
        include_mean_bias_correction_baseline=True,
        train_start=TRAIN_START, train_end=TRAIN_END,
        test_start=TEST_START,   test_end=TEST_END,
        save_dir=None,           # we handle saving ourselves
        **mkw,
    )
    if fig is None:
        print(f"  No figure returned for {out_name}; skipping.")
        return

    _post_process_binscatter(
        fig,
        model_label="Post-processing model",
        baseline_label="Mean bias correction",
    )
    out = os.path.join(save_dir, out_name)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def make_figure_3_binscatter_equator(dirs, save_dir):
    """Binscatter: RMSE and improvement vs distance from equator (Pangu)."""
    print("\n=== Figure 3: binscatter vs equator distance (Pangu) ===")
    _make_binscatter(dirs, save_dir, "equator_distance",
                     "erl_fig3_binscatter_equator_pangu.png")


def make_figure_4_binscatter_sdor(dirs, save_dir):
    """Binscatter: RMSE and improvement vs SDOR (Pangu)."""
    print("\n=== Figure 4: binscatter vs SDOR (Pangu) ===")
    _make_binscatter(dirs, save_dir, "sdor",
                     "erl_fig4_binscatter_sdor_pangu.png")


# ---------------------------------------------------------------------------
# Figure 5 — architecture comparison
# ---------------------------------------------------------------------------

def make_figure_5_arch_comparison(dirs, save_dir):
    """
    Architecture and training-procedure comparison bar charts.

    One chart per variable (temperature, wind), saved as
    erl_fig5_arch_comparison_temperature.png and
    erl_fig5_arch_comparison_wind.png.

    The function uses ``plot_arch_experiment_results`` from figures_finetuning
    with save_dir routed to erl_figures/ and then renames to the ERL scheme.
    """
    print("\n=== Figure 5: architecture comparison ===")
    eval_cells = sample_continent_patches(
        dirs["processed"], fraction=0.05, seed=42, split="eval")

    var_configs = [
        {"label": "2m Temperature",
         "training_vars": ["2m_temperature"],
         "output_vars":   ["2m_temperature"],
         "out_name": "erl_fig5_arch_comparison_temperature.png"},
        {"label": "10m Wind Speed",
         "training_vars": ["10m_wind_speed"],
         "output_vars":   ["10m_wind_speed"],
         "out_name": "erl_fig5_arch_comparison_wind.png"},
    ]

    for vc in var_configs:
        plot_arch_experiment_results(
            dirs=dirs, label=vc["label"],
            training_vars=vc["training_vars"],
            output_vars=vc["output_vars"],
            model="pangu", subregion="6x6",
            train_start=TRAIN_START, train_end=TRAIN_END,
            test_start=TEST_START,   test_end=TEST_END,
            eval_cells=eval_cells,
            save_dir=save_dir,
        )
        # plot_arch_experiment_results saves with its own naming convention;
        # rename to the ERL scheme
        label_slug = vc["label"].lower().replace(" ", "_")
        src = os.path.join(
            save_dir, f"arch_comparison_{label_slug}_global_eval_6x6.png")
        dst = os.path.join(save_dir, vc["out_name"])
        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"  Saved: {dst}")
        else:
            # filename might differ slightly; fall back to searching
            candidates = [f for f in os.listdir(save_dir)
                          if f.startswith("arch_comparison") and f.endswith(".png")]
            if candidates:
                shutil.move(os.path.join(save_dir, candidates[-1]), dst)
                print(f"  Saved (renamed): {dst}")
            else:
                print(f"  Warning: could not locate output for {vc['label']}")


# ---------------------------------------------------------------------------
# Appendix A1 — IFS vs Pangu boxplot
# ---------------------------------------------------------------------------

def make_app_a1_model_compare_boxplot(dirs, save_dir):
    """Grouped boxplot comparing IFS and Pangu post-processing improvement."""
    print("\n=== Appendix A1: IFS vs Pangu boxplot ===")
    mkw = _model_kwargs()
    fig = model_compare_boxplot(
        dirs=dirs, models=["pangu", "ifs"],
        variables=["2m_temperature", "10m_wind_speed"],
        train_start=TRAIN_START, train_end=TRAIN_END,
        test_start=TEST_START,   test_end=TEST_END,
        save_dir=None,
        **mkw,
    )
    if fig is None:
        print("  No figure returned; skipping.")
        return
    fig.suptitle("")
    fig.tight_layout()
    out = os.path.join(save_dir, "erl_appA1_model_compare_boxplot.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Appendix A2 — Pangu 5-day maps
# ---------------------------------------------------------------------------

def make_app_a2_pangu_5day_maps(dirs, save_dir):
    """Pixel-level improvement maps for Pangu at the 5 day lead time."""
    print("\n=== Appendix A2: Pangu 5-day maps ===")
    mkw = _model_kwargs()

    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, 2, wspace=0.06)

    for col, variable in enumerate(["2m_temperature", "10m_wind_speed"]):
        apd = load_region_data(
            dirs=dirs, model="pangu", variable=variable, regions=None,
            lead_times=LEAD_TIMES, sdor_da=None, **mkw,
        )
        if apd is None:
            continue
        pdata = _compute_pixel_improvement(apd[120], variable, 120)
        all_v = np.concatenate(
            [p["values"][~np.isnan(p["values"])].flatten() for p in pdata])
        p1, p99 = np.percentile(all_v, [1, 99])
        vmin, vmax = float(p1), float(p99)

        ax = fig.add_subplot(gs[0, col], projection=ccrs.PlateCarree())
        norm, cmap = _draw_improvement_map(ax, pdata, vmin, vmax,
                                           center_at_zero=True)
        ax.set_title(variable.replace("_", " ").title(), fontsize=14, weight="bold")

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        _attach_cbar(fig, sm, ax, "RMSE improvement (%)", max(vmin, 0), vmax,
                     pad=0.07, shrink=0.75)

    fig.suptitle("Pangu-Weather: 5 day lead time RMSE improvement",
                 fontsize=14, weight="bold", y=1.02)
    out = os.path.join(save_dir, "erl_appA2_pangu_5day_maps.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Appendix A3 — IFS maps (all lead times)
# ---------------------------------------------------------------------------

def make_app_a3_ifs_maps(dirs, save_dir):
    """
    Pixel-level improvement maps for IFS at 1, 5, and 9 day lead times
    for both 2m temperature and 10m wind speed (3 × 2 grid).
    """
    print("\n=== Appendix A3: IFS maps (all lead times) ===")
    mkw = _model_kwargs()
    lead_times_plot = [24, 120, 216]
    lead_labels     = {24: "1 day", 120: "5 day", 216: "9 day"}
    variables       = ["2m_temperature", "10m_wind_speed"]
    var_titles      = {"2m_temperature": "2m temperature",
                       "10m_wind_speed": "10m wind speed"}

    panel_data    = {}
    panel_extents = {}
    for variable in variables:
        apd = load_region_data(
            dirs=dirs, model="ifs", variable=variable, regions=None,
            lead_times=LEAD_TIMES, sdor_da=None, **mkw,
        )
        if apd is None:
            print(f"  No IFS data for {variable}; skipping.")
            return
        panel_data[variable] = {lt: _compute_pixel_improvement(apd[lt], variable, lt)
                                 for lt in lead_times_plot}
        all_v = np.concatenate([
            p["values"][~np.isnan(p["values"])].flatten()
            for lt in lead_times_plot for p in panel_data[variable][lt]])
        p1, p99 = np.percentile(all_v, [1, 99])
        panel_extents[variable] = (float(p1), float(p99))

    fig  = plt.figure(figsize=(16, 13))
    gs   = gridspec.GridSpec(3, 2, hspace=0.06, wspace=0.06)
    col_axes_all  = {v: [] for v in variables}
    col_norms     = {}

    for row, lt in enumerate(lead_times_plot):
        for col, variable in enumerate(variables):
            vmin, vmax = panel_extents[variable]
            ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())
            norm, cmap = _draw_improvement_map(ax, panel_data[variable][lt], vmin, vmax,
                                               center_at_zero=True)
            col_axes_all[variable].append(ax)
            col_norms[variable] = (norm, cmap)
            if row == 0:
                ax.set_title(var_titles[variable], fontsize=14, weight="bold")
            ax.text(-0.03, 0.5, lead_labels[lt],
                    transform=ax.transAxes, rotation=90, va="center",
                    fontsize=13, weight="bold")

    for variable in variables:
        norm, cmap = col_norms[variable]
        vmin, vmax = panel_extents[variable]
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        label = ("2m temperature" if variable == "2m_temperature"
                 else "10m wind speed")
        _attach_cbar(fig, sm, col_axes_all[variable],
                     f"IFS {label} RMSE improvement (%)", max(vmin, 0), vmax,
                     pad=0.03, shrink=0.85, aspect=40)

    fig.suptitle("IFS HRES post-processing improvement (1, 5, 9 day lead times)",
                 fontsize=15, weight="bold", y=0.99)
    out = os.path.join(save_dir, "erl_appA3_ifs_maps.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Appendix A4 & A5 — IFS binscatters
# ---------------------------------------------------------------------------

def make_app_a4_ifs_binscatter_equator(dirs, save_dir):
    """IFS binscatter vs distance from equator."""
    print("\n=== Appendix A4: IFS binscatter equator ===")
    _make_binscatter(dirs, save_dir, "equator_distance",
                     "erl_appA4_ifs_binscatter_equator.png", model="ifs")


def make_app_a5_ifs_binscatter_sdor(dirs, save_dir):
    """IFS binscatter vs SDOR."""
    print("\n=== Appendix A5: IFS binscatter SDOR ===")
    _make_binscatter(dirs, save_dir, "sdor",
                     "erl_appA5_ifs_binscatter_sdor.png", model="ifs")


# ---------------------------------------------------------------------------
# Appendix A6 & A7 — region size experiment
# ---------------------------------------------------------------------------

def make_app_a6_region_size_mlp(dirs, save_dir):
    """Training region size ablation for MLP (temperature and wind)."""
    print("\n=== Appendix A6: region size ablation (MLP) ===")
    for variable in ["2m_temperature", "10m_wind_speed"]:
        generate_subregion_comparison_plots(
            dirs=dirs,
            train_start=TRAIN_START, train_end=TRAIN_END,
            test_start=TEST_START,   test_end=TEST_END,
            model="pangu", variable=variable,
            nn_architecture="mlp", snapshot_ensemble=3,
            save_dir=save_dir,
        )
    # Rename to ERL scheme
    for variable, suffix in [("2m_temperature", "temperature"),
                              ("10m_wind_speed", "wind")]:
        src = os.path.join(save_dir,
            f"region_size_comparison_{variable}_mlp_snapshot3_pangu.png")
        dst = os.path.join(save_dir, f"erl_appA6_region_size_mlp_{suffix}.png")
        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"  Saved: {dst}")


def make_app_a7_region_size_unet(dirs, save_dir):
    """Training region size ablation for U-Net (temperature and wind)."""
    print("\n=== Appendix A7: region size ablation (UNet) ===")
    for variable in ["2m_temperature", "10m_wind_speed"]:
        generate_subregion_comparison_plots(
            dirs=dirs,
            train_start=TRAIN_START, train_end=TRAIN_END,
            test_start=TEST_START,   test_end=TEST_END,
            model="pangu", variable=variable,
            nn_architecture="unet",
            save_dir=save_dir,
        )
    for variable, suffix in [("2m_temperature", "temperature"),
                              ("10m_wind_speed", "wind")]:
        src = os.path.join(save_dir,
            f"region_size_comparison_{variable}_unet_pangu.png")
        dst = os.path.join(save_dir, f"erl_appA7_region_size_unet_{suffix}.png")
        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"  Saved: {dst}")


# ---------------------------------------------------------------------------
# Appendix A8 — architecture eval regions map
# ---------------------------------------------------------------------------

def make_app_a8_arch_eval_regions(dirs, save_dir):
    """Map showing the 5% sample of patches used for architecture evaluation."""
    print("\n=== Appendix A8: arch eval regions map ===")
    eval_cells = sample_continent_patches(
        dirs["processed"], fraction=0.05, seed=42, split="eval")
    map_arch_exeriment_regions(
        dirs=dirs, eval_cells=eval_cells,
        fraction=0.05, seed=42, split="eval",
        model="pangu", subregion="6x6",
        save_dir=save_dir,
    )
    src = os.path.join(save_dir, "arch_eval_regions_map_pangu_6x6.png")
    dst = os.path.join(save_dir, "erl_appA8_arch_eval_regions.png")
    if os.path.exists(src):
        shutil.move(src, dst)
    print(f"  Saved: {dst}")


# ---------------------------------------------------------------------------
# Appendix A9 — World Bank income group map
# ---------------------------------------------------------------------------

def make_app_a9_income_group_map(dirs, save_dir):
    """
    Appendix A9: world map coloured by World Bank 2024 (FY25) income group.

    Acts as a spatial legend for figure 1 — readers can see which countries
    fall in each income group before interpreting the bar chart.
    """
    print("\n=== Appendix A9: World Bank income group map ===")
    income_gdf = _load_income_geodataframe(dirs)

    fig = plt.figure(figsize=(13, 6.5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_global()
    ax.add_feature(cfeature.OCEAN, facecolor="white",   zorder=0)
    ax.add_feature(cfeature.LAND,  facecolor="#dddddd", alpha=0.4, zorder=0)

    for ig in INCOME_GROUP_ORDER:
        sub = income_gdf[income_gdf["income_group"] == ig]
        if sub.empty:
            continue
        ax.add_geometries(
            sub.geometry, crs=ccrs.PlateCarree(),
            facecolor=INCOME_GROUP_COLORS[ig],
            edgecolor="#333333", linewidth=0.3, zorder=2,
        )

    ax.add_feature(cfeature.COASTLINE, linewidth=0.4,
                   edgecolor="black", zorder=3)

    legend_handles = [
        Line2D([0], [0], marker="s", linestyle="",
               markerfacecolor=INCOME_GROUP_COLORS[ig],
               markeredgecolor="#333333", markersize=10, label=ig)
        for ig in INCOME_GROUP_ORDER
    ]
    ax.legend(handles=legend_handles,
              title="World Bank 2024 income group",
              loc="lower left", frameon=True, fontsize=10)
    ax.set_title("World Bank 2024 income group classification",
                 fontsize=14, weight="bold")

    out = os.path.join(save_dir, "erl_appA9_income_group_map.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Build every ERL figure and save to {root}/erl_figures/."""
    dirs     = setup_directories()
    save_dir = _erl_output_dir(dirs)
    print(f"Writing all ERL figures to: {save_dir}\n")

    # make_figure_1_summary_equity(dirs, save_dir)
    # make_figure_2_unified_global_maps(dirs, save_dir)
    # make_figure_3_binscatter_equator(dirs, save_dir)
    # make_figure_4_binscatter_sdor(dirs, save_dir)
    # make_figure_5_arch_comparison(dirs, save_dir)

    # make_app_a1_model_compare_boxplot(dirs, save_dir)
    make_app_a2_pangu_5day_maps(dirs, save_dir)
    # make_app_a3_ifs_maps(dirs, save_dir)
    # make_app_a4_ifs_binscatter_equator(dirs, save_dir)
    # make_app_a5_ifs_binscatter_sdor(dirs, save_dir)
    # make_app_a6_region_size_mlp(dirs, save_dir)
    # make_app_a7_region_size_unet(dirs, save_dir)
    # make_app_a8_arch_eval_regions(dirs, save_dir)
    # make_app_a9_income_group_map(dirs, save_dir)

    print("\nDone. All figures are in:", save_dir)


if __name__ == "__main__":
    main()
