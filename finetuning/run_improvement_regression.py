#!/usr/bin/env python3
"""
Regression analysis of post-processing improvement on geographic features.

Runs OLS: improvement_pct ~ abs(latitude) + SDOR + climate_zone_dummies
          + baseline_rmse + lead_time_dummies

Uses vectorized xarray lookups instead of pixel-by-pixel loops.

Usage:
    python3 finetuning/run_improvement_regression.py
"""

import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import statsmodels.api as sm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from helper_funcs import setup_directories
from finetuning.figures_finetuning import load_region_data


CLIMATE_ZONE_NAMES = {1: "tropical", 2: "arid", 3: "temperate", 4: "cold", 5: "polar"}


def build_pixel_dataframe(dirs, model, variable, lead_times, nn_architecture="mlp",
                          subregion="6x6", train_start="2018-01-01", train_end="2021-12-31",
                          test_start="2022-01-01", test_end="2022-12-31"):
    """
    Build a pixel-level DataFrame with improvement, latitude, SDOR, and climate zone.
    Uses vectorized xarray lookups for speed.
    """
    print(f"\nLoading ancillary data...")

    # Load SDOR
    era5_static_path = os.path.join(dirs["raw"], "era5_static.nc")
    sdor_da = xr.open_dataset(era5_static_path, engine="netcdf4")["sdor"]

    # Load climate zones (pre-regridded to 0.25°)
    climate_zones_path = os.path.join(dirs["processed"], "climate_zones_0p25.nc")
    cz_ds = xr.open_dataset(climate_zones_path, engine="netcdf4")
    cz_da = cz_ds["climate_zones"] if "climate_zones" in cz_ds else cz_ds[list(cz_ds.data_vars)[0]]

    # Load region data (returns dict keyed by lead time)
    print(f"Loading region data for {variable}...")
    all_patch_data = load_region_data(
        dirs=dirs, model=model, variable=variable,
        regions=None,  # all continents
        train_start=train_start, train_end=train_end,
        test_start=test_start, test_end=test_end,
        nn_architecture=nn_architecture, subregion=subregion,
        lead_times=lead_times, sdor_da=None  # we'll do SDOR ourselves vectorized
    )

    if all_patch_data is None:
        print(f"No data loaded for {variable}")
        return None

    var_suffix_map = {lt: f"_lt{lt}h" for lt in lead_times}

    all_rows = []

    for lead_time in lead_times:
        patch_list = all_patch_data[lead_time]
        if not patch_list:
            continue
        print(f"  Processing {len(patch_list)} patches for lead time {lead_time}h...")

        for patch in patch_list:
            ds = patch['ds']
            var_suffix = var_suffix_map[lead_time]

            gt = ds[f"{variable}_ground_truth{var_suffix}"].values   # (time, lat, lon)
            orig = ds[f"{variable}_original{var_suffix}"].values
            corr = ds[f"{variable}_corrected{var_suffix}"].values

            if gt.ndim != 3:
                continue

            lats = ds.latitude.values
            lons = ds.longitude.values

            # Pixel-level RMSE: (lat, lon)
            pixel_rmse_orig = np.sqrt(np.nanmean((orig - gt) ** 2, axis=0))
            pixel_rmse_corr = np.sqrt(np.nanmean((corr - gt) ** 2, axis=0))
            pixel_improvement = ((pixel_rmse_orig - pixel_rmse_corr) /
                                 (pixel_rmse_orig + 1e-10)) * 100

            # Vectorized SDOR lookup using xarray sel
            sdor_patch = sdor_da.sel(latitude=xr.DataArray(lats, dims="lat"),
                                     longitude=xr.DataArray(lons, dims="lon"),
                                     method="nearest").values  # (lat, lon)

            # Vectorized climate zone lookup
            cz_patch = cz_da.sel(latitude=xr.DataArray(lats, dims="lat"),
                                  longitude=xr.DataArray(lons, dims="lon"),
                                  method="nearest").values  # (lat, lon)

            # Build meshgrid for lat
            lon_grid, lat_grid = np.meshgrid(lons, lats)

            # Flatten all arrays
            flat_lat = lat_grid.flatten()
            flat_imp = pixel_improvement.flatten()
            flat_rmse_orig = pixel_rmse_orig.flatten()
            flat_sdor = sdor_patch.flatten()
            flat_cz = cz_patch.flatten()

            # Remove NaNs
            valid = (~np.isnan(flat_imp) & ~np.isnan(flat_rmse_orig) &
                     ~np.isnan(flat_sdor) & ~np.isnan(flat_cz))

            df_patch = pd.DataFrame({
                'improvement_pct': flat_imp[valid],
                'abs_latitude': np.abs(flat_lat[valid]),
                'sdor': flat_sdor[valid],
                'baseline_rmse': flat_rmse_orig[valid],
                'climate_zone': flat_cz[valid].astype(int),
                'lead_time': lead_time,
                'variable': variable
            })
            all_rows.append(df_patch)

    if not all_rows:
        return None

    df = pd.concat(all_rows, ignore_index=True)
    print(f"  Total pixels: {len(df):,}")
    return df


def run_regression(df, variable_name):
    """Run OLS regression with HC1 robust standard errors."""
    # Climate zone dummies (tropical=1 is omitted reference)
    cz_dummies = pd.get_dummies(df['climate_zone'], prefix='cz', drop_first=True, dtype=float)
    rename_map = {f"cz_{k}": f"cz_{v}" for k, v in CLIMATE_ZONE_NAMES.items()
                  if f"cz_{k}" in cz_dummies.columns}
    cz_dummies = cz_dummies.rename(columns=rename_map)

    # Lead time dummies (24h is omitted reference)
    lt_dummies = pd.get_dummies(df['lead_time'], prefix='lt', drop_first=True, dtype=float)

    X = pd.concat([df[['abs_latitude', 'sdor', 'baseline_rmse']], cz_dummies, lt_dummies], axis=1)
    X = sm.add_constant(X)
    y = df['improvement_pct']

    model = sm.OLS(y, X).fit(cov_type='HC1')

    print(f"\n{'='*70}")
    print(f"Regression Results: {variable_name}")
    print(f"  N = {int(model.nobs):,}   R² = {model.rsquared:.4f}")
    print(f"{'='*70}")
    summary_df = pd.DataFrame({
        'coef': model.params,
        'se': model.bse,
        't': model.tvalues,
        'p': model.pvalues
    }).round(4)
    summary_df['stars'] = summary_df['p'].apply(
        lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
    )
    print(summary_df.to_string())

    return model


def save_latex_table(models, variable_names, save_path):
    """Save a two-column regression table as CSV."""
    all_params = set()
    for m in models:
        all_params.update(m.params.index)

    rows = []
    for param in sorted(all_params):
        row = {'predictor': param}
        for model, var_name in zip(models, variable_names):
            if param in model.params:
                coef = model.params[param]
                se = model.bse[param]
                p = model.pvalues[param]
                stars = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
                row[f'{var_name}_coef'] = f"{coef:.4f}{stars}"
                row[f'{var_name}_se'] = f"({se:.4f})"
            else:
                row[f'{var_name}_coef'] = ''
                row[f'{var_name}_se'] = ''
        rows.append(row)

    # Append N and R²
    for label, values in [('N', [f"{int(m.nobs):,}" for m in models]),
                           ('R²', [f"{m.rsquared:.4f}" for m in models])]:
        row = {'predictor': label}
        for var_name, val in zip(variable_names, values):
            row[f'{var_name}_coef'] = val
            row[f'{var_name}_se'] = ''
        rows.append(row)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(save_path, index=False)
    print(f"\nRegression table saved to: {save_path}")
    return result_df


def main():
    dirs = setup_directories()
    lead_times = [24, 120, 216]
    model = "pangu"

    reg_models = []
    var_names = []

    for variable in ["2m_temperature", "10m_wind_speed"]:
        print(f"\n{'='*60}")
        print(f"Building dataframe: {variable}")
        print(f"{'='*60}")
        df = build_pixel_dataframe(dirs, model, variable, lead_times)
        if df is not None and len(df) > 0:
            m = run_regression(df, variable.replace('_', ' ').title())
            reg_models.append(m)
            var_names.append(variable)

    if reg_models:
        out_path = os.path.join(dirs["fig"], "regression_results.csv")
        save_latex_table(reg_models, var_names, out_path)


if __name__ == "__main__":
    main()
