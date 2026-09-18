# scratch.py
# Verifies that the `*_ground_truth_lt{N}h` variables stored in a finetuning
# output zarr match the raw ERA5 analysis, and empirically determines whether
# the output's `time` coordinate is the forecast INIT time or the VALID time
# (init + lead). Compares the stored ground truth against ERA5 under both
# interpretations and reports which one matches.

import numpy as np
import xarray as xr


PANGU_PATH = (
    "/Users/ohouck/globus/forecast_data/processed/finetuning_output/pangu/africa/"
    "train_2m_temperature_test_2m_temperature_dim6x6_leadtime_24_120_216h_"
    "train2018-01-01-2021-12-31_test2022-01-01-2022-12-31_mlp_snapshot3_africa_bs47.zarr"
)
ERA5_PATH = "/Users/ohouck/globus/forecast_data/raw/era5_2022.zarr"
VARIABLE = "2m_temperature"
LEAD_TIMES_H = [24, 120, 216]


def align_era5_to_grid(era5_ds, ref_ds, variable):
    """Subset ERA5 onto the reference dataset's lat/lon grid.

    Inputs:
        era5_ds (xr.Dataset): raw ERA5 analysis (latitude may be descending).
        ref_ds (xr.Dataset): output dataset whose latitude/longitude define the
            target grid.
        variable (str): variable name to extract from ERA5.

    Returns:
        xr.DataArray: ERA5 `variable` on the reference grid, dims (time, latitude,
        longitude), latitude ascending to match `ref_ds`.
    """
    da = era5_ds[variable].sel(
        latitude=ref_ds.latitude.values,
        longitude=ref_ds.longitude.values,
        method="nearest",
    )
    # Force exact-matching coordinates so xarray alignment is by index, not value.
    da = da.assign_coords(latitude=ref_ds.latitude.values,
                          longitude=ref_ds.longitude.values)
    return da.sortby("latitude")


def compare_at_times(gt, era5_da, target_times):
    """Compare stored ground truth against ERA5 sampled at `target_times`.

    Inputs:
        gt (xr.DataArray): stored ground truth, dims (time, latitude, longitude).
        era5_da (xr.DataArray): ERA5 field on the same grid, dims (time, latitude,
            longitude).
        target_times (np.ndarray): datetime64 array, same length/order as gt.time,
            giving the ERA5 timestamp to compare each gt time against.

    Returns:
        dict with keys: n_compared (int), max_abs_diff (float), mean_abs_diff
        (float), all_close (bool). Values are NaN/0 if no times overlap.
    """
    available = np.isin(target_times, era5_da.time.values)
    if not available.any():
        return {"n_compared": 0, "max_abs_diff": np.nan,
                "mean_abs_diff": np.nan, "all_close": False}

    gt_sub = gt.isel(time=available)
    era5_sub = era5_da.sel(time=target_times[available])
    # Align both to a shared integer time index so the subtraction is positional.
    gt_vals = gt_sub.values
    era5_vals = era5_sub.values
    diff = np.abs(gt_vals - era5_vals)
    return {
        "n_compared": int(available.sum()),
        "max_abs_diff": float(np.nanmax(diff)),
        "mean_abs_diff": float(np.nanmean(diff)),
        "all_close": bool(np.allclose(gt_vals, era5_vals, atol=1e-3, equal_nan=True)),
    }


def main():
    """Run the ground-truth vs ERA5 comparison and print a verdict per lead time."""
    ds = xr.open_zarr(PANGU_PATH)
    era5_ds = xr.open_zarr(ERA5_PATH)

    era5_da = align_era5_to_grid(era5_ds, ds, VARIABLE)

    print(f"Output time coordinate: {ds.time.values[0]} .. {ds.time.values[-1]} "
          f"(hours present: {np.unique(ds.time.dt.hour.values)})")
    print(f"Grid: lat {ds.latitude.values.min()}..{ds.latitude.values.max()}, "
          f"lon {ds.longitude.values.min()}..{ds.longitude.values.max()}\n")

    # Test 1: are the ground-truth fields identical across lead times at a shared
    # time value? If yes, `time` is VALID time (obs at time T regardless of lead).
    gts = {lt: ds[f"{VARIABLE}_ground_truth_lt{lt}h"] for lt in LEAD_TIMES_H}
    base = gts[LEAD_TIMES_H[0]]
    identical_across_lt = all(
        np.allclose(base.values, gts[lt].values, atol=1e-3, equal_nan=True)
        for lt in LEAD_TIMES_H[1:]
    )
    print(f"Ground truth identical across lead times at same `time`? "
          f"{identical_across_lt}")
    print("  (True => `time` is VALID time; False => `time` is INIT time)\n")

    # Test 2: compare each lead time's ground truth to ERA5 under both
    # interpretations of the `time` coordinate.
    for lt in LEAD_TIMES_H:
        gt = gts[lt]
        times = gt.time.values
        lt_td = np.timedelta64(lt, "h")

        as_valid = compare_at_times(gt, era5_da, times)              # time == valid
        as_init = compare_at_times(gt, era5_da, times + lt_td)       # time == init

        print(f"lead {lt}h:")
        print(f"  time as VALID (ERA5 @ time):        "
              f"n={as_valid['n_compared']:4d}  "
              f"max|Δ|={as_valid['max_abs_diff']:.4g}  "
              f"match={as_valid['all_close']}")
        print(f"  time as INIT  (ERA5 @ time+{lt}h): "
              f"n={as_init['n_compared']:4d}  "
              f"max|Δ|={as_init['max_abs_diff']:.4g}  "
              f"match={as_init['all_close']}")


if __name__ == "__main__":
    main()
