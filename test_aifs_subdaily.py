"""
test_aifs_subdaily.py

Quick check on one AIFS forecast to see whether 2m_temperature,
geopotential_500_avg, and specific_humidity_1000 carry multiple distinct
sub-daily values within the first 24 hours of the forecast, or are effectively
daily aggregates. Prints the lead-time coordinates plus the per-lead values at
a single grid point near Portland, OR.
"""

import numpy as np
import xarray as xr

path = "/net/monsoon/reforecast/aifs/2003-06-07T00.zarr"
variables = ["2m_temperature", "geopotential_500_avg", "specific_humidity_1000"]
probe_lat, probe_lon = 45.0, -122.0 % 360  # 238.0
one_day = np.timedelta64(24, "h")

ds = xr.open_zarr(path)
print(ds)

td = np.unique(ds["prediction_timedelta"].values)
td_daily = np.unique(ds["prediction_timedelta_daily"].values)
print("\nprediction_timedelta unique values:")
print(td)
print("\nprediction_timedelta_daily unique values:")
print(td_daily)

# Leads within the first 24 h for each coord.
td_24h = td[(td > np.timedelta64(0, "h")) & (td <= one_day)]
if np.issubdtype(td_daily.dtype, np.datetime64):
    daily_deltas = td_daily - td_daily.min()
else:
    daily_deltas = td_daily
td_daily_24h = td_daily[(daily_deltas > np.timedelta64(0, "h")) & (daily_deltas <= one_day)]

print(f"\nleads in (0, 24h] on prediction_timedelta:       {td_24h}")
print(f"leads in (0, 24h] on prediction_timedelta_daily: {td_daily_24h}")

for var in variables:
    print(f"\n--- {var} ---")
    if var not in ds.variables:
        print("  NOT PRESENT")
        continue
    da = ds[var]
    print(f"  dims: {da.dims}")

    if "prediction_timedelta" in da.dims:
        lead_dim, leads_24h = "prediction_timedelta", td_24h
    elif "prediction_timedelta_daily" in da.dims:
        lead_dim, leads_24h = "prediction_timedelta_daily", td_daily_24h
    else:
        print("  no lead-time dim")
        continue

    if len(leads_24h) <= 1:
        print(f"  only {len(leads_24h)} lead value within 24 h on {lead_dim};"
              f" no intraday variation possible")
        continue

    point = da.sel(latitude=probe_lat, longitude=probe_lon, method="nearest")
    point_24h = point.sel({lead_dim: leads_24h}).squeeze().values
    print(f"  values at ({probe_lat}, {probe_lon}) for the 24 h leads:")
    for lead, val in zip(leads_24h, point_24h):
        print(f"    {lead} -> {val}")
    finite = point_24h[np.isfinite(point_24h)]
    n_unique = len(np.unique(finite))
    print(f"  distinct finite values within 24 h: {n_unique}"
          f"  -> {'VARIES intraday' if n_unique > 1 else 'CONSTANT (daily aggregate)'}")
