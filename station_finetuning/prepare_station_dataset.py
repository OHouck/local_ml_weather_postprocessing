"""Build the station-level training dataset used by train_station_models.py.

Running this script end to end does four things:

1. Selects ISD stations inside the study regions that reported throughout the
   study period.
2. Downloads their ISD-Lite records and reads the 00 UTC observations.
3. Extracts the Pangu forecast neighbourhood and the ERA5 reference value at
   each station's nearest grid cell.
4. Builds the day-of-year climatology and records how completely each station
   reported during the training years.

Everything is written to a single npz plus a station metadata csv, so that the
training step does no I/O against the global archives.

Example:
    uv run python station_finetuning/prepare_station_dataset.py \
        --regions_file station_finetuning/configs/study_regions.json \
        --output_prefix station_dataset
"""

import argparse
import json
import os
import sys

import numpy as np

# The shared project helpers live one directory up, and supply the machine
# specific data paths that the rest of the repository uses.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helper_funcs import setup_directories  # noqa: E402

from forecast_features import (NEIGHBOURHOOD_HALF_WIDTH,  # noqa: E402
                               build_day_of_year_climatology,
                               extract_era5_at_stations,
                               extract_forecast_neighbourhoods,
                               read_station_grid_elevation)
from station_observations import (ISD_HISTORY_URL, download_station_files,
                                  find_stations_in_boxes,
                                  load_station_observations)

# Study period. Pangu forecasts are archived locally for these years, and the
# split matches the main paper so results stay comparable.
TRAINING_YEARS = [2018, 2019, 2020, 2021]
TEST_YEAR = 2022
ALL_YEARS = TRAINING_YEARS + [TEST_YEAR]

LEAD_TIMES_HOURS = [24, 120, 216]
FORECAST_VARIABLES = ["2m_temperature", "10m_u_component_of_wind",
                      "10m_v_component_of_wind"]

# Stations below this fraction of 00 UTC training-day coverage are reported as
# unusable here and dropped at training time. Kept in step with
# train_station_models.MINIMUM_TRAINING_COVERAGE.
MINIMUM_TRAINING_COVERAGE = 0.70


def main():
    """Run the full dataset preparation and write the npz and metadata csv.

    Inputs: read from the command line; see the module docstring for an example.

    Returns:
        None. Writes '<output_prefix>.npz' and '<output_prefix>_stations.csv'
        into the processed data directory.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regions_file", required=True,
                        help="json mapping region name to [lat0, lat1, lon0, lon1]")
    parser.add_argument("--output_prefix", default="station_dataset")
    parser.add_argument("--model_name", default="pangu",
                        help="forecast source to post-process: pangu, ifs or aifs")
    parser.add_argument("--work_directory", default=None,
                        help="where to cache ISD downloads; defaults to the "
                             "processed data directory")
    parsed_arguments = parser.parse_args()

    # Paths always come from the shared helper so the code runs unchanged on a
    # laptop and on Midway.
    directories = setup_directories()

    work_directory = (parsed_arguments.work_directory
                      or os.path.join(directories["processed"], "station_finetuning"))
    isd_directory = os.path.join(work_directory, "isd_lite")
    os.makedirs(isd_directory, exist_ok=True)

    # ---- 1. choose stations --------------------------------------------------
    isd_history_path = os.path.join(work_directory, "isd-history.csv")
    if not os.path.exists(isd_history_path):
        print(f"downloading station metadata from {ISD_HISTORY_URL}", flush=True)
        os.system(f'curl -sf -o "{isd_history_path}" "{ISD_HISTORY_URL}"')

    with open(parsed_arguments.regions_file) as regions_file:
        region_boxes = {name: tuple(box)
                        for name, box in json.load(regions_file).items()}

    stations = find_stations_in_boxes(
        isd_history_path, region_boxes,
        first_required_day=int(f"{ALL_YEARS[0]}0101"),
        last_required_day=int(f"{TEST_YEAR}1231"))
    stations = stations.drop_duplicates(subset=["usaf", "wban"]).reset_index(drop=True)
    print(f"{len(stations)} candidate stations across "
          f"{len(region_boxes)} regions", flush=True)

    # ---- 2. download and read observations ----------------------------------
    files_per_year = download_station_files(stations, ALL_YEARS, isd_directory)
    print(f"downloaded files per year: {files_per_year}", flush=True)

    # ---- 3. extract forecast predictors -------------------------------------
    # Archive layout is raw/{source}_{year}.zarr for every source in the repo,
    # so the template only needs the source name substituted in.
    forecast_template = os.path.join(
        directories["raw"], f"{parsed_arguments.model_name}_{{year}}.zarr")
    era5_template = os.path.join(directories["raw"], "era5_{year}.zarr")

    verification_times, neighbourhood_values = extract_forecast_neighbourhoods(
        forecast_template, ALL_YEARS, LEAD_TIMES_HOURS, FORECAST_VARIABLES,
        stations["latitude"].to_numpy(), stations["longitude"].to_numpy())

    era5_values = extract_era5_at_stations(
        era5_template, ALL_YEARS, verification_times,
        stations["latitude"].to_numpy(), stations["longitude"].to_numpy())

    station_observations = load_station_observations(
        stations, ALL_YEARS, isd_directory, verification_times)

    # ---- 4. climatology, coverage and elevation -----------------------------
    climatology_by_variable = build_day_of_year_climatology(
        era5_values, verification_times, TRAINING_YEARS)

    grid_elevations = read_station_grid_elevation(
        os.path.join(directories["raw"], "era5_static.nc"),
        stations["latitude"].to_numpy(), stations["longitude"].to_numpy())

    # Coverage is the fraction of 00 UTC training days on which the station
    # actually reported a temperature. Stations reporting only at 03 and 12 UTC
    # score near zero here and are dropped at training time.
    in_training_period = np.isin(verification_times.year, TRAINING_YEARS)
    stations["training_coverage"] = np.isfinite(
        station_observations[in_training_period, 0]).mean(axis=0)
    stations["grid_elevation_metres"] = grid_elevations

    # Attach each station's World Bank income group, reusing the classification
    # already built for the paper's figures so the two agree by construction.
    # Without this the income-gap summary in verification.py has nothing to
    # group by, which is the headline result of the analysis.
    from finetuning.erl_figures import (_classify_pixels_by_income,
                                        _load_income_geodataframe)
    income_geodataframe = _load_income_geodataframe(directories)
    stations["income_group"] = _classify_pixels_by_income(
        stations["latitude"].to_numpy(), stations["longitude"].to_numpy(),
        income_geodataframe)

    output_directory = os.path.join(directories["processed"], "station_finetuning")
    os.makedirs(output_directory, exist_ok=True)
    dataset_path = os.path.join(output_directory,
                                f"{parsed_arguments.output_prefix}.npz")
    metadata_path = os.path.join(output_directory,
                                 f"{parsed_arguments.output_prefix}_stations.csv")

    np.savez_compressed(
        dataset_path,
        verification_times=verification_times.values,
        neighbourhood_values=neighbourhood_values,
        era5_values=era5_values,
        station_observations=station_observations,
        temperature_climatology=climatology_by_variable[0],
        wind_climatology=climatology_by_variable[1],
        grid_elevations=grid_elevations,
        lead_times_hours=np.array(LEAD_TIMES_HOURS),
        neighbourhood_half_width=np.array(NEIGHBOURHOOD_HALF_WIDTH),
        # Station identifiers travel with the arrays so the training step can
        # verify that the metadata csv is row-aligned with the station axis.
        station_usaf=stations["usaf"].to_numpy().astype(str),
        station_wban=stations["wban"].to_numpy().astype(str))
    stations.to_csv(metadata_path, index=False)

    usable = stations["training_coverage"] >= MINIMUM_TRAINING_COVERAGE
    print(f"wrote {dataset_path}", flush=True)
    print(f"neighbourhood array shape: {neighbourhood_values.shape}", flush=True)
    print(f"{int(usable.sum())} of {len(stations)} stations clear the "
          f"{MINIMUM_TRAINING_COVERAGE:.0%} coverage bar", flush=True)
    print(stations.loc[usable, "income_group"].value_counts(dropna=False).to_string(),
          flush=True)


if __name__ == "__main__":
    main()
