"""Build the station-level training dataset used by train_station_models.py.

Running this script end to end does four things:

1. Selects HadISD stations inside the study regions that reported throughout
   the study period.
2. Downloads their quality-controlled HadISD records and reads the 00 UTC
   observations.
3. Extracts the Pangu forecast neighbourhood and the ERA5 reference value at
   each station's nearest grid cell.
4. Reads the WeatherBench2 30-year ERA5 day-of-year climatology at each
   station (caching it locally on first use) and records how completely each
   station reported during the training years.

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
                               download_weatherbench_climatology,
                               load_station_climatology,
                               neighbourhood_array_path,
                               extract_era5_at_stations,
                               extract_forecast_neighbourhoods,
                               read_station_grid_elevation)
from station_sampling import classify_stations  # noqa: E402
from station_observations import (MINIMUM_TRAINING_COVERAGE,  # noqa: E402
                                  download_station_files,
                                  read_station_metadata,
                                  find_stations_globally,
                                  find_stations_in_boxes,
                                  load_station_observations,
                                  training_coverage)

# Study period. Pangu forecasts are archived locally for these years, and the
# split matches the main paper so results stay comparable.
TRAINING_YEARS = [2018, 2019, 2020, 2021]
TEST_YEAR = 2022
ALL_YEARS = TRAINING_YEARS + [TEST_YEAR]
# A station's record must span this whole window, as YYYYMMDD.
FIRST_REQUIRED_DAY = int(f"{ALL_YEARS[0]}0101")
LAST_REQUIRED_DAY = int(f"{TEST_YEAR}1231")

LEAD_TIMES_HOURS = [24, 120, 216]
FORECAST_VARIABLES = ["2m_temperature", "10m_u_component_of_wind",
                      "10m_v_component_of_wind"]


def main():
    """Run the full dataset preparation and write the npz and metadata csv.

    Inputs: read from the command line; see the module docstring for an example.

    Returns:
        None. Writes '<output_prefix>.npz' and '<output_prefix>_stations.csv'
        into the processed data directory.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=None,
                        help="csv from select_station_sample.py naming the "
                             "stations to use; overrides --station_selection")
    parser.add_argument("--continent", default=None,
                        help="with --manifest, process only this continent "
                             "(africa, asia, europe, north_america, "
                             "south_america, oceania). This is what keeps one "
                             "chunk small enough to build on a laptop")
    parser.add_argument("--station_selection", choices=["boxes", "global"],
                        default="boxes",
                        help="'boxes' draws only from --regions_file; 'global' "
                             "draws every active HadISD station on Earth, which is "
                             "what raises the sample from hundreds to thousands")
    parser.add_argument("--regions_file", default=None,
                        help="json mapping region name to [lat0, lat1, lon0, "
                             "lon1]; required for --station_selection boxes")
    parser.add_argument("--latitude_band", type=float, nargs=2, default=None,
                        metavar=("MIN_LAT", "MAX_LAT"),
                        help="restrict a global draw to a latitude band, e.g. "
                             "-23.5 23.5 for the tropics")
    parser.add_argument("--max_stations", type=int, default=None,
                        help="cap the draw at this many stations, sampled "
                             "reproducibly rather than taken in order")
    parser.add_argument("--parallel_downloads", type=int, default=6,
                        help="concurrent HadISD downloads")
    parser.add_argument("--output_prefix", default="station_dataset")
    parser.add_argument("--model_name", default="pangu",
                        help="forecast source to post-process: pangu, ifs or aifs")
    parser.add_argument("--work_directory", default=None,
                        help="where to cache HadISD downloads; defaults to the "
                             "processed data directory")
    parsed_arguments = parser.parse_args()

    # Paths always come from the shared helper so the code runs unchanged on a
    # laptop and on Midway.
    directories = setup_directories()

    work_directory = (parsed_arguments.work_directory
                      or os.path.join(directories["processed"], "station_finetuning"))
    hadisd_directory = os.path.join(work_directory, "hadisd")

    # ---- 1. choose stations --------------------------------------------------


    if parsed_arguments.manifest:
        # The sample was already chosen globally, so this run only has to pick
        # its slice out of it. Reading the manifest rather than re-drawing is
        # what keeps the per-continent chunks a partition of one sample.
        stations = read_station_metadata(parsed_arguments.manifest)
        chunk_note = ""
        if parsed_arguments.continent:
            available_chunks = sorted(stations["continent"].dropna().unique())
            stations = stations[
                stations["continent"] == parsed_arguments.continent]
            if stations.empty:
                parser.error(
                    f"no stations in {parsed_arguments.manifest} for continent "
                    f"'{parsed_arguments.continent}'; "
                    f"available: {available_chunks}")
            chunk_note = f" ({parsed_arguments.continent})"
        # training_coverage is recomputed below from the observations this run
        # loads, so the manifest's screening copy is dropped to avoid two
        # columns of the same name disagreeing.
        stations = stations.drop(columns=["training_coverage"], errors="ignore")
        print(f"{len(stations)} stations from "
              f"{parsed_arguments.manifest}{chunk_note}", flush=True)
    elif parsed_arguments.station_selection == "global":
        latitude_band = (tuple(parsed_arguments.latitude_band)
                         if parsed_arguments.latitude_band else None)
        stations = find_stations_globally(
            work_directory, FIRST_REQUIRED_DAY, LAST_REQUIRED_DAY,
            latitude_band=latitude_band,
            maximum_stations=parsed_arguments.max_stations)
        band_note = f" within latitudes {latitude_band}" if latitude_band else ""
        print(f"{len(stations)} candidate stations from a global draw{band_note}",
              flush=True)
    else:
        if not parsed_arguments.regions_file:
            parser.error("--regions_file is required for --station_selection boxes")
        with open(parsed_arguments.regions_file) as regions_file:
            # Keys starting with '_' document the file rather than naming a
            # region, so they are not boxes and must not be unpacked as one.
            region_boxes = {name: tuple(box)
                            for name, box in json.load(regions_file).items()
                            if not name.startswith("_")}
        stations = find_stations_in_boxes(
            work_directory, region_boxes, FIRST_REQUIRED_DAY, LAST_REQUIRED_DAY)
        print(f"{len(stations)} candidate stations across "
              f"{len(region_boxes)} regions", flush=True)

    stations = stations.drop_duplicates(subset=["usaf", "wban"]).reset_index(drop=True)

    # ---- 2. download and read observations ----------------------------------
    # A chunk run off a manifest usually has every station cached already by
    # select_station_sample.py, in which case nothing is fetched.
    stations_present = download_station_files(
        stations, hadisd_directory,
        parallel_downloads=parsed_arguments.parallel_downloads)
    print(f"{stations_present} of {len(stations)} stations have HadISD data",
          flush=True)

    # ---- 3. extract forecast predictors -------------------------------------
    # Archive layout is raw/{source}_{year}.zarr for every source in the repo,
    # so the template only needs the source name substituted in.
    forecast_template = os.path.join(
        directories["raw"], f"{parsed_arguments.model_name}_{{year}}.zarr")
    era5_template = os.path.join(directories["raw"], "era5_{year}.zarr")

    # Output paths are resolved before extraction, because the neighbourhood
    # array is written straight into its file as a memory map rather than being
    # built in RAM and saved afterwards.
    os.makedirs(work_directory, exist_ok=True)
    dataset_path = os.path.join(work_directory,
                                f"{parsed_arguments.output_prefix}.npz")
    metadata_path = os.path.join(work_directory,
                                 f"{parsed_arguments.output_prefix}_stations.csv")
    neighbourhood_path = neighbourhood_array_path(dataset_path)

    verification_times, neighbourhood_values = extract_forecast_neighbourhoods(
        forecast_template, ALL_YEARS, LEAD_TIMES_HOURS, FORECAST_VARIABLES,
        stations["latitude"].to_numpy(), stations["longitude"].to_numpy(),
        output_path=neighbourhood_path)

    era5_values = extract_era5_at_stations(
        era5_template, ALL_YEARS, verification_times,
        stations["latitude"].to_numpy(), stations["longitude"].to_numpy())

    station_observations = load_station_observations(
        stations, hadisd_directory, verification_times)

    # ---- 4. climatology, coverage and elevation -----------------------------
    climatology_path = download_weatherbench_climatology(os.path.join(
        directories["raw"], "era5_climatology_1990-2019_00utc.zarr"))
    temperature_climatology, wind_climatology = load_station_climatology(
        climatology_path, stations["latitude"].to_numpy(),
        stations["longitude"].to_numpy())

    grid_elevations = read_station_grid_elevation(
        os.path.join(directories["raw"], "era5_static.nc"),
        stations["latitude"].to_numpy(), stations["longitude"].to_numpy())

    # Coverage is the fraction of 00 UTC training days on which the station
    # actually reported a temperature. Stations reporting only at 03 and 12 UTC
    # score near zero here and are dropped at training time.
    stations["training_coverage"] = training_coverage(
        station_observations, verification_times, TRAINING_YEARS)
    stations["grid_elevation_metres"] = grid_elevations

    # Attach each station's World Bank income group, reusing the classification
    # already built for the paper's figures so the two agree by construction.
    # Without this the income-gap summary in verification.py has nothing to
    # group by, which is the headline result of the analysis.
    # A manifest already carries these; a fresh draw does not.
    if "income_group" not in stations:
        stations = classify_stations(stations, directories)

    # The neighbourhood array is already on disk beside the npz, so only the
    # small arrays go in here; compressing 15 GB of float32 into an npz member
    # would have to be undone in full on every read.
    np.savez_compressed(
        dataset_path,
        verification_times=verification_times.values,
        era5_values=era5_values,
        station_observations=station_observations,
        temperature_climatology=temperature_climatology,
        wind_climatology=wind_climatology,
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
    print(f"wrote {neighbourhood_path}, shape {neighbourhood_values.shape} "
          f"({neighbourhood_values.nbytes / 1e9:.1f} GB)", flush=True)
    print(f"{int(usable.sum())} of {len(stations)} stations clear the "
          f"{MINIMUM_TRAINING_COVERAGE:.0%} coverage bar", flush=True)
    print(stations.loc[usable, "income_group"].value_counts(dropna=False).to_string(),
          flush=True)


if __name__ == "__main__":
    main()
