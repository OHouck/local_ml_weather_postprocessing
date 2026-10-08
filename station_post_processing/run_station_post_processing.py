"""
Section 7: run the station-level post-processing.

For every station that reported on at least 70% of the training days, and for
each forecast model (Pangu and IFS), this trains on 2018-2021 and scores four
forecasts of 2022 against the station's observations, separately at each lead
time:
    raw                          the raw forecast at the station's grid cell
                                 (temperature moved to the station's height
                                 with the dry adiabatic lapse rate)
    linear_mos                   the station observation regressed on the raw
                                 forecast, fitted per lead time (a classic
                                 model output statistics baseline)
    network_trained_on_stations  the post-processing network trained on the
                                 station's observed error
    network_trained_on_era5      the same network trained on the error against
                                 ERA5 (as in the gridded analysis)
The two networks differ only in their training target, which isolates the
effect of training on station data. Each is the mean of an ensemble of
networks with different seeds, with the specification of each variable read
from tuned_specifications.json (see specification_search.py).

Results go to processed/station_finetuning/station_results/{model}_{continent}.csv,
one file per continent so that an interrupted run resumes where it stopped,
and are then combined into station_results_{model}.csv: one row per station,
variable, method and lead time with the verification metrics of
station_training.decompose_forecast_error.

Run on its own with:
    uv run python station_post_processing/run_station_post_processing.py
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (FORECAST_MODELS, LEAD_TIMES_HOURS, MINIMUM_TRAINING_COVERAGE,  # noqa: E402
                    TEST_YEAR, TRAINING_YEARS, VARIABLES, setup_directories)
from station_post_processing.station_training import (  # noqa: E402
    DRY_ADIABATIC_LAPSE_RATE, MINIMUM_SAMPLES_PER_LEAD, MINIMUM_SAMPLES_TO_SCORE,
    SPECIFICATION_FILE, build_station_features, decompose_forecast_error, load_station_dataset,
    split_station_samples, train_and_predict, validation_week_blocks)

TORCH_THREADS = 8


def post_process_station(dataset, station_position, specifications, in_training_period,
                         is_validation_block, in_test_period):
    """
    Train and score every method at one station, for both variables.

    Inputs:
        dataset (dict): output of station_training.load_station_dataset().
        station_position (int): the station's position on the station axis.
        specifications (dict): variable -> network specification, from
            tuned_specifications.json.
        in_training_period, is_validation_block, in_test_period
            (numpy.ndarray of bool): masks over the verification times.

    Returns:
        list of dict, one per variable, method and lead time, holding the
        verification metrics and the station's identifiers.
    """
    station = dataset["stations"].iloc[station_position]
    features, raw_forecasts, time_position, lead_hours = build_station_features(
        dataset["neighbourhoods"], station_position, dataset["verification_times"])
    day_of_year_index = dataset["verification_times"].dayofyear.values[time_position] - 1
    features_are_finite = np.isfinite(features).all(axis=1)

    result_rows = []
    for variable_position, variable in enumerate(VARIABLES):
        raw_forecast = raw_forecasts[variable_position]
        station_truth = dataset["observations"][time_position, variable_position, station_position]
        era5_truth = dataset["era5"][time_position, variable_position, station_position]

        # Samples with every value present, split into fitting, validation and test
        is_usable = (features_are_finite & np.isfinite(raw_forecast)
                     & np.isfinite(station_truth) & np.isfinite(era5_truth))
        sample_split = split_station_samples(is_usable, time_position, in_training_period,
                                             is_validation_block, in_test_period)
        if sample_split is None:
            continue
        is_training, is_fitting, is_validation, is_test = sample_split
        test_lead_hours = lead_hours[is_test]
        fitting_features, validation_features = features[is_fitting], features[is_validation]
        test_features = features[is_test]

        # Raw forecast, with temperature moved from the grid cell's height to the station's
        elevation_offset = 0.0
        if variable == "2m_temperature":
            elevation_offset = DRY_ADIABATIC_LAPSE_RATE * (
                dataset["grid_elevations"][station_position] - station["elevation_metres"])
        corrected_forecasts = {"raw": raw_forecast[is_test] + elevation_offset}

        # Linear MOS: regress the observation on the raw forecast, per lead time
        linear_mos = raw_forecast[is_test].copy()
        for lead_time in LEAD_TIMES_HOURS:
            training_at_lead = is_training & (lead_hours == lead_time)
            test_at_lead = test_lead_hours == lead_time
            if training_at_lead.sum() < MINIMUM_SAMPLES_PER_LEAD or not test_at_lead.any():
                continue
            design_matrix = np.vstack([np.ones(training_at_lead.sum()),
                                       raw_forecast[training_at_lead]]).T
            intercept, slope = np.linalg.lstsq(design_matrix, station_truth[training_at_lead],
                                               rcond=None)[0]
            linear_mos[test_at_lead] = intercept + slope * raw_forecast[is_test][test_at_lead]
        corrected_forecasts["linear_mos"] = linear_mos

        # The same network ensemble trained on two targets: the error against
        # the station, and the error against ERA5
        specification = specifications[variable]
        for method, truth in [("network_trained_on_stations", station_truth),
                              ("network_trained_on_era5", era5_truth)]:
            forecast_error = truth - raw_forecast
            ensemble_total = np.zeros(int(is_test.sum()))
            for seed in specification["seeds"]:
                ensemble_total += train_and_predict(
                    fitting_features, forecast_error[is_fitting], validation_features,
                    forecast_error[is_validation], test_features, specification, seed)
            corrected_forecasts[method] = (raw_forecast[is_test]
                                           + ensemble_total / len(specification["seeds"]))

        # Score every forecast against the station's observations, per lead time
        test_truth = station_truth[is_test]
        test_climatology = dataset["climatology"][day_of_year_index[is_test], variable_position,
                                                  station_position]
        for method, corrected in corrected_forecasts.items():
            for lead_time in LEAD_TIMES_HOURS:
                at_lead = test_lead_hours == lead_time
                if at_lead.sum() < MINIMUM_SAMPLES_TO_SCORE:
                    continue
                metrics = decompose_forecast_error(corrected[at_lead], test_truth[at_lead],
                                                   test_climatology[at_lead])
                if metrics is None:
                    continue
                metrics.update(usaf=station["usaf"], wban=station["wban"],
                               income_group=station["income_group"],
                               latitude=station["latitude"], longitude=station["longitude"],
                               variable=variable, method=method, lead_hours=lead_time,
                               n_test=int(at_lead.sum()))
                result_rows.append(metrics)
    return result_rows


def run():
    """
    Post-process every qualifying station, for both forecast models.

    Inputs:
        None.

    Returns:
        None. Writes station_results_{model}.csv for each forecast model.
    """
    torch.set_num_threads(TORCH_THREADS)
    directories = setup_directories()
    results_directory = os.path.join(directories["station_output"], "station_results")
    os.makedirs(results_directory, exist_ok=True)
    with open(SPECIFICATION_FILE) as specification_file:
        specifications = json.load(specification_file)

    for forecast_model in FORECAST_MODELS:
        dataset = load_station_dataset(directories, forecast_model)
        stations = dataset["stations"]
        in_training_period, is_validation_block = validation_week_blocks(
            dataset["verification_times"], TRAINING_YEARS)
        in_test_period = dataset["verification_times"].year.values == TEST_YEAR

        # One results file per continent, skipped if it already exists
        for continent in sorted(stations["continent"].unique()):
            continent_path = os.path.join(results_directory, f"{forecast_model}_{continent}.csv")
            if os.path.exists(continent_path):
                continue
            station_positions = np.flatnonzero(
                (stations["continent"] == continent)
                & (stations["training_coverage"] >= MINIMUM_TRAINING_COVERAGE))
            print(f"{forecast_model} {continent}: {len(station_positions)} stations", flush=True)
            continent_rows = []
            for count, station_position in enumerate(station_positions, start=1):
                continent_rows += post_process_station(dataset, station_position, specifications,
                                                       in_training_period, is_validation_block,
                                                       in_test_period)
                if count % 25 == 0:
                    print(f"  {count} of {len(station_positions)} stations", flush=True)
            pd.DataFrame(continent_rows).assign(forecast_model=forecast_model).to_csv(
                continent_path, index=False)

        # Combine the continents into one results table per forecast model
        continent_tables = [pd.read_csv(path, dtype={"usaf": str, "wban": str}) for path in
                            sorted(glob.glob(os.path.join(results_directory,
                                                          f"{forecast_model}_*.csv")))]
        combined = pd.concat([table for table in continent_tables if not table.empty],
                             ignore_index=True)
        results_path = os.path.join(directories["station_output"],
                                    f"station_results_{forecast_model}.csv")
        combined.to_csv(results_path, index=False)
        print(f"Wrote {results_path}: {len(combined)} rows", flush=True)


if __name__ == "__main__":
    run()
