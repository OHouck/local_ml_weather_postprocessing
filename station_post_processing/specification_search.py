"""
Section 6: choose the network specification for the station post-processing.

Every combination of hidden-layer width, number of hidden layers and dropout
rate in SEARCH_GRID is trained, as a 5-seed ensemble, on a development split
that never touches the 2022 test year: train on 2018-2020 (early stopping on
held-out weeks), score on 2021. Training uses the same code and loss (mean
squared error) as the production run.

The search runs on the Pangu dataset over a reproducible, income-balanced
sample of stations (SEARCH_STATIONS_PER_INCOME_GROUP of each group). A
specification's score is its mean anomaly correlation (ACC) against the
station observations over every sampled station and lead time, with the
station-trained network.

Writes:
    processed/station_finetuning/station_specification_search.csv
        one row per variable and specification: mean development-year ACC,
        its standard error and the number of station-leads, best first
    station_post_processing/tuned_specifications.json
        the best specification of each variable, read by
        run_station_post_processing.py

Run on its own with:
    uv run python station_post_processing/specification_search.py
"""

import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (LEAD_TIMES_HOURS, MINIMUM_TRAINING_COVERAGE, TRAINING_YEARS,  # noqa: E402
                    VARIABLES, setup_directories)
from data_preparation.select_stations import select_income_balanced_sample  # noqa: E402
from station_post_processing.station_training import (  # noqa: E402
    MINIMUM_SAMPLES_TO_SCORE, SPECIFICATION_FILE, build_station_features,
    decompose_forecast_error, load_station_dataset, split_station_samples, train_and_predict,
    validation_week_blocks)

# Train on all but the last training year, and score on that last year
SEARCH_TRAINING_YEARS = TRAINING_YEARS[:-1]
DEVELOPMENT_YEAR = TRAINING_YEARS[-1]
SEARCH_STATIONS_PER_INCOME_GROUP = 50
TORCH_THREADS = 8

# The specifications compared; the other settings are fixed
SEARCH_GRID = {"hidden_units": [32, 64, 128, 256], "layers": [1, 2, 3], "dropout": [0.1, 0.2]}
FIXED_SETTINGS = {"learning_rate": 0.001, "weight_decay": 0.0001, "batch_size": 256,
                  "seeds": [58, 59, 60, 61, 62]}


def run():
    """
    Score every specification on the development year and save the best per variable.

    Inputs:
        None.

    Returns:
        None. Writes the two files listed in the module description.
    """
    torch.set_num_threads(TORCH_THREADS)
    directories = setup_directories()
    dataset = load_station_dataset(directories, "pangu")
    stations = dataset["stations"]
    verification_times = dataset["verification_times"]
    in_training_period, is_validation_block = validation_week_blocks(verification_times,
                                                                     SEARCH_TRAINING_YEARS)
    in_development_year = verification_times.year.values == DEVELOPMENT_YEAR
    specifications = [dict(zip(SEARCH_GRID, values), **FIXED_SETTINGS)
                      for values in itertools.product(*SEARCH_GRID.values())]

    # An income-balanced sample of the stations that clear the coverage bar
    qualifying = stations[stations["training_coverage"] >= MINIMUM_TRAINING_COVERAGE]
    sample_positions = np.sort(select_income_balanced_sample(
        qualifying, SEARCH_STATIONS_PER_INCOME_GROUP).index.to_numpy())
    print(f"Searching {len(specifications)} specifications on {len(sample_positions)} stations")

    score_rows = []
    for count, station_position in enumerate(sample_positions, start=1):
        features, raw_forecasts, time_position, lead_hours = build_station_features(
            dataset["neighbourhoods"], station_position, verification_times)
        day_of_year_index = verification_times.dayofyear.values[time_position] - 1
        for variable_position, variable in enumerate(VARIABLES):
            raw_forecast = raw_forecasts[variable_position]
            station_truth = dataset["observations"][time_position, variable_position,
                                                    station_position]
            era5_truth = dataset["era5"][time_position, variable_position, station_position]

            # Fitting and validation samples in the search years, scored in
            # the development year (the same samples production would use)
            is_usable = (np.isfinite(features).all(axis=1) & np.isfinite(raw_forecast)
                         & np.isfinite(station_truth) & np.isfinite(era5_truth))
            sample_split = split_station_samples(is_usable, time_position, in_training_period,
                                                 is_validation_block, in_development_year)
            if sample_split is None:
                continue
            _, is_fitting, is_validation, is_development = sample_split
            fitting_features, validation_features = features[is_fitting], features[is_validation]
            development_features = features[is_development]
            forecast_error = station_truth - raw_forecast
            development_climatology = dataset["climatology"][
                day_of_year_index[is_development], variable_position, station_position]

            # Each specification's ensemble forecast, scored per lead time
            for specification_index, specification in enumerate(specifications):
                ensemble_total = np.zeros(int(is_development.sum()))
                for seed in specification["seeds"]:
                    ensemble_total += train_and_predict(
                        fitting_features, forecast_error[is_fitting], validation_features,
                        forecast_error[is_validation], development_features, specification, seed)
                corrected = (raw_forecast[is_development]
                             + ensemble_total / len(specification["seeds"]))
                for lead_time in LEAD_TIMES_HOURS:
                    at_lead = lead_hours[is_development] == lead_time
                    if at_lead.sum() < MINIMUM_SAMPLES_TO_SCORE:
                        continue
                    metrics = decompose_forecast_error(corrected[at_lead],
                                                       station_truth[is_development][at_lead],
                                                       development_climatology[at_lead])
                    if metrics is not None:
                        score_rows.append({"variable": variable,
                                           "specification": specification_index,
                                           "acc": metrics["acc"]})
        print(f"  {count} of {len(sample_positions)} stations", flush=True)

    # Rank the specifications by their mean development-year ACC
    scores = pd.DataFrame(score_rows).groupby(["variable", "specification"])["acc"].agg(
        mean_acc="mean", standard_error="sem", station_leads="count").reset_index()
    for setting in SEARCH_GRID:
        scores[setting] = [specifications[index][setting] for index in scores["specification"]]
    scores = scores.sort_values(["variable", "mean_acc"], ascending=[True, False])
    scores_path = os.path.join(directories["station_output"], "station_specification_search.csv")
    scores.drop(columns="specification").to_csv(scores_path, index=False)
    print(scores.drop(columns="specification").to_string(index=False))

    # Save the best specification of each variable for the production run
    best_specifications = {variable: specifications[int(
        scores.loc[scores["variable"] == variable, "specification"].iloc[0])]
        for variable in VARIABLES}
    with open(SPECIFICATION_FILE, "w") as specification_file:
        json.dump(best_specifications, specification_file, indent=2)
    print(f"Wrote {scores_path} and {SPECIFICATION_FILE}")


if __name__ == "__main__":
    run()
