"""Train and evaluate per-station forecast post-processing models.

Each station gets its own small network that predicts the ERROR of the raw
forecast at that station. The corrected forecast is then

    corrected = raw_forecast + predicted_error

which matches the convention used elsewhere in this project.

The experiment that motivates the module is a single controlled contrast. Two
networks are trained with identical architecture, identical inputs and identical
training protocol; the only difference is what they are asked to predict:

    'station' target - the error against real ISD station observations
    'era5' target    - the error against ERA5, as in the main paper

Three reference methods bracket the neural networks. A constant offset and a
linear rescaling are both affine, so they cannot change anomaly correlation and
serve as controls that the metric behaves as expected. A day-of-year varying
offset is not affine and can move ACC, which separates seasonal bias structure
from genuinely state-dependent correction.

Training runs jointly across lead times with a one-hot lead indicator. Splitting
into separate per-lead models was tested and performed worse, because each
station has only around 1,450 usable days per lead.
"""

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from forecast_features import CLIMATOLOGY_SMOOTHING_DAYS
from verification import decompose_forecast_error

RANDOM_SEED = 58

# Variables that can be corrected, in the order they appear on the observation
# and climatology arrays.
VARIABLE_NAMES = {0: "2m_temperature", 1: "10m_wind_speed"}

# Dry adiabatic lapse rate in K per metre, used to adjust a grid-cell forecast
# to the station's own elevation before verifying temperature (Trotta et al.).
# This shifts the mean only, so it changes bias and RMSE but never ACC.
DRY_ADIABATIC_LAPSE_RATE = 0.0098

# Validation blocks are whole weeks. Random daily splits leak, because weather
# is strongly autocorrelated from one day to the next.
VALIDATION_BLOCK_DAYS = 7
VALIDATION_FRACTION = 0.2

MAXIMUM_EPOCHS = 150
EARLY_STOPPING_PATIENCE = 15

# Minimum sample counts. A station-variable that falls below any of these is
# skipped rather than fitted on too little data to be meaningful.
MINIMUM_TRAINING_SAMPLES = 300
MINIMUM_TEST_SAMPLES = 100
MINIMUM_FITTING_SAMPLES = 200
MINIMUM_VALIDATION_SAMPLES = 50
MINIMUM_SAMPLES_PER_LEAD = 50
MINIMUM_SAMPLES_TO_SCORE = 30

# Stations reporting on fewer than this fraction of 00 UTC training days are
# dropped. Shared with prepare_station_dataset.py, which reports the same bar.
MINIMUM_TRAINING_COVERAGE = 0.70


def build_training_and_validation_masks(verification_times, training_years):
    """Split the training period into fitting and early-stopping subsets.

    Inputs:
        verification_times (pd.DatetimeIndex): 00 UTC times of every sample.
        training_years (list[int]): years that make up the training period.

    Returns:
        tuple: (in_training_period, is_validation_block) as boolean arrays over
            verification_times. in_training_period marks the training years;
            is_validation_block marks the week-long blocks held out for early
            stopping within those years.
    """
    in_training_period = np.isin(verification_times.year, training_years)

    # Assign each day to a week-long block, then hold out whole blocks. Holding
    # out individual days would put near-duplicate weather in both subsets.
    block_index = np.arange(len(verification_times)) // VALIDATION_BLOCK_DAYS
    training_blocks = np.unique(block_index[in_training_period])

    random_generator = np.random.default_rng(RANDOM_SEED)
    number_to_hold_out = max(1, int(len(training_blocks) * VALIDATION_FRACTION))
    held_out_blocks = set(random_generator.choice(
        training_blocks, size=number_to_hold_out, replace=False).tolist())

    is_validation_block = np.array(
        [block in held_out_blocks for block in block_index])
    return in_training_period, is_validation_block


def build_feature_matrix(neighbourhood_values, station_position, verification_times,
                         lead_times_hours, neighbourhood_half_width):
    """Assemble the network inputs and both raw forecasts for one station.

    Samples are stacked over lead times, so a station contributes
    (n_times x n_leads) rows. Each row holds the neighbourhood of four fields
    (temperature, eastward wind, northward wind, wind speed), two seasonal
    terms, and a one-hot lead indicator.

    The feature block does not depend on which variable is being corrected, so
    it is built once per station and both raw forecasts are returned alongside
    it. Only the station's own slice of the (large) neighbourhood array is read,
    which keeps the working set at a few megabytes instead of gigabytes.

    Inputs:
        neighbourhood_values (np.ndarray): shape
            (n_times, n_leads, n_variables, n_stations, window, window) from
            extract_forecast_neighbourhoods, with variables ordered
            [2m_temperature, 10m_u_component_of_wind, 10m_v_component_of_wind].
        station_position (int): index of the station along the station axis.
        verification_times (pd.DatetimeIndex): times matching axis 0.
        lead_times_hours (list[int]): lead times, in the order stored on axis 1.
        neighbourhood_half_width (int): half-width of the stored window in grid
            cells, used to locate the station's own centre cell.

    Returns:
        tuple: (features, raw_forecast_by_variable, time_position, lead_hours)
            where features is float64 of shape (n_samples, n_features),
            raw_forecast_by_variable maps variable position (0 temperature,
            1 wind speed) to the uncorrected forecast at the station's own grid
            cell, time_position indexes back into verification_times, and
            lead_hours records which lead each sample came from.
    """
    day_of_year_index = verification_times.dayofyear.values - 1
    number_of_times = len(verification_times)
    centre_cell = neighbourhood_half_width

    # Take this station's slice up front. Deriving wind speed from the full
    # array would allocate hundreds of megabytes per call and then discard all
    # but one station's worth of it.
    station_window = neighbourhood_values[:, :, :, station_position]

    # Wind speed is derived at native resolution before any averaging, because
    # the speed of the mean wind is not the mean of the speeds.
    eastward_wind = station_window[:, :, 1]
    northward_wind = station_window[:, :, 2]
    wind_speed = np.sqrt(eastward_wind ** 2 + northward_wind ** 2)
    all_fields = np.stack(
        [station_window[:, :, 0], eastward_wind, northward_wind, wind_speed],
        axis=2)

    # Sine and cosine of day-of-year let the network represent a smooth seasonal
    # cycle without a discontinuity at the new year. These do not vary by lead,
    # so they are built once outside the loop.
    seasonal_terms = np.stack([
        np.sin(2 * np.pi * day_of_year_index / 365),
        np.cos(2 * np.pi * day_of_year_index / 365)], axis=1)

    feature_rows, time_rows, lead_rows = [], [], []
    raw_rows_by_variable = {0: [], 1: []}
    for lead_position, lead_hours in enumerate(lead_times_hours):
        flattened_neighbourhood = all_fields[:, lead_position].reshape(
            number_of_times, -1)

        lead_indicator = np.zeros((number_of_times, len(lead_times_hours)))
        lead_indicator[:, lead_position] = 1

        feature_rows.append(np.concatenate(
            [flattened_neighbourhood, seasonal_terms, lead_indicator], axis=1))

        # Field 0 is temperature and field 3 is the derived wind speed; the
        # centre cell of the window is the station's own grid cell.
        raw_rows_by_variable[0].append(
            all_fields[:, lead_position, 0, centre_cell, centre_cell].astype(float))
        raw_rows_by_variable[1].append(
            all_fields[:, lead_position, 3, centre_cell, centre_cell].astype(float))

        time_rows.append(np.arange(number_of_times))
        lead_rows.append(np.full(number_of_times, lead_hours))

    raw_forecast_by_variable = {
        variable_position: np.concatenate(rows)
        for variable_position, rows in raw_rows_by_variable.items()}
    return (np.concatenate(feature_rows), raw_forecast_by_variable,
            np.concatenate(time_rows), np.concatenate(lead_rows))


def train_one_network(training_features, training_targets, validation_features,
                      validation_targets, specification, seed):
    """Fit one post-processing network and return it with its scaling constants.

    Inputs and targets are standardised using training-set statistics only.
    Training stops early when validation loss has not improved for
    EARLY_STOPPING_PATIENCE epochs, and the best-scoring weights are restored.

    Inputs:
        training_features (np.ndarray): (n_train, n_features).
        training_targets (np.ndarray): (n_train,) forecast errors to predict.
        validation_features (np.ndarray): (n_validation, n_features).
        validation_targets (np.ndarray): (n_validation,).
        specification (dict): keys 'hidden_units' (int), 'layers' (int),
            'dropout' (float), 'learning_rate' (float), 'weight_decay' (float),
            'batch_size' (int) and 'use_huber_loss' (bool).
        seed (int): torch seed, varied across ensemble members.

    Returns:
        tuple: (network, feature_means, feature_deviations, target_mean,
            target_deviation). The scaling constants are returned rather than
            captured in a closure so that nothing holds a reference to the
            training tensors once fitting is done.
    """
    torch.manual_seed(seed)

    # Standardise so that no single input dominates the first layer purely
    # because of its units. Statistics come from training data only.
    feature_means = training_features.mean(axis=0)
    feature_deviations = training_features.std(axis=0) + 1e-8
    target_mean = training_targets.mean()
    target_deviation = training_targets.std() + 1e-8

    training_input = torch.tensor(
        (training_features - feature_means) / feature_deviations, dtype=torch.float32)
    training_output = torch.tensor(
        (training_targets - target_mean) / target_deviation, dtype=torch.float32)
    validation_input = torch.tensor(
        (validation_features - feature_means) / feature_deviations, dtype=torch.float32)
    validation_output = torch.tensor(
        (validation_targets - target_mean) / target_deviation, dtype=torch.float32)

    # Build the stack explicitly rather than in a helper, so the architecture is
    # visible at the point where it is trained.
    layer_stack, input_width = [], training_features.shape[1]
    for _ in range(specification["layers"]):
        layer_stack += [nn.Linear(input_width, specification["hidden_units"]),
                        nn.ReLU(),
                        nn.Dropout(specification["dropout"])]
        input_width = specification["hidden_units"]
    layer_stack.append(nn.Linear(input_width, 1))
    network = nn.Sequential(*layer_stack)

    optimiser = torch.optim.Adam(network.parameters(),
                                 lr=specification["learning_rate"],
                                 weight_decay=specification["weight_decay"])
    # Huber is less sensitive than squared error to the occasional bad station
    # report, and tested better on the development year for both variables.
    loss_function = (nn.HuberLoss(delta=1.0) if specification["use_huber_loss"]
                     else nn.MSELoss())

    best_validation_loss, best_weights, epochs_without_improvement = np.inf, None, 0
    number_of_samples = len(training_input)
    for _ in range(MAXIMUM_EPOCHS):
        network.train()
        shuffled_order = torch.randperm(number_of_samples)
        for batch_start in range(0, number_of_samples, specification["batch_size"]):
            batch_rows = shuffled_order[
                batch_start:batch_start + specification["batch_size"]]
            optimiser.zero_grad()
            batch_prediction = network(training_input[batch_rows]).squeeze(-1)
            loss_function(batch_prediction, training_output[batch_rows]).backward()
            optimiser.step()

        network.eval()
        with torch.no_grad():
            validation_loss = float(loss_function(
                network(validation_input).squeeze(-1), validation_output))

        if validation_loss < best_validation_loss - 1e-5:
            best_validation_loss = validation_loss
            best_weights = {name: value.clone()
                            for name, value in network.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                break

    network.load_state_dict(best_weights)
    network.eval()
    return (network, feature_means, feature_deviations, target_mean,
            target_deviation)


def fit_reference_methods(raw_forecast, station_truth, day_of_year_index,
                          lead_hours, is_training, is_test, lead_times_hours):
    """Fit the three non-neural reference corrections and predict on the test set.

    Each is fitted separately per lead time, because forecast bias grows with
    lead. The first two are affine in the forecast and therefore cannot change
    anomaly correlation; that makes them a useful check on the metric.

    Inputs:
        raw_forecast (np.ndarray): uncorrected forecast for every sample.
        station_truth (np.ndarray): verifying station observation, may contain NaN.
        day_of_year_index (np.ndarray): 0-based day-of-year for every sample.
        lead_hours (np.ndarray): lead time of every sample.
        is_training (np.ndarray): boolean mask of usable training samples.
        is_test (np.ndarray): boolean mask of test samples.
        lead_times_hours (list[int]): lead times to fit separately.

    Returns:
        dict: maps method name ('mean_bias', 'linear_mos', 'seasonal_bias') to
            corrected forecasts on the test samples.
    """
    raw_forecast_on_test = raw_forecast[is_test]
    test_day_of_year = day_of_year_index[is_test]
    test_lead_hours = lead_hours[is_test]

    # Build the corrected forecasts directly rather than storing increments,
    # which avoids subtracting the raw forecast only to add it back later.
    corrected_by_method = {
        "mean_bias": raw_forecast_on_test.copy(),
        "linear_mos": raw_forecast_on_test.copy(),
        "seasonal_bias": raw_forecast_on_test.copy(),
    }

    for lead in lead_times_hours:
        training_at_lead = is_training & (lead_hours == lead)
        test_rows_at_lead = test_lead_hours == lead
        if training_at_lead.sum() < MINIMUM_SAMPLES_PER_LEAD or not test_rows_at_lead.any():
            continue

        training_forecast = raw_forecast[training_at_lead]
        training_truth = station_truth[training_at_lead]
        forecast_at_lead = raw_forecast_on_test[test_rows_at_lead]

        # A single constant offset, the simplest possible bias correction.
        corrected_by_method["mean_bias"][test_rows_at_lead] = (
            forecast_at_lead + np.mean(training_truth - training_forecast))

        # Classic model output statistics: regress the observation on the
        # forecast, then apply the fitted line to the test forecasts.
        design_matrix = np.vstack(
            [np.ones(training_at_lead.sum()), training_forecast]).T
        coefficients, *_ = np.linalg.lstsq(design_matrix, training_truth, rcond=None)
        corrected_by_method["linear_mos"][test_rows_at_lead] = (
            coefficients[0] + coefficients[1] * forecast_at_lead)

        # A bias that varies with day of year. bincount accumulates the per-day
        # sums and counts in a single pass, instead of scanning the training
        # rows once for each of the 366 days.
        training_errors = training_truth - training_forecast
        training_days = day_of_year_index[training_at_lead]
        error_sum_by_day = np.bincount(training_days, weights=training_errors,
                                       minlength=366)
        error_count_by_day = np.bincount(training_days, minlength=366)
        error_by_day = np.divide(
            error_sum_by_day, error_count_by_day,
            out=np.full(366, np.nan), where=error_count_by_day > 0)
        error_by_day = pd.Series(error_by_day).interpolate(
            limit_direction="both").to_numpy()

        # Smooth with the same wrapped window used for the climatology, so days
        # either side of the new year are treated as adjacent.
        half_window = CLIMATOLOGY_SMOOTHING_DAYS // 2
        wrapped = np.concatenate(
            [error_by_day[-half_window:], error_by_day, error_by_day[:half_window]])
        smoothing_kernel = (np.ones(CLIMATOLOGY_SMOOTHING_DAYS)
                            / CLIMATOLOGY_SMOOTHING_DAYS)
        smoothed_by_day = np.convolve(wrapped, smoothing_kernel, mode="valid")
        corrected_by_method["seasonal_bias"][test_rows_at_lead] = (
            forecast_at_lead + smoothed_by_day[test_day_of_year[test_rows_at_lead]])

    return corrected_by_method


def run_station_experiment(dataset_path, station_metadata_path, specification_path,
                           output_path, training_years, test_year,
                           minimum_training_coverage=MINIMUM_TRAINING_COVERAGE):
    """Train every method at every qualifying station and write per-station metrics.

    Inputs:
        dataset_path (str): npz written by prepare_station_dataset.py, holding
            'verification_times', 'neighbourhood_values', 'era5_values',
            'station_observations', 'temperature_climatology',
            'wind_climatology', 'grid_elevations', 'lead_times_hours',
            'neighbourhood_half_width' and the 'station_usaf'/'station_wban'
            identifiers used to verify alignment with the metadata csv.
        station_metadata_path (str): csv of station metadata, one row per
            station in the same order as the dataset's station axis. Must carry
            'usaf', 'wban', 'elevation_metres' and 'training_coverage', and
            optionally 'income_group' and 'latitude'.
        specification_path (str): json list of network specifications; see
            configs/ for the tuned settings. A specification may carry a
            'variables' list to restrict it to certain variables.
        output_path (str): csv to write per-station, per-lead metrics to.
        training_years (list[int]): years to fit on.
        test_year (int): year to evaluate on; never seen during fitting or
            model selection.
        minimum_training_coverage (float): drop stations reporting on fewer than
            this fraction of training-period days.

    Returns:
        pd.DataFrame: the same table that is written to output_path.

    Raises:
        ValueError: if the metadata csv is not row-aligned with the station
            axis of the dataset.
    """
    dataset = np.load(dataset_path, allow_pickle=True)
    station_metadata = pd.read_csv(station_metadata_path)
    verification_times = pd.DatetimeIndex(dataset["verification_times"])
    neighbourhood_values = dataset["neighbourhood_values"]
    era5_values = dataset["era5_values"]
    station_observations = dataset["station_observations"]
    climatology_by_variable = {0: dataset["temperature_climatology"],
                               1: dataset["wind_climatology"]}
    grid_elevations = dataset["grid_elevations"]

    # Leads and window geometry come from the dataset rather than being
    # re-declared here, so the two scripts cannot silently disagree.
    lead_times_hours = [int(lead) for lead in dataset["lead_times_hours"]]
    neighbourhood_half_width = int(dataset["neighbourhood_half_width"])

    # The station axis of every array is positional, so the metadata csv must be
    # row-aligned with it. Check that explicitly: pairing a csv with a
    # differently-filtered npz would otherwise train one station's forecast
    # against another station's observations, with no visible error.
    dataset_usaf = [str(value) for value in dataset["station_usaf"]]
    dataset_wban = [str(value) for value in dataset["station_wban"]]
    metadata_usaf = [str(value) for value in station_metadata["usaf"]]
    metadata_wban = [str(value) for value in station_metadata["wban"]]
    if dataset_usaf != metadata_usaf or dataset_wban != metadata_wban:
        raise ValueError(
            f"station metadata ({len(metadata_usaf)} rows) is not aligned with "
            f"the dataset station axis ({len(dataset_usaf)} entries); they must "
            f"come from the same prepare_station_dataset.py run")

    with open(specification_path) as specification_file:
        specifications = json.load(specification_file)

    day_of_year_index = verification_times.dayofyear.values - 1
    in_training_period, is_validation_block = build_training_and_validation_masks(
        verification_times, training_years)
    in_test_period = verification_times.year.values == test_year

    qualifying_stations = station_metadata.index[
        station_metadata["training_coverage"] >= minimum_training_coverage].tolist()
    print(f"training on {len(qualifying_stations)} stations "
          f"(years {training_years}), testing on {test_year}", flush=True)

    result_rows = []
    for stations_done, station_position in enumerate(qualifying_stations):
        station = station_metadata.iloc[station_position]

        # The predictor block is the same for both variables, so it is built
        # once per station and reused below.
        (features, raw_forecast_by_variable, time_position,
         lead_hours) = build_feature_matrix(
            neighbourhood_values, station_position, verification_times,
            lead_times_hours, neighbourhood_half_width)
        features_are_finite = np.isfinite(features).all(axis=1)

        for variable_position, variable_name in VARIABLE_NAMES.items():
            raw_forecast = raw_forecast_by_variable[variable_position]
            station_truth = station_observations[:, variable_position,
                                                 station_position][time_position]
            era5_truth = era5_values[:, variable_position,
                                     station_position][time_position]
            station_climatology = climatology_by_variable[variable_position][
                :, station_position]

            # A sample is usable only when the forecast, both candidate truths
            # and every feature are present.
            is_usable = (features_are_finite
                         & np.isfinite(raw_forecast)
                         & np.isfinite(station_truth)
                         & np.isfinite(era5_truth))
            is_training = is_usable & in_training_period[time_position]
            is_test = is_usable & in_test_period[time_position]
            if (is_training.sum() < MINIMUM_TRAINING_SAMPLES
                    or is_test.sum() < MINIMUM_TEST_SAMPLES):
                continue

            is_fitting = is_training & ~is_validation_block[time_position]
            is_validation = is_training & is_validation_block[time_position]
            if (is_fitting.sum() < MINIMUM_FITTING_SAMPLES
                    or is_validation.sum() < MINIMUM_VALIDATION_SAMPLES):
                continue

            # Slice the feature matrix once per station-variable. Re-slicing it
            # inside the target/specification/seed loops would copy tens of
            # megabytes on every iteration.
            fitting_features = features[is_fitting]
            validation_features = features[is_validation]
            test_features = features[is_test]

            # Temperature is adjusted from the grid cell's mean elevation to the
            # station's own elevation. A station below the grid mean sits in
            # warmer air than the grid value implies, and vice versa.
            elevation_offset = 0.0
            if variable_position == 0:
                elevation_difference = (grid_elevations[station_position]
                                        - station["elevation_metres"])
                elevation_offset = DRY_ADIABATIC_LAPSE_RATE * elevation_difference

            corrected_forecasts = {"raw": raw_forecast[is_test] + elevation_offset}
            corrected_forecasts.update(fit_reference_methods(
                raw_forecast, station_truth, day_of_year_index[time_position],
                lead_hours, is_training, is_test, lead_times_hours))

            # The controlled contrast: identical networks, different targets.
            for target_name, target_values in [("station", station_truth),
                                               ("era5", era5_truth)]:
                forecast_error_target = target_values - raw_forecast
                for specification in specifications:
                    # A specification may declare which variables it applies to,
                    # so a network tuned for wind is not also run on
                    # temperature. Omitting the field means "all variables".
                    applies_to = specification.get("variables")
                    if applies_to is not None and variable_name not in applies_to:
                        continue

                    prediction_total = np.zeros(int(is_test.sum()))
                    ensemble_seeds = specification.get("seeds", [RANDOM_SEED])
                    for seed in ensemble_seeds:
                        (network, feature_means, feature_deviations, target_mean,
                         target_deviation) = train_one_network(
                            fitting_features, forecast_error_target[is_fitting],
                            validation_features,
                            forecast_error_target[is_validation],
                            specification, seed)

                        # Apply the same standardisation the network was fitted
                        # with, then undo the target scaling on the output.
                        standardised_test = (
                            (test_features - feature_means) / feature_deviations)
                        with torch.no_grad():
                            scaled_prediction = network(torch.tensor(
                                standardised_test,
                                dtype=torch.float32)).squeeze(-1).numpy()
                        prediction_total += (
                            scaled_prediction * target_deviation + target_mean)

                    method_name = f"mlp_{target_name}_{specification['name']}"
                    corrected_forecasts[method_name] = (
                        raw_forecast[is_test]
                        + prediction_total / len(ensemble_seeds))

            # Score everything against the station observations, per lead time.
            test_climatology = station_climatology[
                day_of_year_index[time_position][is_test]]
            test_lead_hours = lead_hours[is_test]
            test_truth = station_truth[is_test]

            for method_name, corrected in corrected_forecasts.items():
                for lead in lead_times_hours:
                    at_this_lead = test_lead_hours == lead
                    if at_this_lead.sum() < MINIMUM_SAMPLES_TO_SCORE:
                        continue
                    metrics = decompose_forecast_error(
                        corrected[at_this_lead], test_truth[at_this_lead],
                        test_climatology[at_this_lead])
                    if metrics is None:
                        continue
                    metrics.update(
                        usaf=station["usaf"], wban=station["wban"],
                        income_group=station.get("income_group"),
                        latitude=station.get("latitude"),
                        variable=variable_name, method=method_name,
                        lead_hours=lead, n_test=int(at_this_lead.sum()))
                    result_rows.append(metrics)

        if (stations_done + 1) % 25 == 0:
            print(f"  {stations_done + 1}/{len(qualifying_stations)} stations",
                  flush=True)

    results = pd.DataFrame(result_rows)
    results.to_csv(output_path, index=False)
    print(f"wrote {len(results)} rows to {output_path}", flush=True)
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True,
                        help="npz from prepare_station_dataset.py")
    parser.add_argument("--station_metadata", required=True,
                        help="csv of station metadata")
    parser.add_argument("--specifications", required=True,
                        help="json list of network specifications")
    parser.add_argument("--output", required=True, help="csv to write")
    parser.add_argument("--training_years", type=int, nargs="+",
                        default=[2018, 2019, 2020, 2021])
    parser.add_argument("--test_year", type=int, default=2022)
    parsed_arguments = parser.parse_args()

    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "8")))
    run_station_experiment(
        parsed_arguments.dataset, parsed_arguments.station_metadata,
        parsed_arguments.specifications, parsed_arguments.output,
        parsed_arguments.training_years, parsed_arguments.test_year)
