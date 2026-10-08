"""
Shared pieces of the station-level post-processing (sections 6 and 7).

Each station gets its own small network that predicts the error of the raw
forecast at that station, from the 5x5 grid-cell forecast neighbourhood
around it (2m temperature, 10m u and v wind and wind speed), the day of year
and the lead time. The corrected forecast is the raw forecast plus that
predicted error. One network is trained jointly over the three lead times.

Networks are trained with a mean squared error loss on standardised targets,
stopping early on whole held-out weeks of the training period: weather is
strongly autocorrelated from one day to the next, so random held-out days
would leak.

Forecasts are scored with the error decomposition of Bonavita & Geer (2026):
    RMSE^2 = bias^2 + information_error^2 + noise_error^2
where the noise error shrinks when a forecast is damped (lowering RMSE
without adding skill) and only a real gain in predictive information lowers
the information error. The anomaly correlation (ACC) is reported alongside.
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import LEAD_TIMES_HOURS, NEIGHBOURHOOD_HALF_WIDTH  # noqa: E402

RANDOM_SEED = 58
VALIDATION_BLOCK_DAYS = 7
VALIDATION_FRACTION = 0.2
MAXIMUM_EPOCHS = 150
EARLY_STOPPING_PATIENCE = 15

# Dry adiabatic lapse rate (K per metre), used to move a grid-cell temperature
# to the station's elevation (Trotta et al. 2025). It shifts the mean only, so
# it changes bias and RMSE but never ACC.
DRY_ADIABATIC_LAPSE_RATE = 0.0098

# A station-variable below any of these sample counts is skipped
MINIMUM_TRAINING_SAMPLES = 300
MINIMUM_TEST_SAMPLES = 100
MINIMUM_FITTING_SAMPLES = 200
MINIMUM_VALIDATION_SAMPLES = 50
MINIMUM_SAMPLES_PER_LEAD = 50
MINIMUM_SAMPLES_TO_SCORE = 30

SPECIFICATION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "tuned_specifications.json")


def load_station_dataset(directories, forecast_model):
    """
    Open one forecast model's station dataset, written by prepare_station_data.py.

    Inputs:
        directories (dict): output of common.setup_directories().
        forecast_model (str): 'pangu' or 'ifs'.

    Returns:
        dict with
            'stations'            pandas.DataFrame of station metadata, one row
                                  per position on the station axis
            'verification_times'  pandas.DatetimeIndex of 00 UTC valid times
            'neighbourhoods'      memory-mapped (station, time, lead, variable, 5, 5)
            'observations'        (time, variable, station) station observations
            'era5'                (time, variable, station) ERA5 at the stations
            'climatology'         (day of year, variable, station) ERA5 climatology
            'grid_elevations'     (station,) grid-cell elevation in metres
    """
    station_directory = directories["station_output"]
    stations = pd.read_csv(os.path.join(station_directory, "station_dataset_stations.csv"),
                           dtype={"usaf": str, "wban": str})
    arrays = np.load(os.path.join(station_directory, f"station_dataset_{forecast_model}.npz"))

    # The station axis is positional, so check the metadata lines up with it
    if (list(arrays["station_usaf"]) != list(stations["usaf"])
            or list(arrays["station_wban"]) != list(stations["wban"])):
        raise ValueError("station_dataset_stations.csv does not match the station axis of "
                         f"station_dataset_{forecast_model}.npz; rerun prepare_station_data.py")
    return {
        "stations": stations,
        "verification_times": pd.DatetimeIndex(arrays["verification_times"]),
        "neighbourhoods": np.load(os.path.join(
            station_directory, f"station_dataset_{forecast_model}_neighbourhoods.npy"),
            mmap_mode="r"),
        "observations": arrays["station_observations"],
        "era5": arrays["era5_values"],
        "climatology": np.stack([arrays["temperature_climatology"],
                                 arrays["wind_climatology"]], axis=1),
        "grid_elevations": arrays["grid_elevations"],
    }


def validation_week_blocks(verification_times, training_years):
    """
    Mark the training period and the whole weeks held out from it for early stopping.

    The days are cut into consecutive 7-day blocks and 20% of the blocks in
    the training period are held out at random (the same blocks for every
    station and variable).

    Inputs:
        verification_times (pandas.DatetimeIndex): the 00 UTC valid times.
        training_years (list of int): the years to train on.

    Returns:
        tuple (in_training_period, is_validation_block) of boolean arrays over
        verification_times.
    """
    in_training_period = np.isin(verification_times.year, training_years)
    block_number = np.arange(len(verification_times)) // VALIDATION_BLOCK_DAYS
    training_blocks = np.unique(block_number[in_training_period])
    held_out_blocks = np.random.default_rng(RANDOM_SEED).choice(
        training_blocks, size=max(1, int(len(training_blocks) * VALIDATION_FRACTION)),
        replace=False)
    return in_training_period, np.isin(block_number, held_out_blocks)


def build_station_features(neighbourhoods, station_position, verification_times):
    """
    Assemble one station's network inputs and raw forecasts, stacked over lead times.

    Each sample (one valid time and lead time) holds the 5x5 neighbourhood of
    temperature, u wind, v wind and wind speed, the sine and cosine of the day
    of year, and a one-hot lead time indicator.

    Inputs:
        neighbourhoods (numpy.ndarray): the dataset's (station, time, lead,
            variable, 5, 5) array, usually memory-mapped.
        station_position (int): the station's position on the station axis.
        verification_times (pandas.DatetimeIndex): the valid times.

    Returns:
        tuple (features, raw_forecast, time_position, lead_hours): features of
        shape (n_samples, n_features); raw_forecast of shape (2, n_samples),
        the forecast at the station's own grid cell for temperature and wind
        speed; and the time position and lead time of every sample.
    """
    number_of_times = len(verification_times)
    day_of_year_index = verification_times.dayofyear.values - 1

    # Read this station's block once, and add wind speed computed from the
    # components at full resolution
    station_window = np.array(neighbourhoods[station_position])
    wind_speed = np.sqrt(station_window[:, :, 1] ** 2 + station_window[:, :, 2] ** 2)
    fields = np.stack([station_window[:, :, 0], station_window[:, :, 1],
                       station_window[:, :, 2], wind_speed], axis=2)
    seasonal_terms = np.stack([np.sin(2 * np.pi * day_of_year_index / 365),
                               np.cos(2 * np.pi * day_of_year_index / 365)], axis=1)

    feature_blocks, temperature_blocks, wind_blocks = [], [], []
    for lead_position in range(len(LEAD_TIMES_HOURS)):
        lead_indicator = np.zeros((number_of_times, len(LEAD_TIMES_HOURS)))
        lead_indicator[:, lead_position] = 1
        feature_blocks.append(np.concatenate([fields[:, lead_position].reshape(number_of_times, -1),
                                              seasonal_terms, lead_indicator], axis=1))
        # The centre of the window is the station's own grid cell
        temperature_blocks.append(fields[:, lead_position, 0, NEIGHBOURHOOD_HALF_WIDTH,
                                         NEIGHBOURHOOD_HALF_WIDTH].astype(float))
        wind_blocks.append(fields[:, lead_position, 3, NEIGHBOURHOOD_HALF_WIDTH,
                                  NEIGHBOURHOOD_HALF_WIDTH].astype(float))
    raw_forecast = np.stack([np.concatenate(temperature_blocks), np.concatenate(wind_blocks)])
    time_position = np.tile(np.arange(number_of_times), len(LEAD_TIMES_HOURS))
    lead_hours = np.repeat(LEAD_TIMES_HOURS, number_of_times)
    return np.concatenate(feature_blocks), raw_forecast, time_position, lead_hours


def split_station_samples(is_usable, time_position, in_training_period, is_validation_block,
                          in_scored_period):
    """
    Split one station-variable's usable samples into fitting, validation and
    scored samples, or reject it when any of them is too small.

    Inputs:
        is_usable (numpy.ndarray of bool): samples with every needed value present.
        time_position (numpy.ndarray): each sample's position on the time axis.
        in_training_period, is_validation_block, in_scored_period
            (numpy.ndarray of bool): masks over the time axis, from
            validation_week_blocks() and the year being scored.

    Returns:
        tuple (is_training, is_fitting, is_validation, is_scored) of boolean
        masks over the samples, or None if the station-variable has fewer
        samples than the MINIMUM_* counts require.
    """
    is_training = is_usable & in_training_period[time_position]
    is_scored = is_usable & in_scored_period[time_position]
    is_fitting = is_training & ~is_validation_block[time_position]
    is_validation = is_training & is_validation_block[time_position]
    if (is_training.sum() < MINIMUM_TRAINING_SAMPLES or is_scored.sum() < MINIMUM_TEST_SAMPLES
            or is_fitting.sum() < MINIMUM_FITTING_SAMPLES
            or is_validation.sum() < MINIMUM_VALIDATION_SAMPLES):
        return None
    return is_training, is_fitting, is_validation, is_scored


def train_and_predict(fitting_features, fitting_targets, validation_features, validation_targets,
                      prediction_features, specification, seed):
    """
    Train one network and predict the forecast error on new samples.

    Inputs and targets are standardised with the fitting set's statistics.
    Training uses Adam and a mean squared error loss, and stops once the
    validation loss has not improved for EARLY_STOPPING_PATIENCE epochs; the
    best weights are restored.

    Inputs:
        fitting_features, validation_features, prediction_features
            (numpy.ndarray): (n_samples, n_features) inputs.
        fitting_targets, validation_targets (numpy.ndarray): forecast errors
            (truth minus raw forecast) to learn.
        specification (dict): 'hidden_units', 'layers', 'dropout',
            'learning_rate', 'weight_decay' and 'batch_size'.
        seed (int): torch seed, varied across ensemble members.

    Returns:
        numpy.ndarray of predicted errors for prediction_features, in the
        variable's units.
    """
    torch.manual_seed(seed)
    feature_mean = fitting_features.mean(axis=0)
    feature_deviation = fitting_features.std(axis=0) + 1e-8
    target_mean = fitting_targets.mean()
    target_deviation = fitting_targets.std() + 1e-8

    fitting_input = torch.tensor((fitting_features - feature_mean) / feature_deviation,
                                 dtype=torch.float32)
    fitting_output = torch.tensor((fitting_targets - target_mean) / target_deviation,
                                  dtype=torch.float32)
    validation_input = torch.tensor((validation_features - feature_mean) / feature_deviation,
                                    dtype=torch.float32)
    validation_output = torch.tensor((validation_targets - target_mean) / target_deviation,
                                     dtype=torch.float32)

    # 'layers' hidden layers of ReLU units with dropout, then a linear output
    layers, layer_input_width = [], fitting_features.shape[1]
    for _ in range(specification["layers"]):
        layers += [nn.Linear(layer_input_width, specification["hidden_units"]), nn.ReLU(),
                   nn.Dropout(specification["dropout"])]
        layer_input_width = specification["hidden_units"]
    layers.append(nn.Linear(layer_input_width, 1))
    network = nn.Sequential(*layers)

    optimizer = torch.optim.Adam(network.parameters(), lr=specification["learning_rate"],
                                 weight_decay=specification["weight_decay"])
    loss_function = nn.MSELoss()
    best_validation_loss, best_weights, epochs_without_improvement = np.inf, None, 0
    for _ in range(MAXIMUM_EPOCHS):
        # One pass over the fitting samples in random batches
        network.train()
        shuffled_rows = torch.randperm(len(fitting_input))
        for batch_start in range(0, len(fitting_input), specification["batch_size"]):
            batch_rows = shuffled_rows[batch_start:batch_start + specification["batch_size"]]
            optimizer.zero_grad()
            loss_function(network(fitting_input[batch_rows]).squeeze(-1),
                          fitting_output[batch_rows]).backward()
            optimizer.step()

        # Early stopping on the held-out weeks
        network.eval()
        with torch.no_grad():
            validation_loss = float(loss_function(network(validation_input).squeeze(-1),
                                                  validation_output))
        if validation_loss < best_validation_loss - 1e-5:
            best_validation_loss = validation_loss
            best_weights = {name: value.clone() for name, value in network.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                break

    network.load_state_dict(best_weights)
    network.eval()
    with torch.no_grad():
        prediction_input = torch.tensor((prediction_features - feature_mean) / feature_deviation,
                                        dtype=torch.float32)
        scaled_prediction = network(prediction_input).squeeze(-1).numpy()
    return scaled_prediction * target_deviation + target_mean


def decompose_forecast_error(forecast, truth, climatology):
    """
    Split a forecast's error into bias, information error and noise error.

    Anomalies are taken from the day-of-year climatology and centred, so a
    constant offset only affects the bias (Bonavita & Geer 2026, appendix A).
    The error along the true anomaly is the information error and the error
    perpendicular to it the noise error.

    Inputs:
        forecast, truth, climatology (numpy.ndarray): 1D arrays of the same
            length, in the variable's units.

    Returns:
        dict with 'bias', 'rmse', 'acc', 'forecast_activity',
        'truth_activity', 'information_error' and 'noise_error', or None when
        the forecast or the truth has no anomaly variance.
    """
    bias = float(np.mean(forecast - truth))
    root_mean_square_error = float(np.sqrt(np.mean((forecast - truth) ** 2)))

    # Centred anomalies from the same climatology for both
    forecast_anomaly = forecast - climatology
    truth_anomaly = truth - climatology
    forecast_anomaly = forecast_anomaly - forecast_anomaly.mean()
    truth_anomaly = truth_anomaly - truth_anomaly.mean()

    # Activity: how much each series moves around its climatology
    forecast_activity = float(np.sqrt(np.mean(forecast_anomaly ** 2)))
    truth_activity = float(np.sqrt(np.mean(truth_anomaly ** 2)))
    if forecast_activity == 0.0 or truth_activity == 0.0:
        return None
    anomaly_correlation = float(np.mean(forecast_anomaly * truth_anomaly)
                                / (forecast_activity * truth_activity))
    return {
        "bias": bias,
        "rmse": root_mean_square_error,
        "acc": anomaly_correlation,
        "forecast_activity": forecast_activity,
        "truth_activity": truth_activity,
        "information_error": float(abs(truth_activity - forecast_activity * anomaly_correlation)),
        "noise_error": float(forecast_activity
                             * np.sqrt(max(0.0, 1.0 - anomaly_correlation ** 2))),
    }
