"""
Training and applying the gridded post-processing networks.

Every network predicts the error of the raw forecast on a 6x6 degree patch;
the corrected forecast is the raw forecast plus the predicted error. Inputs,
targets and corrections are all normalised with the mean and standard
deviation of the raw training forecast at each grid cell.

Model variants (the keys of common.MODEL_VARIANTS, which also name the output files):
    mlp                    one MLP, trained until the validation loss stops improving
    unet                   one U-Net, trained the same way
    mlp_snapshot3          3 MLP training runs, each saving a "snapshot" of the
                           network at the end of every cosine learning-rate
                           cycle; the corrected forecast is the mean over all
                           snapshots (the paper's production model)
    mlp_snapshot3_perlt    the same, with a separate set of runs per lead time
    mlp_blockk3_snapshot1  block leave-time-out ensemble: one snapshot run per
                           training year, each validated on the other training
                           years, weighted by each snapshot's loss on them

Validation samples are whole weeks of valid days, never single days: weather
is strongly autocorrelated from one day to the next, so a random daily split
would put near-copies of each validation day in the training set. (The block
leave-time-out ensemble holds out whole years instead.)
"""

import copy
import itertools
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import LEAD_TIMES_HOURS, MODEL_VARIANTS  # noqa: E402
from gridded_post_processing.models import MultilayerPerceptron, UNet  # noqa: E402

BASE_SEED = 58
VALIDATION_FRACTION = 0.2
VALIDATION_BLOCK_DAYS = 7
SNAPSHOT_EPOCHS = 210
MAXIMUM_EPOCHS = {"mlp": 750, "unet": 500}
DEFAULT_LEAD_TIME_EMBEDDING_DIM = 4
PREDICTION_BATCH_SIZE = 128
HYPERPARAMETER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "tuned_hyperparameters.json")

def select_device():
    """
    Pick the fastest available torch device.

    Inputs:
        None.

    Returns:
        torch.device: CUDA if available, then Apple's MPS, then the CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_tuned_hyperparameters(model_suffix, variable):
    """
    Read the tuned hyperparameters of one model variant and variable.

    Inputs:
        model_suffix (str): a key of MODEL_VARIANTS.
        variable (str): '2m_temperature' or '10m_wind_speed'.

    Returns:
        dict of hyperparameter name -> value, from tuned_hyperparameters.json.
    """
    with open(HYPERPARAMETER_FILE) as hyperparameter_file:
        return json.load(hyperparameter_file)[model_suffix][variable]


def split_training_and_validation_by_week(valid_times, seed):
    """
    Hold out a random 20% of the weeks as validation samples.

    Weeks are consecutive 7-day blocks counted from the first valid day, so
    every lead time of a valid day lands on the same side of the split.

    Inputs:
        valid_times (numpy.ndarray of datetime64): valid time of each sample.
        seed (int): seed of the random choice of weeks.

    Returns:
        tuple (training_rows, validation_rows) of integer sample positions.
    """
    days_since_start = (valid_times - valid_times.min()) // np.timedelta64(1, "D")
    week_number = days_since_start // VALIDATION_BLOCK_DAYS
    all_weeks = np.unique(week_number)
    validation_weeks = np.random.default_rng(seed).choice(
        all_weeks, size=int(round(VALIDATION_FRACTION * len(all_weeks))), replace=False)
    is_validation = np.isin(week_number, validation_weeks)
    return np.flatnonzero(~is_validation), np.flatnonzero(is_validation)


def create_data_loader(arrays, rows, batch_size):
    """
    Serve some rows of the normalised arrays in random batches.

    A whole batch is taken from each array with one index, rather than one
    sample at a time; the batches (and the random numbers drawn) are the same
    as torch's default per-sample loading.

    Inputs:
        arrays (tuple of numpy.ndarray): (forecast, truth, lead_time_index,
            day_of_year_features), each with one row per sample.
        rows (numpy.ndarray): positions of the samples to serve.
        batch_size (int): samples per batch.

    Returns:
        torch.utils.data.DataLoader yielding (forecast, truth, lead_time_index,
        day_of_year_features) batches.
    """
    forecast, truth, lead_time_index, day_of_year_features = arrays
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(forecast[rows]).float(),
                                             torch.from_numpy(truth[rows]).float(),
                                             torch.from_numpy(lead_time_index[rows]).long(),
                                             torch.from_numpy(day_of_year_features[rows]).float())
    batches = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(dataset), batch_size,
                                            drop_last=False)
    return torch.utils.data.DataLoader(dataset, sampler=batches, batch_size=None)


def mean_squared_error_on(model, data_loader, device):
    """
    Mean squared error of the corrected forecast over every sample of a loader.

    Inputs:
        model (torch.nn.Module): the network.
        data_loader (DataLoader): output of create_data_loader().
        device (torch.device): where the model lives.

    Returns:
        float, the mean squared error in normalised units.
    """
    model.eval()
    loss_function = nn.MSELoss()
    total_loss = 0.0
    with torch.no_grad():
        for forecast, truth, lead_time_index, day_of_year in data_loader:
            forecast, truth = forecast.to(device), truth.to(device)
            corrected = forecast + model(forecast, lead_time_index.to(device),
                                         day_of_year.to(device))
            total_loss += loss_function(corrected, truth).item() * forecast.size(0)
    return total_loss / len(data_loader.dataset)


def train_with_early_stopping(model, training_loader, validation_loader, hyperparameters,
                              maximum_epochs, device):
    """
    Train one network until its validation loss stops improving.

    Uses Adam with the learning rate halved whenever the validation loss has
    not improved for 10 epochs, and stops once it has not improved by at least
    min_delta for 'patience' epochs. The best weights seen are restored.

    Inputs:
        model (torch.nn.Module): a freshly built network.
        training_loader, validation_loader (DataLoader): output of create_data_loader().
        hyperparameters (dict): with 'learning_rate', 'weight_decay',
            'patience' and 'min_delta'.
        maximum_epochs (int): upper limit on training epochs.
        device (torch.device): where the model lives.

    Returns:
        tuple (model with the best weights, training time in minutes).
    """
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters["learning_rate"],
                           weight_decay=hyperparameters["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                                     patience=10, min_lr=1e-7)
    loss_function = nn.MSELoss()
    best_validation_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    start_time = time.time()

    for _ in range(maximum_epochs):
        # One pass over the training data
        model.train()
        for forecast, truth, lead_time_index, day_of_year in training_loader:
            forecast, truth = forecast.to(device), truth.to(device)
            optimizer.zero_grad()
            corrected = forecast + model(forecast, lead_time_index.to(device),
                                         day_of_year.to(device))
            loss_function(corrected, truth).backward()
            optimizer.step()

        # Reduce the learning rate or stop when the validation loss stalls
        validation_loss = mean_squared_error_on(model, validation_loader, device)
        scheduler.step(validation_loss)
        if validation_loss + hyperparameters["min_delta"] < best_validation_loss:
            best_validation_loss = validation_loss
            best_weights = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= hyperparameters["patience"]:
            break

    model.load_state_dict(best_weights)
    return model, (time.time() - start_time) / 60.0


def train_snapshot_run(model, training_loader, validation_loader, hyperparameters, device):
    """
    Train one network for a fixed number of epochs, saving a snapshot at the
    end of every cosine learning-rate cycle.

    The learning rate follows a cosine from its peak down to 1e-6 over each
    cycle, then restarts. The first cycle lasts snapshot_T0 epochs and each
    later one snapshot_T_mult times longer than the one before. At the end of
    a cycle the network sits in a different minimum, so averaging the
    snapshots gives an ensemble from a single run. Uses AdamW and clips the
    gradient norm at 1.

    Inputs:
        model (torch.nn.Module): a freshly built network.
        training_loader, validation_loader (DataLoader): output of create_data_loader().
        hyperparameters (dict): with 'learning_rate', 'weight_decay',
            'snapshot_T0' and 'snapshot_T_mult'.
        device (torch.device): where the model lives.

    Returns:
        tuple (snapshots, training time in minutes), where snapshots is a list
        of (state_dict, validation loss) pairs, one per completed cycle.
    """
    cycle_length = hyperparameters["snapshot_T0"]
    cycle_multiplier = hyperparameters["snapshot_T_mult"]
    optimizer = optim.AdamW(model.parameters(), lr=hyperparameters["learning_rate"],
                            weight_decay=hyperparameters["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cycle_length, T_mult=cycle_multiplier, eta_min=1e-6)
    loss_function = nn.MSELoss()

    # Epochs on which a cycle ends
    cycle_end_epochs = set()
    cycle_end, this_cycle_length = 0, cycle_length
    while cycle_end + this_cycle_length <= SNAPSHOT_EPOCHS:
        cycle_end += this_cycle_length
        cycle_end_epochs.add(cycle_end)
        this_cycle_length *= cycle_multiplier

    snapshots = []
    start_time = time.time()
    for epoch in range(1, SNAPSHOT_EPOCHS + 1):
        model.train()
        for forecast, truth, lead_time_index, day_of_year in training_loader:
            forecast, truth = forecast.to(device), truth.to(device)
            optimizer.zero_grad()
            corrected = forecast + model(forecast, lead_time_index.to(device),
                                         day_of_year.to(device))
            loss_function(corrected, truth).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        if epoch in cycle_end_epochs:
            snapshots.append((copy.deepcopy(model.state_dict()),
                              mean_squared_error_on(model, validation_loader, device)))
    return snapshots, (time.time() - start_time) / 60.0


def predict_corrected_forecast(model, forecast, lead_time_index, day_of_year_features,
                               forecast_mean, forecast_deviation, device):
    """
    Apply a trained network to normalised forecasts.

    Inputs:
        model (torch.nn.Module): the trained network.
        forecast (numpy.ndarray): (n_samples, n_cells) normalised raw forecast.
        lead_time_index (numpy.ndarray): (n_samples,) lead time positions.
        day_of_year_features (numpy.ndarray): (n_samples, 2).
        forecast_mean, forecast_deviation (numpy.ndarray): (n_cells,) the
            normalisation, undone on the output.
        device (torch.device): where the model lives.

    Returns:
        numpy.ndarray of shape (n_samples, n_cells): the corrected forecast in
        the variable's units.
    """
    model.eval()
    corrected_batches = []
    with torch.no_grad():
        for start in range(0, len(forecast), PREDICTION_BATCH_SIZE):
            rows = slice(start, start + PREDICTION_BATCH_SIZE)
            forecast_batch = torch.from_numpy(forecast[rows]).float().to(device)
            predicted_error = model(forecast_batch,
                                    torch.from_numpy(lead_time_index[rows]).long().to(device),
                                    torch.from_numpy(day_of_year_features[rows]).float().to(device))
            corrected_batches.append((forecast_batch + predicted_error).cpu().numpy())
    return np.concatenate(corrected_batches, axis=0) * forecast_deviation + forecast_mean


def build_network(architecture, hyperparameters, number_of_lead_times, n_latitudes, n_longitudes,
                  device):
    """
    Build a fresh MLP or U-Net for one patch.

    Inputs:
        architecture (str): 'mlp' or 'unet'.
        hyperparameters (dict): tuned hyperparameters of the variant.
        number_of_lead_times (int): lead times the network sees (1 for
            per-lead-time networks).
        n_latitudes, n_longitudes (int): patch size in grid cells.
        device (torch.device): where to put the network.

    Returns:
        torch.nn.Module.
    """
    embedding_dim = hyperparameters.get("lead_time_embedding_dim", DEFAULT_LEAD_TIME_EMBEDDING_DIM)
    if architecture == "unet":
        network = UNet(hyperparameters["hidden_dim"], n_latitudes, n_longitudes,
                       number_of_lead_times, embedding_dim, hyperparameters["dropout_rate"])
    else:
        number_of_cells = n_latitudes * n_longitudes
        network = MultilayerPerceptron(number_of_cells, hyperparameters["hidden_dim"],
                                       number_of_cells, hyperparameters["num_layers"],
                                       number_of_lead_times, embedding_dim,
                                       hyperparameters["dropout_rate"])
    return network.to(device)


def post_process_patch(training_data, test_data, model_suffix, hyperparameters, device):
    """
    Train a model variant on a patch's training period and correct its test period.

    Inputs:
        training_data, test_data (dict): output of
            training_data.load_patch_data() for the two periods.
        model_suffix (str): a key of common.MODEL_VARIANTS.
        hyperparameters (dict): output of load_tuned_hyperparameters().
        device (torch.device): output of select_device().

    Returns:
        tuple (corrected test forecast in physical units with the shape of
        test_data['forecast'], total training time in minutes).
    """
    variant = MODEL_VARIANTS[model_suffix]
    n_latitudes, n_longitudes = len(training_data["latitudes"]), len(training_data["longitudes"])
    batch_size = hyperparameters["batch_size"]

    # Normalise everything with the training forecast's statistics at each cell
    forecast_mean = training_data["forecast"].mean(0)
    forecast_deviation = training_data["forecast"].std(0) + 1e-8
    training_forecast = (training_data["forecast"] - forecast_mean) / forecast_deviation
    training_truth = (training_data["truth"] - forecast_mean) / forecast_deviation
    test_forecast = (test_data["forecast"] - forecast_mean) / forecast_deviation
    training_lead, test_lead = training_data["lead_time_index"], test_data["lead_time_index"]
    training_day, test_day = (training_data["day_of_year_features"],
                              test_data["day_of_year_features"])
    training_arrays = (training_forecast, training_truth, training_lead, training_day)
    training_times = training_data["valid_times"]
    number_of_lead_times = len(LEAD_TIMES_HOURS)

    torch.manual_seed(BASE_SEED)
    np.random.seed(BASE_SEED)
    random.seed(BASE_SEED)
    training_minutes = 0.0

    if variant["snapshot_count"] is None:
        # A single network, trained until the validation loss stops improving
        training_rows, validation_rows = split_training_and_validation_by_week(training_times,
                                                                               BASE_SEED)
        training_loader = create_data_loader(training_arrays, training_rows, batch_size)
        validation_loader = create_data_loader(training_arrays, validation_rows, batch_size)
        model = build_network(variant["architecture"], hyperparameters, number_of_lead_times,
                              n_latitudes, n_longitudes, device)
        model, training_minutes = train_with_early_stopping(
            model, training_loader, validation_loader, hyperparameters,
            MAXIMUM_EPOCHS[variant["architecture"]], device)
        corrected = predict_corrected_forecast(model, test_forecast, test_lead, test_day,
                                               forecast_mean, forecast_deviation, device)

    elif variant["block_ensemble"]:
        # Block leave-time-out: train on each training year in turn, holding
        # out all the others for validation, and weight each snapshot by
        # 1 / its validation loss
        training_years = pd.DatetimeIndex(training_times).year.values
        all_years = sorted(set(training_years))
        snapshot_predictions, snapshot_weights = [], []
        for held_out_years in itertools.combinations(all_years, len(all_years) - 1):
            torch.manual_seed(BASE_SEED + sum(held_out_years))
            is_held_out = np.isin(training_years, held_out_years)
            training_loader = create_data_loader(training_arrays, np.flatnonzero(~is_held_out),
                                                 batch_size)
            validation_loader = create_data_loader(training_arrays, np.flatnonzero(is_held_out),
                                                   batch_size)
            model = build_network(variant["architecture"], hyperparameters, number_of_lead_times,
                                  n_latitudes, n_longitudes, device)
            snapshots, run_minutes = train_snapshot_run(model, training_loader,
                                                        validation_loader, hyperparameters,
                                                        device)
            training_minutes += run_minutes
            for weights, validation_loss in snapshots:
                model.load_state_dict(weights)
                snapshot_predictions.append(predict_corrected_forecast(
                    model, test_forecast, test_lead, test_day, forecast_mean,
                    forecast_deviation, device))
                snapshot_weights.append(1.0 / max(validation_loss, 1e-12))
        snapshot_weights = np.array(snapshot_weights) / np.sum(snapshot_weights)
        corrected = np.average(snapshot_predictions, weights=snapshot_weights, axis=0)

    else:
        # Snapshot ensemble: 3 runs, each on its own random choice of
        # validation weeks; average every snapshot of every run. With
        # per_lead_time, each lead time gets its own runs and networks, which
        # have no lead time input (every lead index is set to 0).
        corrected = np.zeros_like(test_data["forecast"])
        if variant["per_lead_time"]:
            lead_time_groups = [[lead_position] for lead_position in range(number_of_lead_times)]
        else:
            lead_time_groups = [list(range(number_of_lead_times))]
        for lead_group_index, lead_positions in enumerate(lead_time_groups):
            in_group = np.isin(training_lead, lead_positions)
            in_test_group = np.isin(test_lead, lead_positions)
            group_lead, group_test_lead = training_lead[in_group], test_lead[in_test_group]
            if variant["per_lead_time"]:
                group_lead, group_test_lead = (np.zeros_like(group_lead),
                                               np.zeros_like(group_test_lead))
            group_arrays = (training_forecast[in_group], training_truth[in_group], group_lead,
                            training_day[in_group])
            snapshot_predictions = []
            for run_index in range(variant["snapshot_count"]):
                run_seed = BASE_SEED + run_index * 17 + lead_group_index * 1000
                torch.manual_seed(run_seed)
                training_rows, validation_rows = split_training_and_validation_by_week(
                    training_times[in_group], run_seed)
                training_loader = create_data_loader(group_arrays, training_rows, batch_size)
                validation_loader = create_data_loader(group_arrays, validation_rows, batch_size)
                model = build_network(variant["architecture"], hyperparameters,
                                      len(lead_positions), n_latitudes, n_longitudes, device)
                snapshots, run_minutes = train_snapshot_run(model, training_loader,
                                                            validation_loader, hyperparameters,
                                                            device)
                training_minutes += run_minutes
                for weights, _ in snapshots:
                    model.load_state_dict(weights)
                    snapshot_predictions.append(predict_corrected_forecast(
                        model, test_forecast[in_test_group], group_test_lead,
                        test_day[in_test_group], forecast_mean, forecast_deviation, device))
            corrected[in_test_group] = np.mean(snapshot_predictions, axis=0)

    return corrected, training_minutes


def build_output_dataset(training_data, test_data, corrected, variable, forecast_model,
                         model_suffix, training_minutes):
    """
    Collect a patch's raw, corrected and truth fields for the test period.

    Also stores the mean bias correction baseline: the raw forecast minus its
    mean error over the training period, at each grid cell and lead time.

    Inputs:
        training_data, test_data (dict): output of training_data.load_patch_data().
        corrected (numpy.ndarray): output of post_process_patch().
        variable (str): the corrected variable.
        forecast_model (str): 'pangu' or 'ifs'.
        model_suffix (str): the model variant, a key of common.MODEL_VARIANTS.
        training_minutes (float): training time, stored as an attribute.

    Returns:
        xarray.Dataset with '{variable}_{original,corrected,mean_corrected,
        ground_truth}_lt{N}h' arrays of shape (time, latitude, longitude) for
        each lead time N.
    """
    latitudes, longitudes = test_data["latitudes"], test_data["longitudes"]
    patch_shape = (len(latitudes), len(longitudes))
    output_arrays = {}
    for lead_position, lead_time in enumerate(LEAD_TIMES_HOURS):
        in_training_lead = training_data["lead_time_index"] == lead_position
        in_test_lead = test_data["lead_time_index"] == lead_position
        coordinates = {"time": test_data["valid_times"][in_test_lead], "latitude": latitudes,
                       "longitude": longitudes}

        # Mean forecast error over the training period at this lead time
        mean_training_error = (
            training_data["forecast"][in_training_lead].reshape(-1, *patch_shape).mean(axis=0)
            - training_data["truth"][in_training_lead].reshape(-1, *patch_shape).mean(axis=0))

        original = test_data["forecast"][in_test_lead].reshape(-1, *patch_shape)
        fields = {
            "original": original,
            "corrected": corrected[in_test_lead].reshape(-1, *patch_shape),
            "mean_corrected": original - mean_training_error,
            "ground_truth": test_data["truth"][in_test_lead].reshape(-1, *patch_shape),
        }
        for kind, values in fields.items():
            output_arrays[f"{variable}_{kind}_lt{lead_time}h"] = xr.DataArray(
                values, dims=["time", "latitude", "longitude"], coords=coordinates)

    output = xr.Dataset(output_arrays)
    output.attrs["description"] = f"Original and post-processed {forecast_model} forecasts"
    output.attrs["lead_times_hours"] = LEAD_TIMES_HOURS
    output.attrs["training_time_minutes"] = training_minutes
    is_block_ensemble = MODEL_VARIANTS[model_suffix]["block_ensemble"]
    output.attrs["validation_split"] = "held_out_years" if is_block_ensemble else "weekly"
    return output
