"""Verification metrics for comparing forecasts against point observations.

This module implements the error decomposition of Bonavita & Geer (2026, QJRMS),
which splits forecast error into three additive pieces:

    RMSE^2 = bias^2 + information_error^2 + noise_error^2

The split matters because the three pieces respond to different interventions.
Bias is removable by any constant offset correction. Noise error shrinks when a
forecast is damped or smoothed, which lowers RMSE without the forecast actually
becoming more skilful. Information error measures how well the forecast
discriminates between outcomes, and is the piece that only a genuine gain in
predictive information can reduce.

Anomaly correlation (ACC) is reported alongside, because that is the quantity in
which Linsenmeier & Shrader (2025) measure the rich/poor forecast quality gap.
"""

import numpy as np
import pandas as pd


def decompose_forecast_error(forecast_values, truth_values, climatology_values):
    """Split forecast error into bias, information error and noise error.

    Anomalies are taken relative to a smoothed day-of-year climatology and then
    centred (their mean is removed), following Appendix A of Bonavita & Geer.
    Centring means a constant offset between the forecast and the truth affects
    only the bias term, leaving ACC, information error and noise error untouched.

    Inputs:
        forecast_values (np.ndarray): 1-D array of forecast values, one per
            verification time, in the variable's native units (K or m/s).
        truth_values (np.ndarray): 1-D array of verifying observations, same
            length and units as forecast_values.
        climatology_values (np.ndarray): 1-D array giving the climatological
            value for the day-of-year of each verification time, same length
            and units.

    Returns:
        dict or None: keys 'bias', 'rmse', 'acc', 'forecast_activity',
            'truth_activity', 'information_error', 'noise_error'. Returns None
            when either the forecast or the truth has zero anomaly variance,
            because the correlation is then undefined.
    """
    # Bias is the mean signed offset; RMSE is the total error magnitude. Both
    # are in the variable's native units.
    mean_bias = float(np.mean(forecast_values - truth_values))
    root_mean_square_error = float(
        np.sqrt(np.mean((forecast_values - truth_values) ** 2)))

    # Work in anomaly space so that the easily-predicted seasonal cycle does not
    # inflate the correlation. Both series use the SAME climatology, so the
    # correlation below is a like-for-like comparison of departures from normal.
    forecast_anomaly = forecast_values - climatology_values
    truth_anomaly = truth_values - climatology_values

    # Centre the anomalies. This is what makes the decomposition insensitive to
    # a constant forecast offset, which is already captured by mean_bias.
    forecast_anomaly = forecast_anomaly - forecast_anomaly.mean()
    truth_anomaly = truth_anomaly - truth_anomaly.mean()

    # "Activity" is the standard deviation of the anomalies: how much the
    # forecast (or the truth) actually moves around its climatology. A damped or
    # over-smoothed forecast has activity well below that of the truth.
    forecast_activity = float(np.sqrt(np.mean(forecast_anomaly ** 2)))
    truth_activity = float(np.sqrt(np.mean(truth_anomaly ** 2)))
    if forecast_activity == 0.0 or truth_activity == 0.0:
        return None

    anomaly_correlation = float(
        np.mean(forecast_anomaly * truth_anomaly)
        / (forecast_activity * truth_activity))

    # Geometrically, the error vector splits into a component along the true
    # anomaly (information error) and one perpendicular to it (noise error).
    # Damping the forecast shrinks noise error but grows information error,
    # which is how an over-smoothed forecast can post a flattering RMSE.
    information_error = float(
        abs(truth_activity - forecast_activity * anomaly_correlation))
    noise_error = float(
        forecast_activity * np.sqrt(max(0.0, 1.0 - anomaly_correlation ** 2)))

    return {
        "bias": mean_bias,
        "rmse": root_mean_square_error,
        "acc": anomaly_correlation,
        "forecast_activity": forecast_activity,
        "truth_activity": truth_activity,
        "information_error": information_error,
        "noise_error": noise_error,
    }


def summarise_income_gap(results_frame, variable_name, metric_name="acc"):
    """Report the high-income minus low/middle-income gap for each method.

    This reproduces the comparison that motivates the analysis: Linsenmeier &
    Shrader document a forecast-quality gap between rich and poor countries, and
    the question is how much of it each post-processing method closes.

    Inputs:
        results_frame (pd.DataFrame): per-station results containing at least
            the columns 'variable', 'method', 'lead_hours', 'income_group' and
            the column named by metric_name.
        variable_name (str): which variable to summarise, e.g. '2m_temperature'.
        metric_name (str): column to average, default 'acc'.

    Returns:
        pd.DataFrame: one row per (lead_hours, method) with columns
            'high_income', 'low_and_middle_income', 'gap', and
            'percent_of_raw_gap_closed' (relative to the 'raw' method at the
            same lead time).
    """
    selected = results_frame[
        (results_frame["variable"] == variable_name)
        & results_frame["income_group"].notna()]

    summary_rows = []
    for lead_hours in sorted(selected["lead_hours"].unique()):
        at_this_lead = selected[selected["lead_hours"] == lead_hours]

        # Compute each method's gap once, then read the raw forecast's gap back
        # out of the result to use as the denominator.
        gap_by_method = {}
        for method_name in at_this_lead["method"].unique():
            method_rows = at_this_lead[at_this_lead["method"] == method_name]
            high_income_mean = method_rows[
                method_rows["income_group"] == "High income"][metric_name].mean()
            low_middle_mean = method_rows[
                method_rows["income_group"] == "Low and middle income"][metric_name].mean()
            gap_by_method[method_name] = (high_income_mean, low_middle_mean,
                                          high_income_mean - low_middle_mean)

        # The raw forecast gap is how big the inequality is before any
        # post-processing is applied.
        raw_gap = gap_by_method.get("raw", (np.nan, np.nan, np.nan))[2]

        for method_name, (high_income_mean, low_middle_mean,
                          gap) in gap_by_method.items():
            summary_rows.append({
                "lead_hours": lead_hours,
                "method": method_name,
                "high_income": high_income_mean,
                "low_and_middle_income": low_middle_mean,
                "gap": gap,
                # Positive means the method shrank the inequality; above 100
                # means it reversed the sign of the gap.
                "percent_of_raw_gap_closed":
                    100.0 * (raw_gap - gap) / raw_gap if raw_gap else np.nan,
            })

    return pd.DataFrame(summary_rows).set_index(["lead_hours", "method"])
