"""Common holdout evaluation and final-refit workflow."""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any

import numpy as np
import pandas as pd

from .data import future_dates
from .contracts import validate_forecast_context, validate_series


ACCURACY_METRIC_KEYS = (
    "MAE",
    "RMSE",
    "MASE",
    "RMSSE",
    "MAPE",
    "sMAPE",
    "WAPE",
)


def forecast_model(
    model_id: str,
    values: Any,
    steps: int,
    params: dict[str, Any],
    *,
    dates: pd.DatetimeIndex,
    forecast_dates: pd.DatetimeIndex,
) -> dict[str, Any]:
    """Fit one registered method and forecast a fixed number of steps."""
    from .registry import get_method

    spec = get_method(model_id)
    y = validate_series(values, spec.minimum_observations)
    if steps < 1:
        raise ValueError("Forecast steps must be at least 1.")
    observed_dates, requested_dates = validate_forecast_context(
        dates,
        forecast_dates,
        observations=len(y),
        steps=int(steps),
    )
    result = spec.forecast(
        values=y,
        steps=int(steps),
        params=dict(params or {}),
        dates=observed_dates,
        forecast_dates=requested_dates,
    )
    if len(np.asarray(result.get("forecast", [])).reshape(-1)) != int(steps):
        raise ValueError("The model did not return the requested forecast horizon.")
    return result


def regression_metrics(
    actual: Any,
    predicted: Any,
    *,
    training_actual: Any | None = None,
    scale_period: int = 1,
) -> dict[str, float]:
    """Calculate the shared holdout metrics without leaking holdout observations.

    MASE and RMSSE use naive differences from ``training_actual`` only. A
    ``scale_period`` of one compares against an ordinary naive forecast; larger
    values compare against a seasonal-naive forecast at that lag. Scaled metrics
    are unavailable when no training series is supplied, the requested lag cannot
    be formed, or the training benchmark has zero error.

    MAPE is unavailable if any evaluated actual is zero. Chrono Stream does not
    silently delete undefined percentage-error terms. For sMAPE, an actual and
    forecast that are both zero contribute zero error. WAPE is unavailable when
    the sum of absolute holdout actuals is zero.
    """
    actual_array = np.asarray(actual, dtype=float).reshape(-1)
    predicted_array = np.asarray(predicted, dtype=float).reshape(-1)
    if actual_array.size != predicted_array.size:
        raise ValueError("Actual and predicted values must have the same length.")
    mask = np.isfinite(actual_array) & np.isfinite(predicted_array)
    if not mask.any():
        raise ValueError("No finite predictions are available for evaluation.")
    actual_array = actual_array[mask]
    predicted_array = predicted_array[mask]
    errors = actual_array - predicted_array
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))

    has_zero_actual = bool(np.any(actual_array == 0.0))
    mape = (
        math.nan
        if has_zero_actual
        else float(np.mean(np.abs(errors / actual_array)) * 100)
    )

    denominator = np.abs(actual_array) + np.abs(predicted_array)
    smape_terms = np.zeros_like(errors, dtype=float)
    valid_smape = denominator > 0.0
    smape_terms[valid_smape] = (
        2 * np.abs(errors[valid_smape]) / denominator[valid_smape]
    )
    smape = float(np.mean(smape_terms) * 100)

    wape_denominator = float(np.sum(np.abs(actual_array)))
    wape = (
        float(np.sum(np.abs(errors)) / wape_denominator * 100)
        if wape_denominator > 0.0
        else math.nan
    )

    if isinstance(scale_period, bool):
        raise ValueError("Metric scale period must be a positive integer.")
    try:
        normalized_scale_period = int(scale_period)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Metric scale period must be a positive integer.") from exc
    if normalized_scale_period < 1 or normalized_scale_period != scale_period:
        raise ValueError("Metric scale period must be a positive integer.")

    mase = math.nan
    rmsse = math.nan
    if training_actual is not None:
        training_array = np.asarray(training_actual, dtype=float).reshape(-1)
        if not np.isfinite(training_array).all():
            raise ValueError("Metric training values must all be finite.")
        if training_array.size > normalized_scale_period:
            naive_errors = (
                training_array[normalized_scale_period:]
                - training_array[:-normalized_scale_period]
            )
            absolute_scale = float(np.mean(np.abs(naive_errors)))
            squared_scale = float(np.mean(naive_errors**2))
            if absolute_scale > 0.0:
                mase = float(mae / absolute_scale)
            if squared_scale > 0.0:
                rmsse = float(np.sqrt(np.mean(errors**2) / squared_scale))

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MASE": mase,
        "RMSSE": rmsse,
        "MAPE": mape,
        "sMAPE": smape,
        "WAPE": wape,
    }


def evaluate_and_forecast(
    model_id: str,
    frame: pd.DataFrame,
    *,
    horizon: int,
    holdout: int,
    params: dict[str, Any] | None = None,
    frequency: str | None = None,
    metric_scale_period: int = 1,
) -> dict[str, Any]:
    """Backtest a method, refit it independently, and forecast future dates."""
    from .registry import get_method

    spec = get_method(model_id)
    if frame.shape[1] < 2:
        raise ValueError(
            "Expected a dataframe containing a date column and a value column."
        )

    parameters = dict(params or {})
    dates = pd.DatetimeIndex(pd.to_datetime(frame.iloc[:, 0]))
    values = validate_series(frame.iloc[:, 1], spec.minimum_observations)
    if len(dates) != len(values):
        raise ValueError("Date and value columns must have the same length.")
    if not 1 <= holdout < len(values) - spec.minimum_observations + 1:
        maximum_holdout = len(values) - spec.minimum_observations
        raise ValueError(f"Holdout must be between 1 and {maximum_holdout}.")

    train_values = values[:-holdout]
    train_dates = dates[:-holdout]
    test_dates = dates[-holdout:]
    backtest = forecast_model(
        model_id,
        train_values,
        holdout,
        parameters,
        dates=train_dates,
        forecast_dates=test_dates,
    )
    metrics = regression_metrics(
        values[-holdout:],
        backtest["forecast"],
        training_actual=train_values,
        scale_period=metric_scale_period,
    )

    requested_future_dates = future_dates(dates, horizon, frequency)
    final = forecast_model(
        model_id,
        values,
        horizon,
        parameters,
        dates=dates,
        forecast_dates=requested_future_dates,
    )

    date_name, value_name = map(str, frame.columns[:2])
    fitted_frame = pd.DataFrame(
        {"Date": dates, "Actual": values, "Fitted": final["fitted"]}
    )
    forecast_frame = pd.DataFrame(
        {
            "Date": requested_future_dates,
            "Forecast": final["forecast"],
            "Lower interval": final["lower"],
            "Upper interval": final["upper"],
        }
    )
    backtest_frame = pd.DataFrame(
        {
            "Date": test_dates,
            "Actual": values[-holdout:],
            "Predicted": backtest["forecast"],
            "Lower interval": backtest["lower"],
            "Upper interval": backtest["upper"],
        }
    )
    return {
        "model_id": model_id,
        "model_name": spec.display_name,
        "date_name": date_name,
        "value_name": value_name,
        "parameters": parameters,
        "model_details": final.get("details", {}),
        "backtest_model_details": backtest.get("details", {}),
        "metrics": metrics,
        "metric_context": {
            "scale_period": int(metric_scale_period),
            "scale_source": "Pre-holdout training observations",
            "scale_benchmark": (
                "Naive one-step differences"
                if int(metric_scale_period) == 1
                else f"Seasonal-naive differences at lag {int(metric_scale_period)}"
            ),
        },
        "holdout": int(holdout),
        "horizon": int(horizon),
        "fitted": fitted_frame,
        "forecast": forecast_frame,
        "backtest": backtest_frame,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
