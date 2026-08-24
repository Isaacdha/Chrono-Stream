"""Forecasting implementations with a consistent evaluation contract."""

from __future__ import annotations

from datetime import datetime, timezone
import math
import os
from typing import Any, Callable

import numpy as np
import pandas as pd

from .data import future_dates


MODEL_NAMES = {
    "moving_average": "Moving Average",
    "weighted_moving_average": "Weighted Moving Average",
    "single_exponential_smoothing": "Single Exponential Smoothing",
    "double_exponential_smoothing": "Double Exponential Smoothing (Holt)",
    "triple_exponential_smoothing": "Triple Exponential Smoothing (Holt-Winters)",
    "arima": "ARIMA",
    "sarima": "SARIMA",
    "x11": "STL Decomposition Forecast (X-11-inspired)",
    "prophet": "Prophet",
    "lstm": "LSTM Neural Network",
    "cnn": "1D CNN Neural Network",
    "xgboost": "XGBoost",
    "linear": "Linear Trend",
    "quadratic": "Quadratic Trend",
    "exponential": "Exponential Trend",
    "logarithmic": "Logarithmic Trend",
}


def _as_float_array(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if len(array) < 4 or not np.isfinite(array).all():
        raise ValueError("The model requires at least 4 finite numeric observations.")
    return array


def _align_fitted(fitted: Any, length: int) -> np.ndarray:
    array = np.asarray(fitted, dtype=float).reshape(-1)
    if len(array) == length:
        return array
    aligned = np.full(length, np.nan)
    if len(array) < length:
        aligned[-len(array) :] = array
    else:
        aligned[:] = array[-length:]
    return aligned


def _residual_intervals(
    values: np.ndarray,
    fitted: np.ndarray,
    forecast: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(values) & np.isfinite(fitted)
    residuals = values[mask] - fitted[mask]
    if len(residuals) > 1:
        sigma = float(np.std(residuals, ddof=1))
    else:
        sigma = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    if not np.isfinite(sigma):
        sigma = 0.0
    scale = np.sqrt(1.0 + np.arange(1, len(forecast) + 1) / max(len(values), 1))
    margin = 1.96 * sigma * scale
    return forecast - margin, forecast + margin


def _output(
    values: np.ndarray,
    fitted: Any,
    forecast: Any,
    lower: Any | None = None,
    upper: Any | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fitted_array = _align_fitted(fitted, len(values))
    forecast_array = np.asarray(forecast, dtype=float).reshape(-1)
    if not np.isfinite(forecast_array).all():
        raise ValueError("The model produced non-finite forecast values.")
    if lower is None or upper is None:
        lower_array, upper_array = _residual_intervals(
            values, fitted_array, forecast_array
        )
    else:
        lower_array = np.asarray(lower, dtype=float).reshape(-1)
        upper_array = np.asarray(upper, dtype=float).reshape(-1)
    result: dict[str, Any] = {
        "fitted": fitted_array,
        "forecast": forecast_array,
        "lower": lower_array,
        "upper": upper_array,
    }
    if details:
        result["details"] = details
    return result


def _weighted_fitted_values(
    values: np.ndarray, window: int, weights: np.ndarray
) -> np.ndarray:
    fitted = np.full(len(values), np.nan)
    for index in range(window, len(values)):
        fitted[index] = float(np.dot(values[index - window : index], weights))
    return fitted


def _select_window(
    values: np.ndarray,
    maximum: int,
    weights_for_window: Callable[[int], np.ndarray],
) -> int:
    """Select a rolling window by one-step-ahead RMSE on the available history."""
    maximum = max(2, min(int(maximum), len(values) - 1))
    best_window = 2
    best_rmse = math.inf
    for window in range(2, maximum + 1):
        weights = weights_for_window(window)
        fitted = _weighted_fitted_values(values, window, weights)
        mask = np.isfinite(fitted)
        if not mask.any():
            continue
        rmse = float(np.sqrt(np.mean((values[mask] - fitted[mask]) ** 2)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_window = window
    return best_window


def _moving_average(
    values: np.ndarray, steps: int, params: dict[str, Any], **_: Any
) -> dict[str, Any]:
    automatic = bool(params.get("automatic_window", False))
    if automatic:
        window = _select_window(
            values,
            int(params.get("max_window", min(24, len(values) // 2))),
            lambda candidate: np.full(candidate, 1.0 / candidate),
        )
    else:
        window = int(params.get("window", 3))
    if not 2 <= window < len(values):
        raise ValueError(f"Window size must be between 2 and {len(values) - 1}.")
    weights = np.full(window, 1.0 / window)
    fitted = _weighted_fitted_values(values, window, weights)
    history = values.astype(float).tolist()
    forecasts: list[float] = []
    for _step in range(steps):
        prediction = float(np.mean(history[-window:]))
        forecasts.append(prediction)
        history.append(prediction)
    return _output(
        values,
        fitted,
        forecasts,
        details={
            "selection": "Automatic" if automatic else "Manual",
            "selected_window": window,
        },
    )


def _weighted_moving_average(
    values: np.ndarray, steps: int, params: dict[str, Any], **_: Any
) -> dict[str, Any]:
    automatic = bool(params.get("automatic_window", False))
    weighting = params.get("weighting", "Linear")
    decay = float(params.get("decay", 0.8))

    def create_weights(candidate: int) -> np.ndarray:
        if weighting == "Exponential":
            candidate_weights = decay ** np.arange(candidate - 1, -1, -1)
        else:
            candidate_weights = np.arange(1, candidate + 1, dtype=float)
        return candidate_weights / candidate_weights.sum()

    if automatic:
        window = _select_window(
            values,
            int(params.get("max_window", min(24, len(values) // 2))),
            create_weights,
        )
    else:
        window = int(params.get("window", 3))
    if not 2 <= window < len(values):
        raise ValueError(f"Window size must be between 2 and {len(values) - 1}.")
    weights = create_weights(window)

    fitted = _weighted_fitted_values(values, window, weights)
    history = values.astype(float).tolist()
    forecasts: list[float] = []
    for _step in range(steps):
        prediction = float(np.dot(np.asarray(history[-window:]), weights))
        forecasts.append(prediction)
        history.append(prediction)
    return _output(
        values,
        fitted,
        forecasts,
        details={
            "selection": "Automatic" if automatic else "Manual",
            "selected_window": window,
            "weighting": weighting,
            **({"decay": decay} if weighting == "Exponential" else {}),
        },
    )


def _single_exponential(
    values: np.ndarray, steps: int, params: dict[str, Any], **_: Any
) -> dict[str, Any]:
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing

    alpha = params.get("alpha")
    model = SimpleExpSmoothing(values, initialization_method="estimated")
    fit = model.fit(optimized=alpha is None, smoothing_level=alpha)
    return _output(
        values,
        fit.fittedvalues,
        fit.forecast(steps),
        details={
            "selection": "Automatic" if alpha is None else "Manual",
            "smoothing_level": float(fit.params["smoothing_level"]),
        },
    )


def _double_exponential(
    values: np.ndarray, steps: int, params: dict[str, Any], **_: Any
) -> dict[str, Any]:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    damped = bool(params.get("damped", False))
    alpha = params.get("alpha")
    beta = params.get("beta")
    automatic = alpha is None or beta is None
    model = ExponentialSmoothing(
        values,
        trend="add",
        damped_trend=damped,
        initialization_method="estimated",
    )
    fit = model.fit(
        optimized=automatic,
        smoothing_level=alpha,
        smoothing_trend=beta,
        damping_trend=params.get("phi") if damped and not automatic else None,
    )
    return _output(
        values,
        fit.fittedvalues,
        fit.forecast(steps),
        details={
            "selection": "Automatic" if automatic else "Manual",
            "smoothing_level": float(fit.params["smoothing_level"]),
            "smoothing_trend": float(fit.params["smoothing_trend"]),
            **({"damping_trend": float(fit.params["damping_trend"])} if damped else {}),
        },
    )


def _triple_exponential(
    values: np.ndarray, steps: int, params: dict[str, Any], **_: Any
) -> dict[str, Any]:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    period = int(params.get("seasonal_period", 12))
    if period < 2 or len(values) < 2 * period:
        raise ValueError(
            f"Holt-Winters needs at least two full seasons ({2 * period} observations)."
        )
    trend = params.get("trend", "add")
    seasonal = params.get("seasonal", "add")
    alpha = params.get("alpha")
    beta = params.get("beta")
    gamma = params.get("gamma")
    automatic = alpha is None or beta is None or gamma is None
    damped = bool(params.get("damped", False))
    if (trend == "mul" or seasonal == "mul") and np.any(values <= 0):
        raise ValueError(
            "Multiplicative components require all values to be greater than zero."
        )
    model = ExponentialSmoothing(
        values,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=period,
        damped_trend=damped,
        initialization_method="estimated",
    )
    fit = model.fit(
        optimized=automatic,
        smoothing_level=alpha,
        smoothing_trend=beta,
        smoothing_seasonal=gamma,
        damping_trend=params.get("phi") if damped and not automatic else None,
    )
    return _output(
        values,
        fit.fittedvalues,
        fit.forecast(steps),
        details={
            "selection": "Automatic" if automatic else "Manual",
            "seasonal_period": period,
            "smoothing_level": float(fit.params["smoothing_level"]),
            "smoothing_trend": float(fit.params["smoothing_trend"]),
            "smoothing_seasonal": float(fit.params["smoothing_seasonal"]),
            **({"damping_trend": float(fit.params["damping_trend"])} if damped else {}),
        },
    )


def _arima(
    values: np.ndarray, steps: int, params: dict[str, Any], **_: Any
) -> dict[str, Any]:
    from .arima_pipeline import fit_arima_pipeline

    return fit_arima_pipeline(values, steps, params, seasonal=False)


def _sarima(
    values: np.ndarray, steps: int, params: dict[str, Any], **_: Any
) -> dict[str, Any]:
    from .arima_pipeline import fit_arima_pipeline

    return fit_arima_pipeline(values, steps, params, seasonal=True)


def _stl_decomposition_forecast(
    values: np.ndarray, steps: int, params: dict[str, Any], **_: Any
) -> dict[str, Any]:
    from statsmodels.tsa.seasonal import STL

    period = int(params.get("seasonal_period", 12))
    if period < 2 or len(values) < 2 * period:
        raise ValueError(
            f"Seasonal decomposition needs at least two full seasons ({2 * period} observations)."
        )
    robust = bool(params.get("robust", True))
    decomposition = STL(values, period=period, robust=robust).fit()
    fitted = decomposition.trend + decomposition.seasonal

    tail_length = min(len(values), max(2 * period, 8))
    tail_x = np.arange(len(values) - tail_length, len(values), dtype=float)
    tail_trend = decomposition.trend[-tail_length:]
    valid = np.isfinite(tail_trend)
    slope, intercept = np.polyfit(tail_x[valid], tail_trend[valid], 1)
    future_x = np.arange(len(values), len(values) + steps, dtype=float)
    trend_forecast = intercept + slope * future_x

    seasonal_pattern = np.zeros(period)
    positions = np.arange(len(values)) % period
    for phase in range(period):
        seasonal_pattern[phase] = float(
            np.nanmean(decomposition.seasonal[positions == phase])
        )
    seasonal_forecast = seasonal_pattern[
        np.arange(len(values), len(values) + steps) % period
    ]
    return _output(
        values,
        fitted,
        trend_forecast + seasonal_forecast,
        details={
            "decomposition": "STL",
            "official_x11": False,
            "seasonal_period": period,
            "robust": robust,
            "trend_extension": (
                f"Least-squares line fitted to the final {tail_length} STL trend values"
            ),
            "seasonal_extension": "Mean STL seasonal value for each cycle phase",
        },
    )


def _trend(
    values: np.ndarray, steps: int, params: dict[str, Any], *, kind: str, **_: Any
) -> dict[str, Any]:
    x = np.arange(1, len(values) + 1, dtype=float)
    future_x = np.arange(len(values) + 1, len(values) + steps + 1, dtype=float)
    details: dict[str, Any]
    if kind == "linear":
        coefficients = np.polyfit(x, values, 1)
        fitted = np.polyval(coefficients, x)
        forecast = np.polyval(coefficients, future_x)
        details = {
            "equation": "y(t) = slope * t + intercept",
            "slope": float(coefficients[0]),
            "intercept": float(coefficients[1]),
        }
    elif kind == "quadratic":
        if len(values) < 5:
            raise ValueError("Quadratic trend requires at least 5 observations.")
        coefficients = np.polyfit(x, values, 2)
        fitted = np.polyval(coefficients, x)
        forecast = np.polyval(coefficients, future_x)
        details = {
            "equation": "y(t) = a * t^2 + b * t + c",
            "a": float(coefficients[0]),
            "b": float(coefficients[1]),
            "c": float(coefficients[2]),
        }
    elif kind == "exponential":
        if np.any(values <= 0):
            raise ValueError(
                "Exponential trend requires all values to be greater than zero."
            )
        log_values = np.log(values)
        coefficients = np.polyfit(x, log_values, 1)
        fitted_log = np.polyval(coefficients, x)
        log_residuals = log_values - fitted_log
        with np.errstate(over="ignore", invalid="ignore"):
            smearing_factor = float(np.mean(np.exp(log_residuals)))
        if not math.isfinite(smearing_factor) or smearing_factor <= 0:
            raise ValueError(
                "The exponential trend's retransformation correction is not finite."
            )
        fitted = smearing_factor * np.exp(fitted_log)
        forecast = smearing_factor * np.exp(np.polyval(coefficients, future_x))
        details = {
            "equation": "E[y(t)] = smearing_factor * exp(log_slope * t + log_intercept)",
            "log_slope": float(coefficients[0]),
            "log_intercept": float(coefficients[1]),
            "smearing_factor": smearing_factor,
            "retransformation": "Duan nonparametric smearing estimate",
        }
    elif kind == "logarithmic":
        coefficients = np.polyfit(np.log(x), values, 1)
        fitted = np.polyval(coefficients, np.log(x))
        forecast = np.polyval(coefficients, np.log(future_x))
        details = {
            "equation": "y(t) = slope * log(t) + intercept",
            "slope": float(coefficients[0]),
            "intercept": float(coefficients[1]),
        }
    else:
        raise ValueError(f"Unknown trend type: {kind}")
    return _output(values, fitted, forecast, details=details)


def _prophet(
    values: np.ndarray,
    steps: int,
    params: dict[str, Any],
    *,
    dates: pd.DatetimeIndex,
    forecast_dates: pd.DatetimeIndex,
    **_: Any,
) -> dict[str, np.ndarray]:
    try:
        from prophet import Prophet
    except ImportError as exc:
        raise ImportError(
            "Prophet is not installed. Install the project requirements and try again."
        ) from exc

    frame = pd.DataFrame({"ds": pd.to_datetime(dates), "y": values})
    model = Prophet(
        seasonality_mode=params.get("seasonality_mode", "additive"),
        changepoint_prior_scale=float(params.get("changepoint_prior_scale", 0.05)),
        yearly_seasonality=params.get("yearly_seasonality", "auto"),
        weekly_seasonality=params.get("weekly_seasonality", "auto"),
        daily_seasonality=False,
        interval_width=0.95,
    )
    model.fit(frame)
    prediction_dates = pd.DataFrame(
        {"ds": pd.DatetimeIndex(dates).append(pd.DatetimeIndex(forecast_dates))}
    )
    prediction = model.predict(prediction_dates)
    fitted = prediction["yhat"].iloc[: len(values)].to_numpy()
    future = prediction.iloc[len(values) :]
    return _output(
        values,
        fitted,
        future["yhat"].to_numpy(),
        future["yhat_lower"].to_numpy(),
        future["yhat_upper"].to_numpy(),
    )


def _lagged_arrays(values: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    if lookback < 2 or len(values) <= lookback + 2:
        raise ValueError(
            f"Lookback must leave at least 3 training samples; got {len(values)} observations."
        )
    features = np.asarray(
        [values[index - lookback : index] for index in range(lookback, len(values))]
    )
    targets = values[lookback:]
    return features, targets


def _xgboost(
    values: np.ndarray, steps: int, params: dict[str, Any], **_: Any
) -> dict[str, np.ndarray]:
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError(
            "XGBoost is not installed. Install the project requirements and try again."
        ) from exc

    lookback = int(params.get("lookback", 12))
    features, targets = _lagged_arrays(values, lookback)
    model = XGBRegressor(
        n_estimators=int(params.get("n_estimators", 250)),
        max_depth=int(params.get("max_depth", 3)),
        learning_rate=float(params.get("learning_rate", 0.05)),
        objective="reg:squarederror",
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=1,
    )
    model.fit(features, targets)
    fitted = np.full(len(values), np.nan)
    fitted[lookback:] = model.predict(features)
    history = values.astype(float).tolist()
    forecasts: list[float] = []
    for _step in range(steps):
        prediction = float(
            model.predict(np.asarray(history[-lookback:]).reshape(1, -1))[0]
        )
        forecasts.append(prediction)
        history.append(prediction)
    return _output(values, fitted, forecasts)


def _neural_network(
    values: np.ndarray,
    steps: int,
    params: dict[str, Any],
    *,
    architecture: str,
    **_: Any,
) -> dict[str, np.ndarray]:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    try:
        import tensorflow as tf
        from sklearn.preprocessing import MinMaxScaler
    except ImportError as exc:
        raise ImportError(
            "TensorFlow and scikit-learn are required for the neural-network models."
        ) from exc

    lookback = int(params.get("lookback", 12))
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values.reshape(-1, 1)).reshape(-1)
    features, targets = _lagged_arrays(scaled, lookback)
    features_3d = features.reshape(-1, lookback, 1)

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)
    if architecture == "lstm":
        network = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(lookback, 1)),
                tf.keras.layers.LSTM(int(params.get("units", 32))),
                tf.keras.layers.Dense(1),
            ]
        )
    else:
        kernel_size = int(params.get("kernel_size", 3))
        if kernel_size > lookback:
            raise ValueError("CNN kernel size cannot exceed the lookback window.")
        network = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(lookback, 1)),
                tf.keras.layers.Conv1D(
                    int(params.get("filters", 32)), kernel_size, activation="relu"
                ),
                tf.keras.layers.GlobalAveragePooling1D(),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(1),
            ]
        )
    network.compile(optimizer="adam", loss="mse")
    network.fit(
        features_3d,
        targets,
        epochs=int(params.get("epochs", 50)),
        batch_size=min(int(params.get("batch_size", 16)), len(features_3d)),
        verbose=0,
        shuffle=False,
    )

    fitted_scaled = network(features_3d, training=False).numpy().reshape(-1, 1)
    fitted = np.full(len(values), np.nan)
    fitted[lookback:] = scaler.inverse_transform(fitted_scaled).reshape(-1)

    history = scaled.astype(float).tolist()
    forecast_scaled: list[float] = []
    for _step in range(steps):
        window = np.asarray(history[-lookback:]).reshape(1, lookback, 1)
        prediction = float(network(window, training=False).numpy().reshape(-1)[0])
        forecast_scaled.append(prediction)
        history.append(prediction)
    forecast = scaler.inverse_transform(
        np.asarray(forecast_scaled).reshape(-1, 1)
    ).reshape(-1)
    tf.keras.backend.clear_session()
    return _output(values, fitted, forecast)


def forecast_model(
    model_id: str,
    values: Any,
    steps: int,
    params: dict[str, Any],
    *,
    dates: pd.DatetimeIndex,
    forecast_dates: pd.DatetimeIndex,
) -> dict[str, Any]:
    """Fit one model and forecast a fixed number of steps."""
    y = _as_float_array(values)
    if steps < 1:
        raise ValueError("Forecast steps must be at least 1.")

    common = {
        "values": y,
        "steps": int(steps),
        "params": params,
        "dates": pd.DatetimeIndex(dates),
        "forecast_dates": pd.DatetimeIndex(forecast_dates),
    }
    handlers: dict[str, Callable[..., dict[str, Any]]] = {
        "moving_average": _moving_average,
        "weighted_moving_average": _weighted_moving_average,
        "single_exponential_smoothing": _single_exponential,
        "double_exponential_smoothing": _double_exponential,
        "triple_exponential_smoothing": _triple_exponential,
        "arima": _arima,
        "sarima": _sarima,
        "x11": _stl_decomposition_forecast,
        "prophet": _prophet,
        "xgboost": _xgboost,
        "lstm": lambda **kwargs: _neural_network(architecture="lstm", **kwargs),
        "cnn": lambda **kwargs: _neural_network(architecture="cnn", **kwargs),
        "linear": lambda **kwargs: _trend(kind="linear", **kwargs),
        "quadratic": lambda **kwargs: _trend(kind="quadratic", **kwargs),
        "exponential": lambda **kwargs: _trend(kind="exponential", **kwargs),
        "logarithmic": lambda **kwargs: _trend(kind="logarithmic", **kwargs),
    }
    if model_id not in handlers:
        raise ValueError(f"Unknown model: {model_id}")
    return handlers[model_id](**common)


def regression_metrics(actual: Any, predicted: Any) -> dict[str, float]:
    actual_array = np.asarray(actual, dtype=float).reshape(-1)
    predicted_array = np.asarray(predicted, dtype=float).reshape(-1)
    mask = np.isfinite(actual_array) & np.isfinite(predicted_array)
    if not mask.any():
        raise ValueError("No finite predictions are available for evaluation.")
    actual_array = actual_array[mask]
    predicted_array = predicted_array[mask]
    errors = actual_array - predicted_array
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    nonzero = np.abs(actual_array) > np.finfo(float).eps
    mape = (
        float(np.mean(np.abs(errors[nonzero] / actual_array[nonzero])) * 100)
        if nonzero.any()
        else math.nan
    )
    denominator = np.abs(actual_array) + np.abs(predicted_array)
    valid_smape = denominator > np.finfo(float).eps
    smape = (
        float(np.mean(2 * np.abs(errors[valid_smape]) / denominator[valid_smape]) * 100)
        if valid_smape.any()
        else math.nan
    )
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "sMAPE": smape}


def evaluate_and_forecast(
    model_id: str,
    frame: pd.DataFrame,
    *,
    horizon: int,
    holdout: int,
    params: dict[str, Any] | None = None,
    frequency: str | None = None,
) -> dict[str, Any]:
    """Backtest a model, refit it on all observations, and create future forecasts."""
    if model_id not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {model_id}")
    if frame.shape[1] < 2:
        raise ValueError(
            "Expected a dataframe containing a date column and a value column."
        )

    params = dict(params or {})
    dates = pd.DatetimeIndex(pd.to_datetime(frame.iloc[:, 0]))
    values = _as_float_array(frame.iloc[:, 1])
    if len(dates) != len(values):
        raise ValueError("Date and value columns must have the same length.")
    if not 1 <= holdout < len(values) - 3:
        raise ValueError(f"Holdout must be between 1 and {len(values) - 4}.")

    train_values = values[:-holdout]
    train_dates = dates[:-holdout]
    test_dates = dates[-holdout:]
    backtest = forecast_model(
        model_id,
        train_values,
        holdout,
        params,
        dates=train_dates,
        forecast_dates=test_dates,
    )
    metrics = regression_metrics(values[-holdout:], backtest["forecast"])

    requested_future_dates = future_dates(dates, horizon, frequency)
    final = forecast_model(
        model_id,
        values,
        horizon,
        params,
        dates=dates,
        forecast_dates=requested_future_dates,
    )

    date_name, value_name = map(str, frame.columns[:2])
    fitted_frame = pd.DataFrame(
        {
            "Date": dates,
            "Actual": values,
            "Fitted": final["fitted"],
        }
    )
    forecast_frame = pd.DataFrame(
        {
            "Date": requested_future_dates,
            "Forecast": final["forecast"],
            "Lower 95%": final["lower"],
            "Upper 95%": final["upper"],
        }
    )
    backtest_frame = pd.DataFrame(
        {
            "Date": test_dates,
            "Actual": values[-holdout:],
            "Predicted": backtest["forecast"],
            "Lower 95%": backtest["lower"],
            "Upper 95%": backtest["upper"],
        }
    )
    return {
        "model_id": model_id,
        "model_name": MODEL_NAMES[model_id],
        "date_name": date_name,
        "value_name": value_name,
        "parameters": params,
        "model_details": final.get("details", {}),
        "backtest_model_details": backtest.get("details", {}),
        "metrics": metrics,
        "holdout": int(holdout),
        "horizon": int(horizon),
        "fitted": fitted_frame,
        "forecast": forecast_frame,
        "backtest": backtest_frame,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
