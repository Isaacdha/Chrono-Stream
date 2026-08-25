"""Leakage-safe feature construction for lag-based forecasting methods.

This module owns neutral row construction and expanding-window split mechanics.  It
does not fit an estimator or implement a forecasting engine; each registered method
retains those responsibilities in its own module.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterator

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class LagFeatureConfig:
    """Configuration for one causal, univariate supervised feature matrix."""

    lookback: int = 12
    use_rolling: bool = False
    rolling_window: int = 3
    seasonal_lag: int | None = None
    use_calendar: bool = False

    @property
    def required_history(self) -> int:
        requirements = [self.lookback]
        if self.use_rolling:
            requirements.append(self.rolling_window)
        if self.seasonal_lag is not None:
            requirements.append(self.seasonal_lag)
        return max(requirements)


def config_from_params(
    params: dict[str, Any], *, default_lookback: int = 12
) -> LagFeatureConfig:
    """Read and validate the common lag-feature parameters used by regressors."""
    lookback = int(params.get("lookback", default_lookback))
    use_rolling = bool(params.get("use_rolling_features", False))
    rolling_window = int(params.get("rolling_window", 3))
    use_seasonal = bool(params.get("use_seasonal_lag", False))
    seasonal_lag = (
        int(params.get("seasonal_lag", params.get("seasonal_period", 12)))
        if use_seasonal
        else None
    )
    use_calendar = bool(params.get("use_calendar_features", False))
    if lookback < 1:
        raise ValueError("Lookback must be at least 1.")
    if use_rolling and rolling_window < 2:
        raise ValueError("The rolling-summary window must be at least 2.")
    if seasonal_lag is not None and seasonal_lag < 2:
        raise ValueError("The optional seasonal lag must be at least 2.")
    return LagFeatureConfig(
        lookback=lookback,
        use_rolling=use_rolling,
        rolling_window=rolling_window,
        seasonal_lag=seasonal_lag,
        use_calendar=use_calendar,
    )


def with_lookback(config: LagFeatureConfig, lookback: int) -> LagFeatureConfig:
    """Return a candidate configuration without mutating the caller's settings."""
    return replace(config, lookback=int(lookback))


def validate_regular_dates(
    dates: Any, forecast_dates: Any | None = None
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex | None, str]:
    """Require unique, increasing, regularly spaced observed and future dates."""
    observed = pd.DatetimeIndex(pd.to_datetime(dates))
    future = (
        pd.DatetimeIndex(pd.to_datetime(forecast_dates))
        if forecast_dates is not None
        else None
    )
    if len(observed) < 3:
        raise ValueError("This forecasting method needs at least 3 dated observations.")
    if not observed.is_monotonic_increasing or observed.has_duplicates:
        raise ValueError("Dates must be unique and strictly increasing.")
    try:
        frequency = pd.infer_freq(observed)
    except (TypeError, ValueError):
        frequency = None
    if frequency is None:
        raise ValueError(
            "This forecasting method requires regularly spaced dates. Regularize the "
            "series on the Data Input page before fitting."
        )
    regular_observed = pd.date_range(
        start=observed[0], periods=len(observed), freq=frequency
    )
    if not regular_observed.equals(observed):
        raise ValueError(
            "Observed dates do not exactly match their inferred regular frequency."
        )
    observed = regular_observed
    if future is not None:
        if len(future) and (not future.is_monotonic_increasing or future.has_duplicates):
            raise ValueError("Future dates must be unique and strictly increasing.")
        expected = pd.date_range(
            start=observed[-1] + pd.tseries.frequencies.to_offset(frequency),
            periods=len(future),
            freq=frequency,
        )
        if not expected.equals(future):
            raise ValueError(
                "Future dates must continue the observed regular frequency without gaps."
            )
        future = expected
    return observed, future, str(frequency)


def feature_names(config: LagFeatureConfig) -> list[str]:
    """Return stable, human-readable names in exactly the generated column order."""
    names = [f"lag_{lag}" for lag in range(1, config.lookback + 1)]
    if config.use_rolling:
        names.extend(
            [
                f"rolling_mean_{config.rolling_window}",
                f"rolling_std_{config.rolling_window}",
            ]
        )
    if (
        config.seasonal_lag is not None
        and config.seasonal_lag > config.lookback
    ):
        names.append(f"seasonal_lag_{config.seasonal_lag}")
    if config.use_calendar:
        names.extend(
            ["month_sin", "month_cos", "weekday_sin", "weekday_cos"]
        )
    return names


def feature_row(
    history: Any, target_date: Any, config: LagFeatureConfig
) -> np.ndarray:
    """Construct one predictor row using only values known before ``target_date``."""
    values = np.asarray(history, dtype=float).reshape(-1)
    if len(values) < config.required_history:
        raise ValueError(
            f"Feature settings require {config.required_history} prior observations; "
            f"only {len(values)} are available."
        )
    row = [float(values[-lag]) for lag in range(1, config.lookback + 1)]
    if config.use_rolling:
        trailing = values[-config.rolling_window :]
        row.extend([float(np.mean(trailing)), float(np.std(trailing, ddof=0))])
    if (
        config.seasonal_lag is not None
        and config.seasonal_lag > config.lookback
    ):
        row.append(float(values[-config.seasonal_lag]))
    if config.use_calendar:
        timestamp = pd.Timestamp(target_date)
        month_angle = 2.0 * np.pi * (timestamp.month - 1) / 12.0
        weekday_angle = 2.0 * np.pi * timestamp.weekday() / 7.0
        row.extend(
            [
                float(np.sin(month_angle)),
                float(np.cos(month_angle)),
                float(np.sin(weekday_angle)),
                float(np.cos(weekday_angle)),
            ]
        )
    result = np.asarray(row, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError("Lag feature construction produced non-finite values.")
    return result


def supervised_rows(
    values: Any,
    dates: Any,
    config: LagFeatureConfig,
    *,
    minimum_rows: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], str]:
    """Create past-only rows, targets, and target indices for one training series."""
    series = np.asarray(values, dtype=float).reshape(-1)
    observed, _future, frequency = validate_regular_dates(dates)
    if len(series) != len(observed):
        raise ValueError("Date and value arrays must have the same length.")
    start = config.required_history
    count = len(series) - start
    if count < minimum_rows:
        raise ValueError(
            "Feature settings leave too few supervised examples: "
            f"need at least {minimum_rows}, got {max(count, 0)}. Reduce the lookback, "
            "rolling window, or seasonal lag."
        )
    target_indices = np.arange(start, len(series), dtype=int)
    features = np.vstack(
        [feature_row(series[:index], observed[index], config) for index in target_indices]
    )
    targets = series[target_indices]
    return features, targets, target_indices, feature_names(config), frequency


def expanding_window_splits(
    sample_count: int, *, maximum_splits: int = 4, minimum_train: int = 6
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return bounded, ordered CV folds with no shuffled or future training rows."""
    sample_count = int(sample_count)
    if sample_count < minimum_train + 2:
        raise ValueError(
            f"Automatic tuning needs at least {minimum_train + 2} supervised rows; "
            f"got {sample_count}."
        )
    split_count = min(int(maximum_splits), sample_count - minimum_train)
    split_count = max(2, split_count)
    test_size = max(1, (sample_count - minimum_train) // split_count)
    first_origin = sample_count - split_count * test_size
    while split_count > 2 and first_origin < minimum_train:
        split_count -= 1
        first_origin = sample_count - split_count * test_size
    if first_origin < minimum_train:
        test_size = 1
        split_count = min(maximum_splits, sample_count - minimum_train)
        first_origin = sample_count - split_count
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in range(split_count):
        origin = first_origin + fold * test_size
        train = np.arange(origin, dtype=int)
        validation = np.arange(origin, origin + test_size, dtype=int)
        folds.append((train, validation))
    return folds


def iter_recursive_rows(
    values: Any,
    forecast_dates: Any,
    config: LagFeatureConfig,
) -> Iterator[tuple[list[float], pd.Timestamp, np.ndarray]]:
    """Yield mutable history and causal rows; callers append their own predictions."""
    history = np.asarray(values, dtype=float).reshape(-1).tolist()
    for target_date in pd.DatetimeIndex(forecast_dates):
        yield history, pd.Timestamp(target_date), feature_row(history, target_date, config)


def feature_details(
    config: LagFeatureConfig, names: list[str], frequency: str
) -> dict[str, Any]:
    """Return serializable audit metadata shared by lag-based method results."""
    return {
        "lookback": config.lookback,
        "rolling_features": config.use_rolling,
        "rolling_window": config.rolling_window if config.use_rolling else None,
        "seasonal_lag": config.seasonal_lag,
        "calendar_features": config.use_calendar,
        "feature_names": list(names),
        "training_row_rule": "Every target uses only observations strictly before it",
        "date_frequency": frequency,
    }


def render_feature_controls(
    data_length: int,
    seasonal_period: int,
    *,
    key_prefix: str,
    include_lookback: bool = True,
) -> dict[str, Any]:
    """Render the consistent optional predictors used by lag-based method pages."""
    import streamlit as st

    maximum = max(2, min(60, data_length - 8))
    parameters: dict[str, Any] = {}
    if include_lookback:
        parameters["lookback"] = st.slider(
            "Lookback lags",
            min_value=1,
            max_value=maximum,
            value=max(1, min(seasonal_period, maximum)),
            key=f"{key_prefix}_lookback",
        )
    rolling = st.toggle(
        "Add past-only rolling mean and standard deviation",
        value=False,
        key=f"{key_prefix}_rolling",
    )
    parameters["use_rolling_features"] = rolling
    if rolling:
        parameters["rolling_window"] = st.slider(
            "Rolling-summary window",
            min_value=2,
            max_value=maximum,
            value=max(2, min(seasonal_period, maximum)),
            key=f"{key_prefix}_rolling_window",
        )
    seasonal = st.toggle(
        "Add a separate seasonal lag",
        value=False,
        key=f"{key_prefix}_seasonal_toggle",
    )
    parameters["use_seasonal_lag"] = seasonal
    if seasonal:
        parameters["seasonal_lag"] = st.number_input(
            "Seasonal lag",
            min_value=2,
            max_value=max(2, data_length - 8),
            value=max(2, min(seasonal_period, data_length - 8)),
            step=1,
            key=f"{key_prefix}_seasonal_lag",
        )
    parameters["use_calendar_features"] = st.toggle(
        "Add known month and weekday cycle features",
        value=False,
        key=f"{key_prefix}_calendar",
        help="Calendar values come from each target date and are known in advance.",
    )
    return parameters
