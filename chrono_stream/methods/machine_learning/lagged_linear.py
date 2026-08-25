"""Ordinary least-squares forecasting from causal lag features."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ...contracts import MethodSpec
from ...features import (
    config_from_params,
    expanding_window_splits,
    feature_details,
    feature_row,
    render_feature_controls,
    supervised_rows,
    validate_regular_dates,
    with_lookback,
)
from ...intervals import build_output


def _candidate_score(
    values: np.ndarray,
    dates: Any,
    config,
    *,
    fit_intercept: bool,
) -> float:
    from sklearn.linear_model import LinearRegression

    features, targets, _indices, _names, _frequency = supervised_rows(
        values, dates, config, minimum_rows=8
    )
    squared_errors: list[float] = []
    for train, validation in expanding_window_splits(len(targets)):
        estimator = LinearRegression(fit_intercept=fit_intercept)
        estimator.fit(features[train], targets[train])
        errors = targets[validation] - estimator.predict(features[validation])
        squared_errors.extend(np.square(errors).tolist())
    return float(np.sqrt(np.mean(squared_errors)))


def forecast(
    values: np.ndarray,
    steps: int,
    params: dict[str, Any],
    *,
    dates: Any,
    forecast_dates: Any,
    **_: Any,
) -> dict[str, Any]:
    """Fit one lagged OLS equation and forecast it recursively."""
    from sklearn.linear_model import LinearRegression

    observed, future_dates, _frequency = validate_regular_dates(dates, forecast_dates)
    base_config = config_from_params(params)
    fit_intercept = bool(params.get("fit_intercept", True))
    automatic = bool(params.get("automatic", True))
    candidates: list[dict[str, Any]] = []
    if automatic:
        maximum = min(
            int(params.get("max_lookback", min(24, len(values) - 8))),
            len(values) - 8,
        )
        if maximum < 1:
            raise ValueError("Automatic lag selection needs more observations.")
        best_score = math.inf
        selected_config = None
        for lookback in range(1, maximum + 1):
            candidate_config = with_lookback(base_config, lookback)
            try:
                score = _candidate_score(
                    values,
                    observed,
                    candidate_config,
                    fit_intercept=fit_intercept,
                )
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                candidates.append(
                    {"lookback": lookback, "CV RMSE": None, "fit_error": str(exc)}
                )
                continue
            candidates.append(
                {"lookback": lookback, "CV RMSE": score, "fit_error": None}
            )
            if score < best_score:
                best_score = score
                selected_config = candidate_config
        if selected_config is None:
            raise ValueError(
                "No lag count could be fitted during expanding-window validation. "
                "Reduce the optional feature windows or use manual settings."
            )
        config = selected_config
    else:
        config = base_config

    features, targets, target_indices, names, frequency = supervised_rows(
        values, observed, config
    )
    estimator = LinearRegression(fit_intercept=fit_intercept)
    estimator.fit(features, targets)
    fitted = np.full(len(values), np.nan)
    fitted[target_indices] = estimator.predict(features)

    history = values.astype(float).tolist()
    future: list[float] = []
    for target_date in future_dates:
        row = feature_row(history, target_date, config).reshape(1, -1)
        prediction = float(estimator.predict(row)[0])
        future.append(prediction)
        history.append(prediction)

    coefficient_map = {
        name: float(value) for name, value in zip(names, estimator.coef_, strict=True)
    }
    details = {
        **feature_details(config, names, frequency),
        "selection": "Automatic expanding-window CV" if automatic else "Manual",
        "selected_lookback": config.lookback,
        "fit_intercept": fit_intercept,
        "intercept": float(estimator.intercept_) if fit_intercept else 0.0,
        "coefficients": coefficient_map,
        "candidate_results": candidates,
        "top_candidates": sorted(
            [item for item in candidates if item["CV RMSE"] is not None],
            key=lambda item: item["CV RMSE"],
        )[:10],
        "multi_step_strategy": "Recursive",
        "recursive_disclosure": (
            "Each forecast is appended to history, so errors can propagate across horizons."
        ),
    }
    return build_output(values, fitted, future, details=details)


def render_parameters(data_length: int, seasonal_period: int) -> dict[str, Any]:
    """Render lagged OLS controls."""
    import streamlit as st

    automatic = st.toggle(
        "Automatically select the lag count with expanding-window CV", value=True
    )
    parameters: dict[str, Any] = {"automatic": automatic}
    if automatic:
        maximum = max(1, min(36, data_length - 8))
        parameters["max_lookback"] = st.slider(
            "Maximum lag count to search",
            min_value=1,
            max_value=maximum,
            value=max(1, min(seasonal_period, maximum)),
        )
        parameters.update(
            render_feature_controls(
                data_length,
                seasonal_period,
                key_prefix="lagged_linear",
                include_lookback=False,
            )
        )
        parameters["lookback"] = min(seasonal_period, maximum)
    else:
        parameters.update(
            render_feature_controls(
                data_length, seasonal_period, key_prefix="lagged_linear"
            )
        )
    parameters["fit_intercept"] = st.toggle("Fit an intercept", value=True)
    st.caption(
        "CV rows and every rolling summary use earlier values only. Future forecasts "
        "are recursive."
    )
    return parameters


SPEC = MethodSpec(
    model_id="lagged_linear",
    display_name="Lagged Linear Regression",
    icon="📐",
    navigation_group="Machine Learning",
    description="Ordinary least squares applied to past-only lag, rolling, seasonal, and known-calendar predictors.",
    guidance="Use the automatic lag search as a transparent linear benchmark; correlated lags can make individual coefficients unstable.",
    forecast=forecast,
    render_parameters=render_parameters,
    multi_step_strategy="Recursive",
    interval_capability="Descriptive in-sample residual band; unavailable when degenerate",
    minimum_observations=12,
    statistical_test_keys=("rolling_origin_cv",),
)
