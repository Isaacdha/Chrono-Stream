"""Simple trailing moving-average forecast."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ...contracts import MethodSpec
from ...intervals import build_output


def _fitted_values(values: np.ndarray, window: int) -> np.ndarray:
    fitted = np.full(len(values), np.nan)
    weights = np.full(window, 1.0 / window)
    for index in range(window, len(values)):
        fitted[index] = float(np.dot(values[index - window : index], weights))
    return fitted


def _select_window(values: np.ndarray, maximum: int) -> int:
    maximum = max(2, min(int(maximum), len(values) - 1))
    best_window = 2
    best_rmse = math.inf
    for window in range(2, maximum + 1):
        fitted = _fitted_values(values, window)
        mask = np.isfinite(fitted)
        if not mask.any():
            continue
        rmse = float(np.sqrt(np.mean((values[mask] - fitted[mask]) ** 2)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_window = window
    return best_window


def forecast(
    values: np.ndarray, steps: int, params: dict[str, Any], **_: Any
) -> dict[str, Any]:
    """Forecast recursively from an equal-weight trailing window."""
    automatic = bool(params.get("automatic_window", False))
    if automatic:
        window = _select_window(
            values, int(params.get("max_window", min(24, len(values) // 2)))
        )
    else:
        window = int(params.get("window", 3))
    if not 2 <= window < len(values):
        raise ValueError(f"Window size must be between 2 and {len(values) - 1}.")

    fitted = _fitted_values(values, window)
    history = values.astype(float).tolist()
    forecasts: list[float] = []
    for _step in range(steps):
        prediction = float(np.mean(history[-window:]))
        forecasts.append(prediction)
        history.append(prediction)
    return build_output(
        values,
        fitted,
        forecasts,
        details={
            "selection": "Automatic" if automatic else "Manual",
            "selected_window": window,
            "multi_step_strategy": "Recursive",
        },
    )


def render_parameters(data_length: int, seasonal_period: int) -> dict[str, Any]:
    """Render moving-average controls inside the shared model form."""
    import streamlit as st

    maximum = max(2, min(60, data_length // 2))
    automatic = st.toggle("Automatically find the optimal window size", value=True)
    parameters: dict[str, Any] = {"automatic_window": automatic}
    if automatic:
        parameters["max_window"] = st.number_input(
            "Maximum window to search",
            min_value=2,
            max_value=maximum,
            value=min(24, maximum),
            step=1,
        )
        st.caption(
            "The window with the lowest one-step-ahead training RMSE is selected."
        )
    else:
        parameters["window"] = st.slider(
            "Window size",
            min_value=2,
            max_value=maximum,
            value=max(2, min(seasonal_period, maximum)),
        )
    return parameters


SPEC = MethodSpec(
    model_id="moving_average",
    display_name="Moving Average",
    icon="📎",
    navigation_group="Smoothing",
    description="Uses the average of the most recent observations. It is a transparent baseline for locally stable series.",
    guidance="Good baseline for smooth, level series. It does not model a sustained trend or seasonality.",
    forecast=forecast,
    render_parameters=render_parameters,
    multi_step_strategy="Recursive",
    interval_capability="Descriptive in-sample residual band; unavailable when degenerate",
)
