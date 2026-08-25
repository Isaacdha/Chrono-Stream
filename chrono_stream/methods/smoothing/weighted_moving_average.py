"""Recency-weighted trailing moving-average forecast."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ...contracts import MethodSpec
from ...intervals import build_output


def _weights(window: int, weighting: str, decay: float) -> np.ndarray:
    if weighting == "Exponential":
        weights = decay ** np.arange(window - 1, -1, -1)
    else:
        weights = np.arange(1, window + 1, dtype=float)
    return weights / weights.sum()


def _fitted_values(
    values: np.ndarray, window: int, weighting: str, decay: float
) -> np.ndarray:
    weights = _weights(window, weighting, decay)
    fitted = np.full(len(values), np.nan)
    for index in range(window, len(values)):
        fitted[index] = float(np.dot(values[index - window : index], weights))
    return fitted


def _select_window(
    values: np.ndarray, maximum: int, weighting: str, decay: float
) -> int:
    maximum = max(2, min(int(maximum), len(values) - 1))
    best_window = 2
    best_rmse = math.inf
    for window in range(2, maximum + 1):
        fitted = _fitted_values(values, window, weighting, decay)
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
    """Forecast recursively with a declared finite recency-weight pattern."""
    automatic = bool(params.get("automatic_window", False))
    weighting = str(params.get("weighting", "Linear"))
    decay = float(params.get("decay", 0.8))
    if weighting not in {"Linear", "Exponential"}:
        raise ValueError("Weight pattern must be Linear or Exponential.")
    if weighting == "Exponential" and not 0 < decay < 1:
        raise ValueError("Exponential decay must be greater than 0 and less than 1.")
    if automatic:
        window = _select_window(
            values,
            int(params.get("max_window", min(24, len(values) // 2))),
            weighting,
            decay,
        )
    else:
        window = int(params.get("window", 3))
    if not 2 <= window < len(values):
        raise ValueError(f"Window size must be between 2 and {len(values) - 1}.")

    weights = _weights(window, weighting, decay)
    fitted = _fitted_values(values, window, weighting, decay)
    history = values.astype(float).tolist()
    forecasts: list[float] = []
    for _step in range(steps):
        prediction = float(np.dot(np.asarray(history[-window:]), weights))
        forecasts.append(prediction)
        history.append(prediction)
    return build_output(
        values,
        fitted,
        forecasts,
        details={
            "selection": "Automatic" if automatic else "Manual",
            "selected_window": window,
            "weighting": weighting,
            **({"decay": decay} if weighting == "Exponential" else {}),
            "multi_step_strategy": "Recursive",
        },
    )


def render_parameters(data_length: int, seasonal_period: int) -> dict[str, Any]:
    """Render weighted-moving-average controls."""
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
    parameters["weighting"] = st.selectbox(
        "Weight pattern", ["Linear", "Exponential"]
    )
    if parameters["weighting"] == "Exponential":
        parameters["decay"] = st.slider("Decay", 0.10, 0.99, 0.80, 0.01)
    return parameters


SPEC = MethodSpec(
    model_id="weighted_moving_average",
    display_name="Weighted Moving Average",
    icon="🖇️",
    navigation_group="Smoothing",
    description="Weights recent observations more heavily than older observations within a rolling window.",
    guidance="Useful when recent observations should influence the forecast more strongly.",
    forecast=forecast,
    render_parameters=render_parameters,
    multi_step_strategy="Recursive",
    interval_capability="Descriptive in-sample residual band; unavailable when degenerate",
)
