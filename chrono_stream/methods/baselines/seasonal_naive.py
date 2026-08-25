"""Seasonal-phase persistence forecast."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...contracts import MethodSpec
from ...intervals import build_output


def forecast(
    values: np.ndarray, steps: int, params: dict[str, Any], **_: Any
) -> dict[str, Any]:
    """Repeat the last observed value from each matching seasonal phase."""
    period = int(params.get("seasonal_period", 12))
    if period < 2:
        raise ValueError("Seasonal period must be at least 2.")
    if len(values) < 2 * period:
        raise ValueError(
            f"Seasonal naive needs at least two full seasons ({2 * period} observations)."
        )
    fitted = np.full(len(values), np.nan)
    fitted[period:] = values[:-period]
    last_cycle = values[-period:]
    future = np.asarray(
        [last_cycle[step % period] for step in range(steps)], dtype=float
    )
    innovations = values[period:] - values[:-period]
    sigma = float(np.sqrt(np.mean(innovations**2)))
    horizons = np.arange(1, steps + 1)
    completed_cycles = (horizons - 1) // period
    margin = 1.96 * sigma * np.sqrt(completed_cycles + 1)
    return build_output(
        values,
        fitted,
        future,
        future - margin,
        future + margin,
        details={
            "rule": "Repeat the latest observation from the same seasonal phase",
            "seasonal_period": period,
            "multi_step_strategy": "Direct seasonal persistence",
            "innovation_standard_deviation": sigma,
            "interval_method": (
                "Gaussian seasonal-random-walk 95% predictive interval"
            ),
            "interval_assumptions": (
                "Independent, constant-variance Gaussian seasonal differences; "
                "uncertainty accumulates after each additional seasonal cycle."
            ),
        },
    )


def render_parameters(data_length: int, seasonal_period: int) -> dict[str, Any]:
    """Render the seasonal-period control."""
    import streamlit as st

    maximum = max(2, data_length // 2)
    return {
        "seasonal_period": st.number_input(
            "Seasonal period",
            min_value=2,
            max_value=maximum,
            value=max(2, min(seasonal_period, maximum)),
            step=1,
            help="Number of observations in one repeating seasonal cycle.",
        )
    }


SPEC = MethodSpec(
    model_id="seasonal_naive",
    display_name="Seasonal Naive Forecast",
    icon="🗓️",
    navigation_group="Baselines",
    description="Repeats the latest observed value from the same phase of the seasonal cycle.",
    guidance="Use as the essential benchmark for regularly spaced data with stable seasonality and at least two complete cycles.",
    forecast=forecast,
    render_parameters=render_parameters,
    multi_step_strategy="Direct seasonal persistence",
    interval_capability="Gaussian seasonal-random-walk predictive intervals",
)
