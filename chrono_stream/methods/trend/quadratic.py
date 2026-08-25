"""Least-squares quadratic trend projection."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...contracts import MethodSpec, empty_parameters
from ...intervals import build_output


def forecast(
    values: np.ndarray, steps: int, params: dict[str, Any], **_: Any
) -> dict[str, Any]:
    """Fit and extrapolate a second-degree polynomial."""
    x = np.arange(1, len(values) + 1, dtype=float)
    future_x = np.arange(len(values) + 1, len(values) + steps + 1, dtype=float)
    coefficients = np.polyfit(x, values, 2)
    return build_output(
        values,
        np.polyval(coefficients, x),
        np.polyval(coefficients, future_x),
        details={
            "equation": "y(t) = a * t^2 + b * t + c",
            "a": float(coefficients[0]),
            "b": float(coefficients[1]),
            "c": float(coefficients[2]),
            "multi_step_strategy": "Direct deterministic extrapolation",
        },
    )


SPEC = MethodSpec(
    model_id="quadratic",
    display_name="Quadratic Trend",
    icon="➿",
    navigation_group="Trend Projection",
    description="Projects a second-degree polynomial trend into the future.",
    guidance="Use cautiously: polynomial forecasts can grow or fall very quickly outside the observed range.",
    forecast=forecast,
    render_parameters=empty_parameters,
    multi_step_strategy="Direct deterministic extrapolation",
    interval_capability="Descriptive in-sample residual band; unavailable when degenerate",
    minimum_observations=5,
)
