"""Least-squares linear trend projection."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...contracts import MethodSpec, empty_parameters
from ...intervals import build_output


def forecast(
    values: np.ndarray, steps: int, params: dict[str, Any], **_: Any
) -> dict[str, Any]:
    """Fit and extrapolate one least-squares straight line."""
    x = np.arange(1, len(values) + 1, dtype=float)
    future_x = np.arange(len(values) + 1, len(values) + steps + 1, dtype=float)
    coefficients = np.polyfit(x, values, 1)
    return build_output(
        values,
        np.polyval(coefficients, x),
        np.polyval(coefficients, future_x),
        details={
            "equation": "y(t) = slope * t + intercept",
            "slope": float(coefficients[0]),
            "intercept": float(coefficients[1]),
            "multi_step_strategy": "Direct deterministic extrapolation",
        },
    )


SPEC = MethodSpec(
    model_id="linear",
    display_name="Linear Trend",
    icon="↗️",
    navigation_group="Trend Projection",
    description="Projects a least-squares straight-line trend into the future.",
    guidance="A useful benchmark when the trend changes by a roughly constant amount per period.",
    forecast=forecast,
    render_parameters=empty_parameters,
    multi_step_strategy="Direct deterministic extrapolation",
    interval_capability="Descriptive in-sample residual band; unavailable when degenerate",
)
