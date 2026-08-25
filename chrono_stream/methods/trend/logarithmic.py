"""Least-squares logarithmic trend projection."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...contracts import MethodSpec, empty_parameters
from ...intervals import build_output


def forecast(
    values: np.ndarray, steps: int, params: dict[str, Any], **_: Any
) -> dict[str, Any]:
    """Fit values against log(time) and extrapolate at future indexes."""
    x = np.arange(1, len(values) + 1, dtype=float)
    future_x = np.arange(len(values) + 1, len(values) + steps + 1, dtype=float)
    coefficients = np.polyfit(np.log(x), values, 1)
    return build_output(
        values,
        np.polyval(coefficients, np.log(x)),
        np.polyval(coefficients, np.log(future_x)),
        details={
            "equation": "y(t) = slope * log(t) + intercept",
            "slope": float(coefficients[0]),
            "intercept": float(coefficients[1]),
            "multi_step_strategy": "Direct deterministic extrapolation",
        },
    )


SPEC = MethodSpec(
    model_id="logarithmic",
    display_name="Logarithmic Trend",
    icon="❇️",
    navigation_group="Trend Projection",
    description="Fits a trend that changes quickly at first and gradually flattens over time.",
    guidance="Useful for saturating growth where changes become smaller over time.",
    forecast=forecast,
    render_parameters=empty_parameters,
    multi_step_strategy="Direct deterministic extrapolation",
    interval_capability="Descriptive in-sample residual band; unavailable when degenerate",
)
