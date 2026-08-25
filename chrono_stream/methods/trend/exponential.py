"""Log-linear exponential trend with Duan smearing."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ...contracts import MethodSpec, empty_parameters
from ...intervals import build_output


def forecast(
    values: np.ndarray, steps: int, params: dict[str, Any], **_: Any
) -> dict[str, Any]:
    """Fit a log-linear trend and return original-scale conditional means."""
    if np.any(values <= 0):
        raise ValueError(
            "Exponential trend requires all values to be greater than zero."
        )
    x = np.arange(1, len(values) + 1, dtype=float)
    future_x = np.arange(len(values) + 1, len(values) + steps + 1, dtype=float)
    log_values = np.log(values)
    coefficients = np.polyfit(x, log_values, 1)
    fitted_log = np.polyval(coefficients, x)
    with np.errstate(over="ignore", invalid="ignore"):
        smearing_factor = float(np.mean(np.exp(log_values - fitted_log)))
    if not math.isfinite(smearing_factor) or smearing_factor <= 0:
        raise ValueError(
            "The exponential trend's retransformation correction is not finite."
        )
    fitted = smearing_factor * np.exp(fitted_log)
    future = smearing_factor * np.exp(np.polyval(coefficients, future_x))
    return build_output(
        values,
        fitted,
        future,
        details={
            "equation": "E[y(t)] = smearing_factor * exp(log_slope * t + log_intercept)",
            "log_slope": float(coefficients[0]),
            "log_intercept": float(coefficients[1]),
            "smearing_factor": smearing_factor,
            "retransformation": "Duan nonparametric smearing estimate",
            "multi_step_strategy": "Direct deterministic extrapolation",
        },
    )


SPEC = MethodSpec(
    model_id="exponential",
    display_name="Exponential Trend",
    icon="✴️",
    navigation_group="Trend Projection",
    description="Fits a straight line to the logarithm of positive values and uses Duan's smearing correction when returning forecasts to the original scale.",
    guidance="Only supports strictly positive values, assumes a roughly constant percentage change, and reports its mean-bias smearing factor.",
    forecast=forecast,
    render_parameters=empty_parameters,
    multi_step_strategy="Direct deterministic extrapolation",
    interval_capability="Descriptive in-sample residual band; unavailable when degenerate",
)
