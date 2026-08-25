"""Random-walk-with-drift point forecast."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...contracts import MethodSpec, empty_parameters
from ...intervals import build_output


def forecast(
    values: np.ndarray, steps: int, params: dict[str, Any], **_: Any
) -> dict[str, Any]:
    """Extrapolate the average change from the first to final observation."""
    drift_per_step = float((values[-1] - values[0]) / (len(values) - 1))
    fitted = np.full(len(values), np.nan)
    fitted[1:] = values[:-1] + drift_per_step
    future = values[-1] + drift_per_step * np.arange(1, steps + 1)
    innovations = values[1:] - fitted[1:]
    variance_denominator = max(len(innovations) - 1, 1)
    sigma = float(np.sqrt(np.sum(innovations**2) / variance_denominator))
    horizons = np.arange(1, steps + 1, dtype=float)
    standard_errors = sigma * np.sqrt(
        horizons * (1.0 + horizons / (len(values) - 1))
    )
    margin = 1.96 * standard_errors
    return build_output(
        values,
        fitted,
        future,
        future - margin,
        future + margin,
        details={
            "rule": "Extend the average first-to-last change per observation",
            "drift_per_step": drift_per_step,
            "first_observation": float(values[0]),
            "last_observation": float(values[-1]),
            "fitted_value_method": "One-step random-walk-with-drift forecasts",
            "multi_step_strategy": "Direct linear drift extrapolation",
            "innovation_standard_deviation": sigma,
            "interval_method": (
                "Gaussian random-walk-with-drift 95% predictive interval"
            ),
            "interval_assumptions": (
                "Independent, constant-variance Gaussian one-step innovations, with "
                "uncertainty from both future innovations and the estimated drift."
            ),
        },
    )


SPEC = MethodSpec(
    model_id="drift",
    display_name="Drift Forecast",
    icon="📐",
    navigation_group="Baselines",
    description="Extrapolates the average per-period change between the first and last observations.",
    guidance="Use as a simple trend-sensitive benchmark. It assumes the historical average change continues unchanged through the forecast horizon.",
    forecast=forecast,
    render_parameters=empty_parameters,
    multi_step_strategy="Direct linear drift extrapolation",
    interval_capability="Gaussian random-walk-with-drift predictive intervals",
)
