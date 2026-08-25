"""Last-observation-carried-forward naive forecast."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...contracts import MethodSpec, empty_parameters
from ...intervals import build_output


def forecast(
    values: np.ndarray, steps: int, params: dict[str, Any], **_: Any
) -> dict[str, Any]:
    """Repeat the final observation at every future horizon."""
    fitted = np.full(len(values), np.nan)
    fitted[1:] = values[:-1]
    future = np.full(steps, float(values[-1]))
    innovations = np.diff(values)
    sigma = float(np.sqrt(np.mean(innovations**2)))
    horizons = np.arange(1, steps + 1, dtype=float)
    margin = 1.96 * sigma * np.sqrt(horizons)
    return build_output(
        values,
        fitted,
        future,
        future - margin,
        future + margin,
        details={
            "rule": "Every forecast equals the last observed value",
            "last_observation": float(values[-1]),
            "multi_step_strategy": "Direct persistence",
            "innovation_standard_deviation": sigma,
            "interval_method": "Gaussian random-walk 95% predictive interval",
            "interval_assumptions": (
                "Independent, constant-variance Gaussian first differences; forecast "
                "standard deviation accumulates in proportion to sqrt(h)."
            ),
        },
    )


SPEC = MethodSpec(
    model_id="naive",
    display_name="Naive Forecast",
    icon="🧭",
    navigation_group="Baselines",
    description="Repeats the last observed value for every forecast horizon.",
    guidance="Use as the essential non-seasonal benchmark: a more complex method should justify itself by improving on persistence out of sample.",
    forecast=forecast,
    render_parameters=empty_parameters,
    multi_step_strategy="Direct persistence",
    interval_capability="Gaussian random-walk predictive intervals",
)
