"""STL decomposition with declared trend and seasonal forecast rules."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...contracts import MethodSpec
from ...intervals import build_output


def forecast(
    values: np.ndarray, steps: int, params: dict[str, Any], **_: Any
) -> dict[str, Any]:
    """Decompose with STL, extend the tail trend, and repeat seasonal phases."""
    from statsmodels.tsa.seasonal import STL

    period = int(params.get("seasonal_period", 12))
    if period < 2 or len(values) < 2 * period:
        raise ValueError(
            f"STL decomposition needs at least two full seasons ({2 * period} observations)."
        )
    robust = bool(params.get("robust", True))
    decomposition = STL(values, period=period, robust=robust).fit()
    fitted = decomposition.trend + decomposition.seasonal

    tail_length = min(len(values), max(2 * period, 8))
    tail_x = np.arange(len(values) - tail_length, len(values), dtype=float)
    tail_trend = decomposition.trend[-tail_length:]
    valid = np.isfinite(tail_trend)
    if valid.sum() < 2:
        raise ValueError("STL did not produce enough finite tail-trend values.")
    slope, intercept = np.polyfit(tail_x[valid], tail_trend[valid], 1)
    future_x = np.arange(len(values), len(values) + steps, dtype=float)
    trend_forecast = intercept + slope * future_x

    seasonal_pattern = np.zeros(period)
    positions = np.arange(len(values)) % period
    for phase in range(period):
        phase_values = decomposition.seasonal[positions == phase]
        if not np.isfinite(phase_values).any():
            raise ValueError(f"STL seasonal phase {phase + 1} has no finite values.")
        seasonal_pattern[phase] = float(np.nanmean(phase_values))
    seasonal_forecast = seasonal_pattern[
        np.arange(len(values), len(values) + steps) % period
    ]
    return build_output(
        values,
        fitted,
        trend_forecast + seasonal_forecast,
        details={
            "decomposition": "STL",
            "official_x11": False,
            "seasonal_period": period,
            "robust": robust,
            "trend_extension": (
                f"Least-squares line fitted to the final {tail_length} STL trend values"
            ),
            "seasonal_extension": "Mean STL seasonal value for each cycle phase",
            "multi_step_strategy": "Direct component extrapolation and recombination",
        },
    )


def render_parameters(data_length: int, seasonal_period: int) -> dict[str, Any]:
    """Render STL decomposition-forecast controls."""
    import streamlit as st

    maximum = max(2, data_length // 2)
    return {
        "seasonal_period": st.number_input(
            "Seasonal period",
            min_value=2,
            max_value=maximum,
            value=max(2, min(seasonal_period, maximum)),
            step=1,
        ),
        "robust": st.toggle("Use robust decomposition", value=True),
    }


SPEC = MethodSpec(
    model_id="stl",
    display_name="STL Decomposition Forecast (X-11-inspired)",
    icon="💫",
    navigation_group="Decomposition & Seasonal Adjustment",
    description="A robust STL decomposition followed by explicit trend and seasonal extrapolation; historically related to, but not an implementation of Census X-11.",
    guidance="Use this portable STL forecast as a decomposition benchmark. It is intentionally not official Census X-11/X-13.",
    forecast=forecast,
    render_parameters=render_parameters,
    multi_step_strategy="Direct component extrapolation and recombination",
    interval_capability="Descriptive in-sample residual band; unavailable when degenerate",
)
