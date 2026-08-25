"""Simple exponential-smoothing forecast."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...contracts import MethodSpec
from ...intervals import build_output


def forecast(
    values: np.ndarray, steps: int, params: dict[str, Any], **_: Any
) -> dict[str, Any]:
    """Estimate or apply a simple exponential-smoothing level recursion."""
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing

    alpha = params.get("alpha")
    if alpha is not None and not 0 < float(alpha) <= 1:
        raise ValueError("Smoothing level alpha must be greater than 0 and at most 1.")
    model = SimpleExpSmoothing(values, initialization_method="estimated")
    fit = model.fit(optimized=alpha is None, smoothing_level=alpha)
    return build_output(
        values,
        fit.fittedvalues,
        fit.forecast(steps),
        details={
            "selection": "Automatic" if alpha is None else "Manual",
            "smoothing_level": float(fit.params["smoothing_level"]),
            "multi_step_strategy": "Direct state extrapolation",
        },
    )


def render_parameters(_data_length: int, _seasonal_period: int) -> dict[str, Any]:
    """Render simple-exponential-smoothing controls."""
    import streamlit as st

    optimize = st.toggle("Estimate smoothing level automatically", value=True)
    return {
        "alpha": (
            None
            if optimize
            else st.slider("Smoothing level (alpha)", 0.01, 1.0, 0.30, 0.01)
        )
    }


SPEC = MethodSpec(
    model_id="single_exponential_smoothing",
    display_name="Single Exponential Smoothing",
    icon="1️⃣",
    navigation_group="Smoothing",
    description="Estimates a changing level without an explicit trend or seasonal component.",
    guidance="Use for a series with a changing level but no clear trend or seasonality.",
    forecast=forecast,
    render_parameters=render_parameters,
    multi_step_strategy="Direct state extrapolation",
    interval_capability="Descriptive in-sample residual band; unavailable when degenerate",
)
