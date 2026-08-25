"""Holt linear-trend exponential-smoothing forecast."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...contracts import MethodSpec
from ...intervals import build_output


def forecast(
    values: np.ndarray, steps: int, params: dict[str, Any], **_: Any
) -> dict[str, Any]:
    """Fit Holt's additive trend, optionally with damping."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    damped = bool(params.get("damped", False))
    alpha = params.get("alpha")
    beta = params.get("beta")
    if (alpha is None) != (beta is None):
        raise ValueError("Holt alpha and beta must either both be automatic or both be supplied.")
    automatic = alpha is None
    phi = None
    if not automatic:
        alpha = float(alpha)
        beta = float(beta)
        if not np.isfinite(alpha) or not 0.0 < alpha <= 1.0:
            raise ValueError("Holt alpha must be in (0, 1].")
        if not np.isfinite(beta) or not 0.0 <= beta <= alpha:
            raise ValueError("Holt beta must be in [0, alpha].")
        if damped:
            phi = float(params.get("phi", 0.98))
            if not np.isfinite(phi) or not 0.0 < phi <= 1.0:
                raise ValueError("Holt damping phi must be in (0, 1].")
    model = ExponentialSmoothing(
        values,
        trend="add",
        damped_trend=damped,
        initialization_method="estimated",
    )
    fit = model.fit(
        optimized=automatic,
        smoothing_level=alpha,
        smoothing_trend=beta,
        damping_trend=phi,
    )
    return build_output(
        values,
        fit.fittedvalues,
        fit.forecast(steps),
        details={
            "selection": "Automatic" if automatic else "Manual",
            "smoothing_level": float(fit.params["smoothing_level"]),
            "smoothing_trend": float(fit.params["smoothing_trend"]),
            **(
                {"damping_trend": float(fit.params["damping_trend"])}
                if damped
                else {}
            ),
            "multi_step_strategy": "Direct state extrapolation",
        },
    )


def render_parameters(_data_length: int, _seasonal_period: int) -> dict[str, Any]:
    """Render Holt controls."""
    import streamlit as st

    optimize = st.toggle("Automatically find optimal alpha and beta", value=True)
    parameters: dict[str, Any]
    if optimize:
        parameters = {"alpha": None, "beta": None}
    else:
        col1, col2 = st.columns(2)
        alpha = col1.slider(
            "Smoothing level (alpha)", 0.01, 1.0, 0.30, 0.01
        )
        parameters = {
            "alpha": alpha,
            "beta": col2.slider(
                "Trend smoothing (beta)",
                0.0,
                float(alpha),
                min(0.10, float(alpha)),
                0.01,
                help="Admissibility requires beta to be no greater than alpha.",
            ),
        }
    parameters["damped"] = st.toggle("Damp the projected trend", value=False)
    if parameters["damped"] and not optimize:
        parameters["phi"] = st.slider(
            "Damping coefficient (phi)", 0.80, 0.995, 0.98, 0.005
        )
    return parameters


SPEC = MethodSpec(
    model_id="double_exponential_smoothing",
    display_name="Double Exponential Smoothing (Holt)",
    icon="2️⃣",
    navigation_group="Smoothing",
    description="Holt's method estimates both a level and a linear trend.",
    guidance="Use when the series has a trend but no repeating seasonal pattern.",
    forecast=forecast,
    render_parameters=render_parameters,
    multi_step_strategy="Direct state extrapolation",
    interval_capability="Descriptive in-sample residual band; unavailable when degenerate",
)
