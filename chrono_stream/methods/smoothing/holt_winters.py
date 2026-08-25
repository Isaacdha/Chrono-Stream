"""Holt–Winters level, trend, and seasonal forecast."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...contracts import MethodSpec
from ...intervals import build_output


def forecast(
    values: np.ndarray, steps: int, params: dict[str, Any], **_: Any
) -> dict[str, Any]:
    """Fit additive/multiplicative Holt–Winters with an optional damped trend."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    period = int(params.get("seasonal_period", 12))
    if period < 2 or len(values) < 2 * period:
        raise ValueError(
            f"Holt-Winters needs at least two full seasons ({2 * period} observations)."
        )
    trend = str(params.get("trend", "add"))
    seasonal = str(params.get("seasonal", "add"))
    if trend not in {"add", "mul"} or seasonal not in {"add", "mul"}:
        raise ValueError("Trend and seasonality must be additive or multiplicative.")
    alpha = params.get("alpha")
    beta = params.get("beta")
    gamma = params.get("gamma")
    supplied = (alpha is not None, beta is not None, gamma is not None)
    if any(supplied) and not all(supplied):
        raise ValueError(
            "Holt-Winters alpha, beta, and gamma must all be automatic or all be supplied."
        )
    automatic = not any(supplied)
    damped = bool(params.get("damped", False))
    phi = None
    if not automatic:
        alpha = float(alpha)
        beta = float(beta)
        gamma = float(gamma)
        if not np.isfinite(alpha) or not 0.0 < alpha <= 1.0:
            raise ValueError("Holt-Winters alpha must be in (0, 1].")
        if not np.isfinite(beta) or not 0.0 <= beta <= alpha:
            raise ValueError("Holt-Winters beta must be in [0, alpha].")
        if (
            not np.isfinite(gamma)
            or gamma < 0.0
            or gamma > 1.0 - alpha + 1e-12
        ):
            raise ValueError("Holt-Winters gamma must be in [0, 1 - alpha].")
        if damped:
            phi = float(params.get("phi", 0.98))
            if not np.isfinite(phi) or not 0.0 < phi <= 1.0:
                raise ValueError("Holt-Winters damping phi must be in (0, 1].")
    if (trend == "mul" or seasonal == "mul") and np.any(values <= 0):
        raise ValueError(
            "Multiplicative components require all values to be greater than zero."
        )
    model = ExponentialSmoothing(
        values,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=period,
        damped_trend=damped,
        initialization_method="estimated",
    )
    fit = model.fit(
        optimized=automatic,
        smoothing_level=alpha,
        smoothing_trend=beta,
        smoothing_seasonal=gamma,
        damping_trend=phi,
    )
    return build_output(
        values,
        fit.fittedvalues,
        fit.forecast(steps),
        details={
            "selection": "Automatic" if automatic else "Manual",
            "seasonal_period": period,
            "smoothing_level": float(fit.params["smoothing_level"]),
            "smoothing_trend": float(fit.params["smoothing_trend"]),
            "smoothing_seasonal": float(fit.params["smoothing_seasonal"]),
            **(
                {"damping_trend": float(fit.params["damping_trend"])}
                if damped
                else {}
            ),
            "multi_step_strategy": "Direct state extrapolation",
        },
    )


def render_parameters(data_length: int, seasonal_period: int) -> dict[str, Any]:
    """Render Holt–Winters controls."""
    import streamlit as st

    maximum_period = max(2, data_length // 2)
    parameters: dict[str, Any] = {
        "seasonal_period": st.number_input(
            "Seasonal period",
            min_value=2,
            max_value=maximum_period,
            value=max(2, min(seasonal_period, maximum_period)),
            step=1,
        )
    }
    col1, col2 = st.columns(2)
    with col1:
        parameters["trend"] = st.selectbox(
            "Trend",
            ["add", "mul"],
            format_func=lambda value: {
                "add": "Additive",
                "mul": "Multiplicative",
            }[value],
        )
    with col2:
        parameters["seasonal"] = st.selectbox(
            "Seasonality",
            ["add", "mul"],
            format_func=lambda value: {
                "add": "Additive",
                "mul": "Multiplicative",
            }[value],
        )
    optimize = st.toggle(
        "Automatically find optimal alpha, beta, and gamma", value=True
    )
    if optimize:
        parameters.update({"alpha": None, "beta": None, "gamma": None})
    else:
        columns = st.columns(3)
        alpha = columns[0].slider(
            "Smoothing level (alpha)", 0.01, 0.99, 0.30, 0.01
        )
        gamma_maximum = round(1.0 - float(alpha), 2)
        parameters["alpha"] = alpha
        parameters["beta"] = columns[1].slider(
            "Trend smoothing (beta)",
            0.0,
            float(alpha),
            min(0.10, float(alpha)),
            0.01,
            help="Admissibility requires beta to be no greater than alpha.",
        )
        parameters["gamma"] = columns[2].slider(
            "Seasonal smoothing (gamma)",
            0.0,
            gamma_maximum,
            min(0.10, gamma_maximum),
            0.01,
            help="Admissibility requires gamma to be no greater than 1 - alpha.",
        )
    parameters["damped"] = st.toggle("Damp the projected trend", value=False)
    if parameters["damped"] and not optimize:
        parameters["phi"] = st.slider(
            "Damping coefficient (phi)", 0.80, 0.995, 0.98, 0.005
        )
    return parameters


SPEC = MethodSpec(
    model_id="triple_exponential_smoothing",
    display_name="Triple Exponential Smoothing (Holt-Winters)",
    icon="3️⃣",
    navigation_group="Smoothing",
    description="Holt-Winters estimates level, trend, and a repeating seasonal pattern.",
    guidance="Use when at least two full seasonal cycles are available.",
    forecast=forecast,
    render_parameters=render_parameters,
    multi_step_strategy="Direct state extrapolation",
    interval_capability="Descriptive in-sample residual band; unavailable when degenerate",
)
