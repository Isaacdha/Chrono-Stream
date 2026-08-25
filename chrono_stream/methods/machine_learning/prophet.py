"""Prophet trend-and-calendar forecast."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ...contracts import MethodSpec
from ...intervals import build_output


def forecast(
    values: np.ndarray,
    steps: int,
    params: dict[str, Any],
    *,
    dates: pd.DatetimeIndex,
    forecast_dates: pd.DatetimeIndex,
    **_: Any,
) -> dict[str, Any]:
    """Fit Prophet lazily and forecast the supplied future date index."""
    try:
        from prophet import Prophet
    except ImportError as exc:
        raise ImportError(
            "Prophet is not installed. Install the project requirements and try again."
        ) from exc

    seasonality_mode = str(params.get("seasonality_mode", "additive"))
    if seasonality_mode not in {"additive", "multiplicative"}:
        raise ValueError("Prophet seasonality mode must be additive or multiplicative.")
    changepoint_prior_scale = float(params.get("changepoint_prior_scale", 0.05))
    if changepoint_prior_scale <= 0:
        raise ValueError("Trend flexibility must be greater than zero.")

    frame = pd.DataFrame({"ds": pd.to_datetime(dates), "y": values})
    model = Prophet(
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=changepoint_prior_scale,
        yearly_seasonality=params.get("yearly_seasonality", "auto"),
        weekly_seasonality=params.get("weekly_seasonality", "auto"),
        daily_seasonality=False,
        interval_width=0.95,
    )
    model.fit(frame)
    prediction_dates = pd.DataFrame(
        {"ds": pd.DatetimeIndex(dates).append(pd.DatetimeIndex(forecast_dates))}
    )
    prediction = model.predict(prediction_dates)
    future = prediction.iloc[len(values) :]
    return build_output(
        values,
        prediction["yhat"].iloc[: len(values)].to_numpy(),
        future["yhat"].to_numpy(),
        future["yhat_lower"].to_numpy(),
        future["yhat_upper"].to_numpy(),
        details={
            "seasonality_mode": seasonality_mode,
            "changepoint_prior_scale": changepoint_prior_scale,
            "yearly_seasonality": params.get("yearly_seasonality", "auto"),
            "weekly_seasonality": params.get("weekly_seasonality", "auto"),
            "multi_step_strategy": "Direct date-index forecast",
        },
    )


def render_parameters(_data_length: int, _seasonal_period: int) -> dict[str, Any]:
    """Render Prophet controls."""
    import streamlit as st

    parameters: dict[str, Any] = {
        "seasonality_mode": st.selectbox(
            "Seasonality mode", ["additive", "multiplicative"]
        ),
        "changepoint_prior_scale": st.slider(
            "Trend flexibility", 0.001, 0.50, 0.05, 0.001
        ),
    }
    col1, col2 = st.columns(2)
    with col1:
        yearly = st.selectbox("Yearly seasonality", ["Auto", "On", "Off"])
    with col2:
        weekly = st.selectbox("Weekly seasonality", ["Auto", "On", "Off"])
    mapping: dict[str, str | bool] = {"Auto": "auto", "On": True, "Off": False}
    parameters["yearly_seasonality"] = mapping[yearly]
    parameters["weekly_seasonality"] = mapping[weekly]
    return parameters


SPEC = MethodSpec(
    model_id="prophet",
    display_name="Prophet",
    icon="🔮",
    navigation_group="Machine Learning",
    description="An additive model with trend changepoints and calendar seasonalities, implemented by Prophet.",
    guidance="Best suited to regular calendar data with enough history to estimate seasonal patterns and trend changes.",
    forecast=forecast,
    render_parameters=render_parameters,
    multi_step_strategy="Direct date-index forecast",
    interval_capability="Model-native predictive intervals",
)
