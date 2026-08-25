"""ARIMA method specification."""

from __future__ import annotations

from typing import Any

from ...contracts import MethodSpec
from .box_jenkins import ARIMA_METHOD_TEST_KEYS, forecast as box_jenkins_forecast
from .box_jenkins import render_parameters as render_box_jenkins_parameters


def forecast(values, steps: int, params: dict[str, Any], **kwargs: Any):
    """Run the shared Box–Jenkins engine without seasonal terms."""
    return box_jenkins_forecast(
        values, steps, params, seasonal=False, **kwargs
    )


def render_parameters(data_length: int, seasonal_period: int) -> dict[str, Any]:
    """Render ARIMA-specific use of the shared Box–Jenkins controls."""
    return render_box_jenkins_parameters("arima", data_length, seasonal_period)


SPEC = MethodSpec(
    model_id="arima",
    display_name="ARIMA",
    icon="🌠",
    navigation_group="Statistical",
    description="Combines autoregression, differencing, and moving-average errors for non-seasonal dynamics.",
    guidance="The strict workflow transforms variance first, establishes mean stationarity, examines ACF/PACF, and ranks only candidates that pass every mandatory diagnostic.",
    forecast=forecast,
    render_parameters=render_parameters,
    multi_step_strategy="Direct model forecast",
    interval_capability="Model-native predictive intervals",
    statistical_test_keys=ARIMA_METHOD_TEST_KEYS,
)
