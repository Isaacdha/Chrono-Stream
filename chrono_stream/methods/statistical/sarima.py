"""SARIMA method specification."""

from __future__ import annotations

from typing import Any

from ...contracts import MethodSpec
from .box_jenkins import SARIMA_METHOD_TEST_KEYS, forecast as box_jenkins_forecast
from .box_jenkins import render_parameters as render_box_jenkins_parameters


def forecast(values, steps: int, params: dict[str, Any], **kwargs: Any):
    """Run the shared Box–Jenkins engine with seasonal terms."""
    return box_jenkins_forecast(values, steps, params, seasonal=True, **kwargs)


def render_parameters(data_length: int, seasonal_period: int) -> dict[str, Any]:
    """Render SARIMA-specific use of the shared Box–Jenkins controls."""
    return render_box_jenkins_parameters("sarima", data_length, seasonal_period)


SPEC = MethodSpec(
    model_id="sarima",
    display_name="SARIMA",
    icon="❄️",
    navigation_group="Statistical",
    description="Extends ARIMA with autoregressive, differencing, and moving-average terms at a seasonal interval.",
    guidance="Set the observations per cycle. Seasonal differencing is assessed before regular differencing, and strict selection requires every enabled diagnostic to pass.",
    forecast=forecast,
    render_parameters=render_parameters,
    multi_step_strategy="Direct model forecast",
    interval_capability="Model-native predictive intervals",
    statistical_test_keys=SARIMA_METHOD_TEST_KEYS,
)
