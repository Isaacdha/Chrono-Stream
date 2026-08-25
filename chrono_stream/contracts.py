"""Contracts shared by every forecasting method."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd


ForecastResult = dict[str, Any]
ForecastHandler = Callable[..., ForecastResult]
ParameterRenderer = Callable[[int, int], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class MethodSpec:
    """The single registration record for one user-selectable forecast method."""

    model_id: str
    display_name: str
    icon: str
    navigation_group: str
    description: str
    guidance: str
    forecast: ForecastHandler
    render_parameters: ParameterRenderer
    multi_step_strategy: str
    interval_capability: str
    minimum_observations: int = 4
    random_seed: int | None = None
    statistical_test_keys: tuple[str, ...] = ()
    research_key: str | None = None

    @property
    def method_research_key(self) -> str:
        """Return the catalog key used for practical and scholarly information."""
        return self.research_key or self.model_id

    @property
    def url_path(self) -> str:
        """Return the stable URL slug used by callable Streamlit pages."""
        return self.model_id.replace("_", "-")


def empty_parameters(_data_length: int, _seasonal_period: int) -> dict[str, Any]:
    """Render no controls for a parameter-free forecasting method."""
    return {}


def validate_series(values: Any, minimum: int = 4) -> np.ndarray:
    """Return one finite float vector after enforcing a method's sample minimum."""
    array = np.asarray(values, dtype=float).reshape(-1)
    if len(array) < minimum or not np.isfinite(array).all():
        raise ValueError(
            f"The model requires at least {minimum} finite numeric observations."
        )
    return array


def validate_forecast_context(
    dates: Any,
    forecast_dates: Any,
    *,
    observations: int,
    steps: int,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Normalize and validate the date indexes passed to a method handler."""
    observed = pd.DatetimeIndex(dates)
    future = pd.DatetimeIndex(forecast_dates)
    if len(observed) != observations:
        raise ValueError("Date and value arrays must have the same length.")
    if len(future) != steps:
        raise ValueError("The future date index must match the forecast horizon.")
    return observed, future
