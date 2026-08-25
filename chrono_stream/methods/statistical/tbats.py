"""TBATS forecasting for multiple, long, or non-integer seasonal periods."""

from __future__ import annotations

import importlib
import math
import os
from typing import Any
import warnings

import numpy as np

from ...contracts import MethodSpec
from ...features import validate_regular_dates
from ...intervals import build_output


def _patch_sklearn_compatibility() -> None:
    """Bridge tbats 1.1.3 to sklearn's renamed finite-value keyword.

    tbats imports ``check_array`` into three modules and still passes the removed
    ``force_all_finite`` keyword.  The wrapper maps only that keyword to sklearn's
    current ``ensure_all_finite`` spelling; no data or model behavior is changed.
    """
    from sklearn.utils.validation import check_array as sklearn_check_array

    def compatible_check_array(*args: Any, **kwargs: Any):
        if "force_all_finite" in kwargs:
            kwargs["ensure_all_finite"] = kwargs.pop("force_all_finite")
        return sklearn_check_array(*args, **kwargs)

    for module_name in (
        "tbats.abstract.Estimator",
        "tbats.abstract.Model",
        "tbats.transformation.BoxCox",
    ):
        module = importlib.import_module(module_name)
        module.check_array = compatible_check_array


def _seasonal_periods(raw: Any) -> list[float]:
    if isinstance(raw, str):
        pieces = [piece.strip() for piece in raw.split(",") if piece.strip()]
    elif np.isscalar(raw):
        pieces = [raw]
    else:
        pieces = list(raw)
    try:
        periods = sorted({float(piece) for piece in pieces})
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "TBATS seasonal periods must be comma-separated numeric values."
        ) from exc
    if not periods or any(not np.isfinite(period) or period <= 1 for period in periods):
        raise ValueError("TBATS needs at least one finite seasonal period greater than 1.")
    return periods


def forecast(
    values: np.ndarray,
    steps: int,
    params: dict[str, Any],
    *,
    dates: Any,
    forecast_dates: Any,
    **_: Any,
) -> dict[str, Any]:
    """Fit TBATS and return its analytical point and interval forecasts."""
    try:
        os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
        from tbats import TBATS
    except ImportError as exc:
        raise ImportError(
            "TBATS is not installed. Install the pinned project requirements and try again."
        ) from exc

    _patch_sklearn_compatibility()
    _observed, _future_dates, frequency = validate_regular_dates(dates, forecast_dates)
    periods = _seasonal_periods(params.get("seasonal_periods", [12]))
    required = max(16, int(math.ceil(2 * max(periods))))
    if len(values) < required:
        raise ValueError(
            f"TBATS needs at least two repetitions of its longest period: "
            f"{required} observations for periods {periods}."
        )
    automatic = bool(params.get("automatic", True))
    use_arma_errors = bool(params.get("use_arma_errors", True))
    positive = bool(np.all(values > 0))
    if automatic:
        use_box_cox: bool | None = None if positive else False
        use_trend: bool | None = None
        use_damped_trend: bool | None = None
    else:
        use_box_cox = bool(params.get("use_box_cox", False))
        use_trend = bool(params.get("use_trend", True))
        use_damped_trend = bool(params.get("use_damped_trend", True))
        if use_damped_trend and not use_trend:
            raise ValueError("TBATS trend damping requires the trend component.")
        if use_box_cox and not positive:
            raise ValueError("TBATS Box–Cox transformation requires positive values.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        estimator = TBATS(
            use_box_cox=use_box_cox,
            box_cox_bounds=(0.0, 1.0),
            use_trend=use_trend,
            use_damped_trend=use_damped_trend,
            seasonal_periods=periods,
            use_arma_errors=use_arma_errors,
            show_warnings=False,
            n_jobs=1,
            multiprocessing_start_method="spawn",
        )
        fitted_model = estimator.fit(values)
    point, intervals = fitted_model.forecast(steps=steps, confidence_level=0.95)
    components = fitted_model.params.components
    details = {
        "selection": "Automatic AIC component search" if automatic else "Manual component switches",
        "seasonal_periods": [float(value) for value in components.seasonal_periods],
        "seasonal_harmonics": [int(value) for value in components.seasonal_harmonics],
        "use_box_cox": bool(components.use_box_cox),
        "box_cox_lambda": (
            float(fitted_model.params.box_cox_lambda)
            if fitted_model.params.box_cox_lambda is not None
            else None
        ),
        "box_cox_automatic_omission": (
            "Nonpositive data excluded Box–Cox from automatic search" if automatic and not positive else None
        ),
        "use_trend": bool(components.use_trend),
        "use_damped_trend": bool(components.use_damped_trend),
        "damping_phi": (
            float(fitted_model.params.phi)
            if fitted_model.params.phi is not None
            else None
        ),
        "arma_errors": {
            "enabled_for_search": use_arma_errors,
            "p": int(components.p),
            "q": int(components.q),
        },
        "AIC": float(fitted_model.aic),
        "warnings": [str(item) for item in fitted_model.warnings],
        "model_summary": str(fitted_model.summary()),
        "n_jobs": 1,
        "date_frequency": frequency,
        "compatibility_note": (
            "Maps tbats 1.1.3's legacy force_all_finite keyword to sklearn's "
            "ensure_all_finite keyword at runtime."
        ),
        "multi_step_strategy": "Direct TBATS state-space forecast",
        "interval_assumptions": "Native 95% TBATS intervals under Gaussian errors",
    }
    return build_output(
        values,
        fitted_model.y_hat,
        point,
        intervals["lower_bound"],
        intervals["upper_bound"],
        details,
    )


def render_parameters(data_length: int, seasonal_period: int) -> dict[str, Any]:
    """Render TBATS controls."""
    import streamlit as st

    st.caption(
        "Enter one or more periods in observations, for example 24, 168 for hourly "
        "daily/weekly cycles. Non-integer periods are supported."
    )
    periods = st.text_input("Seasonal periods", value=str(seasonal_period))
    automatic = st.toggle("Automatically select TBATS components by AIC", value=True)
    parameters: dict[str, Any] = {
        "seasonal_periods": periods,
        "automatic": automatic,
    }
    if automatic:
        parameters["use_arma_errors"] = st.toggle(
            "Allow ARMA error correction", value=True
        )
        st.caption(
            "Automatic search considers Box–Cox (for positive data), trend, and damping. "
            "The longest period needs at least two observed repetitions."
        )
    else:
        columns = st.columns(4)
        parameters["use_box_cox"] = columns[0].toggle("Box–Cox", value=False)
        parameters["use_trend"] = columns[1].toggle("Trend", value=True)
        parameters["use_damped_trend"] = columns[2].toggle(
            "Damped trend", value=True
        )
        parameters["use_arma_errors"] = columns[3].toggle(
            "ARMA errors", value=True
        )
    return parameters


SPEC = MethodSpec(
    model_id="tbats",
    display_name="TBATS",
    icon="🌀",
    navigation_group="Statistical",
    description="A trigonometric state-space model for multiple, high-frequency, or non-integer seasonal periods.",
    guidance="TBATS can be computationally expensive. Supply defensible periods and at least two repetitions of the longest cycle.",
    forecast=forecast,
    render_parameters=render_parameters,
    multi_step_strategy="Direct state-space forecast",
    interval_capability="Model-native Gaussian predictive intervals",
    minimum_observations=16,
    statistical_test_keys=("information_criteria",),
)
