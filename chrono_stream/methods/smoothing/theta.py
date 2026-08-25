"""Theta-method forecasting with optional seasonal adjustment."""

from __future__ import annotations

from typing import Any
import warnings

import numpy as np
import pandas as pd

from ...contracts import MethodSpec
from ...features import validate_regular_dates
from ...intervals import build_output


def _settings(params: dict[str, Any]) -> tuple[int, str, bool, bool, str, bool, float]:
    period = int(params.get("seasonal_period", 12))
    mode = str(params.get("seasonality", "Automatic test"))
    if mode not in {"Automatic test", "Force seasonal adjustment", "No seasonal adjustment"}:
        raise ValueError(
            "Theta seasonality must be Automatic test, Force seasonal adjustment, or No seasonal adjustment."
        )
    deseasonalize = mode != "No seasonal adjustment"
    use_test = mode == "Automatic test"
    decomposition = str(params.get("decomposition", "auto")).lower()
    aliases = {"additive": "additive", "multiplicative": "multiplicative", "auto": "auto"}
    if decomposition not in aliases:
        raise ValueError("Theta decomposition must be auto, additive, or multiplicative.")
    use_mle = bool(params.get("use_mle", False))
    theta = float(params.get("theta", 2.0))
    if theta < 1:
        raise ValueError("Theta must be at least 1.")
    return period, mode, deseasonalize, use_test, aliases[decomposition], use_mle, theta


def _fit_model(
    values: np.ndarray,
    dates: pd.DatetimeIndex,
    *,
    period: int,
    deseasonalize: bool,
    use_test: bool,
    decomposition: str,
    use_mle: bool,
):
    from statsmodels.tsa.forecasting.theta import ThetaModel

    series = pd.Series(values, index=dates)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Setting the shape on a NumPy array has been deprecated.*",
            category=DeprecationWarning,
        )
        model = ThetaModel(
            series,
            period=period if deseasonalize else None,
            deseasonalize=deseasonalize,
            use_test=use_test,
            method=decomposition,
        )
        return model.fit(use_mle=use_mle, disp=False)


def _causal_fitted(
    values: np.ndarray,
    dates: pd.DatetimeIndex,
    *,
    period: int,
    deseasonalize: bool,
    use_test: bool,
    decomposition: str,
    use_mle: bool,
    theta: float,
) -> np.ndarray:
    """Generate honest historical one-step forecasts from expanding prefixes."""
    start = max(5, 2 * period if deseasonalize else 5)
    fitted = np.full(len(values), np.nan)
    for index in range(start, len(values)):
        try:
            prefix_fit = _fit_model(
                values[:index],
                dates[:index],
                period=period,
                deseasonalize=deseasonalize,
                use_test=use_test,
                decomposition=decomposition,
                use_mle=use_mle,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Setting the shape on a NumPy array has been deprecated.*",
                    category=DeprecationWarning,
                )
                fitted[index] = float(prefix_fit.forecast(1, theta=theta).iloc[0])
        except (ValueError, np.linalg.LinAlgError):
            # Early prefixes can be numerically unsuitable even when the final fit is valid.
            continue
    return fitted


def _innovation_variance(fit: Any, *, use_mle: bool) -> tuple[float, str]:
    """Return variance from the same deseasonalized series used by Theta."""
    if use_mle:
        return float(fit.sigma2), "Theta maximum-likelihood fit"

    from statsmodels.tsa.statespace.sarimax import SARIMAX

    adjusted, _seasonal = fit.model._deseasonalize_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        variance_fit = SARIMAX(
            adjusted,
            order=(0, 1, 1),
            trend="c",
        ).fit(disp=False)
    variance = float(np.asarray(variance_fit.params, dtype=float)[-1])
    if not np.isfinite(variance) or variance < 0.0:
        raise ValueError("Theta could not estimate a finite innovation variance.")
    return variance, "SARIMAX(0,1,1) refit on Theta's deseasonalized series"


def _prediction_intervals(
    point: np.ndarray,
    *,
    smoothing_alpha: float,
    innovation_variance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the integrated-MA variance expression used by Theta."""
    variance_by_horizon = (
        1.0
        + np.arange(len(point), dtype=float) * (1.0 + (smoothing_alpha - 1.0) ** 2)
    ) * innovation_variance
    margin = 1.959963984540054 * np.sqrt(variance_by_horizon)
    return point - margin, point + margin


def forecast(
    values: np.ndarray,
    steps: int,
    params: dict[str, Any],
    *,
    dates: Any,
    forecast_dates: Any,
    **_: Any,
) -> dict[str, Any]:
    """Fit statsmodels' Theta model and use its model-based Gaussian intervals."""
    observed, future_dates, frequency = validate_regular_dates(dates, forecast_dates)
    period, mode, deseasonalize, use_test, decomposition, use_mle, theta = _settings(params)
    if period < 2:
        raise ValueError("Theta seasonal period must be at least 2.")
    if deseasonalize and len(values) < 2 * period:
        raise ValueError(
            f"Theta seasonal adjustment needs at least two full cycles ({2 * period} observations). "
            "Choose no seasonal adjustment or a shorter valid period."
        )
    if decomposition == "multiplicative" and np.any(values <= 0):
        raise ValueError("Multiplicative Theta decomposition requires positive values.")

    fit = _fit_model(
        values,
        observed,
        period=period,
        deseasonalize=deseasonalize,
        use_test=use_test,
        decomposition=decomposition,
        use_mle=use_mle,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Setting the shape on a NumPy array has been deprecated.*",
            category=DeprecationWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="Non-invertible starting MA parameters found.*",
            category=UserWarning,
        )
        point = np.asarray(fit.forecast(steps, theta=theta), dtype=float)
    smoothing_alpha = float(fit.params["alpha"])
    innovation_variance, variance_source = _innovation_variance(
        fit, use_mle=use_mle
    )
    lower, upper = _prediction_intervals(
        point,
        smoothing_alpha=smoothing_alpha,
        innovation_variance=innovation_variance,
    )
    fitted = _causal_fitted(
        values,
        observed,
        period=period,
        deseasonalize=deseasonalize,
        use_test=use_test,
        decomposition=decomposition,
        use_mle=use_mle,
        theta=theta,
    )
    actual_method = str(fit.model.method)
    seasonality_detected = bool(getattr(fit.model, "_has_seasonality", False))
    fitted_values_available = int(np.isfinite(fitted).sum())
    details = {
        "selection": mode,
        "seasonal_period": period if deseasonalize else None,
        "seasonality_detected": seasonality_detected,
        "decomposition_requested": decomposition,
        "decomposition_used": actual_method,
        "theta": theta,
        "estimation": "Maximum likelihood" if use_mle else "Two-step OLS plus SES",
        "trend_slope_b0": float(fit.params["b0"]),
        "smoothing_alpha": smoothing_alpha,
        "innovation_variance": innovation_variance,
        "innovation_variance_source": variance_source,
        "date_frequency": frequency,
        "fitted_value_method": (
            "Expanding-prefix one-step forecasts where the prefix satisfies the "
            "model's minimum history"
        ),
        "fitted_values_available": fitted_values_available,
        "fitted_value_availability_note": (
            "No causal historical fitted value is available when the series ends at "
            "the minimum two-cycle seasonal history."
            if fitted_values_available == 0
            else None
        ),
        "multi_step_strategy": "Direct Theta forecast",
        "interval_assumptions": (
            "Native Theta intervals use the integrated-MA variance form and Gaussian innovations."
        ),
    }
    return build_output(
        values,
        fitted,
        point,
        lower,
        upper,
        details,
    )


def render_parameters(data_length: int, seasonal_period: int) -> dict[str, Any]:
    """Render Theta controls."""
    import streamlit as st

    seasonality = st.selectbox(
        "Seasonal adjustment",
        ["Automatic test", "Force seasonal adjustment", "No seasonal adjustment"],
    )
    parameters: dict[str, Any] = {"seasonality": seasonality}
    if seasonality != "No seasonal adjustment":
        maximum = max(2, data_length // 2)
        parameters["seasonal_period"] = st.number_input(
            "Seasonal period",
            min_value=2,
            max_value=maximum,
            value=max(2, min(seasonal_period, maximum)),
            step=1,
        )
        parameters["decomposition"] = st.selectbox(
            "Seasonal decomposition", ["auto", "additive", "multiplicative"]
        )
    columns = st.columns(2)
    parameters["theta"] = columns[0].number_input(
        "Theta coefficient", min_value=1.0, max_value=20.0, value=2.0, step=0.1
    )
    parameters["use_mle"] = columns[1].toggle(
        "Use maximum-likelihood estimation", value=False
    )
    st.caption(
        "Theta=2 reproduces the conventional two-line method of Assimakopoulos and "
        "Nikolopoulos. Other values use the later generalized/optimized Theta family. "
        "Intervals assume Gaussian innovations."
    )
    return parameters


SPEC = MethodSpec(
    model_id="theta",
    display_name="Theta Method",
    icon="🔭",
    navigation_group="Smoothing",
    description="A lightweight benchmark combining a long-run linear drift component with simple exponential smoothing.",
    guidance="Theta=2 is the classical default; seasonal adjustment needs at least two complete cycles and a defensible period.",
    forecast=forecast,
    render_parameters=render_parameters,
    multi_step_strategy="Direct model forecast",
    interval_capability="Model-native Gaussian predictive intervals",
    minimum_observations=10,
)
