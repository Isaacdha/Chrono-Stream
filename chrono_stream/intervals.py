"""Shared interval helpers for methods without native predictive intervals."""

from __future__ import annotations

from typing import Any

import numpy as np


def align_fitted(fitted: Any, length: int) -> np.ndarray:
    """Right-align fitted values to the original observation index."""
    array = np.asarray(fitted, dtype=float).reshape(-1)
    if len(array) == length:
        return array
    aligned = np.full(length, np.nan)
    if len(array) < length:
        aligned[-len(array) :] = array
    else:
        aligned[:] = array[-length:]
    return aligned


def residual_intervals(
    values: np.ndarray,
    fitted: np.ndarray,
    forecast: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct a descriptive residual band without claiming nominal coverage.

    In-sample residual spread is not a multi-step forecast-error distribution.  The
    resulting band is therefore deliberately constant across the horizon and is
    presented only as a visual scale reference.  If there are too few residuals or
    their spread is numerically zero, no band is returned.
    """
    mask = np.isfinite(values) & np.isfinite(fitted)
    residuals = values[mask] - fitted[mask]
    unavailable = np.full(len(forecast), np.nan)
    if len(residuals) < 2:
        return unavailable.copy(), unavailable.copy()

    sigma = float(np.std(residuals, ddof=1))
    finite_values = np.asarray(values, dtype=float)[np.isfinite(values)]
    value_scale = max(
        1.0,
        float(np.max(np.abs(finite_values))) if len(finite_values) else 1.0,
    )
    zero_tolerance = 100.0 * np.finfo(float).eps * value_scale
    if not np.isfinite(sigma) or sigma <= zero_tolerance:
        return unavailable.copy(), unavailable.copy()

    margin = np.full(len(forecast), 1.96 * sigma, dtype=float)
    return forecast - margin, forecast + margin


def build_output(
    values: np.ndarray,
    fitted: Any,
    forecast: Any,
    lower: Any | None = None,
    upper: Any | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and validate the result shape shared by every forecast handler."""
    fitted_array = align_fitted(fitted, len(values))
    forecast_array = np.asarray(forecast, dtype=float).reshape(-1)
    if not np.isfinite(forecast_array).all():
        raise ValueError("The model produced non-finite forecast values.")
    if (lower is None) != (upper is None):
        raise ValueError("Forecast lower and upper bounds must be supplied together.")
    if lower is None:
        lower_array, upper_array = residual_intervals(
            values, fitted_array, forecast_array
        )
        interval_available = bool(
            np.isfinite(lower_array).all() and np.isfinite(upper_array).all()
        )
        if interval_available:
            interval_method = (
                "Descriptive in-sample residual band (±1.96 residual standard "
                "deviations; not a calibrated predictive interval)"
            )
            interval_warning = (
                "This band summarizes in-sample residual spread. It has no nominal "
                "coverage guarantee and does not model horizon-specific uncertainty."
            )
        else:
            interval_method = "Unavailable"
            interval_warning = (
                "A residual band is unavailable because fewer than two usable "
                "residuals were present or their in-sample variance was degenerate."
            )
    else:
        lower_array = np.asarray(lower, dtype=float).reshape(-1)
        upper_array = np.asarray(upper, dtype=float).reshape(-1)
        interval_method = "Model-native 95% predictive interval"
        interval_available = True
        interval_warning = None
    if len(lower_array) != len(forecast_array) or len(upper_array) != len(
        forecast_array
    ):
        raise ValueError("Forecast interval arrays must match the forecast horizon.")
    if (lower is not None or upper is not None) and (
        not np.isfinite(lower_array).all() or not np.isfinite(upper_array).all()
    ):
        raise ValueError("The model produced non-finite forecast intervals.")
    if np.isfinite(lower_array).all() and np.any(lower_array > upper_array):
        raise ValueError("Forecast lower bounds cannot exceed upper bounds.")
    result_details = dict(details or {})
    result_details.setdefault("interval_method", interval_method)
    result_details.setdefault("interval_available", interval_available)
    result_details.setdefault(
        "interval_nominal_coverage", 0.95 if lower is not None else None
    )
    if interval_warning is not None:
        result_details.setdefault("interval_warning", interval_warning)
    return {
        "fitted": fitted_array,
        "forecast": forecast_array,
        "lower": lower_array,
        "upper": upper_array,
        "details": result_details,
    }
