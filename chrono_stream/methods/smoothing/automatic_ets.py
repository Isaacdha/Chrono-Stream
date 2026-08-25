"""Automatic innovation-state-space exponential smoothing (ETS)."""

from __future__ import annotations

from itertools import product
from typing import Any
import warnings

import numpy as np
import pandas as pd

from ...contracts import MethodSpec
from ...features import validate_regular_dates
from ...intervals import build_output


def _component(value: Any, *, allow_none: bool) -> str | None:
    normalized = str(value).strip().lower()
    aliases: dict[str, str | None] = {
        "add": "add",
        "additive": "add",
        "mul": "mul",
        "multiplicative": "mul",
    }
    if allow_none:
        aliases.update({"none": None, "n": None, "null": None})
    if normalized not in aliases:
        expected = "none, additive, or multiplicative" if allow_none else "additive or multiplicative"
        raise ValueError(f"ETS component must be {expected}.")
    return aliases[normalized]


def _label(error: str, trend: str | None, damped: bool, seasonal: str | None) -> str:
    symbols = {None: "N", "add": "A", "mul": "M"}
    trend_symbol = symbols[trend] + ("d" if damped and trend else "")
    return f"ETS({symbols[error]},{trend_symbol},{symbols[seasonal]})"


def _fit_candidate(
    values: np.ndarray,
    dates: pd.DatetimeIndex,
    *,
    error: str,
    trend: str | None,
    damped: bool,
    seasonal: str | None,
    period: int,
):
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel

    model = ETSModel(
        pd.Series(values, index=dates),
        error=error,
        trend=trend,
        damped_trend=damped,
        seasonal=seasonal,
        seasonal_periods=period if seasonal is not None else None,
        initialization_method="estimated",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(disp=False)


def _result_record(fit: Any, structure: str) -> dict[str, Any]:
    converged = bool(getattr(fit, "mle_retvals", {}).get("converged", False))
    return {
        "structure": structure,
        "AICc": float(fit.aicc) if np.isfinite(fit.aicc) else None,
        "AIC": float(fit.aic) if np.isfinite(fit.aic) else None,
        "BIC": float(fit.bic) if np.isfinite(fit.bic) else None,
        "converged": converged,
        "fit_error": None,
    }


def _record_selection_status(record: dict[str, Any], criterion: str) -> None:
    """Record why a fitted candidate can or cannot participate in selection."""
    reasons: list[str] = []
    if record.get("fit_error"):
        reasons.append(f"Fit failed: {record['fit_error']}")
    else:
        if not record.get("converged"):
            reasons.append("Optimizer did not converge")
        if record.get(criterion) is None:
            reasons.append(
                f"{criterion} is undefined or non-finite for this sample and parameter count"
            )
    record["selection_eligible"] = not reasons
    record["selection_exclusion_reason"] = "; ".join(reasons) or None


def forecast(
    values: np.ndarray,
    steps: int,
    params: dict[str, Any],
    *,
    dates: Any,
    forecast_dates: Any,
    **_: Any,
) -> dict[str, Any]:
    """Select or fit one ETS error/trend/seasonal state-space structure."""
    observed, _future_dates, frequency = validate_regular_dates(dates, forecast_dates)
    automatic = bool(params.get("automatic", True))
    period = int(params.get("seasonal_period", 12))
    if period < 2:
        raise ValueError("ETS seasonal period must be at least 2.")
    positive = bool(np.all(values > 0))
    criterion = str(params.get("criterion", "AICc"))
    if criterion not in {"AICc", "AIC", "BIC"}:
        raise ValueError("ETS criterion must be AICc, AIC, or BIC.")
    candidates: list[dict[str, Any]] = []
    selected_fit = None
    seasonal_search_available = len(values) >= 2 * period

    if automatic:
        allow_multiplicative = bool(params.get("allow_multiplicative", True)) and positive
        allow_damped = bool(params.get("allow_damped", True))
        errors = ["add", "mul"] if allow_multiplicative else ["add"]
        trends: list[str | None] = [None, "add"] + (["mul"] if allow_multiplicative else [])
        seasonals: list[str | None] = [None]
        if seasonal_search_available:
            seasonals.append("add")
            if allow_multiplicative:
                seasonals.append("mul")
        best_score = np.inf
        for error, trend, seasonal in product(errors, trends, seasonals):
            damping_options = [False, True] if allow_damped and trend else [False]
            for damped in damping_options:
                structure = _label(error, trend, damped, seasonal)
                try:
                    fit = _fit_candidate(
                        values,
                        observed,
                        error=error,
                        trend=trend,
                        damped=damped,
                        seasonal=seasonal,
                        period=period,
                    )
                    record = _result_record(fit, structure)
                except (ValueError, np.linalg.LinAlgError, OverflowError) as exc:
                    failed_record = {
                        "structure": structure,
                        "AICc": None,
                        "AIC": None,
                        "BIC": None,
                        "converged": False,
                        "fit_error": str(exc),
                    }
                    _record_selection_status(failed_record, criterion)
                    candidates.append(failed_record)
                    continue
                _record_selection_status(record, criterion)
                candidates.append(record)
                score = record[criterion]
                if record["selection_eligible"] and score < best_score:
                    best_score = float(score)
                    selected_fit = fit
        if selected_fit is None:
            raise ValueError(
                "No converged ETS candidate had a finite selection criterion. "
                "Review the candidate records or use a simpler manual structure."
            )
        error = str(selected_fit.model.error)
        trend = selected_fit.model.trend
        damped = bool(selected_fit.model.damped_trend)
        seasonal = selected_fit.model.seasonal
    else:
        error = _component(params.get("error", "add"), allow_none=False)
        trend = _component(params.get("trend", "add"), allow_none=True)
        seasonal = _component(params.get("seasonal", "none"), allow_none=True)
        damped = bool(params.get("damped", False))
        if damped and trend is None:
            raise ValueError("A damped ETS trend requires a trend component.")
        if (error == "mul" or trend == "mul" or seasonal == "mul") and not positive:
            raise ValueError("Multiplicative ETS components require strictly positive values.")
        if seasonal is not None and not seasonal_search_available:
            raise ValueError(
                f"Seasonal ETS needs at least two full cycles ({2 * period} observations)."
            )
        selected_fit = _fit_candidate(
            values,
            observed,
            error=error,
            trend=trend,
            damped=damped,
            seasonal=seasonal,
            period=period,
        )
        record = _result_record(selected_fit, _label(error, trend, damped, seasonal))
        _record_selection_status(record, criterion)
        candidates.append(record)

    structure = _label(error, trend, damped, seasonal)
    point = np.asarray(selected_fit.forecast(steps), dtype=float)
    prediction = selected_fit.get_prediction(
        start=len(values), end=len(values) + steps - 1
    ).summary_frame(alpha=0.05)
    parameter_names = list(selected_fit.model.param_names)
    parameter_values = np.asarray(selected_fit.params, dtype=float)
    undefined_criterion_count = sum(
        item.get("fit_error") is None and item.get(criterion) is None
        for item in candidates
    )
    exclusion_count = sum(not item.get("selection_eligible", False) for item in candidates)
    undefined_fraction = (
        undefined_criterion_count / len(candidates) if candidates else 0.0
    )
    details = {
        "selection": "Automatic structure search" if automatic else "Manual",
        "criterion": criterion if automatic else None,
        "selected_structure": structure,
        "error": error,
        "trend": trend,
        "damped_trend": damped,
        "seasonal": seasonal,
        "seasonal_period": period if seasonal is not None else None,
        "seasonal_candidates_included": seasonal_search_available,
        "multiplicative_candidates_included": positive and bool(params.get("allow_multiplicative", True)),
        "AICc": float(selected_fit.aicc),
        "AIC": float(selected_fit.aic),
        "BIC": float(selected_fit.bic),
        "converged": bool(selected_fit.mle_retvals.get("converged", False)),
        "parameters": {
            name: float(value)
            for name, value in zip(parameter_names, parameter_values, strict=True)
        },
        "candidate_results": candidates,
        "top_candidates": sorted(
            [item for item in candidates if item.get("selection_eligible")],
            key=lambda item: item[criterion],
        )[:10]
        if automatic
        else [],
        "candidate_exclusion_summary": {
            "attempted": len(candidates),
            "excluded_from_selection": exclusion_count,
            "undefined_selected_criterion": undefined_criterion_count,
        },
        "candidate_warning": (
            f"{undefined_criterion_count} of {len(candidates)} fitted candidates had "
            f"undefined {criterion} and were excluded; inspect each recorded reason."
            if automatic and undefined_fraction >= 0.25
            else None
        ),
        "date_frequency": frequency,
        "multi_step_strategy": "Direct state-space forecast",
        "interval_assumptions": "Native ETS state-space prediction intervals",
    }
    return build_output(
        values,
        selected_fit.fittedvalues,
        point,
        prediction["pi_lower"],
        prediction["pi_upper"],
        details,
    )


def render_parameters(data_length: int, seasonal_period: int) -> dict[str, Any]:
    """Render automatic/manual ETS controls."""
    import streamlit as st

    automatic = st.toggle("Automatically select the ETS structure", value=True)
    parameters: dict[str, Any] = {"automatic": automatic}
    maximum_period = max(2, data_length // 2)
    parameters["seasonal_period"] = st.number_input(
        "Candidate seasonal period",
        min_value=2,
        max_value=maximum_period,
        value=max(2, min(seasonal_period, maximum_period)),
        step=1,
    )
    if automatic:
        columns = st.columns(3)
        parameters["criterion"] = columns[0].selectbox(
            "Selection criterion", ["AICc", "AIC", "BIC"]
        )
        parameters["allow_multiplicative"] = columns[1].toggle(
            "Allow multiplicative structures", value=True
        )
        parameters["allow_damped"] = columns[2].toggle(
            "Allow damped trends", value=True
        )
        st.caption(
            "All attempted structures and failures are retained in model details. "
            "Seasonal candidates require two complete cycles."
        )
    else:
        columns = st.columns(4)
        parameters["error"] = columns[0].selectbox(
            "Error", ["additive", "multiplicative"]
        )
        parameters["trend"] = columns[1].selectbox(
            "Trend", ["none", "additive", "multiplicative"]
        )
        parameters["seasonal"] = columns[2].selectbox(
            "Seasonality", ["none", "additive", "multiplicative"]
        )
        parameters["damped"] = columns[3].toggle("Damped trend", value=False)
    return parameters


SPEC = MethodSpec(
    model_id="automatic_ets",
    display_name="Automatic ETS",
    icon="✨",
    navigation_group="Smoothing",
    description="Innovation state-space exponential smoothing selected across error, trend, damping, and seasonal structures.",
    guidance="AICc is a relative model-selection criterion, not a goodness-of-fit test; inspect holdout error and convergence as well.",
    forecast=forecast,
    render_parameters=render_parameters,
    multi_step_strategy="Direct state-space forecast",
    interval_capability="Model-native state-space predictive intervals",
    minimum_observations=10,
    statistical_test_keys=("optimizer_convergence", "information_criteria"),
)
