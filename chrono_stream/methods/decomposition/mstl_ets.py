"""MSTL decomposition followed by an explicit nonseasonal ETS forecast."""

from __future__ import annotations

from typing import Any
import warnings

import numpy as np
import pandas as pd

from ...contracts import MethodSpec
from ...features import validate_regular_dates
from ...intervals import build_output


def _periods(raw: Any) -> tuple[int, ...]:
    if isinstance(raw, str):
        pieces = [piece.strip() for piece in raw.split(",") if piece.strip()]
    elif np.isscalar(raw):
        pieces = [raw]
    else:
        pieces = list(raw)
    periods: list[int] = []
    try:
        for piece in pieces:
            value = float(piece)
            integer = int(value)
            if not np.isfinite(value) or integer != value:
                raise ValueError
            periods.append(integer)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "MSTL seasonal periods must be comma-separated whole numbers."
        ) from exc
    normalized = tuple(sorted(set(periods)))
    if len(normalized) < 2 or any(period < 2 for period in normalized):
        raise ValueError(
            "MSTL + ETS requires at least two distinct seasonal periods of 2 or more."
        )
    return normalized


def _fit_adjusted(
    adjusted: np.ndarray,
    dates: pd.DatetimeIndex,
    trend: str | None,
    damped: bool,
):
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel

    model = ETSModel(
        pd.Series(adjusted, index=dates),
        error="add",
        trend=trend,
        damped_trend=damped,
        seasonal=None,
        initialization_method="estimated",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.fit(disp=False)


def _structure(trend: str | None, damped: bool) -> str:
    if trend is None:
        return "ETS(A,N,N)"
    return "ETS(A,Ad,N)" if damped else "ETS(A,A,N)"


def _candidate_record(fit: Any, trend: str | None, damped: bool) -> dict[str, Any]:
    return {
        "structure": _structure(trend, damped),
        "AICc": float(fit.aicc) if np.isfinite(fit.aicc) else None,
        "AIC": float(fit.aic) if np.isfinite(fit.aic) else None,
        "BIC": float(fit.bic) if np.isfinite(fit.bic) else None,
        "converged": bool(fit.mle_retvals.get("converged", False)),
        "fit_error": None,
    }


def _seasonal_future(
    seasonal: np.ndarray,
    periods: tuple[int, ...],
    steps: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    matrix = np.asarray(seasonal, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.shape[1] != len(periods):
        raise ValueError("MSTL returned an unexpected number of seasonal components.")
    future = np.zeros(steps, dtype=float)
    records: list[dict[str, Any]] = []
    for column, period in enumerate(periods):
        terminal_cycle = matrix[-period:, column]
        if len(terminal_cycle) != period or not np.isfinite(terminal_cycle).all():
            raise ValueError(
                f"MSTL did not return a complete finite terminal cycle for period {period}."
            )
        repeated = np.resize(terminal_cycle, steps)
        future += repeated
        records.append(
            {
                "period": int(period),
                "extension": "Repeat the final decomposed seasonal cycle",
                "terminal_cycle": terminal_cycle.astype(float).tolist(),
            }
        )
    return future, records


def forecast(
    values: np.ndarray,
    steps: int,
    params: dict[str, Any],
    *,
    dates: Any,
    forecast_dates: Any,
    **_: Any,
) -> dict[str, Any]:
    """Decompose multiple seasons, forecast adjusted data with ETS, and recombine."""
    from statsmodels.tsa.seasonal import MSTL

    observed, _future, frequency = validate_regular_dates(dates, forecast_dates)
    periods = _periods(params.get("seasonal_periods", "7, 365"))
    # statsmodels intentionally drops periods greater than or equal to half the
    # sample length. Requiring one observation beyond two full cycles prevents
    # that silent removal and preserves the declared multi-seasonal contract.
    required = 2 * max(periods) + 1
    if len(values) < required:
        raise ValueError(
            f"MSTL + ETS needs more than two repetitions of its longest period: "
            f"{required} observations for periods {list(periods)}."
        )
    robust = bool(params.get("robust", True))
    iterations = int(params.get("iterations", 2))
    if not 1 <= iterations <= 10:
        raise ValueError("MSTL seasonal-refinement iterations must be between 1 and 10.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        decomposition = MSTL(
            values,
            periods=periods,
            iterate=iterations,
            stl_kwargs={"robust": robust},
        ).fit()
    seasonal_matrix = np.asarray(decomposition.seasonal, dtype=float)
    if seasonal_matrix.ndim == 1:
        seasonal_matrix = seasonal_matrix.reshape(-1, 1)
    seasonal_total = np.sum(seasonal_matrix, axis=1)
    adjusted = values - seasonal_total
    seasonal_forecast, component_records = _seasonal_future(
        seasonal_matrix, periods, steps
    )

    automatic = bool(params.get("automatic_ets", True))
    criterion = str(params.get("criterion", "AICc"))
    if criterion not in {"AICc", "AIC", "BIC"}:
        raise ValueError("MSTL + ETS selection criterion must be AICc, AIC, or BIC.")
    candidates: list[dict[str, Any]] = []
    selected_fit = None
    selected_trend: str | None = None
    selected_damped = False
    structures: list[tuple[str | None, bool]]
    if automatic:
        structures = [(None, False), ("add", False), ("add", True)]
    else:
        trend_label = str(params.get("ets_trend", "Damped additive"))
        manual = {
            "None": (None, False),
            "Additive": ("add", False),
            "Damped additive": ("add", True),
        }
        if trend_label not in manual:
            raise ValueError(
                "MSTL downstream ETS trend must be None, Additive, or Damped additive."
            )
        structures = [manual[trend_label]]

    best_score = np.inf
    for trend, damped in structures:
        try:
            fit = _fit_adjusted(adjusted, observed, trend, damped)
            record = _candidate_record(fit, trend, damped)
        except (ValueError, np.linalg.LinAlgError, OverflowError) as exc:
            candidates.append(
                {
                    "structure": _structure(trend, damped),
                    "AICc": None,
                    "AIC": None,
                    "BIC": None,
                    "converged": False,
                    "fit_error": str(exc),
                }
            )
            continue
        candidates.append(record)
        score = record[criterion]
        if record["converged"] and score is not None and score < best_score:
            best_score = float(score)
            selected_fit = fit
            selected_trend = trend
            selected_damped = damped
    if selected_fit is None:
        raise ValueError(
            "No converged nonseasonal ETS component model had a finite selection "
            "criterion. Review the periods or use a simpler decomposition."
        )

    adjusted_point = np.asarray(selected_fit.forecast(steps), dtype=float)
    prediction = selected_fit.get_prediction(
        start=len(values), end=len(values) + steps - 1
    ).summary_frame(alpha=0.05)
    point = adjusted_point + seasonal_forecast
    lower = np.asarray(prediction["pi_lower"], dtype=float) + seasonal_forecast
    upper = np.asarray(prediction["pi_upper"], dtype=float) + seasonal_forecast
    fitted = np.asarray(selected_fit.fittedvalues, dtype=float) + seasonal_total
    structure = _structure(selected_trend, selected_damped)
    details = {
        "decomposition": "MSTL",
        "seasonal_periods": list(periods),
        "robust": robust,
        "refinement_iterations": iterations,
        "component_forecast_rule": (
            "Forecast the seasonally adjusted series with nonseasonal additive ETS; "
            "repeat each final MSTL seasonal cycle; add the components"
        ),
        "seasonal_component_rules": component_records,
        "selection": "Automatic downstream ETS search" if automatic else "Manual downstream ETS",
        "criterion": criterion if automatic else None,
        "selected_ets_structure": structure,
        "AICc": float(selected_fit.aicc),
        "AIC": float(selected_fit.aic),
        "BIC": float(selected_fit.bic),
        "converged": bool(selected_fit.mle_retvals.get("converged", False)),
        "candidate_results": candidates,
        "top_candidates": sorted(
            [item for item in candidates if item[criterion] is not None],
            key=lambda item: item[criterion],
        ),
        "date_frequency": frequency,
        "multi_step_strategy": "Direct component forecast and additive recombination",
        "interval_method": "Conditional ETS 95% interval with fixed repeated MSTL seasonality",
        "interval_assumptions": (
            "ETS state-space uncertainty for the adjusted component only; decomposition "
            "and repeated seasonal paths are treated as fixed"
        ),
    }
    return build_output(values, fitted, point, lower, upper, details)


def render_parameters(data_length: int, seasonal_period: int) -> dict[str, Any]:
    """Render MSTL periods and the declared downstream ETS rule."""
    import streamlit as st

    short_period = max(2, seasonal_period // 2)
    long_period = max(seasonal_period, short_period + 1)
    default_periods = f"{short_period}, {long_period}"
    periods = st.text_input(
        "MSTL seasonal periods",
        value=default_periods,
        help="Enter at least two comma-separated integer periods, such as 24, 168.",
    )
    columns = st.columns(2)
    robust = columns[0].toggle("Use robust MSTL decomposition", value=True)
    iterations = columns[1].slider("Seasonal refinement iterations", 1, 10, 2)
    automatic = st.toggle(
        "Automatically select the downstream ETS trend", value=True
    )
    parameters: dict[str, Any] = {
        "seasonal_periods": periods,
        "robust": robust,
        "iterations": iterations,
        "automatic_ets": automatic,
    }
    if automatic:
        parameters["criterion"] = st.selectbox(
            "Downstream ETS criterion", ["AICc", "AIC", "BIC"]
        )
    else:
        parameters["ets_trend"] = st.selectbox(
            "Downstream ETS trend", ["None", "Additive", "Damped additive"], index=2
        )
    st.caption(
        "This is a forecast, not decomposition alone: nonseasonal additive ETS forecasts "
        "the adjusted series, and each final MSTL seasonal cycle is repeated and recombined."
    )
    return parameters


SPEC = MethodSpec(
    model_id="mstl_ets",
    display_name="MSTL + ETS Forecast",
    icon="🧩",
    navigation_group="Decomposition & Seasonal Adjustment",
    description=(
        "Multiple Seasonal-Trend decomposition using Loess followed by an explicit "
        "nonseasonal ETS forecast and seasonal recombination."
    ),
    guidance=(
        "Use for regularly spaced data with at least two credible, well-sampled seasonal "
        "cycles. MSTL alone is a decomposition; the named ETS rule produces the forecast."
    ),
    forecast=forecast,
    render_parameters=render_parameters,
    multi_step_strategy="Direct component forecast and additive recombination",
    interval_capability="Conditional ETS intervals with fixed decomposed seasonality",
    minimum_observations=24,
    statistical_test_keys=("optimizer_convergence", "information_criteria"),
)
