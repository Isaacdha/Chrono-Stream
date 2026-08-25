"""Croston-family forecasting for nonnegative intermittent demand."""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np

from ...contracts import MethodSpec
from ...features import validate_regular_dates
from ...intervals import build_output


VARIANT_LABELS = {
    "croston": "Original Croston",
    "sba": "Syntetos–Boylan Approximation (SBA)",
    "tsb": "Teunter–Syntetos–Babai (TSB)",
}
AUTOMATIC_SMOOTHING_GRID = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50)


def _variant(raw: Any) -> str:
    normalized = str(raw).strip().lower()
    aliases = {
        "croston": "croston",
        "original croston": "croston",
        "sba": "sba",
        "syntetos-boylan": "sba",
        "syntetos–boylan approximation (sba)": "sba",
        "tsb": "tsb",
        "teunter-syntetos-babai": "tsb",
        "teunter–syntetos–babai (tsb)": "tsb",
    }
    if normalized not in aliases:
        raise ValueError("Croston-family variant must be Croston, SBA, or TSB.")
    return aliases[normalized]


def _smoothing(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number in (0, 1].") from exc
    if not np.isfinite(result) or not 0.0 < result <= 1.0:
        raise ValueError(f"{name} must be a number in (0, 1].")
    return result


def _level(
    variant: str,
    demand_size: float,
    interval: float,
    probability: float | None,
    alpha: float,
) -> float:
    if variant == "tsb":
        return float(demand_size * float(probability))
    correction = 1.0 - alpha / 2.0 if variant == "sba" else 1.0
    return float(correction * demand_size / interval)


def _run_filter(
    values: np.ndarray,
    *,
    variant: str,
    alpha: float,
    beta: float,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Return causal one-step fits, terminal forecast, and auditable states."""
    nonzero_positions = np.flatnonzero(values > 0.0)
    if len(nonzero_positions) == 0:
        return (
            np.zeros(len(values), dtype=float),
            0.0,
            {
                "all_zero_series": True,
                "first_nonzero_index": None,
                "demand_size_state": 0.0,
                "interval_state": None,
                "occurrence_probability_state": 0.0 if variant == "tsb" else None,
            },
        )

    first = int(nonzero_positions[0])
    demand_size = float(values[first])
    interval = float(first + 1)
    probability = 1.0 / interval if variant == "tsb" else None
    elapsed = 1
    fitted = np.full(len(values), np.nan, dtype=float)

    for index in range(first + 1, len(values)):
        fitted[index] = _level(
            variant, demand_size, interval, probability, alpha
        )
        occurrence = values[index] > 0.0
        if variant == "tsb":
            probability = float(
                probability + beta * (float(occurrence) - probability)
            )
            if occurrence:
                demand_size = float(
                    demand_size + alpha * (values[index] - demand_size)
                )
        elif occurrence:
            demand_size = float(
                demand_size + alpha * (values[index] - demand_size)
            )
            interval = float(interval + alpha * (elapsed - interval))
            elapsed = 1
        else:
            elapsed += 1

    terminal = _level(variant, demand_size, interval, probability, alpha)
    return (
        fitted,
        terminal,
        {
            "all_zero_series": False,
            "first_nonzero_index": first,
            "initial_interval": float(first + 1),
            "demand_size_state": demand_size,
            "interval_state": interval if variant != "tsb" else None,
            "occurrence_probability_state": probability if variant == "tsb" else None,
            "periods_since_last_nonzero": elapsed if variant != "tsb" else None,
        },
    )


def _candidate_score(
    values: np.ndarray,
    fitted: np.ndarray,
    criterion: str,
) -> float:
    mask = np.isfinite(fitted)
    if mask.sum() < 3:
        return float("inf")
    errors = values[mask] - fitted[mask]
    if criterion == "MAE":
        return float(np.mean(np.abs(errors)))
    return float(np.sqrt(np.mean(errors**2)))


def _select_smoothing(
    values: np.ndarray,
    *,
    variant: str,
    criterion: str,
) -> tuple[float, float, list[dict[str, Any]]]:
    beta_grid = AUTOMATIC_SMOOTHING_GRID if variant == "tsb" else (0.10,)
    candidates: list[dict[str, Any]] = []
    for alpha, beta in product(AUTOMATIC_SMOOTHING_GRID, beta_grid):
        fitted, _terminal, _states = _run_filter(
            values,
            variant=variant,
            alpha=float(alpha),
            beta=float(beta),
        )
        score = _candidate_score(values, fitted, criterion)
        candidates.append(
            {
                "alpha": float(alpha),
                "beta": float(beta) if variant == "tsb" else None,
                criterion: score if np.isfinite(score) else None,
            }
        )
    successful = [item for item in candidates if item[criterion] is not None]
    if not successful:
        raise ValueError(
            "Automatic Croston smoothing needs at least three causal one-step "
            "predictions after the first nonzero demand. Use manual smoothing or "
            "provide more history."
        )
    selected = min(successful, key=lambda item: item[criterion])
    return float(selected["alpha"]), float(selected.get("beta") or 0.10), candidates


def forecast(
    values: np.ndarray,
    steps: int,
    params: dict[str, Any],
    *,
    dates: Any,
    forecast_dates: Any,
    **_: Any,
) -> dict[str, Any]:
    """Fit one Croston-family recursion and forecast demand per period."""
    _observed, _future, frequency = validate_regular_dates(dates, forecast_dates)
    if np.any(values < 0.0):
        raise ValueError(
            "Croston-family methods require nonnegative demand. Negative values "
            "cannot represent demand occurrences."
        )
    variant = _variant(params.get("variant", "sba"))
    automatic = bool(params.get("automatic", True))
    criterion = str(params.get("criterion", "RMSE")).upper()
    if criterion not in {"MAE", "RMSE"}:
        raise ValueError("Croston automatic criterion must be MAE or RMSE.")

    all_zero = not np.any(values > 0.0)
    candidates: list[dict[str, Any]] = []
    if automatic and not all_zero:
        alpha, beta, candidates = _select_smoothing(
            values,
            variant=variant,
            criterion=criterion,
        )
    else:
        alpha = _smoothing(params.get("alpha", 0.10), "Demand-size smoothing alpha")
        beta = (
            _smoothing(params.get("beta", 0.10), "Occurrence smoothing beta")
            if variant == "tsb"
            else 0.10
        )

    fitted, terminal, states = _run_filter(
        values,
        variant=variant,
        alpha=alpha,
        beta=beta,
    )
    point = np.full(steps, terminal, dtype=float)
    sorted_candidates = sorted(
        [item for item in candidates if item[criterion] is not None],
        key=lambda item: item[criterion],
    )
    details = {
        "variant": VARIANT_LABELS[variant],
        "selection": (
            f"Automatic causal one-step {criterion}"
            if automatic and not all_zero
            else "Manual smoothing"
            if not all_zero
            else "Explicit all-zero demand rule"
        ),
        "alpha": alpha,
        "beta": beta if variant == "tsb" else None,
        "criterion": criterion if automatic and not all_zero else None,
        "candidate_results": candidates,
        "top_candidates": sorted_candidates[:10],
        "nonzero_demands": int(np.count_nonzero(values > 0.0)),
        "zero_periods": int(np.count_nonzero(values == 0.0)),
        "zero_demand_definition": "An exact value of 0 denotes no demand in that period",
        "initialization": (
            "First positive demand initializes size; its one-based position initializes "
            "the first inter-demand interval"
        ),
        "all_zero_rule": (
            "An all-zero training series returns a structural zero forecast"
            if all_zero
            else None
        ),
        **states,
        "date_frequency": frequency,
        "multi_step_strategy": "Constant expected demand per future period",
        "interval_assumptions": (
            "Approximate residual intervals clipped at zero; Croston-family recursions "
            "do not provide calibrated predictive intervals here"
        ),
    }
    if all_zero:
        structural_bounds = np.zeros(len(point), dtype=float)
        details.update(
            {
                "interval_method": (
                    "Degenerate structural-zero range (not a probabilistic interval)"
                ),
                "interval_nominal_coverage": None,
                "interval_assumptions": (
                    "The explicit all-zero rule treats future demand as structurally "
                    "zero; no probability coverage is asserted."
                ),
            }
        )
        output = build_output(
            values,
            fitted,
            point,
            structural_bounds,
            structural_bounds,
            details,
        )
    else:
        output = build_output(values, fitted, point, details=details)
    output["lower"] = np.maximum(0.0, output["lower"])
    if not all_zero and output["details"].get("interval_available"):
        output["details"]["interval_method"] = (
            "Descriptive nonnegative in-sample residual band (not calibrated)"
        )
    return output


def render_parameters(_data_length: int, _seasonal_period: int) -> dict[str, Any]:
    """Render Croston-family variant and smoothing controls."""
    import streamlit as st

    selected_label = st.selectbox(
        "Intermittent-demand variant", list(VARIANT_LABELS.values()), index=1
    )
    variant = next(key for key, label in VARIANT_LABELS.items() if label == selected_label)
    automatic = st.toggle(
        "Automatically select smoothing by causal one-step error", value=True
    )
    parameters: dict[str, Any] = {"variant": variant, "automatic": automatic}
    if automatic:
        parameters["criterion"] = st.selectbox(
            "Automatic smoothing criterion", ["RMSE", "MAE"]
        )
        st.caption(
            "The bounded smoothing grid is scored sequentially on the current training "
            "partition only. Zeros remain evaluation periods."
        )
    else:
        columns = st.columns(2 if variant == "tsb" else 1)
        parameters["alpha"] = columns[0].slider(
            "Demand-size smoothing alpha", 0.01, 1.0, 0.10, 0.01
        )
        if variant == "tsb":
            parameters["beta"] = columns[1].slider(
                "Occurrence smoothing beta", 0.01, 1.0, 0.10, 0.01
            )
    st.caption(
        "Exact zeros mean no demand. Negative values are rejected. Forecasts are "
        "expected demand per period, not forecasts of the next nonzero transaction size."
    )
    return parameters


SPEC = MethodSpec(
    model_id="croston_family",
    display_name="Croston Family (Croston / SBA / TSB)",
    icon="📦",
    navigation_group="Statistical",
    description=(
        "Intermittent-demand forecasts that separately smooth positive demand sizes "
        "and demand timing or occurrence probability."
    ),
    guidance=(
        "Use only for regularly spaced, nonnegative demand with genuine zero-demand "
        "periods. Compare SBA/TSB with naive baselines using the zero-safe metrics."
    ),
    forecast=forecast,
    render_parameters=render_parameters,
    multi_step_strategy="Constant expected demand per future period",
    interval_capability="Approximate nonnegative residual-based intervals",
    minimum_observations=8,
)
