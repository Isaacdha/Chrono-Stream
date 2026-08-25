"""Ridge, Lasso, and Elastic Net forecasting from causal lag features."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...contracts import MethodSpec
from ...features import (
    config_from_params,
    expanding_window_splits,
    feature_details,
    feature_row,
    render_feature_controls,
    supervised_rows,
    validate_regular_dates,
)
from ...intervals import build_output


def _estimator(penalty: str, *, alpha: float, l1_ratio: float):
    from sklearn.linear_model import ElasticNet, Lasso, Ridge

    if penalty == "Ridge":
        return Ridge(alpha=alpha)
    if penalty == "Lasso":
        return Lasso(alpha=alpha, max_iter=20_000, selection="cyclic")
    if penalty == "Elastic Net":
        return ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=20_000,
            selection="cyclic",
        )
    raise ValueError("Penalty must be Ridge, Lasso, or Elastic Net.")


def forecast(
    values: np.ndarray,
    steps: int,
    params: dict[str, Any],
    *,
    dates: Any,
    forecast_dates: Any,
    **_: Any,
) -> dict[str, Any]:
    """Scale lag predictors, estimate one penalized equation, and recurse."""
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    observed, future_dates, _frequency = validate_regular_dates(dates, forecast_dates)
    config = config_from_params(params)
    features, targets, target_indices, names, frequency = supervised_rows(
        values, observed, config, minimum_rows=8
    )
    penalty = str(params.get("penalty", "Ridge"))
    automatic = bool(params.get("automatic", True))
    l1_ratio = float(params.get("l1_ratio", 0.5))
    alpha = float(params.get("alpha", 1.0))
    if alpha <= 0:
        raise ValueError("Regularization strength alpha must be greater than zero.")
    if not 0 < l1_ratio <= 1:
        raise ValueError("Elastic Net L1 ratio must be in (0, 1].")

    pipeline = Pipeline(
        [("scale", StandardScaler()), ("model", _estimator(penalty, alpha=alpha, l1_ratio=l1_ratio))]
    )
    candidates: list[dict[str, Any]] = []
    if automatic:
        alpha_grid = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        grid: dict[str, list[float]] = {"model__alpha": alpha_grid}
        if penalty == "Elastic Net":
            grid["model__l1_ratio"] = [0.2, 0.5, 0.8]
        search = GridSearchCV(
            pipeline,
            grid,
            scoring="neg_root_mean_squared_error",
            cv=expanding_window_splits(len(targets)),
            n_jobs=1,
            refit=True,
            error_score=np.nan,
        )
        search.fit(features, targets)
        if not np.isfinite(search.best_score_):
            raise ValueError("No regularization candidate produced a finite CV score.")
        pipeline = search.best_estimator_
        alpha = float(search.best_params_["model__alpha"])
        if penalty == "Elastic Net":
            l1_ratio = float(search.best_params_["model__l1_ratio"])
        for parameters, score in zip(
            search.cv_results_["params"],
            search.cv_results_["mean_test_score"],
            strict=True,
        ):
            candidates.append(
                {
                    "alpha": float(parameters["model__alpha"]),
                    "l1_ratio": (
                        float(parameters["model__l1_ratio"])
                        if "model__l1_ratio" in parameters
                        else None
                    ),
                    "CV RMSE": float(-score) if np.isfinite(score) else None,
                }
            )
    else:
        pipeline.fit(features, targets)

    fitted = np.full(len(values), np.nan)
    fitted[target_indices] = pipeline.predict(features)
    history = values.astype(float).tolist()
    future: list[float] = []
    for target_date in future_dates:
        row = feature_row(history, target_date, config).reshape(1, -1)
        prediction = float(pipeline.predict(row)[0])
        future.append(prediction)
        history.append(prediction)

    model = pipeline.named_steps["model"]
    coefficient_map = {
        name: float(value) for name, value in zip(names, model.coef_, strict=True)
    }
    details = {
        **feature_details(config, names, frequency),
        "selection": "Automatic expanding-window CV" if automatic else "Manual",
        "penalty": penalty,
        "selected_alpha": alpha,
        "selected_l1_ratio": l1_ratio if penalty == "Elastic Net" else None,
        "scaling": "StandardScaler fitted inside every training fold/pipeline",
        "coefficients_on_standardized_features": coefficient_map,
        "intercept": float(model.intercept_),
        "candidate_results": candidates,
        "top_candidates": sorted(
            [item for item in candidates if item["CV RMSE"] is not None],
            key=lambda item: item["CV RMSE"],
        )[:10],
        "multi_step_strategy": "Recursive",
        "recursive_disclosure": (
            "Each forecast is appended to history, so errors can propagate across horizons."
        ),
    }
    return build_output(values, fitted, future, details=details)


def render_parameters(data_length: int, seasonal_period: int) -> dict[str, Any]:
    """Render regularized lag-regression controls."""
    import streamlit as st

    penalty = st.selectbox("Penalty", ["Ridge", "Lasso", "Elastic Net"])
    automatic = st.toggle(
        "Automatically tune regularization with expanding-window CV", value=True
    )
    parameters: dict[str, Any] = {
        "penalty": penalty,
        "automatic": automatic,
    }
    if not automatic:
        parameters["alpha"] = st.number_input(
            "Regularization strength (alpha)",
            min_value=0.0001,
            max_value=1000.0,
            value=1.0,
            format="%.4f",
        )
    if penalty == "Elastic Net" and not automatic:
        parameters["l1_ratio"] = st.slider(
            "L1 ratio", 0.05, 1.0, 0.5, 0.05
        )
    parameters.update(
        render_feature_controls(
            data_length, seasonal_period, key_prefix="regularized_regression"
        )
    )
    st.caption(
        "Scaling is refitted inside every expanding training fold; validation and the "
        "outer holdout never influence it."
    )
    return parameters


SPEC = MethodSpec(
    model_id="regularized_regression",
    display_name="Regularized Lag Regression",
    icon="🧲",
    navigation_group="Machine Learning",
    description="Ridge, Lasso, or Elastic Net regression on leakage-safe lag and optional known-time features.",
    guidance="Ridge stabilizes correlated lags; Lasso can remove lag coefficients; Elastic Net combines shrinkage and grouped retention.",
    forecast=forecast,
    render_parameters=render_parameters,
    multi_step_strategy="Recursive",
    interval_capability="Descriptive in-sample residual band; unavailable when degenerate",
    minimum_observations=14,
    statistical_test_keys=("rolling_origin_cv",),
)
