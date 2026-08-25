"""Support Vector Regression forecasting from causal lag features."""

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


def forecast(
    values: np.ndarray,
    steps: int,
    params: dict[str, Any],
    *,
    dates: Any,
    forecast_dates: Any,
    **_: Any,
) -> dict[str, Any]:
    """Fit epsilon-insensitive SVR and recursively construct future lag rows."""
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR

    observed, future_dates, _frequency = validate_regular_dates(dates, forecast_dates)
    config = config_from_params(params)
    features, targets, target_indices, names, frequency = supervised_rows(
        values, observed, config, minimum_rows=8
    )
    automatic = bool(params.get("automatic", True))
    kernel = str(params.get("kernel", "rbf")).lower()
    c_value = float(params.get("C", 1.0))
    epsilon = float(params.get("epsilon", 0.1))
    gamma_value: str | float = params.get("gamma", "scale")
    if kernel not in {"linear", "rbf"}:
        raise ValueError("SVR kernel must be linear or RBF.")
    if c_value <= 0 or epsilon < 0:
        raise ValueError("SVR C must be positive and epsilon cannot be negative.")
    if gamma_value not in {"scale", "auto"}:
        gamma_value = float(gamma_value)
        if gamma_value <= 0:
            raise ValueError("Numeric SVR gamma must be positive.")
    estimator = TransformedTargetRegressor(
        regressor=Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    SVR(
                        kernel=kernel,
                        C=c_value,
                        epsilon=epsilon,
                        gamma=gamma_value,
                    ),
                ),
            ]
        ),
        transformer=StandardScaler(),
    )
    candidates: list[dict[str, Any]] = []
    if automatic:
        search = GridSearchCV(
            estimator,
            [
                {
                    "regressor__model__kernel": ["linear"],
                    "regressor__model__C": [0.5, 2.0, 10.0],
                    "regressor__model__epsilon": [0.05, 0.2],
                },
                {
                    "regressor__model__kernel": ["rbf"],
                    "regressor__model__C": [0.5, 2.0, 10.0],
                    "regressor__model__epsilon": [0.05, 0.2],
                    "regressor__model__gamma": ["scale", 0.1],
                },
            ],
            scoring="neg_root_mean_squared_error",
            cv=expanding_window_splits(len(targets)),
            n_jobs=1,
            refit=True,
            error_score=np.nan,
        )
        search.fit(features, targets)
        if not np.isfinite(search.best_score_):
            raise ValueError("No SVR candidate produced a finite CV score.")
        estimator = search.best_estimator_
        kernel = str(search.best_params_["regressor__model__kernel"])
        c_value = float(search.best_params_["regressor__model__C"])
        epsilon = float(search.best_params_["regressor__model__epsilon"])
        gamma_value = search.best_params_.get(
            "regressor__model__gamma", "scale"
        )
        for settings, score in zip(
            search.cv_results_["params"],
            search.cv_results_["mean_test_score"],
            strict=True,
        ):
            candidates.append(
                {
                    "kernel": settings["regressor__model__kernel"],
                    "C": float(settings["regressor__model__C"]),
                    "epsilon": float(settings["regressor__model__epsilon"]),
                    "gamma": settings.get("regressor__model__gamma"),
                    "CV RMSE": float(-score) if np.isfinite(score) else None,
                }
            )
    else:
        estimator.fit(features, targets)

    fitted = np.full(len(values), np.nan)
    fitted[target_indices] = estimator.predict(features)
    history = values.astype(float).tolist()
    future: list[float] = []
    for target_date in future_dates:
        prediction = float(
            estimator.predict(
                feature_row(history, target_date, config).reshape(1, -1)
            )[0]
        )
        future.append(prediction)
        history.append(prediction)

    fitted_pipeline = estimator.regressor_
    model = fitted_pipeline.named_steps["model"]
    target_scaler = estimator.transformer_
    details = {
        **feature_details(config, names, frequency),
        "selection": "Automatic expanding-window CV" if automatic else "Manual",
        "selected_kernel": kernel,
        "selected_C": c_value,
        "selected_epsilon": epsilon,
        "selected_gamma": gamma_value,
        "support_vector_count": int(len(model.support_)),
        "scaling": (
            "Separate feature and target StandardScalers fitted inside every "
            "training fold/pipeline; predictions are inverse-transformed"
        ),
        "epsilon_units": "Training-target standard deviations",
        "target_scaler_mean": float(target_scaler.mean_[0]),
        "target_scaler_scale": float(target_scaler.scale_[0]),
        "candidate_results": candidates,
        "top_candidates": sorted(
            [item for item in candidates if item["CV RMSE"] is not None],
            key=lambda item: item["CV RMSE"],
        )[:10],
        "multi_step_strategy": "Recursive",
        "recursive_disclosure": "Forecasts feed later lag rows, so errors can compound.",
    }
    return build_output(values, fitted, future, details=details)


def render_parameters(data_length: int, seasonal_period: int) -> dict[str, Any]:
    """Render SVR controls."""
    import streamlit as st

    automatic = st.toggle(
        "Automatically tune SVR with expanding-window CV", value=True
    )
    parameters: dict[str, Any] = {"automatic": automatic}
    if not automatic:
        columns = st.columns(4)
        parameters["kernel"] = columns[0].selectbox("Kernel", ["rbf", "linear"])
        parameters["C"] = columns[1].number_input(
            "C", min_value=0.01, max_value=1000.0, value=1.0,
            help="Penalty applied after the target is standardized.",
        )
        parameters["epsilon"] = columns[2].number_input(
            "Epsilon", min_value=0.0, max_value=10.0, value=0.1,
            help="Width of the insensitive tube in target-standard-deviation units.",
        )
        parameters["gamma"] = columns[3].selectbox("Gamma", ["scale", "auto"])
    parameters.update(
        render_feature_controls(
            data_length, seasonal_period, key_prefix="support_vector"
        )
    )
    return parameters


SPEC = MethodSpec(
    model_id="support_vector_regression",
    display_name="Support Vector Regression",
    icon="🛡️",
    navigation_group="Machine Learning",
    description="Epsilon-insensitive linear or radial-basis regression with standardized causal features and target.",
    guidance="SVR can model smooth nonlinear lag relationships, but C, epsilon, kernel scale, and feature scaling strongly affect it.",
    forecast=forecast,
    render_parameters=render_parameters,
    multi_step_strategy="Recursive",
    interval_capability="Descriptive in-sample residual band; unavailable when degenerate",
    minimum_observations=14,
    statistical_test_keys=("rolling_origin_cv",),
)
