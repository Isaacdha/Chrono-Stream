"""CART regression-tree forecasting from causal lag features."""

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


RANDOM_SEED = 42


def forecast(
    values: np.ndarray,
    steps: int,
    params: dict[str, Any],
    *,
    dates: Any,
    forecast_dates: Any,
    **_: Any,
) -> dict[str, Any]:
    """Fit one seeded regression tree and forecast recursively."""
    from sklearn.model_selection import GridSearchCV
    from sklearn.tree import DecisionTreeRegressor

    observed, future_dates, _frequency = validate_regular_dates(dates, forecast_dates)
    config = config_from_params(params)
    features, targets, target_indices, names, frequency = supervised_rows(
        values, observed, config, minimum_rows=8
    )
    automatic = bool(params.get("automatic", True))
    max_depth_value = params.get("max_depth", 5)
    max_depth = None if max_depth_value in {None, 0, "None"} else int(max_depth_value)
    min_samples_leaf = int(params.get("min_samples_leaf", 2))
    if max_depth is not None and max_depth < 1:
        raise ValueError("Maximum tree depth must be positive or unrestricted.")
    if min_samples_leaf < 1:
        raise ValueError("Minimum leaf size must be at least 1.")
    estimator = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=RANDOM_SEED,
    )
    candidates: list[dict[str, Any]] = []
    if automatic:
        search = GridSearchCV(
            estimator,
            {
                "max_depth": [2, 4, 8, None],
                "min_samples_leaf": [1, 2, 4],
            },
            scoring="neg_root_mean_squared_error",
            cv=expanding_window_splits(len(targets)),
            n_jobs=1,
            refit=True,
            error_score=np.nan,
        )
        search.fit(features, targets)
        if not np.isfinite(search.best_score_):
            raise ValueError("No CART candidate produced a finite CV score.")
        estimator = search.best_estimator_
        max_depth = search.best_params_["max_depth"]
        min_samples_leaf = int(search.best_params_["min_samples_leaf"])
        for settings, score in zip(
            search.cv_results_["params"],
            search.cv_results_["mean_test_score"],
            strict=True,
        ):
            candidates.append(
                {
                    "max_depth": settings["max_depth"],
                    "min_samples_leaf": int(settings["min_samples_leaf"]),
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
            estimator.predict(feature_row(history, target_date, config).reshape(1, -1))[0]
        )
        future.append(prediction)
        history.append(prediction)

    importances = {
        name: float(value)
        for name, value in zip(names, estimator.feature_importances_, strict=True)
    }
    details = {
        **feature_details(config, names, frequency),
        "selection": "Automatic expanding-window CV" if automatic else "Manual",
        "selected_max_depth": max_depth,
        "selected_min_samples_leaf": min_samples_leaf,
        "tree_depth": int(estimator.get_depth()),
        "leaf_count": int(estimator.get_n_leaves()),
        "feature_importances": importances,
        "candidate_results": candidates,
        "top_candidates": sorted(
            [item for item in candidates if item["CV RMSE"] is not None],
            key=lambda item: item["CV RMSE"],
        )[:10],
        "random_seed": RANDOM_SEED,
        "multi_step_strategy": "Recursive",
        "recursive_disclosure": "Forecasts feed later lag rows, so errors can compound.",
        "extrapolation_warning": (
            "A regression tree is piecewise constant and does not extrapolate a new trend."
        ),
    }
    return build_output(values, fitted, future, details=details)


def render_parameters(data_length: int, seasonal_period: int) -> dict[str, Any]:
    """Render CART controls."""
    import streamlit as st

    automatic = st.toggle(
        "Automatically tune tree complexity with expanding-window CV", value=True
    )
    parameters: dict[str, Any] = {"automatic": automatic}
    if not automatic:
        columns = st.columns(2)
        unrestricted = columns[0].toggle("Unrestricted depth", value=False)
        parameters["max_depth"] = (
            None
            if unrestricted
            else columns[0].slider("Maximum depth", 1, 30, 5)
        )
        parameters["min_samples_leaf"] = columns[1].slider(
            "Minimum samples per leaf", 1, 20, 2
        )
    parameters.update(
        render_feature_controls(data_length, seasonal_period, key_prefix="cart")
    )
    return parameters


SPEC = MethodSpec(
    model_id="cart",
    display_name="CART / Decision Tree Regression",
    icon="🌳",
    navigation_group="Machine Learning",
    description="One interpretable regression tree that partitions leakage-safe lag features into constant forecasts.",
    guidance="Prune depth or enlarge leaves to reduce variance; trees cannot extrapolate a sustained trend beyond learned target regions.",
    forecast=forecast,
    render_parameters=render_parameters,
    multi_step_strategy="Recursive",
    interval_capability="Descriptive in-sample residual band; unavailable when degenerate",
    minimum_observations=14,
    random_seed=RANDOM_SEED,
    statistical_test_keys=("rolling_origin_cv",),
)
