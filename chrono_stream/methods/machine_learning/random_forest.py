"""Random-forest forecasting from causal lag features."""

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
    """Fit a reproducible bootstrap forest and recurse through future lag rows."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV

    observed, future_dates, _frequency = validate_regular_dates(dates, forecast_dates)
    config = config_from_params(params)
    features, targets, target_indices, names, frequency = supervised_rows(
        values, observed, config, minimum_rows=8
    )
    automatic = bool(params.get("automatic", True))
    n_estimators = int(params.get("n_estimators", 250))
    depth_value = params.get("max_depth", None)
    max_depth = None if depth_value in {None, 0, "None"} else int(depth_value)
    min_samples_leaf = int(params.get("min_samples_leaf", 1))
    max_features = float(params.get("max_features", 1.0))
    if n_estimators < 10:
        raise ValueError("Random Forest needs at least 10 trees.")
    if max_depth is not None and max_depth < 1:
        raise ValueError("Maximum depth must be positive or unrestricted.")
    if min_samples_leaf < 1 or not 0 < max_features <= 1:
        raise ValueError("Leaf size must be positive and max-features must be in (0, 1].")
    estimator = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=True,
        random_state=RANDOM_SEED,
        n_jobs=1,
    )
    candidates: list[dict[str, Any]] = []
    if automatic:
        estimator.set_params(n_estimators=100)
        search = GridSearchCV(
            estimator,
            {
                "max_depth": [4, None],
                "min_samples_leaf": [1, 3],
                "max_features": [0.7, 1.0],
            },
            scoring="neg_root_mean_squared_error",
            cv=expanding_window_splits(len(targets)),
            n_jobs=1,
            refit=True,
            error_score=np.nan,
        )
        search.fit(features, targets)
        if not np.isfinite(search.best_score_):
            raise ValueError("No Random Forest candidate produced a finite CV score.")
        estimator = search.best_estimator_
        n_estimators = 100
        max_depth = search.best_params_["max_depth"]
        min_samples_leaf = int(search.best_params_["min_samples_leaf"])
        max_features = float(search.best_params_["max_features"])
        for settings, score in zip(
            search.cv_results_["params"],
            search.cv_results_["mean_test_score"],
            strict=True,
        ):
            candidates.append(
                {
                    "max_depth": settings["max_depth"],
                    "min_samples_leaf": int(settings["min_samples_leaf"]),
                    "max_features": float(settings["max_features"]),
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

    details = {
        **feature_details(config, names, frequency),
        "selection": "Automatic expanding-window CV" if automatic else "Manual",
        "n_estimators": n_estimators,
        "selected_max_depth": max_depth,
        "selected_min_samples_leaf": min_samples_leaf,
        "selected_max_features": max_features,
        "bootstrap": True,
        "feature_importances": {
            name: float(value)
            for name, value in zip(names, estimator.feature_importances_, strict=True)
        },
        "candidate_results": candidates,
        "top_candidates": sorted(
            [item for item in candidates if item["CV RMSE"] is not None],
            key=lambda item: item["CV RMSE"],
        )[:10],
        "random_seed": RANDOM_SEED,
        "multi_step_strategy": "Recursive",
        "recursive_disclosure": "Forecasts feed later lag rows, so errors can compound.",
        "extrapolation_warning": (
            "Tree ensembles average learned leaves and generally extrapolate trends poorly."
        ),
    }
    return build_output(values, fitted, future, details=details)


def render_parameters(data_length: int, seasonal_period: int) -> dict[str, Any]:
    """Render Random Forest controls."""
    import streamlit as st

    automatic = st.toggle(
        "Automatically tune forest complexity with expanding-window CV", value=True
    )
    parameters: dict[str, Any] = {"automatic": automatic}
    if not automatic:
        columns = st.columns(4)
        parameters["n_estimators"] = columns[0].slider("Trees", 50, 1000, 250, 50)
        unrestricted = columns[1].toggle("Unrestricted depth", value=True)
        parameters["max_depth"] = (
            None if unrestricted else columns[1].slider("Maximum depth", 2, 30, 8)
        )
        parameters["min_samples_leaf"] = columns[2].slider(
            "Minimum leaf size", 1, 20, 1
        )
        parameters["max_features"] = columns[3].slider(
            "Feature fraction", 0.2, 1.0, 1.0, 0.1
        )
    parameters.update(
        render_feature_controls(
            data_length, seasonal_period, key_prefix="random_forest"
        )
    )
    return parameters


SPEC = MethodSpec(
    model_id="random_forest",
    display_name="Random Forest Regression",
    icon="🌲",
    navigation_group="Machine Learning",
    description="A seeded bootstrap ensemble of randomized trees trained on causal lag and known-time features.",
    guidance="Forests reduce the variance of one tree but remain weak extrapolators; judge them against seasonal and drift baselines.",
    forecast=forecast,
    render_parameters=render_parameters,
    multi_step_strategy="Recursive",
    interval_capability="Descriptive in-sample residual band; not a native forest interval and unavailable when degenerate",
    minimum_observations=14,
    random_seed=RANDOM_SEED,
    statistical_test_keys=("rolling_origin_cv",),
)
