"""k-nearest-neighbor regression forecasting from causal lag features."""

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
    """Fit scaled kNN regression and recursively query future lag states."""
    from sklearn.model_selection import GridSearchCV
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    observed, future_dates, _frequency = validate_regular_dates(dates, forecast_dates)
    config = config_from_params(params)
    features, targets, target_indices, names, frequency = supervised_rows(
        values, observed, config, minimum_rows=8
    )
    automatic = bool(params.get("automatic", True))
    neighbors = int(params.get("n_neighbors", 5))
    weights = str(params.get("weights", "distance"))
    distance_power = int(params.get("p", 2))
    if not 1 <= neighbors <= len(targets):
        raise ValueError(f"Neighbor count must be between 1 and {len(targets)}.")
    if weights not in {"uniform", "distance"} or distance_power not in {1, 2}:
        raise ValueError("kNN weights must be uniform/distance and p must be 1 or 2.")
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                KNeighborsRegressor(
                    n_neighbors=neighbors,
                    weights=weights,
                    p=distance_power,
                ),
            ),
        ]
    )
    candidates: list[dict[str, Any]] = []
    if automatic:
        folds = expanding_window_splits(len(targets))
        smallest_training_fold = len(folds[0][0])
        neighbor_grid = [
            value
            for value in [1, 3, 5, 7, 9]
            if value <= smallest_training_fold
        ]
        search = GridSearchCV(
            pipeline,
            {
                "model__n_neighbors": neighbor_grid,
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
            },
            scoring="neg_root_mean_squared_error",
            cv=folds,
            n_jobs=1,
            refit=True,
            error_score=np.nan,
        )
        search.fit(features, targets)
        if not np.isfinite(search.best_score_):
            raise ValueError("No kNN candidate produced a finite CV score.")
        pipeline = search.best_estimator_
        neighbors = int(search.best_params_["model__n_neighbors"])
        weights = str(search.best_params_["model__weights"])
        distance_power = int(search.best_params_["model__p"])
        for settings, score in zip(
            search.cv_results_["params"],
            search.cv_results_["mean_test_score"],
            strict=True,
        ):
            candidates.append(
                {
                    "n_neighbors": int(settings["model__n_neighbors"]),
                    "weights": settings["model__weights"],
                    "p": int(settings["model__p"]),
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
        prediction = float(
            pipeline.predict(feature_row(history, target_date, config).reshape(1, -1))[0]
        )
        future.append(prediction)
        history.append(prediction)

    details = {
        **feature_details(config, names, frequency),
        "selection": "Automatic expanding-window CV" if automatic else "Manual",
        "selected_n_neighbors": neighbors,
        "selected_weights": weights,
        "selected_distance_power": distance_power,
        "distance_metric": "Manhattan" if distance_power == 1 else "Euclidean",
        "scaling": "StandardScaler fitted inside every training fold/pipeline",
        "candidate_results": candidates,
        "top_candidates": sorted(
            [item for item in candidates if item["CV RMSE"] is not None],
            key=lambda item: item["CV RMSE"],
        )[:10],
        "multi_step_strategy": "Recursive",
        "recursive_disclosure": "Forecasts feed later lag rows, so errors can compound.",
        "extrapolation_warning": (
            "kNN averages observed targets near a lag state and does not extrapolate a new trend."
        ),
    }
    return build_output(values, fitted, future, details=details)


def render_parameters(data_length: int, seasonal_period: int) -> dict[str, Any]:
    """Render kNN controls."""
    import streamlit as st

    automatic = st.toggle(
        "Automatically tune neighbors with expanding-window CV", value=True
    )
    parameters: dict[str, Any] = {"automatic": automatic}
    if not automatic:
        columns = st.columns(3)
        parameters["n_neighbors"] = columns[0].slider(
            "Neighbors", 1, max(1, min(30, data_length - 8)), 5
        )
        parameters["weights"] = columns[1].selectbox(
            "Neighbor weights", ["distance", "uniform"]
        )
        metric = columns[2].selectbox("Distance", ["Euclidean", "Manhattan"])
        parameters["p"] = 2 if metric == "Euclidean" else 1
    parameters.update(
        render_feature_controls(data_length, seasonal_period, key_prefix="knn")
    )
    return parameters


SPEC = MethodSpec(
    model_id="knn_regression",
    display_name="k-Nearest-Neighbor Regression",
    icon="🧩",
    navigation_group="Machine Learning",
    description="A local nonparametric forecast formed from historically similar standardized lag states.",
    guidance="kNN is intuitive for recurring regimes, but distance degrades with many features and the method cannot extrapolate unseen levels well.",
    forecast=forecast,
    render_parameters=render_parameters,
    multi_step_strategy="Recursive",
    interval_capability="Descriptive in-sample residual band; unavailable when degenerate",
    minimum_observations=14,
    statistical_test_keys=("rolling_origin_cv",),
)
