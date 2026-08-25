"""XGBoost recursive lag forecast."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...contracts import MethodSpec
from ...intervals import build_output


RANDOM_SEED = 42


def _supervised_rows(
    values: np.ndarray, lookback: int
) -> tuple[np.ndarray, np.ndarray]:
    if lookback < 2 or len(values) <= lookback + 2:
        raise ValueError(
            f"Lookback must leave at least 3 training samples; got {len(values)} observations."
        )
    features = np.asarray(
        [values[index - lookback : index] for index in range(lookback, len(values))]
    )
    return features, values[lookback:]


def forecast(
    values: np.ndarray, steps: int, params: dict[str, Any], **_: Any
) -> dict[str, Any]:
    """Fit boosted trees to lag rows and forecast recursively."""
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError(
            "XGBoost is not installed. Install the project requirements and try again."
        ) from exc

    lookback = int(params.get("lookback", 12))
    features, targets = _supervised_rows(values, lookback)
    n_estimators = int(params.get("n_estimators", 250))
    max_depth = int(params.get("max_depth", 3))
    learning_rate = float(params.get("learning_rate", 0.05))
    if n_estimators < 1 or max_depth < 1 or learning_rate <= 0:
        raise ValueError("Trees, maximum depth, and learning rate must be positive.")
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        objective="reg:squarederror",
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_SEED,
        n_jobs=1,
    )
    model.fit(features, targets)
    fitted = np.full(len(values), np.nan)
    fitted[lookback:] = model.predict(features)
    history = values.astype(float).tolist()
    forecasts: list[float] = []
    for _step in range(steps):
        prediction = float(
            model.predict(np.asarray(history[-lookback:]).reshape(1, -1))[0]
        )
        forecasts.append(prediction)
        history.append(prediction)
    return build_output(
        values,
        fitted,
        forecasts,
        details={
            "lookback": lookback,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "random_seed": RANDOM_SEED,
            "multi_step_strategy": "Recursive",
        },
    )


def render_parameters(data_length: int, seasonal_period: int) -> dict[str, Any]:
    """Render XGBoost controls."""
    import streamlit as st

    maximum = max(2, min(60, data_length // 3))
    lookback = st.slider(
        "Lookback window",
        2,
        maximum,
        max(2, min(seasonal_period, maximum)),
    )
    col1, col2, col3 = st.columns(3)
    return {
        "lookback": lookback,
        "n_estimators": col1.slider("Trees", 50, 1000, 250, 50),
        "max_depth": col2.slider("Maximum depth", 1, 10, 3, 1),
        "learning_rate": col3.slider(
            "Learning rate", 0.01, 0.30, 0.05, 0.01
        ),
    }


SPEC = MethodSpec(
    model_id="xgboost",
    display_name="XGBoost",
    icon="🔥",
    navigation_group="Machine Learning",
    description="Gradient-boosted decision trees trained on lagged observations for recursive multi-step forecasts.",
    guidance="A longer lookback can capture seasonality, but it also reduces the number of training examples.",
    forecast=forecast,
    render_parameters=render_parameters,
    multi_step_strategy="Recursive",
    interval_capability="Descriptive in-sample residual band; unavailable when degenerate",
    random_seed=RANDOM_SEED,
)
