"""Compact 1D convolutional recursive lag forecast."""

from __future__ import annotations

import os
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
    """Scale, train, and recursively forecast with a compact 1D CNN."""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    try:
        import tensorflow as tf
        from sklearn.preprocessing import MinMaxScaler
    except ImportError as exc:
        raise ImportError(
            "TensorFlow and scikit-learn are required for the 1D CNN method."
        ) from exc

    lookback = int(params.get("lookback", 12))
    filters = int(params.get("filters", 32))
    kernel_size = int(params.get("kernel_size", 3))
    epochs = int(params.get("epochs", 50))
    batch_size = int(params.get("batch_size", 16))
    if kernel_size > lookback:
        raise ValueError("CNN kernel size cannot exceed the lookback window.")
    if filters < 1 or kernel_size < 1 or epochs < 1 or batch_size < 1:
        raise ValueError("CNN filters, kernel size, epochs, and batch size must be positive.")

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values.reshape(-1, 1)).reshape(-1)
    features, targets = _supervised_rows(scaled, lookback)
    features_3d = features.reshape(-1, lookback, 1)

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    network = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(lookback, 1)),
            tf.keras.layers.Conv1D(filters, kernel_size, activation="relu"),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    network.compile(optimizer="adam", loss="mse")
    network.fit(
        features_3d,
        targets,
        epochs=epochs,
        batch_size=min(batch_size, len(features_3d)),
        verbose=0,
        shuffle=False,
    )

    fitted_scaled = network(features_3d, training=False).numpy().reshape(-1, 1)
    fitted = np.full(len(values), np.nan)
    fitted[lookback:] = scaler.inverse_transform(fitted_scaled).reshape(-1)
    history = scaled.astype(float).tolist()
    forecast_scaled: list[float] = []
    for _step in range(steps):
        window = np.asarray(history[-lookback:]).reshape(1, lookback, 1)
        prediction = float(network(window, training=False).numpy().reshape(-1)[0])
        forecast_scaled.append(prediction)
        history.append(prediction)
    future = scaler.inverse_transform(
        np.asarray(forecast_scaled).reshape(-1, 1)
    ).reshape(-1)
    tf.keras.backend.clear_session()
    return build_output(
        values,
        fitted,
        future,
        details={
            "lookback": lookback,
            "filters": filters,
            "kernel_size": kernel_size,
            "epochs": epochs,
            "batch_size": batch_size,
            "random_seed": RANDOM_SEED,
            "scaling": "Training-partition MinMax scaling",
            "multi_step_strategy": "Recursive",
        },
    )


def render_parameters(data_length: int, seasonal_period: int) -> dict[str, Any]:
    """Render 1D CNN controls."""
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
        "filters": col1.slider("Filters", 8, 128, 32, 8),
        "kernel_size": col2.slider(
            "Kernel size", 2, min(7, lookback), min(3, lookback), 1
        ),
        "epochs": col3.slider("Training epochs", 10, 300, 50, 10),
        "batch_size": 16,
    }


SPEC = MethodSpec(
    model_id="cnn",
    display_name="1D CNN Neural Network",
    icon="🗃️",
    navigation_group="Machine Learning",
    description="A compact one-dimensional convolutional neural network trained on rolling windows of the series.",
    guidance="The convolution kernel detects local patterns inside each lookback window.",
    forecast=forecast,
    render_parameters=render_parameters,
    multi_step_strategy="Recursive",
    interval_capability="Descriptive in-sample residual band; unavailable when degenerate",
    random_seed=RANDOM_SEED,
)
