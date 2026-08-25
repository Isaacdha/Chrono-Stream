"""Causal dilated temporal convolutional recursive forecast."""

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
    if lookback < 2 or len(values) <= lookback + 5:
        raise ValueError(
            "TCN lookback must leave at least 6 causal training examples; "
            f"got {len(values)} observations and lookback {lookback}."
        )
    features = np.asarray(
        [values[index - lookback : index] for index in range(lookback, len(values))]
    )
    return features, values[lookback:]


def _network(
    tf: Any,
    *,
    lookback: int,
    filters: int,
    kernel_size: int,
    dilation_levels: int,
    dropout: float,
):
    inputs = tf.keras.layers.Input(shape=(lookback, 1), name="causal_window")
    residual = inputs
    for level in range(dilation_levels):
        dilation = 2**level
        hidden = tf.keras.layers.Conv1D(
            filters,
            kernel_size,
            padding="causal",
            dilation_rate=dilation,
            activation="relu",
            name=f"dilated_causal_{dilation}_a",
        )(residual)
        if dropout > 0.0:
            hidden = tf.keras.layers.SpatialDropout1D(
                dropout, name=f"dropout_{dilation}"
            )(hidden)
        hidden = tf.keras.layers.Conv1D(
            filters,
            kernel_size,
            padding="causal",
            dilation_rate=dilation,
            activation="relu",
            name=f"dilated_causal_{dilation}_b",
        )(hidden)
        shortcut = residual
        if int(shortcut.shape[-1]) != filters:
            shortcut = tf.keras.layers.Conv1D(
                filters, 1, padding="same", name=f"shortcut_{dilation}"
            )(shortcut)
        residual = tf.keras.layers.Activation(
            "relu", name=f"residual_activation_{dilation}"
        )(tf.keras.layers.Add(name=f"residual_add_{dilation}")([shortcut, hidden]))
    last_state = tf.keras.layers.Lambda(
        lambda tensor: tensor[:, -1, :], name="last_causal_state"
    )(residual)
    output = tf.keras.layers.Dense(1, name="next_value")(last_state)
    return tf.keras.Model(inputs=inputs, outputs=output, name="chrono_stream_tcn")


def forecast(
    values: np.ndarray,
    steps: int,
    params: dict[str, Any],
    **_: Any,
) -> dict[str, Any]:
    """Train causal residual convolutions and recursively forecast."""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    try:
        import tensorflow as tf
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError(
            "TensorFlow and scikit-learn are required for the TCN method."
        ) from exc

    lookback = int(params.get("lookback", 24))
    filters = int(params.get("filters", 32))
    kernel_size = int(params.get("kernel_size", 3))
    dilation_levels = int(params.get("dilation_levels", 3))
    dropout = float(params.get("dropout", 0.0))
    epochs = int(params.get("epochs", 50))
    batch_size = int(params.get("batch_size", 16))
    learning_rate = float(params.get("learning_rate", 0.001))
    if kernel_size < 2 or kernel_size > lookback:
        raise ValueError("TCN kernel size must be between 2 and the lookback window.")
    if filters < 1 or not 1 <= dilation_levels <= 8:
        raise ValueError("TCN filters must be positive and dilation levels must be 1–8.")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("TCN dropout must be in [0, 1).")
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("TCN epochs, batch size, and learning rate must be positive.")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(values.reshape(-1, 1)).reshape(-1)
    features, targets = _supervised_rows(scaled, lookback)
    features_3d = features.reshape(-1, lookback, 1)

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass
    network = _network(
        tf,
        lookback=lookback,
        filters=filters,
        kernel_size=kernel_size,
        dilation_levels=dilation_levels,
        dropout=dropout,
    )
    network.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    network.fit(
        features_3d,
        targets,
        epochs=epochs,
        batch_size=min(batch_size, len(features_3d)),
        verbose=0,
        shuffle=False,
    )

    fitted_scaled = network(features_3d, training=False).numpy().reshape(-1, 1)
    fitted = np.full(len(values), np.nan, dtype=float)
    fitted[lookback:] = scaler.inverse_transform(fitted_scaled).reshape(-1)
    history = scaled.astype(float).tolist()
    forecast_scaled: list[float] = []
    for _step in range(steps):
        window = np.asarray(history[-lookback:], dtype=float).reshape(1, lookback, 1)
        prediction = float(network(window, training=False).numpy().reshape(-1)[0])
        forecast_scaled.append(prediction)
        history.append(prediction)
    future = scaler.inverse_transform(
        np.asarray(forecast_scaled).reshape(-1, 1)
    ).reshape(-1)
    receptive_field = 1 + 2 * (kernel_size - 1) * sum(
        2**level for level in range(dilation_levels)
    )
    parameter_count = int(network.count_params())
    tf.keras.backend.clear_session()

    return build_output(
        values,
        fitted,
        future,
        details={
            "architecture": "Residual causal dilated temporal convolutional network",
            "lookback": lookback,
            "filters": filters,
            "kernel_size": kernel_size,
            "dilations": [2**level for level in range(dilation_levels)],
            "theoretical_receptive_field": receptive_field,
            "zero_padding_note": (
                "The receptive field exceeds the lookback, so the oldest convolution "
                "positions include causal zero padding"
                if receptive_field > lookback
                else None
            ),
            "dropout": dropout,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "parameter_count": parameter_count,
            "random_seed": RANDOM_SEED,
            "scaling": "Training-partition StandardScaler",
            "causality": "Every convolution uses causal left padding and past-only windows",
            "multi_step_strategy": "Recursive",
            "interval_assumptions": (
                "Approximate residual intervals; the network does not estimate a "
                "predictive distribution"
            ),
        },
    )


def render_parameters(data_length: int, seasonal_period: int) -> dict[str, Any]:
    """Render bounded TCN architecture and training controls."""
    import streamlit as st

    maximum = max(2, min(120, data_length - 6))
    lookback = st.slider(
        "Causal lookback window",
        2,
        maximum,
        max(2, min(2 * seasonal_period, maximum)),
    )
    architecture = st.columns(4)
    filters = architecture[0].slider("TCN filters", 8, 128, 32, 8)
    kernel_size = architecture[1].slider(
        "TCN kernel size", 2, min(7, lookback), min(3, lookback)
    )
    dilation_levels = architecture[2].slider("Dilation levels", 1, 6, 3)
    dropout = architecture[3].slider("Spatial dropout", 0.0, 0.5, 0.0, 0.05)
    training = st.columns(2)
    epochs = training[0].slider("Training epochs", 10, 300, 50, 10)
    learning_rate = training[1].selectbox(
        "Learning rate", [0.0003, 0.001, 0.003], index=1
    )
    st.caption(
        "Causal padding prevents future leakage; exponentially increasing dilations "
        "expand the receptive field. Multi-step forecasts are recursive."
    )
    return {
        "lookback": lookback,
        "filters": filters,
        "kernel_size": kernel_size,
        "dilation_levels": dilation_levels,
        "dropout": dropout,
        "epochs": epochs,
        "batch_size": 16,
        "learning_rate": learning_rate,
    }


SPEC = MethodSpec(
    model_id="tcn",
    display_name="Temporal Convolutional Network (TCN)",
    icon="🔗",
    navigation_group="Machine Learning",
    description=(
        "A residual one-dimensional network with causal, exponentially dilated "
        "convolutions trained on past-only windows."
    ),
    guidance=(
        "TCNs can model long lag patterns but remain data- and compute-intensive. "
        "Treat dilation depth and lookback as substantive choices, not free accuracy."
    ),
    forecast=forecast,
    render_parameters=render_parameters,
    multi_step_strategy="Recursive",
    interval_capability="Descriptive in-sample residual band; unavailable when degenerate",
    minimum_observations=16,
    random_seed=RANDOM_SEED,
)
