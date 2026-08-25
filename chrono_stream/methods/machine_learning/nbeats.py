"""Generic N-BEATS direct multi-horizon univariate forecast."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from ...contracts import MethodSpec
from ...intervals import build_output


RANDOM_SEED = 42


def _direct_rows(
    values: np.ndarray,
    lookback: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    if lookback < 2:
        raise ValueError("N-BEATS lookback must be at least 2.")
    sample_count = len(values) - lookback - horizon + 1
    if sample_count < 6:
        raise ValueError(
            "N-BEATS direct training needs at least 6 complete lookback/forecast "
            f"examples; got {max(sample_count, 0)}. Reduce the lookback or horizon, "
            "or provide more history."
        )
    features = np.vstack(
        [values[start : start + lookback] for start in range(sample_count)]
    )
    targets = np.vstack(
        [
            values[start + lookback : start + lookback + horizon]
            for start in range(sample_count)
        ]
    )
    return features, targets


def _network(
    tf: Any,
    *,
    lookback: int,
    horizon: int,
    blocks: int,
    hidden_layers: int,
    hidden_units: int,
):
    inputs = tf.keras.layers.Input(shape=(lookback,), name="backcast_input")
    residual = inputs
    forecast_parts = []
    for block in range(blocks):
        hidden = residual
        for layer in range(hidden_layers):
            hidden = tf.keras.layers.Dense(
                hidden_units,
                activation="relu",
                name=f"block_{block + 1}_dense_{layer + 1}",
            )(hidden)
        backcast = tf.keras.layers.Dense(
            lookback, name=f"block_{block + 1}_backcast"
        )(hidden)
        block_forecast = tf.keras.layers.Dense(
            horizon, name=f"block_{block + 1}_forecast"
        )(hidden)
        residual = tf.keras.layers.Subtract(name=f"block_{block + 1}_residual")(
            [residual, backcast]
        )
        forecast_parts.append(block_forecast)
    output = (
        forecast_parts[0]
        if len(forecast_parts) == 1
        else tf.keras.layers.Add(name="additive_forecast")(forecast_parts)
    )
    return tf.keras.Model(inputs=inputs, outputs=output, name="chrono_stream_nbeats")


def forecast(
    values: np.ndarray,
    steps: int,
    params: dict[str, Any],
    **_: Any,
) -> dict[str, Any]:
    """Train generic N-BEATS blocks and produce one direct horizon vector."""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    try:
        import tensorflow as tf
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError(
            "TensorFlow and scikit-learn are required for N-BEATS."
        ) from exc

    lookback = int(params.get("lookback", 24))
    blocks = int(params.get("blocks", 3))
    hidden_layers = int(params.get("hidden_layers", 2))
    hidden_units = int(params.get("hidden_units", 64))
    epochs = int(params.get("epochs", 50))
    batch_size = int(params.get("batch_size", 16))
    learning_rate = float(params.get("learning_rate", 0.001))
    if not 1 <= blocks <= 12:
        raise ValueError("N-BEATS blocks must be between 1 and 12.")
    if not 1 <= hidden_layers <= 8:
        raise ValueError("N-BEATS hidden layers per block must be between 1 and 8.")
    if hidden_units < 4 or epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError(
            "N-BEATS hidden units, epochs, batch size, and learning rate must be positive."
        )

    scaler = StandardScaler()
    scaled = scaler.fit_transform(values.reshape(-1, 1)).reshape(-1)
    features, targets = _direct_rows(scaled, lookback, steps)

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass
    network = _network(
        tf,
        lookback=lookback,
        horizon=steps,
        blocks=blocks,
        hidden_layers=hidden_layers,
        hidden_units=hidden_units,
    )
    network.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    network.fit(
        features,
        targets,
        epochs=epochs,
        batch_size=min(batch_size, len(features)),
        verbose=0,
        shuffle=False,
    )

    training_predictions = network(features, training=False).numpy()
    fitted = np.full(len(values), np.nan, dtype=float)
    first_step_scaled = training_predictions[:, 0].reshape(-1, 1)
    fitted[lookback : lookback + len(first_step_scaled)] = scaler.inverse_transform(
        first_step_scaled
    ).reshape(-1)
    final_input = scaled[-lookback:].reshape(1, lookback)
    future_scaled = network(final_input, training=False).numpy().reshape(-1, 1)
    future = scaler.inverse_transform(future_scaled).reshape(-1)
    parameter_count = int(network.count_params())
    tf.keras.backend.clear_session()

    return build_output(
        values,
        fitted,
        future,
        details={
            "architecture": "Generic N-BEATS backward/forward residual blocks",
            "basis": "Learned generic backcast and forecast heads",
            "interpretability_note": (
                "This page implements the generic N-BEATS architecture, not the "
                "paper's constrained trend/seasonality basis stacks"
            ),
            "lookback": lookback,
            "direct_horizon": steps,
            "blocks": blocks,
            "hidden_layers_per_block": hidden_layers,
            "hidden_units": hidden_units,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "parameter_count": parameter_count,
            "training_examples": int(len(features)),
            "random_seed": RANDOM_SEED,
            "scaling": "Training-partition StandardScaler",
            "fitted_value_method": "First horizon output from each complete training window",
            "multi_step_strategy": "Direct multi-output horizon",
            "interval_assumptions": (
                "Approximate residual intervals; the network does not estimate a "
                "predictive distribution"
            ),
        },
    )


def render_parameters(data_length: int, seasonal_period: int) -> dict[str, Any]:
    """Render bounded generic N-BEATS controls."""
    import streamlit as st

    maximum = max(2, min(120, data_length - 8))
    lookback = st.slider(
        "Backcast lookback",
        2,
        maximum,
        max(2, min(2 * seasonal_period, maximum)),
    )
    columns = st.columns(3)
    blocks = columns[0].slider("Residual blocks", 1, 6, 3)
    hidden_layers = columns[1].slider("Dense layers per block", 1, 4, 2)
    hidden_units = columns[2].slider("Hidden units", 16, 256, 64, 16)
    training = st.columns(2)
    epochs = training[0].slider("Training epochs", 10, 300, 50, 10)
    learning_rate = training[1].selectbox(
        "Learning rate", [0.0003, 0.001, 0.003], index=1
    )
    st.caption(
        "N-BEATS predicts the complete requested horizon directly. Larger horizons "
        "leave fewer complete training windows and can require substantially more data."
    )
    return {
        "lookback": lookback,
        "blocks": blocks,
        "hidden_layers": hidden_layers,
        "hidden_units": hidden_units,
        "epochs": epochs,
        "batch_size": 16,
        "learning_rate": learning_rate,
    }


SPEC = MethodSpec(
    model_id="nbeats",
    display_name="N-BEATS Neural Forecast",
    icon="🧱",
    navigation_group="Machine Learning",
    description=(
        "A generic fully connected neural architecture with backward and forward "
        "residual links that predicts the complete horizon directly."
    ),
    guidance=(
        "N-BEATS is data- and compute-hungry. Start with small blocks and one short "
        "horizon, then compare against naive and classical methods on the holdout."
    ),
    forecast=forecast,
    render_parameters=render_parameters,
    multi_step_strategy="Direct multi-output horizon",
    interval_capability="Descriptive in-sample residual band; unavailable when degenerate",
    minimum_observations=20,
    random_seed=RANDOM_SEED,
)
