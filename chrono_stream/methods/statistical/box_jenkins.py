"""Shared ARIMA/SARIMA Box–Jenkins controls and engine facade.

ARIMA and SARIMA are the deliberate exception to one independent mathematical
implementation per method. They share the same validated search, transformation,
diagnostic, and inverse-transformation pipeline.
"""

from __future__ import annotations

from typing import Any

from ...statistical_tests import (
    ARIMA_DECISION_AID_KEYS,
    ARIMA_TEST_KEYS,
    SARIMA_DECISION_AID_KEYS,
    SARIMA_TEST_KEYS,
)

ARIMA_METHOD_TEST_KEYS = (*ARIMA_TEST_KEYS, *ARIMA_DECISION_AID_KEYS)
SARIMA_METHOD_TEST_KEYS = (*SARIMA_TEST_KEYS, *SARIMA_DECISION_AID_KEYS)


def forecast(values, steps: int, params: dict[str, Any], *, seasonal: bool, **_: Any):
    """Run the shared, lazy-loaded Box–Jenkins pipeline."""
    from .box_jenkins_pipeline import fit_arima_pipeline

    result = fit_arima_pipeline(values, steps, params, seasonal=seasonal)
    details = result.setdefault("details", {})
    details.setdefault("multi_step_strategy", "Direct model forecast")
    details.setdefault("interval_method", "Model-native 95% predictive interval")
    return result


def _bounded_default(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))


def render_parameters(
    model_id: str, data_length: int, seasonal_period: int
) -> dict[str, Any]:
    """Render the shared, auditable ARIMA/SARIMA workflow controls."""
    import streamlit as st

    from .box_jenkins_pipeline import (
        DEFAULT_MAX_AR_MA_ORDER,
        DEFAULT_MAX_REGULAR_DIFFERENCING,
        DEFAULT_MAX_SEASONAL_AR_MA_ORDER,
        DEFAULT_MAX_SEASONAL_DIFFERENCING,
        DEFAULT_MAX_TOTAL_ORDER,
        candidate_search_space_size,
        candidate_search_warning,
    )

    parameters: dict[str, Any] = {}
    seasonal = model_id == "sarima"
    if seasonal:
        maximum_period = max(2, data_length // 2)
        parameters["seasonal_period"] = st.number_input(
            "Seasonal period",
            min_value=2,
            max_value=maximum_period,
            value=_bounded_default(seasonal_period, 2, maximum_period),
            step=1,
            help="Number of observations in one complete seasonal cycle.",
        )
    else:
        parameters["seasonal_period"] = 1

    automatic_label = (
        "Automatically select non-seasonal and seasonal orders"
        if seasonal
        else "Automatically select AR (p), differencing (d), and MA (q)"
    )
    automatic = st.toggle(automatic_label, value=True)
    parameters["automatic"] = automatic
    parameters["diagnostic_policy"] = st.selectbox(
        "Diagnostic policy",
        ["Strict Box-Jenkins", "Forecast-oriented", "Custom"],
        help=(
            "Strict mode requires convergence, valid roots, significant coefficients, "
            "zero-mean normal residuals, and white-noise residuals."
        ),
    )
    allow_unbounded_orders = st.checkbox(
        "Allow unbounded differencing and AR/MA orders",
        value=False,
        help=(
            "Removes the conservative upper limits from d, D, p, q, P, Q, and total "
            "order controls. Large orders can make candidate selection very slow, use "
            "substantial memory, fail to converge, overfit, or over-difference the data."
        ),
    )
    parameters["allow_unbounded_orders"] = allow_unbounded_orders
    regular_difference_maximum = (
        None
        if allow_unbounded_orders
        else DEFAULT_MAX_REGULAR_DIFFERENCING
    )
    seasonal_difference_maximum = (
        None
        if allow_unbounded_orders
        else DEFAULT_MAX_SEASONAL_DIFFERENCING
    )
    ar_ma_maximum = None if allow_unbounded_orders else DEFAULT_MAX_AR_MA_ORDER
    seasonal_ar_ma_maximum = (
        None if allow_unbounded_orders else DEFAULT_MAX_SEASONAL_AR_MA_ORDER
    )
    total_order_maximum = None if allow_unbounded_orders else DEFAULT_MAX_TOTAL_ORDER
    if allow_unbounded_orders:
        st.warning(
            "Unbounded orders are enabled. High differencing or large candidate grids can "
            "be extremely slow and may produce unstable or overfit models."
        )

    with st.expander("1. Variance stabilization", expanded=True):
        stabilize = st.toggle("Stabilize variance before differencing", value=True)
        if stabilize:
            transformation = st.selectbox(
                "Transformation",
                ["Auto", "Box-Cox", "Yeo-Johnson", "Log", "Square root"],
                help=(
                    "Auto estimates Box-Cox for positive series and Yeo-Johnson when "
                    "zero or negative values are present."
                ),
            )
        else:
            transformation = "None"
        parameters["transformation"] = transformation

        if transformation in {"Box-Cox", "Yeo-Johnson"}:
            estimate_lambda = st.toggle(
                "Estimate transformation lambda automatically", value=True
            )
            parameters["transformation_lambda"] = (
                None
                if estimate_lambda
                else st.slider("Transformation lambda", -2.0, 2.0, 1.0, 0.05)
            )
        else:
            parameters["transformation_lambda"] = None

        parameters["allow_transform_shift"] = (
            st.toggle(
                "Allow an automatic positive shift when required",
                value=False,
                help="The fitted shift is stored and removed during inverse transformation.",
            )
            if transformation in {"Box-Cox", "Log", "Square root"}
            else False
        )
        if transformation == "Auto":
            parameters["identity_tolerance"] = st.slider(
                "Keep original data when lambda is this close to 1",
                0.0,
                0.30,
                0.10,
                0.01,
            )
        parameters["bias_adjust"] = st.toggle(
            "Bias-adjust forecasts during inverse transformation", value=True
        )
        if parameters["bias_adjust"] and transformation != "None":
            parameters["inverse_simulations"] = st.number_input(
                "Inverse-transform simulation draws",
                min_value=500,
                max_value=10_000,
                value=2_000,
                step=500,
            )

    with st.expander("2. Mean stationarity and differencing", expanded=True):
        if seasonal:
            seasonal_mode = st.selectbox(
                "Seasonal differencing (D)",
                ["Automatic", "Manual", "Disabled"],
                index=0 if automatic else 1,
            )
            parameters["seasonal_difference_mode"] = seasonal_mode
            if seasonal_mode == "Automatic":
                seasonal_columns = st.columns(2)
                parameters["seasonal_differencing_test"] = seasonal_columns[
                    0
                ].selectbox(
                    "Seasonal stationarity test",
                    ["OCSB", "Canova-Hansen", "Seasonal-lag ACF heuristic"],
                )
                parameters["max_D"] = seasonal_columns[1].number_input(
                    "Maximum seasonal differencing (D)",
                    min_value=0,
                    max_value=seasonal_difference_maximum,
                    value=1,
                    step=1,
                )
            elif seasonal_mode == "Manual":
                parameters["D"] = st.number_input(
                    "Seasonal differencing order (D)",
                    min_value=0,
                    max_value=seasonal_difference_maximum,
                    value=1,
                    step=1,
                )
                parameters["max_D"] = parameters["D"]
            else:
                parameters["D"] = 0
                parameters["max_D"] = 0
        else:
            parameters["seasonal_difference_mode"] = "Disabled"
            parameters["D"] = 0
            parameters["max_D"] = 0

        difference_mode = st.selectbox(
            "Regular differencing (d)",
            ["Automatic", "Manual", "Disabled"],
            index=0 if automatic else 1,
        )
        parameters["difference_mode"] = difference_mode
        differencing_columns = st.columns(2)
        parameters["differencing_test"] = differencing_columns[0].selectbox(
            "Regular stationarity test",
            ["ADF + KPSS consensus", "ADF", "KPSS", "Phillips-Perron"],
            help="Consensus requires both ADF and KPSS to indicate stationarity.",
        )
        if difference_mode == "Automatic":
            parameters["max_d"] = differencing_columns[1].number_input(
                "Maximum regular differencing (d)",
                min_value=0,
                max_value=regular_difference_maximum,
                value=2,
                step=1,
            )
        elif difference_mode == "Manual":
            parameters["d"] = differencing_columns[1].number_input(
                "Regular differencing order (d)",
                min_value=0,
                max_value=regular_difference_maximum,
                value=1,
                step=1,
            )
            parameters["max_d"] = parameters["d"]
        else:
            parameters["d"] = 0
            parameters["max_d"] = 0
        parameters["stationarity_alpha"] = st.slider(
            "Stationarity-test significance level", 0.01, 0.10, 0.05, 0.01
        )

    with st.expander("3. Order candidates", expanded=True):
        if automatic:
            parameters["search_strategy"] = st.selectbox(
                "Candidate search strategy",
                ["Guided ACF/PACF", "Exhaustive grid", "Stepwise"],
            )
            nonseasonal_limits = st.columns(2)
            parameters["max_p"] = nonseasonal_limits[0].number_input(
                "Maximum AR order (p)",
                min_value=0,
                max_value=ar_ma_maximum,
                value=3,
                step=1,
            )
            parameters["max_q"] = nonseasonal_limits[1].number_input(
                "Maximum MA order (q)",
                min_value=0,
                max_value=ar_ma_maximum,
                value=3,
                step=1,
            )
            if seasonal:
                seasonal_limits = st.columns(2)
                parameters["max_P"] = seasonal_limits[0].number_input(
                    "Maximum seasonal AR order (P)",
                    min_value=0,
                    max_value=seasonal_ar_ma_maximum,
                    value=1,
                    step=1,
                )
                parameters["max_Q"] = seasonal_limits[1].number_input(
                    "Maximum seasonal MA order (Q)",
                    min_value=0,
                    max_value=seasonal_ar_ma_maximum,
                    value=1,
                    step=1,
                )
            else:
                parameters["max_P"] = 0
                parameters["max_Q"] = 0
            parameters["max_order"] = st.number_input(
                "Maximum total AR and MA order",
                min_value=1,
                max_value=total_order_maximum,
                value=6,
                step=1,
            )
            parameters["guided_fallback"] = st.toggle(
                "Expand to the full grid if the initial search finds no eligible model",
                value=True,
            )
        else:
            manual_mode = st.selectbox(
                "Manual order input", ["Single order", "Candidate lists"]
            )
            if manual_mode == "Single order":
                parameters["search_strategy"] = "Manual order"
                nonseasonal_order = st.columns(2)
                parameters["p"] = nonseasonal_order[0].number_input(
                    "AR order (p)",
                    min_value=0,
                    max_value=ar_ma_maximum,
                    value=1,
                    step=1,
                )
                parameters["q"] = nonseasonal_order[1].number_input(
                    "MA order (q)",
                    min_value=0,
                    max_value=ar_ma_maximum,
                    value=1,
                    step=1,
                )
                parameters["max_p"] = parameters["p"]
                parameters["max_q"] = parameters["q"]
                if seasonal:
                    seasonal_order = st.columns(2)
                    parameters["P"] = seasonal_order[0].number_input(
                        "Seasonal AR order (P)",
                        min_value=0,
                        max_value=seasonal_ar_ma_maximum,
                        value=1,
                        step=1,
                    )
                    parameters["Q"] = seasonal_order[1].number_input(
                        "Seasonal MA order (Q)",
                        min_value=0,
                        max_value=seasonal_ar_ma_maximum,
                        value=1,
                        step=1,
                    )
                    parameters["max_P"] = parameters["P"]
                    parameters["max_Q"] = parameters["Q"]
                else:
                    parameters["max_P"] = 0
                    parameters["max_Q"] = 0
            else:
                parameters["search_strategy"] = "Manual candidate list"
                manual_nonseasonal = st.columns(2)
                parameters["manual_p_values"] = manual_nonseasonal[0].text_input(
                    "Candidate AR orders (p)", value="0,1,2"
                )
                parameters["manual_q_values"] = manual_nonseasonal[1].text_input(
                    "Candidate MA orders (q)", value="0,1,2"
                )
                if seasonal:
                    manual_seasonal = st.columns(2)
                    parameters["manual_P_values"] = manual_seasonal[0].text_input(
                        "Candidate seasonal AR orders (P)", value="0,1"
                    )
                    parameters["manual_Q_values"] = manual_seasonal[1].text_input(
                        "Candidate seasonal MA orders (Q)", value="0,1"
                    )
                parameters["max_order"] = st.number_input(
                    "Maximum total order for manual candidates",
                    min_value=1,
                    max_value=total_order_maximum,
                    value=6,
                    step=1,
                )
            parameters["guided_fallback"] = False

        try:
            candidate_count = candidate_search_space_size(parameters, seasonal=seasonal)
        except ValueError as exc:
            st.warning(f"Candidate count unavailable: {exc}")
        else:
            candidate_warning = candidate_search_warning(candidate_count)
            if candidate_warning:
                st.warning(candidate_warning)
            else:
                st.caption(
                    f"Configured order space: up to {candidate_count:,} tentative "
                    "candidate models."
                )

        maximum_acf_lag = max(5, min(80, data_length // 2 - 1))
        default_acf_lag = min(
            maximum_acf_lag,
            max(20, 2 * int(parameters["seasonal_period"])) if seasonal else 40,
        )
        selection_options = st.columns(3)
        parameters["acf_lags"] = selection_options[0].number_input(
            "ACF/PACF lags",
            min_value=5,
            max_value=maximum_acf_lag,
            value=default_acf_lag,
            step=1,
        )
        parameters["criterion"] = selection_options[1].selectbox(
            "Eligible-model ranking",
            ["AICc", "AIC", "BIC", "HQIC", "CV RMSE", "CV MAE"],
        )
        parameters["trend"] = selection_options[2].selectbox(
            "Deterministic term", ["Automatic", "None", "Constant", "Linear trend"]
        )
        if parameters["criterion"] in {"CV RMSE", "CV MAE"}:
            validation_columns = st.columns(2)
            parameters["cv_folds"] = validation_columns[0].number_input(
                "Rolling validation folds", 2, 10, 3, 1
            )
            parameters["cv_horizon"] = validation_columns[1].number_input(
                "Periods per validation fold", 1, 12, 1, 1
            )
            st.caption(
                "Rolling validation stays inside the pre-holdout training partition; "
                "the variance transformer is refitted for every fold."
            )

    with st.expander("4. Mandatory model diagnostics", expanded=True):
        diagnostic_columns = st.columns(3)
        parameters["diagnostic_alpha"] = diagnostic_columns[0].slider(
            "Diagnostic significance level", 0.01, 0.10, 0.05, 0.01
        )
        parameters["normality_test"] = diagnostic_columns[1].selectbox(
            "Residual normality test",
            ["Jarque-Bera", "Shapiro-Wilk", "Anderson-Darling", "Lilliefors"],
        )
        parameters["white_noise_test"] = diagnostic_columns[2].selectbox(
            "Residual white-noise test", ["Ljung-Box", "Box-Pierce"]
        )
        parameters["ljung_box_lags"] = st.text_input(
            "White-noise diagnostic lags (optional)",
            value="",
            placeholder="Automatic, or enter values such as 12,24",
        )
        if parameters["diagnostic_policy"] == "Custom":
            requirements = st.columns(3)
            parameters["require_significance"] = requirements[0].toggle(
                "Require significant coefficients", value=True
            )
            parameters["require_normality"] = requirements[1].toggle(
                "Require normal residuals", value=True
            )
            parameters["require_white_noise"] = requirements[2].toggle(
                "Require white-noise residuals", value=True
            )
            extra_requirements = st.columns(3)
            parameters["require_roots"] = extra_requirements[0].toggle(
                "Require stationary/invertible roots", value=True
            )
            parameters["require_residual_mean"] = extra_requirements[1].toggle(
                "Require zero residual mean", value=True
            )
            parameters["require_heteroskedasticity"] = extra_requirements[2].toggle(
                "Require constant residual variance", value=False
            )
            parameters["require_stationarity"] = st.toggle(
                "Require the transformed/differenced series to pass its stationarity tests",
                value=True,
            )
        else:
            parameters["require_heteroskedasticity"] = st.toggle(
                "Also require the residual ARCH variance test to pass", value=False
            )
        parameters["allow_near_match"] = st.toggle(
            "Explicitly allow the closest model when none passes every mandatory gate",
            value=False,
            help=(
                "When disabled, strict selection stops and reports every rejection rather "
                "than silently choosing a failing model."
            ),
        )
        parameters["maximum_iterations"] = st.number_input(
            "Maximum optimizer iterations", 50, 500, 150, 25
        )

    return parameters
