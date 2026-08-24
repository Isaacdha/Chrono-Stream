"""Reusable Streamlit components for model pages."""

from __future__ import annotations

import json
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from .arima_pipeline import NoEligibleModelError
from .literature_reviews import METHOD_LITERATURE_REVIEWS
from .method_info import METHOD_INFORMATION, copy_ready_method_note
from .models import MODEL_NAMES, evaluate_and_forecast
from .statistical_tests import (
    STATISTICAL_TESTS,
    copy_ready_test_handbook,
    copy_ready_test_note,
    test_keys_for_model,
)


MODEL_DESCRIPTIONS = {
    "moving_average": "Uses the average of the most recent observations. It is a transparent baseline for locally stable series.",
    "weighted_moving_average": "Weights recent observations more heavily than older observations within a rolling window.",
    "single_exponential_smoothing": "Estimates a changing level without an explicit trend or seasonal component.",
    "double_exponential_smoothing": "Holt's method estimates both a level and a linear trend.",
    "triple_exponential_smoothing": "Holt-Winters estimates level, trend, and a repeating seasonal pattern.",
    "arima": "Combines autoregression, differencing, and moving-average errors for non-seasonal dynamics.",
    "sarima": "Extends ARIMA with autoregressive, differencing, and moving-average terms at a seasonal interval.",
    "x11": "A robust STL decomposition followed by explicit trend and seasonal extrapolation; historically related to, but not an implementation of, Census X-11.",
    "prophet": "An additive model with trend changepoints and calendar seasonalities, implemented by Prophet.",
    "lstm": "A compact recurrent neural network trained on rolling windows of the series.",
    "cnn": "A compact one-dimensional convolutional neural network trained on rolling windows of the series.",
    "xgboost": "Gradient-boosted decision trees trained on lagged observations for recursive multi-step forecasts.",
    "linear": "Projects a least-squares straight-line trend into the future.",
    "quadratic": "Projects a second-degree polynomial trend into the future.",
    "exponential": "Fits a straight line to the logarithm of positive values and uses Duan's smearing correction when returning forecasts to the original scale.",
    "logarithmic": "Fits a trend that changes quickly at first and gradually flattens over time.",
}


MODEL_GUIDANCE = {
    "moving_average": "Good baseline for smooth, level series. It does not model a sustained trend or seasonality.",
    "weighted_moving_average": "Useful when recent observations should influence the forecast more strongly.",
    "single_exponential_smoothing": "Use for a series with a changing level but no clear trend or seasonality.",
    "double_exponential_smoothing": "Use when the series has a trend but no repeating seasonal pattern.",
    "triple_exponential_smoothing": "Use when at least two full seasonal cycles are available.",
    "arima": "The strict workflow transforms variance first, establishes mean stationarity, examines ACF/PACF, and ranks only candidates that pass every mandatory diagnostic.",
    "sarima": "Set the observations per cycle (for example, 12 for monthly data). Seasonal differencing is assessed before regular differencing, and strict selection requires every enabled diagnostic to pass.",
    "x11": "Use this portable STL forecast as a decomposition benchmark. It is intentionally not labeled as official Census X-11/X-13.",
    "prophet": "Best suited to regular calendar data with enough history to estimate seasonal patterns and trend changes.",
    "lstm": "Neural forecasts are slower and less interpretable. Increase epochs only after the basic workflow works well.",
    "cnn": "The convolution kernel detects local patterns inside each lookback window.",
    "xgboost": "A longer lookback can capture seasonality, but it also reduces the number of training examples.",
    "linear": "A useful benchmark when the trend changes by a roughly constant amount per period.",
    "quadratic": "Use cautiously: polynomial forecasts can grow or fall very quickly outside the observed range.",
    "exponential": "Only supports strictly positive values, assumes a roughly constant percentage change, and reports its mean-bias smearing factor.",
    "logarithmic": "Useful for saturating growth where changes become smaller over time.",
}


TEST_DISPLAY_NAMES = {
    "adf": "ADF",
    "kpss": "KPSS",
    "pp": "Phillips–Perron",
    "ocsb": "OCSB",
    "canova_hansen": "Canova–Hansen",
    "acf_lag": "ACF",
    "pacf_lag": "PACF",
    "coefficient_wald": "Coefficients",
    "residual_mean_t": "Residual mean",
    "jarque_bera": "Jarque–Bera",
    "shapiro_wilk": "Shapiro–Wilk",
    "anderson_darling": "Anderson–Darling",
    "lilliefors": "Lilliefors",
    "ljung_box": "Ljung–Box",
    "box_pierce": "Box–Pierce",
    "arch_lm": "ARCH LM",
    "adf_kpss_consensus": "ADF + KPSS",
    "seasonal_acf_rule": "Seasonal ACF",
    "roots": "AR/MA roots",
    "optimizer_convergence": "Convergence",
    "information_criteria": "Information criteria",
    "rolling_origin_cv": "Rolling validation",
    "qq_plot": "Q–Q plot",
}

TEST_GROUPS = (
    ("Seasonality", ("ocsb", "canova_hansen", "seasonal_acf_rule")),
    ("Stationarity", ("adf", "kpss", "pp", "adf_kpss_consensus")),
    ("Identification", ("acf_lag", "pacf_lag", "roots")),
    ("Parameters", ("coefficient_wald", "optimizer_convergence")),
    (
        "Normality",
        (
            "jarque_bera",
            "shapiro_wilk",
            "anderson_darling",
            "lilliefors",
            "qq_plot",
        ),
    ),
    (
        "Residuals",
        ("residual_mean_t", "ljung_box", "box_pierce", "arch_lm"),
    ),
    ("Selection", ("information_criteria", "rolling_origin_cv")),
)

TEST_APP_OUTCOMES = {
    "adf": "Reject H0: treat the current difference as stationary. Fail to reject: try the next regular difference, up to max d.",
    "kpss": "Reject H0: the stationarity check fails and another regular difference may be tried. Fail to reject: treat the current difference as stationary.",
    "pp": "Reject H0: treat the current difference as stationary. Fail to reject: try the next regular difference, up to max d.",
    "ocsb": "Reject the seasonal-unit-root H0: D=0. Fail to reject: use D=1, subject to the configured maximum.",
    "canova_hansen": "Reject stable seasonality: use D=1. Fail to reject: use D=0.",
    "acf_lag": "A flagged lag is added to the guided MA candidate neighborhood; it does not force the final q or Q.",
    "pacf_lag": "A flagged lag is added to the guided AR candidate neighborhood; it does not force the final p or P.",
    "coefficient_wald": "In a mandatory gate, every evaluated nonvariance coefficient must reject H0. Otherwise the candidate is rejected.",
    "residual_mean_t": "Reject H0: the zero-mean residual gate fails. Fail to reject: the gate passes.",
    "jarque_bera": "Reject H0: the normality gate fails. Fail to reject: the gate passes.",
    "shapiro_wilk": "Reject H0: the normality gate fails. Fail to reject: the gate passes.",
    "anderson_darling": "Statistic above the selected critical value: the normality gate fails. Otherwise it passes.",
    "lilliefors": "Reject H0: the normality gate fails. Fail to reject: the gate passes.",
    "ljung_box": "Reject H0 at any selected lag: the white-noise gate fails. Every selected lag must fail to reject H0 for the gate to pass.",
    "box_pierce": "Reject H0 at any selected lag: the white-noise gate fails. Every selected lag must fail to reject H0 for the gate to pass.",
    "arch_lm": "Reject H0: the constant-variance gate fails. Fail to reject: the gate passes. The gate uses LM p; the auxiliary F result is also shown.",
}


def _visible_test_groups(test_keys: tuple[str, ...]) -> list[tuple[str, list[str]]]:
    """Group the tests available on one model page into compact UI sections."""
    available = set(test_keys)
    return [
        (label, [key for key in group_keys if key in available])
        for label, group_keys in TEST_GROUPS
        if any(key in available for key in group_keys)
    ]


def _render_test_decision(key: str) -> None:
    """Render the practical definition and app outcome for one diagnostic."""
    item = STATISTICAL_TESTS[key]
    with st.container(border=True):
        st.markdown(f"##### {TEST_DISPLAY_NAMES[key]}")
        st.write(item.purpose)
        if item.formal:
            st.markdown(f"**H₀** {item.null_hypothesis.removeprefix('H0: ')}")
            st.markdown(f"**H₁** {item.alternative_hypothesis.removeprefix('H1: ')}")
            st.markdown(f"**Statistic** {item.statistic}")
            st.markdown(f"**Decision** {item.decision_rule}")
            st.markdown(f"**Result in Chrono Stream** {TEST_APP_OUTCOMES[key]}")
        else:
            st.caption("Decision aid — no H₀, H₁, or p-value")
            st.markdown(f"**Rule** {item.decision_rule}")


def _render_method_popover(model_id: str) -> None:
    """Render practical method and diagnostic guidance."""
    information = METHOD_INFORMATION[model_id]
    title = MODEL_NAMES[model_id]
    statistical_keys = test_keys_for_model(model_id)

    with st.popover(
        "!",
        help="Method information",
        width="content",
        key=f"{model_id}_method_information",
    ):
        st.markdown(f"### {title}")
        tab_labels = ["Method", "Use"]
        if statistical_keys:
            tab_labels.append("Tests")
        method_tabs = st.tabs(tab_labels)

        with method_tabs[0]:
            st.write(MODEL_DESCRIPTIONS[model_id])
            st.markdown("#### Background")
            st.write(information.origin)
            st.markdown("#### How it works")
            st.write(information.how_it_works)
            st.markdown("#### In Chrono Stream")
            st.write(information.chrono_stream)

        with method_tabs[1]:
            st.markdown("#### Good fit")
            st.write(information.when_to_use)
            st.markdown("#### Watch for")
            st.write(information.limitations)

        if statistical_keys:
            visible_groups = _visible_test_groups(statistical_keys)
            with method_tabs[2]:
                group_tabs = st.tabs([label for label, _ in visible_groups])
                for group_tab, (_label, group_keys) in zip(
                    group_tabs, visible_groups, strict=True
                ):
                    with group_tab:
                        for key in group_keys:
                            _render_test_decision(key)


def _render_scholarly_popover(model_id: str) -> None:
    """Render literature reviews, references, and plain-text downloads."""
    information = METHOD_INFORMATION[model_id]
    title = MODEL_NAMES[model_id]
    statistical_keys = test_keys_for_model(model_id)
    complete_note = copy_ready_method_note(model_id, title)

    with st.popover(
        "?",
        help="Scholarly review and references",
        width="content",
        key=f"{model_id}_scholarly_references",
    ):
        st.markdown(f"### {title}: literature")
        tab_labels = ["Method review"]
        if statistical_keys:
            tab_labels.append("Test reviews")
        tab_labels.extend(["References", "Files"])
        research_tabs = st.tabs(tab_labels)

        with research_tabs[0]:
            st.markdown("#### Overview")
            st.code(information.citation_ready, language=None, wrap_lines=True)
            st.markdown("#### Literature review")
            st.code(
                METHOD_LITERATURE_REVIEWS[model_id], language=None, wrap_lines=True
            )

        next_tab = 1
        if statistical_keys:
            visible_groups = _visible_test_groups(statistical_keys)
            with research_tabs[next_tab]:
                group_tabs = st.tabs([label for label, _ in visible_groups])
                for group_tab, (_label, group_keys) in zip(
                    group_tabs, visible_groups, strict=True
                ):
                    with group_tab:
                        test_tabs = st.tabs(
                            [TEST_DISPLAY_NAMES[key] for key in group_keys]
                        )
                        for test_tab, key in zip(test_tabs, group_keys, strict=True):
                            with test_tab:
                                st.code(
                                    copy_ready_test_note(key),
                                    language=None,
                                    wrap_lines=True,
                                )
            next_tab += 1

        with research_tabs[next_tab]:
            for index, reference in enumerate(information.references, start=1):
                st.markdown(f"**{index}.** {reference.apa}")

        with research_tabs[next_tab + 1]:
            st.download_button(
                "Method literature (TXT)",
                data=complete_note,
                file_name=f"chrono_stream_{model_id}_method_note.txt",
                mime="text/plain",
                icon=":material/download:",
                on_click="ignore",
                width="stretch",
                key=f"{model_id}_method_note_download",
            )
            if statistical_keys:
                handbook = copy_ready_test_handbook(
                    statistical_keys,
                    title=f"{title}: statistical tests and decision handbook",
                )
                st.download_button(
                    "Test literature (TXT)",
                    data=handbook,
                    file_name=f"chrono_stream_{model_id}_statistical_handbook.txt",
                    mime="text/plain",
                    icon=":material/download:",
                    on_click="ignore",
                    width="stretch",
                    key=f"{model_id}_statistical_handbook_download",
                )


def _render_method_header(model_id: str) -> None:
    """Place the page title and two compact reference controls on one row."""
    header_columns = st.columns(
        [12, 2], gap="small", vertical_alignment="center"
    )
    with header_columns[0]:
        st.title(MODEL_NAMES[model_id])
    with header_columns[1]:
        with st.container(
            key="method_reference_actions",
            horizontal=True,
            horizontal_alignment="right",
            vertical_alignment="center",
            gap="small",
            wrap=False,
        ):
            _render_method_popover(model_id)
            _render_scholarly_popover(model_id)


def _bounded_default(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))


def _box_jenkins_parameters(
    model_id: str, data_length: int, seasonal_period: int
) -> dict[str, Any]:
    """Render the shared, auditable ARIMA/SARIMA workflow controls."""
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
                    ["OCSB", "Canova-Hansen", "ACF significance"],
                )
                parameters["max_D"] = seasonal_columns[1].number_input(
                    "Maximum seasonal differencing (D)", 0, 2, 1, 1
                )
            elif seasonal_mode == "Manual":
                parameters["D"] = st.number_input(
                    "Seasonal differencing order (D)", 0, 2, 1, 1
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
                "Maximum regular differencing (d)", 0, 2, 2, 1
            )
        elif difference_mode == "Manual":
            parameters["d"] = differencing_columns[1].number_input(
                "Regular differencing order (d)", 0, 2, 1, 1
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
                "Maximum AR order (p)", 0, 6, 3, 1
            )
            parameters["max_q"] = nonseasonal_limits[1].number_input(
                "Maximum MA order (q)", 0, 6, 3, 1
            )
            if seasonal:
                seasonal_limits = st.columns(2)
                parameters["max_P"] = seasonal_limits[0].number_input(
                    "Maximum seasonal AR order (P)", 0, 3, 1, 1
                )
                parameters["max_Q"] = seasonal_limits[1].number_input(
                    "Maximum seasonal MA order (Q)", 0, 3, 1, 1
                )
            else:
                parameters["max_P"] = 0
                parameters["max_Q"] = 0
            parameters["max_order"] = st.number_input(
                "Maximum total AR and MA order",
                min_value=1,
                max_value=12,
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
                    "AR order (p)", 0, 6, 1, 1
                )
                parameters["q"] = nonseasonal_order[1].number_input(
                    "MA order (q)", 0, 6, 1, 1
                )
                parameters["max_p"] = parameters["p"]
                parameters["max_q"] = parameters["q"]
                if seasonal:
                    seasonal_order = st.columns(2)
                    parameters["P"] = seasonal_order[0].number_input(
                        "Seasonal AR order (P)", 0, 3, 1, 1
                    )
                    parameters["Q"] = seasonal_order[1].number_input(
                        "Seasonal MA order (Q)", 0, 3, 1, 1
                    )
                    parameters["max_P"] = parameters["P"]
                    parameters["max_Q"] = parameters["Q"]
                else:
                    parameters["max_P"] = 0
                    parameters["max_Q"] = 0
                parameters["max_order"] = 12
            else:
                parameters["search_strategy"] = "Manual candidate list"
                manual_nonseasonal = st.columns(2)
                parameters["manual_p_values"] = manual_nonseasonal[0].text_input(
                    "Candidate AR orders (p)", value="0,1,2"
                )
                parameters["manual_q_values"] = manual_nonseasonal[1].text_input(
                    "Candidate MA orders (q)", value="0,1,2"
                )
                parameters["max_p"] = 6
                parameters["max_q"] = 6
                if seasonal:
                    manual_seasonal = st.columns(2)
                    parameters["manual_P_values"] = manual_seasonal[0].text_input(
                        "Candidate seasonal AR orders (P)", value="0,1"
                    )
                    parameters["manual_Q_values"] = manual_seasonal[1].text_input(
                        "Candidate seasonal MA orders (Q)", value="0,1"
                    )
                    parameters["max_P"] = 3
                    parameters["max_Q"] = 3
                else:
                    parameters["max_P"] = 0
                    parameters["max_Q"] = 0
                parameters["max_order"] = st.number_input(
                    "Maximum total order for manual candidates", 1, 12, 6, 1
                )
            parameters["guided_fallback"] = False

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
            "Deterministic term",
            ["Automatic", "None", "Constant", "Linear trend"],
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


def _model_parameters(
    model_id: str, data_length: int, seasonal_period: int
) -> dict[str, Any]:
    """Render controls inside the caller's form and return model parameters."""
    parameters: dict[str, Any] = {}
    max_window = max(2, min(60, data_length // 2))

    if model_id in {"moving_average", "weighted_moving_average"}:
        automatic_window = st.toggle(
            "Automatically find the optimal window size", value=True
        )
        parameters["automatic_window"] = automatic_window
        if automatic_window:
            parameters["max_window"] = st.number_input(
                "Maximum window to search",
                min_value=2,
                max_value=max_window,
                value=min(24, max_window),
                step=1,
            )
            st.caption(
                "The window with the lowest one-step-ahead training RMSE is selected."
            )
        else:
            parameters["window"] = st.slider(
                "Window size",
                min_value=2,
                max_value=max_window,
                value=_bounded_default(seasonal_period, 2, max_window),
            )
        if model_id == "weighted_moving_average":
            parameters["weighting"] = st.selectbox(
                "Weight pattern", ["Linear", "Exponential"]
            )
            if parameters["weighting"] == "Exponential":
                parameters["decay"] = st.slider("Decay", 0.10, 0.99, 0.80, 0.01)

    elif model_id == "single_exponential_smoothing":
        optimize = st.toggle("Estimate smoothing level automatically", value=True)
        parameters["alpha"] = (
            None
            if optimize
            else st.slider("Smoothing level (alpha)", 0.01, 1.0, 0.30, 0.01)
        )

    elif model_id == "double_exponential_smoothing":
        optimize = st.toggle("Automatically find optimal alpha and beta", value=True)
        if optimize:
            parameters["alpha"] = None
            parameters["beta"] = None
        else:
            col1, col2 = st.columns(2)
            parameters["alpha"] = col1.slider(
                "Smoothing level (alpha)", 0.01, 1.0, 0.30, 0.01
            )
            parameters["beta"] = col2.slider(
                "Trend smoothing (beta)", 0.01, 1.0, 0.10, 0.01
            )
        parameters["damped"] = st.toggle("Damp the projected trend", value=False)
        if parameters["damped"] and not optimize:
            parameters["phi"] = st.slider(
                "Damping coefficient (phi)", 0.80, 0.995, 0.98, 0.005
            )

    elif model_id == "triple_exponential_smoothing":
        parameters["seasonal_period"] = st.number_input(
            "Seasonal period",
            min_value=2,
            max_value=max(2, data_length // 2),
            value=_bounded_default(seasonal_period, 2, max(2, data_length // 2)),
            step=1,
        )
        col1, col2 = st.columns(2)
        with col1:
            parameters["trend"] = st.selectbox(
                "Trend",
                ["add", "mul"],
                format_func=lambda value: {"add": "Additive", "mul": "Multiplicative"}[
                    value
                ],
            )
        with col2:
            parameters["seasonal"] = st.selectbox(
                "Seasonality",
                ["add", "mul"],
                format_func=lambda value: {"add": "Additive", "mul": "Multiplicative"}[
                    value
                ],
            )
        optimize = st.toggle(
            "Automatically find optimal alpha, beta, and gamma", value=True
        )
        if optimize:
            parameters["alpha"] = None
            parameters["beta"] = None
            parameters["gamma"] = None
        else:
            smoothing_columns = st.columns(3)
            parameters["alpha"] = smoothing_columns[0].slider(
                "Smoothing level (alpha)", 0.01, 1.0, 0.30, 0.01
            )
            parameters["beta"] = smoothing_columns[1].slider(
                "Trend smoothing (beta)", 0.01, 1.0, 0.10, 0.01
            )
            parameters["gamma"] = smoothing_columns[2].slider(
                "Seasonal smoothing (gamma)", 0.01, 1.0, 0.10, 0.01
            )
        parameters["damped"] = st.toggle("Damp the projected trend", value=False)
        if parameters["damped"] and not optimize:
            parameters["phi"] = st.slider(
                "Damping coefficient (phi)", 0.80, 0.995, 0.98, 0.005
            )

    elif model_id in {"arima", "sarima"}:
        return _box_jenkins_parameters(model_id, data_length, seasonal_period)

    elif model_id == "x11":
        parameters["seasonal_period"] = st.number_input(
            "Seasonal period",
            2,
            max(2, data_length // 2),
            _bounded_default(seasonal_period, 2, max(2, data_length // 2)),
            1,
        )
        parameters["robust"] = st.toggle("Use robust decomposition", value=True)

    elif model_id == "prophet":
        parameters["seasonality_mode"] = st.selectbox(
            "Seasonality mode", ["additive", "multiplicative"]
        )
        parameters["changepoint_prior_scale"] = st.slider(
            "Trend flexibility", 0.001, 0.50, 0.05, 0.001
        )
        col1, col2 = st.columns(2)
        with col1:
            yearly = st.selectbox("Yearly seasonality", ["Auto", "On", "Off"])
        with col2:
            weekly = st.selectbox("Weekly seasonality", ["Auto", "On", "Off"])
        mapping: dict[str, str | bool] = {"Auto": "auto", "On": True, "Off": False}
        parameters["yearly_seasonality"] = mapping[yearly]
        parameters["weekly_seasonality"] = mapping[weekly]

    elif model_id in {"lstm", "cnn", "xgboost"}:
        max_lookback = max(2, min(60, data_length // 3))
        parameters["lookback"] = st.slider(
            "Lookback window",
            2,
            max_lookback,
            _bounded_default(seasonal_period, 2, max_lookback),
        )
        if model_id == "lstm":
            col1, col2 = st.columns(2)
            parameters["units"] = col1.slider("LSTM units", 8, 128, 32, 8)
            parameters["epochs"] = col2.slider("Training epochs", 10, 300, 50, 10)
            parameters["batch_size"] = 16
        elif model_id == "cnn":
            col1, col2, col3 = st.columns(3)
            parameters["filters"] = col1.slider("Filters", 8, 128, 32, 8)
            parameters["kernel_size"] = col2.slider(
                "Kernel size",
                2,
                min(7, parameters["lookback"]),
                min(3, parameters["lookback"]),
                1,
            )
            parameters["epochs"] = col3.slider("Training epochs", 10, 300, 50, 10)
            parameters["batch_size"] = 16
        else:
            col1, col2, col3 = st.columns(3)
            parameters["n_estimators"] = col1.slider("Trees", 50, 1000, 250, 50)
            parameters["max_depth"] = col2.slider("Maximum depth", 1, 10, 3, 1)
            parameters["learning_rate"] = col3.slider(
                "Learning rate", 0.01, 0.30, 0.05, 0.01
            )

    return parameters


def _format_metric(value: float, *, percent: bool = False) -> str:
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value:,.2f}%" if percent else f"{value:,.3f}"


def _forecast_chart(result: dict[str, Any]) -> alt.LayerChart:
    fitted = result["fitted"].copy()
    actual = fitted[["Date", "Actual"]].rename(columns={"Actual": "Value"})
    actual["Series"] = "Actual"
    fitted_line = fitted.dropna(subset=["Fitted"])[["Date", "Fitted"]].rename(
        columns={"Fitted": "Value"}
    )
    fitted_line["Series"] = "Fitted"
    forecast = result["forecast"].copy()
    forecast_line = forecast[["Date", "Forecast"]].rename(columns={"Forecast": "Value"})
    forecast_line["Series"] = "Forecast"
    lines = pd.concat([actual, fitted_line, forecast_line], ignore_index=True)

    color = alt.Color(
        "Series:N",
        scale=alt.Scale(
            domain=["Actual", "Fitted", "Forecast"],
            range=["#8ecae6", "#ffb703", "#fb8500"],
        ),
        legend=alt.Legend(orient="top"),
    )
    line_chart = (
        alt.Chart(lines)
        .mark_line()
        .encode(
            x=alt.X("Date:T", title=result["date_name"]),
            y=alt.Y("Value:Q", title=result["value_name"], scale=alt.Scale(zero=False)),
            color=color,
            tooltip=[
                alt.Tooltip("Date:T"),
                "Series:N",
                alt.Tooltip("Value:Q", format=",.4f"),
            ],
        )
    )
    interval = (
        alt.Chart(forecast)
        .mark_area(opacity=0.18, color="#fb8500")
        .encode(
            x="Date:T",
            y=alt.Y("Lower 95%:Q", scale=alt.Scale(zero=False)),
            y2="Upper 95%:Q",
            tooltip=[
                alt.Tooltip("Date:T"),
                alt.Tooltip("Lower 95%:Q", format=",.4f"),
                alt.Tooltip("Upper 95%:Q", format=",.4f"),
            ],
        )
    )
    return alt.layer(interval, line_chart).properties(height=430).interactive()


def _backtest_chart(result: dict[str, Any]) -> alt.LayerChart:
    backtest = result["backtest"]
    long = backtest.melt(
        id_vars="Date",
        value_vars=["Actual", "Predicted"],
        var_name="Series",
        value_name="Value",
    )
    line = (
        alt.Chart(long)
        .mark_line(point=True)
        .encode(
            x="Date:T",
            y=alt.Y("Value:Q", scale=alt.Scale(zero=False)),
            color=alt.Color("Series:N", scale=alt.Scale(range=["#8ecae6", "#fb8500"])),
            tooltip=[
                alt.Tooltip("Date:T"),
                "Series:N",
                alt.Tooltip("Value:Q", format=",.4f"),
            ],
        )
    )
    interval = (
        alt.Chart(backtest)
        .mark_area(opacity=0.15, color="#fb8500")
        .encode(
            x="Date:T",
            y=alt.Y("Lower 95%:Q", scale=alt.Scale(zero=False)),
            y2="Upper 95%:Q",
        )
    )
    return alt.layer(interval, line).properties(height=330).interactive()


def _result_csv(result: dict[str, Any]) -> bytes:
    history = result["fitted"].copy()
    history["Forecast"] = np.nan
    history["Lower 95%"] = np.nan
    history["Upper 95%"] = np.nan
    history["Phase"] = "history"
    future = result["forecast"].copy()
    future["Actual"] = np.nan
    future["Fitted"] = np.nan
    future["Phase"] = "forecast"
    columns = [
        "Date",
        "Actual",
        "Fitted",
        "Forecast",
        "Lower 95%",
        "Upper 95%",
        "Phase",
    ]
    return (
        pd.concat([history[columns], future[columns]], ignore_index=True)
        .to_csv(index=False)
        .encode("utf-8")
    )


def _order_label(details: dict[str, Any]) -> str | None:
    order = details.get("selected_order")
    if not order:
        return None
    order_text = ",".join(map(str, order))
    seasonal_order = details.get("selected_seasonal_order")
    if seasonal_order:
        seasonal_text = ",".join(map(str, seasonal_order))
        return f"SARIMA({order_text}) × ({seasonal_text})"
    return f"ARIMA({order_text})"


def _correlation_chart(records: list[dict[str, Any]], title: str) -> alt.LayerChart:
    frame = pd.DataFrame(records)
    band = (
        alt.Chart(frame)
        .mark_area(opacity=0.15, color="#8ecae6")
        .encode(
            x=alt.X("lag:Q", title="Lag"),
            y=alt.Y("lower:Q", title="Correlation", scale=alt.Scale(zero=False)),
            y2="upper:Q",
        )
    )
    stems = (
        alt.Chart(frame)
        .mark_rule()
        .encode(
            x="lag:Q",
            y=alt.Y("value:Q", title="Correlation"),
            y2=alt.datum(0),
            color=alt.condition(
                "datum.significant",
                alt.value("#fb8500"),
                alt.value("#6c757d"),
            ),
            tooltip=["lag:Q", alt.Tooltip("value:Q", format=".4f"), "significant:N"],
        )
    )
    points = (
        alt.Chart(frame)
        .mark_point(filled=True, size=35)
        .encode(
            x="lag:Q",
            y="value:Q",
            color=alt.condition(
                "datum.significant",
                alt.value("#fb8500"),
                alt.value("#6c757d"),
            ),
        )
    )
    return alt.layer(band, stems, points).properties(title=title, height=260)


def _diagnostic_status(value: Any) -> str:
    return "Pass" if bool(value) else "Fail"


def _display_box_jenkins_details(
    result: dict[str, Any], final_details: dict[str, Any]
) -> None:
    backtest_details = result.get("backtest_model_details", {})
    final_order = _order_label(final_details)
    backtest_order = _order_label(backtest_details)
    if final_details.get("selection_override"):
        st.warning(
            f"Final forecast order: **{final_order}**. This is an explicit near-match "
            f"override and failed: {final_details.get('selected_failures') or 'one or more gates'}."
        )
    else:
        st.success(
            f"Final forecast order: **{final_order}** passed every mandatory gate."
        )
    if backtest_order:
        st.caption(
            f"Holdout evaluation order: {backtest_order}. The entire transformation and "
            "selection pipeline is fitted only on pre-holdout observations and then rerun "
            "on all observations for the final forecast."
        )

    transformation = final_details.get("transformation", {})
    overview = st.columns(4)
    overview[0].metric("Transformation", transformation.get("applied_method", "None"))
    overview[1].metric("Selected d", final_details.get("selected_d", 0))
    overview[2].metric("Selected D", final_details.get("selected_D", 0))
    overview[3].metric(
        "Eligible candidates",
        f"{final_details.get('eligible_models', 0)}/{final_details.get('models_succeeded', 0)}",
    )

    pipeline_tab, candidates_tab, coefficients_tab, residuals_tab = st.tabs(
        ["Pipeline", "Candidate models", "Coefficients", "Residual diagnostics"]
    )
    with pipeline_tab:
        st.markdown("#### Variance transformation")
        transformation_columns = st.columns(2)
        transformation_columns[0].json(transformation)
        inverse_details = final_details.get("inverse_transformation", {})
        transformation_columns[1].json(inverse_details)

        transformed = final_details.get("transformed_series", [])
        if transformed and len(transformed) == len(result["fitted"]):
            comparison = result["fitted"][["Date", "Actual"]].copy()
            comparison["Transformed"] = transformed
            long = comparison.melt(
                id_vars="Date",
                value_vars=["Actual", "Transformed"],
                var_name="Scale",
                value_name="Value",
            )
            chart = (
                alt.Chart(long)
                .mark_line()
                .encode(
                    x="Date:T",
                    y=alt.Y("Value:Q", scale=alt.Scale(zero=False)),
                    color="Scale:N",
                    tooltip=[
                        "Date:T",
                        "Scale:N",
                        alt.Tooltip("Value:Q", format=",.4f"),
                    ],
                )
                .properties(height=280)
                .interactive()
            )
            st.altair_chart(chart, width="stretch")

        st.markdown("#### Stationarity decisions")
        seasonal_history = final_details.get("seasonal_stationarity_history", [])
        if seasonal_history:
            st.caption(
                "Seasonal differencing is determined before regular differencing."
            )
            st.dataframe(
                pd.DataFrame(seasonal_history), hide_index=True, width="stretch"
            )
        regular_history = final_details.get("regular_stationarity_history", [])
        if regular_history:
            st.dataframe(
                pd.DataFrame(regular_history), hide_index=True, width="stretch"
            )
        if not final_details.get("stationarity_achieved", True):
            st.warning(
                "The selected maximum d was reached before every configured stationarity "
                "test agreed. Review the history or increase the limit carefully."
            )

        order_correlations = final_details.get("order_correlations", {})
        if order_correlations.get("acf") and order_correlations.get("pacf"):
            st.markdown("#### ACF and PACF after transformation and differencing")
            correlation_columns = st.columns(2)
            correlation_columns[0].altair_chart(
                _correlation_chart(order_correlations["acf"], "ACF"),
                width="stretch",
            )
            correlation_columns[1].altair_chart(
                _correlation_chart(order_correlations["pacf"], "PACF"),
                width="stretch",
            )

    with candidates_tab:
        if final_details.get("search_expanded"):
            st.info(
                "The guided or stepwise candidates contained no eligible model, so the "
                "search expanded to the configured full grid."
            )
        candidate_records = final_details.get("candidate_results", [])
        if candidate_records:
            candidate_frame = pd.DataFrame(candidate_records)
            candidate_frame["order"] = candidate_frame["order"].apply(
                lambda value: tuple(value) if isinstance(value, list) else value
            )
            if "seasonal_order" in candidate_frame:
                candidate_frame["seasonal_order"] = candidate_frame[
                    "seasonal_order"
                ].apply(
                    lambda value: tuple(value) if isinstance(value, list) else value
                )
            preferred_columns = [
                "order",
                "seasonal_order",
                "eligible",
                "AICc",
                "AIC",
                "BIC",
                "HQIC",
                "CV RMSE",
                "CV MAE",
                "CV folds completed",
                "CV error",
                "converged",
                "stationarity_passed",
                "stable_and_invertible",
                "coefficients_significant",
                "normal_residuals",
                "white_noise",
                "zero_residual_mean",
                "constant_variance",
                "maximum_coefficient_p",
                "normality_p",
                "minimum_white_noise_p",
                "residual_mean_p",
                "failure_reasons",
                "fit_error",
                "search_phase",
            ]
            visible = [
                column
                for column in preferred_columns
                if column in candidate_frame.columns
            ]
            st.dataframe(
                candidate_frame[visible], hide_index=True, width="stretch", height=430
            )
        st.caption(
            f"Candidates are ranked by {final_details.get('criterion', 'AICc')} only after "
            "the mandatory diagnostic gates are applied."
        )

    with coefficients_tab:
        coefficients = final_details.get("coefficients", [])
        if coefficients:
            coefficient_frame = pd.DataFrame(coefficients)
            st.dataframe(coefficient_frame, hide_index=True, width="stretch")
            required = final_details.get("diagnostic_requirements", {}).get(
                "coefficients", False
            )
            status = final_details.get("diagnostic_outcomes", {}).get(
                "coefficients", False
            )
            st.caption(
                f"Coefficient significance: {_diagnostic_status(status)}"
                + (" (mandatory)." if required else " (advisory).")
                + " The innovation-variance parameter is reported but not significance-gated."
            )
        roots = final_details.get("root_diagnostics", {})
        if roots:
            st.markdown("#### Stationarity and invertibility roots")
            st.json(roots)

    with residuals_tab:
        outcomes = final_details.get("diagnostic_outcomes", {})
        normality = final_details.get("normality", {})
        white_noise = final_details.get("white_noise", {})
        heteroskedasticity = final_details.get("heteroskedasticity", {})
        residual_mean = final_details.get("residual_mean", {})
        diagnostic_columns = st.columns(4)
        diagnostic_columns[0].metric(
            "Normality", _diagnostic_status(outcomes.get("normality"))
        )
        diagnostic_columns[1].metric(
            "White noise", _diagnostic_status(outcomes.get("white_noise"))
        )
        diagnostic_columns[2].metric(
            "Constant variance",
            _diagnostic_status(outcomes.get("heteroskedasticity")),
        )
        diagnostic_columns[3].metric(
            "Zero residual mean", _diagnostic_status(residual_mean.get("passed"))
        )
        diagnostic_rows = [
                {
                    "Diagnostic": normality.get("method", "Normality"),
                    "Statistic": normality.get("statistic"),
                    "p-value": normality.get("p_value"),
                    "Result": _diagnostic_status(normality.get("passed")),
                },
                {
                    "Diagnostic": heteroskedasticity.get("method", "ARCH LM"),
                    "Statistic": heteroskedasticity.get("statistic"),
                    "p-value": heteroskedasticity.get("p_value"),
                    "Result": _diagnostic_status(heteroskedasticity.get("passed")),
                },
                {
                    "Diagnostic": residual_mean.get("method", "Residual mean"),
                    "Statistic": residual_mean.get("statistic"),
                    "p-value": residual_mean.get("p_value"),
                    "Result": _diagnostic_status(residual_mean.get("passed")),
                },
            ]
        if heteroskedasticity.get("f_statistic") is not None:
            diagnostic_rows.append(
                {
                    "Diagnostic": "ARCH auxiliary F",
                    "Statistic": heteroskedasticity.get("f_statistic"),
                    "p-value": heteroskedasticity.get("f_p_value"),
                    "Result": _diagnostic_status(
                        heteroskedasticity.get("f_passed")
                    ),
                }
            )
        summary_table = pd.DataFrame(diagnostic_rows)
        st.dataframe(summary_table, hide_index=True, width="stretch")
        if white_noise.get("lags"):
            st.markdown(
                f"#### {white_noise.get('method', 'Ljung-Box')} white-noise test"
            )
            st.caption(
                f"Model degrees of freedom: {white_noise.get('model_degrees_of_freedom', 0)}. "
                "Every displayed lag must pass when this diagnostic is mandatory."
            )
            st.dataframe(
                pd.DataFrame(white_noise["lags"]), hide_index=True, width="stretch"
            )

        residual_values = final_details.get("residuals", [])
        if residual_values:
            residual_frame = pd.DataFrame(
                {
                    "Observation": np.arange(1, len(residual_values) + 1),
                    "Standardized residual": residual_values,
                }
            )
            residual_columns = st.columns(2)
            residual_line = (
                alt.Chart(residual_frame)
                .mark_line()
                .encode(
                    x="Observation:Q",
                    y=alt.Y("Standardized residual:Q", scale=alt.Scale(zero=False)),
                    tooltip=[
                        "Observation:Q",
                        alt.Tooltip("Standardized residual:Q", format=".4f"),
                    ],
                )
                .properties(title="Standardized residuals", height=260)
            )
            residual_histogram = (
                alt.Chart(residual_frame)
                .mark_bar()
                .encode(
                    x=alt.X("Standardized residual:Q", bin=alt.Bin(maxbins=30)),
                    y="count():Q",
                )
                .properties(title="Residual distribution", height=260)
            )
            residual_columns[0].altair_chart(residual_line, width="stretch")
            residual_columns[1].altair_chart(residual_histogram, width="stretch")

        residual_correlations = final_details.get("residual_correlations", {})
        if residual_correlations.get("acf") and residual_correlations.get("pacf"):
            residual_correlation_columns = st.columns(2)
            residual_correlation_columns[0].altair_chart(
                _correlation_chart(residual_correlations["acf"], "Residual ACF"),
                width="stretch",
            )
            residual_correlation_columns[1].altair_chart(
                _correlation_chart(residual_correlations["pacf"], "Residual PACF"),
                width="stretch",
            )

        qq_records = final_details.get("qq_plot", [])
        if qq_records:
            qq_frame = pd.DataFrame(qq_records)
            lower_bound = float(
                min(qq_frame["theoretical"].min(), qq_frame["sample"].min())
            )
            upper_bound = float(
                max(qq_frame["theoretical"].max(), qq_frame["sample"].max())
            )
            reference = pd.DataFrame(
                {
                    "theoretical": [lower_bound, upper_bound],
                    "sample": [lower_bound, upper_bound],
                }
            )
            qq_chart = (
                alt.Chart(qq_frame)
                .mark_point(opacity=0.65)
                .encode(
                    x=alt.X("theoretical:Q", title="Theoretical normal quantile"),
                    y=alt.Y("sample:Q", title="Observed residual quantile"),
                    tooltip=[
                        alt.Tooltip("theoretical:Q", format=".4f"),
                        alt.Tooltip("sample:Q", format=".4f"),
                    ],
                )
                + alt.Chart(reference)
                .mark_line(color="#fb8500")
                .encode(x="theoretical:Q", y="sample:Q")
            ).properties(title="Residual Q-Q plot", height=320)
            st.altair_chart(qq_chart, width="stretch")

    st.download_button(
        "Download ARIMA diagnostic report (JSON)",
        data=json.dumps(final_details, indent=2, allow_nan=False).encode("utf-8"),
        file_name=f"chrono_stream_{result['model_id']}_diagnostics.json",
        mime="application/json",
        width="stretch",
    )


def _display_tuning_details(result: dict[str, Any]) -> None:
    final_details = result.get("model_details", {})
    backtest_details = result.get("backtest_model_details", {})
    if not final_details:
        return
    if final_details.get("pipeline_version") == 2:
        _display_box_jenkins_details(result, final_details)
        return

    final_order = _order_label(final_details)
    backtest_order = _order_label(backtest_details)
    if final_order:
        st.success(f"Final forecast order: **{final_order}**")
        if backtest_order:
            st.caption(
                f"Holdout evaluation order: {backtest_order}. Automatic selection is rerun using "
                "only the pre-holdout observations, so this can differ from the final order."
            )
    elif "selected_window" in final_details:
        st.success(
            f"Selected window size: **{final_details['selected_window']}** "
            f"({final_details.get('selection', 'Automatic')})"
        )
    elif "smoothing_level" in final_details:
        st.success(
            f"Smoothing coefficients: alpha={final_details['smoothing_level']:.4f}"
            + (
                f", beta={final_details['smoothing_trend']:.4f}"
                if "smoothing_trend" in final_details
                else ""
            )
            + (
                f", gamma={final_details['smoothing_seasonal']:.4f}"
                if "smoothing_seasonal" in final_details
                else ""
            )
        )

    with st.expander("Fitted model details"):
        summary = {
            key: value
            for key, value in final_details.items()
            if key != "top_candidates"
        }
        st.json(summary)
        candidates = final_details.get("top_candidates")
        if candidates:
            st.markdown("**Top candidate models**")
            st.dataframe(pd.DataFrame(candidates), hide_index=True, width="stretch")


def display_model_result(result: dict[str, Any]) -> None:
    st.subheader("Results")
    metrics = result["metrics"]
    columns = st.columns(4)
    columns[0].metric("Holdout MAE", _format_metric(metrics["MAE"]))
    columns[1].metric("Holdout RMSE", _format_metric(metrics["RMSE"]))
    columns[2].metric("Holdout MAPE", _format_metric(metrics["MAPE"], percent=True))
    columns[3].metric("Holdout sMAPE", _format_metric(metrics["sMAPE"], percent=True))
    st.caption(
        f"Metrics use the final {result['holdout']} observed periods as an out-of-sample holdout."
    )
    _display_tuning_details(result)

    chart_tab, forecast_tab, backtest_tab = st.tabs(
        ["Forecast", "Forecast table", "Backtest"]
    )
    with chart_tab:
        st.altair_chart(_forecast_chart(result), width="stretch")
    with forecast_tab:
        st.dataframe(result["forecast"], width="stretch", hide_index=True)
    with backtest_tab:
        st.altair_chart(_backtest_chart(result), width="stretch")
        st.dataframe(result["backtest"], width="stretch", hide_index=True)

    st.download_button(
        "Download model output (CSV)",
        data=_result_csv(result),
        file_name=f"chrono_stream_{result['model_id']}_forecast.csv",
        mime="text/csv",
        width="stretch",
    )


def render_model_page(model_id: str) -> None:
    """Render a complete configuration, evaluation, and forecast page."""
    if model_id not in MODEL_NAMES:
        st.error(f"Unknown model page: {model_id}")
        st.stop()

    _render_method_header(model_id)
    st.write(MODEL_DESCRIPTIONS[model_id])
    st.info(MODEL_GUIDANCE[model_id])

    if "filtered_df" not in st.session_state:
        st.warning(
            "Load and save a time series on the Data Input page before fitting a model."
        )
        st.stop()

    data = st.session_state["filtered_df"]
    horizon = max(1, int(st.session_state.get("forecast_period", 12)))
    holdout = int(
        st.session_state.get("evaluation_period", min(horizon, max(1, len(data) // 5)))
    )
    holdout = max(1, min(holdout, len(data) - 4))
    seasonal_period = int(st.session_state.get("seasonal_period", 12))
    frequency = st.session_state.get("data_frequency")

    summary_columns = st.columns(4)
    summary_columns[0].metric("Observations", f"{len(data):,}")
    summary_columns[1].metric("Forecast horizon", horizon)
    summary_columns[2].metric("Evaluation holdout", holdout)
    summary_columns[3].metric("Frequency", frequency or "Observed spacing")

    with st.form(f"{model_id}_form", border=True):
        st.subheader("Model settings")
        parameters = _model_parameters(model_id, len(data) - holdout, seasonal_period)
        submitted = st.form_submit_button(
            "Fit, evaluate, and forecast", type="primary", width="stretch"
        )

    if submitted:
        with st.spinner(
            f"Training {MODEL_NAMES[model_id]} and running the holdout evaluation..."
        ):
            try:
                result = evaluate_and_forecast(
                    model_id,
                    data,
                    horizon=horizon,
                    holdout=holdout,
                    params=parameters,
                    frequency=frequency,
                )
            except NoEligibleModelError as exc:
                st.error(str(exc))
                details = exc.details
                candidates = details.get("candidate_results", [])
                if candidates:
                    st.markdown("#### Candidate diagnostic rejections")
                    rejection_frame = pd.DataFrame(candidates)
                    preferred = [
                        "order",
                        "seasonal_order",
                        "AICc",
                        "AIC",
                        "BIC",
                        "stationarity_passed",
                        "coefficients_significant",
                        "normal_residuals",
                        "white_noise",
                        "stable_and_invertible",
                        "failure_reasons",
                        "fit_error",
                    ]
                    visible = [
                        column
                        for column in preferred
                        if column in rejection_frame.columns
                    ]
                    st.dataframe(
                        rejection_frame[visible],
                        hide_index=True,
                        width="stretch",
                    )
                st.info(
                    "Change the transformation or search bounds, relax selected gates, "
                    "or deliberately enable the near-match override and submit again."
                )
            except Exception as exc:  # Streamlit must turn model/library failures into actionable UI feedback.
                st.error(f"The model could not be fitted: {exc}")
            else:
                results = dict(st.session_state.get("model_results", {}))
                results[model_id] = result
                st.session_state["model_results"] = results
                st.success(
                    "Model result saved. It is now available on the comparison page."
                )

    saved_result = st.session_state.get("model_results", {}).get(model_id)
    if saved_result:
        display_model_result(saved_result)
        if st.button("Remove this saved result", key=f"remove_{model_id}"):
            results = dict(st.session_state.get("model_results", {}))
            results.pop(model_id, None)
            st.session_state["model_results"] = results
            st.rerun()
