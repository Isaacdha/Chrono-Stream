"""Project overview page."""

import pandas as pd
import streamlit as st

from chrono_stream.models import MODEL_NAMES


st.title("⌛ Chrono Stream")
st.subheader("Explore, benchmark, and forecast a univariate time series")
st.write(
    "Chrono Stream turns a CSV or Excel file into a consistent forecasting workflow. "
    "Clean the dates and values once, explore the series, evaluate several methods on "
    "the same holdout window, and compare their future forecasts side by side."
)

steps = st.columns(4)
steps[0].metric("1", "Load data")
steps[1].metric("2", "Explore")
steps[2].metric("3", "Fit models")
steps[3].metric("4", "Compare")

st.divider()

if "filtered_df" in st.session_state:
    data = st.session_state["filtered_df"]
    date_column, value_column = data.columns[:2]
    st.success(
        "A prepared time series is active. You can explore it or open any model page."
    )
    summary = st.columns(4)
    summary[0].metric("Observations", f"{len(data):,}")
    summary[1].metric(
        "From", pd.to_datetime(data.iloc[:, 0]).min().strftime("%Y-%m-%d")
    )
    summary[2].metric("To", pd.to_datetime(data.iloc[:, 0]).max().strftime("%Y-%m-%d"))
    summary[3].metric("Saved models", len(st.session_state.get("model_results", {})))
    st.line_chart(data.set_index(date_column)[value_column], height=300)
else:
    st.info(
        "No data is loaded yet. Open **Data Input & Settings** or use one of the bundled sample datasets."
    )

st.subheader("Included forecasting methods")
groups = {
    "Smoothing": [
        name
        for key, name in MODEL_NAMES.items()
        if key
        in {
            "moving_average",
            "weighted_moving_average",
            "single_exponential_smoothing",
            "double_exponential_smoothing",
            "triple_exponential_smoothing",
        }
    ],
    "Statistical": [MODEL_NAMES[key] for key in ("arima", "sarima", "x11")],
    "Machine learning": [
        MODEL_NAMES[key] for key in ("prophet", "lstm", "cnn", "xgboost")
    ],
    "Trend projection": [
        MODEL_NAMES[key]
        for key in ("linear", "quadratic", "exponential", "logarithmic")
    ],
}
model_table = pd.DataFrame(
    [
        {"Family": family, "Methods": ", ".join(methods)}
        for family, methods in groups.items()
    ]
)
st.dataframe(model_table, hide_index=True, width="stretch")

st.subheader("Built-in literature and statistical guidance")
st.write(
    "Use ! beside a method title for the model explanation, limitations, and test "
    "decisions. Use ? for the literature review, APA references, and TXT downloads. "
    "For ARIMA and SARIMA, the test guide gives H0, H1, the statistic, the decision "
    "rule, and what each outcome means in the model-selection workflow."
)

st.caption(
    "Forecasts are decision-support estimates, not guarantees. Compare holdout errors, inspect assumptions, "
    "and apply domain knowledge before using a result operationally."
)
