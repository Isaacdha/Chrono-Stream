"""Exploratory analysis for the prepared series."""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, adfuller

from chrono_stream.statistical_tests import STATISTICAL_TESTS, copy_ready_test_note


st.title("🔍 Data Exploration")
if "filtered_df" not in st.session_state:
    st.warning("Load and save a time series on the Data Input page first.")
    st.stop()

data = st.session_state["filtered_df"].copy()
date_name, value_name = data.columns[:2]
data[date_name] = pd.to_datetime(data[date_name])
values = data[value_name].astype(float)
plot_data = pd.DataFrame({"Date": data[date_name], "Value": values})

summary = st.columns(5)
summary[0].metric("Observations", f"{len(data):,}")
summary[1].metric("Mean", f"{values.mean():,.3f}")
summary[2].metric("Std. deviation", f"{values.std():,.3f}")
summary[3].metric("Minimum", f"{values.min():,.3f}")
summary[4].metric("Maximum", f"{values.max():,.3f}")

chart = (
    alt.Chart(plot_data)
    .mark_line(color="#8ecae6")
    .encode(
        x=alt.X("Date:T", title=str(date_name)),
        y=alt.Y("Value:Q", title=str(value_name), scale=alt.Scale(zero=False)),
        tooltip=[
            alt.Tooltip("Date:T", title=str(date_name)),
            alt.Tooltip("Value:Q", title=str(value_name), format=",.4f"),
        ],
    )
    .properties(height=380)
    .interactive()
)
st.altair_chart(chart, width="stretch")

tab1, tab2, tab3 = st.tabs(
    ["Distribution", "Seasonal decomposition", "Autocorrelation & stationarity"]
)
with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        histogram = (
            alt.Chart(plot_data)
            .mark_bar(color="#8ecae6")
            .encode(
                x=alt.X("Value:Q", bin=alt.Bin(maxbins=30), title=str(value_name)),
                y=alt.Y("count():Q", title="Observations"),
                tooltip=["count():Q"],
            )
            .properties(height=320)
        )
        st.altair_chart(histogram, width="stretch")
    with col2:
        st.dataframe(values.describe().rename("Value").to_frame(), width="stretch")

with tab2:
    default_period = int(st.session_state.get("seasonal_period", 12))
    max_period = max(2, len(data) // 2)
    period = st.number_input(
        "Seasonal period",
        min_value=2,
        max_value=max_period,
        value=max(2, min(default_period, max_period)),
        step=1,
        key="exploration_period",
    )
    model = st.selectbox("Decomposition type", ["additive", "multiplicative"])
    if model == "multiplicative" and (values <= 0).any():
        st.warning(
            "Multiplicative decomposition requires positive values. Choose additive for this series."
        )
    elif len(values) < 2 * period:
        st.warning(
            f"At least {2 * period} observations are needed for two complete seasonal cycles."
        )
    else:
        decomposition = seasonal_decompose(
            values.to_numpy(), model=model, period=int(period), extrapolate_trend="freq"
        )
        components = pd.DataFrame(
            {
                "Observed": decomposition.observed,
                "Trend": decomposition.trend,
                "Seasonal": decomposition.seasonal,
                "Residual": decomposition.resid,
            },
            index=data[date_name],
        )
        st.line_chart(components, height=500)

with tab3:
    max_lags = min(60, len(values) - 2)
    selected_lags = st.slider("Number of lags", 1, max_lags, min(24, max_lags))
    correlations = acf(values, nlags=selected_lags, fft=True)
    acf_frame = pd.DataFrame(
        {"Lag": np.arange(len(correlations)), "Autocorrelation": correlations}
    )
    st.bar_chart(acf_frame.set_index("Lag"), height=300)
    if len(values) >= 12:
        statistic, p_value, used_lags, observations, critical_values, _ = adfuller(
            values, autolag="AIC"
        )
        adf_information = STATISTICAL_TESTS["adf"]
        with st.popover(
            "ADF hypotheses, decision rule & literature",
            icon=":material/info:",
            width="stretch",
        ):
            st.markdown(
                f"**H0:** {adf_information.null_hypothesis.removeprefix('H0: ')}"
            )
            st.markdown(
                f"**H1:** {adf_information.alternative_hypothesis.removeprefix('H1: ')}"
            )
            st.markdown("**Decision at alpha = .05:** reject H0 when p < .05, or "
                        "when the ADF tau statistic is more negative than the 5% "
                        "critical value. Otherwise fail to reject H0.")
            st.code(copy_ready_test_note("adf"), language=None, wrap_lines=True)
            for index, reference in enumerate(adf_information.references, start=1):
                st.markdown(f"**{index}.** {reference.apa}")
                st.markdown(f"[Open source]({reference.url})")
        stationarity = st.columns(4)
        stationarity[0].metric("ADF statistic", f"{statistic:.3f}")
        stationarity[1].metric("p-value", f"{p_value:.4f}")
        stationarity[2].metric("Lags used", used_lags)
        stationarity[3].metric("Observations", observations)
        st.dataframe(
            pd.DataFrame(
                {
                    "Significance level": list(critical_values),
                    "Critical tau value": list(critical_values.values()),
                }
            ),
            hide_index=True,
            width="stretch",
        )
        if p_value < 0.05:
            st.success(
                "The ADF test rejects a unit root at the 5% level; the series appears stationary."
            )
        else:
            st.info(
                "The ADF test does not reject a unit root at the 5% level; differencing or detrending may help."
            )

st.download_button(
    "Download prepared data (CSV)",
    data=data.to_csv(index=False).encode("utf-8"),
    file_name="chrono_stream_prepared_data.csv",
    mime="text/csv",
    width="stretch",
)
