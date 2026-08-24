"""Compare saved model evaluations and forecasts."""

import altair as alt
import pandas as pd
import streamlit as st


st.title("📊 Result Comparison and Forecasting")
if "filtered_df" not in st.session_state:
    st.warning("Load and save a time series on the Data Input page first.")
    st.stop()

results = st.session_state.get("model_results", {})
if not results:
    st.info(
        "No model results are saved yet. Fit at least one forecasting method, then return here."
    )
    st.stop()

metric_rows = []
for model_id, result in results.items():
    metric_rows.append(
        {
            "Model": result["model_name"],
            "MAE": result["metrics"]["MAE"],
            "RMSE": result["metrics"]["RMSE"],
            "MAPE (%)": result["metrics"]["MAPE"],
            "sMAPE (%)": result["metrics"]["sMAPE"],
            "Holdout": result["holdout"],
            "Forecast horizon": result["horizon"],
            "Model ID": model_id,
        }
    )
metrics = (
    pd.DataFrame(metric_rows)
    .sort_values("RMSE", na_position="last")
    .reset_index(drop=True)
)

best = metrics.iloc[0]
summary = st.columns(3)
summary[0].metric("Models compared", len(metrics))
summary[1].metric("Lowest holdout RMSE", f"{best['RMSE']:,.3f}")
summary[2].metric("Leading model", best["Model"])
st.caption(
    "The leading model is selected by the lowest RMSE on the shared out-of-sample holdout."
)

display_metrics = metrics.drop(columns="Model ID").copy()
st.dataframe(
    display_metrics.style.format(
        {
            "MAE": "{:,.3f}",
            "RMSE": "{:,.3f}",
            "MAPE (%)": "{:,.2f}",
            "sMAPE (%)": "{:,.2f}",
        }
    ),
    hide_index=True,
    width="stretch",
)

selected_names = st.multiselect(
    "Models shown on the forecast chart",
    options=metrics["Model"].tolist(),
    default=metrics["Model"].tolist(),
)
selected_ids = metrics.loc[metrics["Model"].isin(selected_names), "Model ID"].tolist()

history = st.session_state["filtered_df"].copy()
history.columns = ["Date", "Actual"]
history["Date"] = pd.to_datetime(history["Date"])
actual_long = history.rename(columns={"Actual": "Value"})
actual_long["Series"] = "Actual"
forecast_frames = [actual_long]
interval_frame = None
for model_id in selected_ids:
    result = results[model_id]
    model_forecast = result["forecast"][["Date", "Forecast"]].rename(
        columns={"Forecast": "Value"}
    )
    model_forecast["Series"] = result["model_name"]
    forecast_frames.append(model_forecast)
    if model_id == best["Model ID"]:
        interval_frame = result["forecast"]

combined = pd.concat(forecast_frames, ignore_index=True)
line_chart = (
    alt.Chart(combined)
    .mark_line()
    .encode(
        x=alt.X("Date:T", title="Date"),
        y=alt.Y(
            "Value:Q",
            title=str(st.session_state["filtered_df"].columns[1]),
            scale=alt.Scale(zero=False),
        ),
        color=alt.Color("Series:N", legend=alt.Legend(orient="top")),
        strokeDash=alt.condition(
            alt.datum.Series == "Actual", alt.value([1, 0]), alt.value([6, 3])
        ),
        tooltip=[
            alt.Tooltip("Date:T"),
            "Series:N",
            alt.Tooltip("Value:Q", format=",.4f"),
        ],
    )
)
layers = [line_chart]
if interval_frame is not None:
    layers.insert(
        0,
        alt.Chart(interval_frame)
        .mark_area(opacity=0.12, color="#fb8500")
        .encode(
            x="Date:T",
            y=alt.Y("Lower 95%:Q", scale=alt.Scale(zero=False)),
            y2="Upper 95%:Q",
        ),
    )
st.altair_chart(
    alt.layer(*layers).properties(height=480).interactive(), width="stretch"
)
st.caption("The shaded interval belongs to the current lowest-RMSE model.")

with st.expander("Inspect a saved result", expanded=False):
    inspected_name = st.selectbox("Model", metrics["Model"].tolist())
    inspected_id = metrics.loc[metrics["Model"] == inspected_name, "Model ID"].iloc[0]
    inspected = results[inspected_id]
    st.write("Parameters", inspected["parameters"])
    st.write("Selected model details", inspected.get("model_details", {}))
    st.dataframe(inspected["forecast"], hide_index=True, width="stretch")

download_forecasts = []
for result in results.values():
    frame = result["forecast"].copy()
    frame.insert(0, "Model", result["model_name"])
    download_forecasts.append(frame)
all_forecasts = pd.concat(download_forecasts, ignore_index=True)
downloads = st.columns(2)
downloads[0].download_button(
    "Download comparison metrics",
    metrics.drop(columns="Model ID").to_csv(index=False).encode("utf-8"),
    "chrono_stream_model_metrics.csv",
    "text/csv",
    width="stretch",
)
downloads[1].download_button(
    "Download all forecasts",
    all_forecasts.to_csv(index=False).encode("utf-8"),
    "chrono_stream_all_forecasts.csv",
    "text/csv",
    width="stretch",
)

with st.expander("Remove saved results"):
    remove_names = st.multiselect("Results to remove", metrics["Model"].tolist())
    if st.button("Remove selected results", disabled=not remove_names):
        remove_ids = set(metrics.loc[metrics["Model"].isin(remove_names), "Model ID"])
        st.session_state["model_results"] = {
            model_id: result
            for model_id, result in results.items()
            if model_id not in remove_ids
        }
        st.rerun()
