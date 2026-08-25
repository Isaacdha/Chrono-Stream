"""Compare saved model evaluations and forecasts."""

import altair as alt
import pandas as pd
import streamlit as st

from chrono_stream.ui import render_metric_reference_actions


title_columns = st.columns([12, 4], gap="small", vertical_alignment="center")
with title_columns[0]:
    st.title("📊 Result Comparison and Forecasting")
with title_columns[1]:
    render_metric_reference_actions("comparison")
if "filtered_df" not in st.session_state:
    st.warning("Load and save a time series on the Data Input page first.")
    st.stop()

results = st.session_state.get("model_results", {})
if not results:
    st.info(
        "No model results are saved yet. Fit at least one forecasting method, then return here."
    )
    st.stop()

saved_holdouts = {int(result["holdout"]) for result in results.values()}
if len(saved_holdouts) != 1:
    st.error(
        "The saved results use different holdout windows and cannot be ranked fairly. "
        "Clear or refit them with the shared holdout from Data Input & Settings."
    )
    st.stop()

metric_rows = []
for model_id, result in results.items():
    metric_rows.append(
        {
            "Model": result["model_name"],
            "MAE": result["metrics"].get("MAE"),
            "RMSE": result["metrics"].get("RMSE"),
            "MASE": result["metrics"].get("MASE"),
            "RMSSE": result["metrics"].get("RMSSE"),
            "MAPE (%)": result["metrics"].get("MAPE"),
            "sMAPE (%)": result["metrics"].get("sMAPE"),
            "WAPE (%)": result["metrics"].get("WAPE"),
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
            "MASE": "{:,.3f}",
            "RMSSE": "{:,.3f}",
            "MAPE (%)": "{:,.2f}",
            "sMAPE (%)": "{:,.2f}",
            "WAPE (%)": "{:,.2f}",
        },
        na_rep="N/A",
    ),
    hide_index=True,
    width="stretch",
)

scale_periods = {
    result.get("metric_context", {}).get("scale_period")
    for result in results.values()
    if result.get("metric_context", {}).get("scale_period") is not None
}
if len(scale_periods) == 1:
    scale_period = next(iter(scale_periods))
    benchmark = "naive" if scale_period == 1 else f"seasonal-naive lag {scale_period}"
    st.caption(
        f"MASE and RMSSE use the shared {benchmark} scale calculated only from "
        "pre-holdout training observations. N/A denotes an undefined metric, not zero error."
    )
elif len(scale_periods) > 1:
    st.warning(
        "Saved results use different MASE/RMSSE scale periods. Refit them with shared "
        "settings before comparing scaled metrics."
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
            y=alt.Y("Lower interval:Q", scale=alt.Scale(zero=False)),
            y2="Upper interval:Q",
        ),
    )
st.altair_chart(
    alt.layer(*layers).properties(height=480).interactive(), width="stretch"
)
best_details = results[best["Model ID"]].get("model_details", {})
if best_details.get("interval_available", True):
    st.caption(
        "The shaded interval belongs to the current lowest-RMSE model: "
        f"{best_details.get('interval_method', 'method not reported')}."
    )
else:
    st.caption("The current lowest-RMSE model has no available forecast interval.")

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
