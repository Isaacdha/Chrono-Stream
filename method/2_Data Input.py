"""Data loading, validation, and shared forecast settings."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from chrono_stream.data import data_signature, infer_frequency, prepare_time_series


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = {
    "Sample Data 1 (Excel)": ROOT / "Sample_Data_1.xlsx",
    "Sample Data 2 (CSV)": ROOT / "Sample_Data_2.csv",
    "Sample Data 3 (CSV)": ROOT / "Sample_Data_3.csv",
}
FREQUENCIES = {
    "Auto-detect": None,
    "Hourly": "h",
    "Daily": "D",
    "Weekly": "W",
    "Month start": "MS",
    "Month end": "ME",
    "Quarter start": "QS",
    "Quarter end": "QE",
    "Year start": "YS",
    "Year end": "YE",
}


@st.cache_data(show_spinner=False)
def _read_csv(contents: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(contents))


@st.cache_data(show_spinner=False)
def _excel_sheets(contents: bytes) -> list[str]:
    return pd.ExcelFile(BytesIO(contents)).sheet_names


@st.cache_data(show_spinner=False)
def _read_excel(contents: bytes, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(BytesIO(contents), sheet_name=sheet_name)


def _load_source() -> tuple[pd.DataFrame | None, str | None]:
    source = st.radio(
        "Data source",
        ["Upload a file", *SAMPLES],
        horizontal=True,
        help="Bundled samples are useful for trying the app immediately.",
    )
    if source in SAMPLES:
        path = SAMPLES[source]
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path), path.name
        return pd.read_excel(path), path.name

    uploaded = st.file_uploader("Choose a CSV or XLSX file", type=["csv", "xlsx"])
    if uploaded is None:
        return None, None
    contents = uploaded.getvalue()
    if uploaded.name.lower().endswith(".csv"):
        return _read_csv(contents), uploaded.name
    sheets = _excel_sheets(contents)
    sheet = st.selectbox("Excel sheet", sheets) if len(sheets) > 1 else sheets[0]
    return _read_excel(contents, sheet), uploaded.name


st.title("📝 Data Input & Forecast Settings")
st.write(
    "Select one timestamp column and one numeric value column. Your source file is never modified."
)

with st.container(border=True):
    st.subheader("1. Load data")
    try:
        raw_data, source_name = _load_source()
    except Exception as exc:
        st.error(f"The file could not be read: {exc}")
        raw_data, source_name = None, None

if raw_data is not None and not raw_data.empty:
    with st.container(border=True):
        st.subheader("2. Prepare the time series")
        date_candidates = [
            column
            for column in raw_data.columns
            if pd.api.types.is_datetime64_any_dtype(raw_data[column])
            or any(
                token in str(column).lower()
                for token in ("date", "time", "month", "year")
            )
        ]
        numeric_candidates = list(raw_data.select_dtypes(include="number").columns)
        date_default = (
            raw_data.columns.get_loc(date_candidates[0]) if date_candidates else 0
        )
        value_default = (
            raw_data.columns.get_loc(numeric_candidates[0])
            if numeric_candidates
            else min(1, len(raw_data.columns) - 1)
        )

        col1, col2 = st.columns(2)
        date_column = col1.selectbox(
            "Date/time column", raw_data.columns, index=date_default
        )
        value_column = col2.selectbox(
            "Value column", raw_data.columns, index=value_default
        )

        day_first = st.toggle(
            "Interpret ambiguous dates as day-first",
            value=False,
            help="Enable this for dates such as 31/12/2025.",
        )
        parsed_dates = pd.to_datetime(
            raw_data[date_column], errors="coerce", format="mixed", dayfirst=day_first
        )
        detected_frequency = infer_frequency(parsed_dates)
        st.caption(
            f"Detected frequency: **{detected_frequency or 'not regular enough to infer'}**"
        )

        option_columns = st.columns(4)
        frequency_label = option_columns[0].selectbox("Frequency", list(FREQUENCIES))
        regularize = option_columns[1].toggle("Fill missing periods", value=True)
        missing_method = option_columns[2].selectbox(
            "Missing periods",
            ["Interpolate", "Forward fill", "Backward fill", "Drop gaps"],
        )
        duplicate_method = option_columns[3].selectbox(
            "Duplicate dates", ["Mean", "Sum", "First", "Last"]
        )

        requested_frequency = FREQUENCIES[frequency_label]
        if frequency_label == "Auto-detect":
            requested_frequency = detected_frequency
        try:
            prepared, report = prepare_time_series(
                raw_data,
                date_column,
                value_column,
                frequency=requested_frequency,
                regularize=regularize,
                missing_method=missing_method,
                duplicate_method=duplicate_method,
                day_first=day_first,
            )
        except ValueError as exc:
            st.error(str(exc))
            prepared = None
        if prepared is not None:
            report_columns = st.columns(4)
            report_columns[0].metric("Prepared rows", f"{report.output_rows:,}")
            report_columns[1].metric(
                "Invalid rows removed", report.invalid_rows_removed
            )
            report_columns[2].metric(
                "Duplicates combined", report.duplicate_timestamps_combined
            )
            report_columns[3].metric(
                "Missing periods filled", report.missing_periods_created
            )
            st.dataframe(prepared.head(20), hide_index=True, width="stretch")

            max_horizon = max(1, min(365, len(prepared) // 2))
            default_horizon = min(12, max_horizon)
            default_season = min(12, max(2, len(prepared) // 2))
            settings = st.columns(3)
            horizon = settings[0].number_input(
                "Forecast horizon",
                min_value=1,
                max_value=max_horizon,
                value=default_horizon,
                step=1,
            )
            holdout = settings[1].number_input(
                "Evaluation holdout",
                min_value=1,
                max_value=max(1, len(prepared) - 4),
                value=min(default_horizon, max(1, len(prepared) // 5)),
                step=1,
            )
            seasonal_period = settings[2].number_input(
                "Default seasonal period",
                min_value=2,
                max_value=max(2, len(prepared) // 2),
                value=default_season,
                step=1,
            )

            if st.button("Save data and settings", type="primary", width="stretch"):
                new_signature = data_signature(prepared)
                configuration_signature = (
                    new_signature,
                    report.frequency,
                    int(horizon),
                    int(holdout),
                    int(seasonal_period),
                )
                old_configuration = st.session_state.get("configuration_signature")
                st.session_state["filtered_df"] = prepared
                st.session_state["data_frequency"] = report.frequency
                st.session_state["forecast_period"] = int(horizon)
                st.session_state["evaluation_period"] = int(holdout)
                st.session_state["seasonal_period"] = int(seasonal_period)
                st.session_state["source_name"] = source_name
                st.session_state["data_signature"] = new_signature
                st.session_state["configuration_signature"] = configuration_signature
                if old_configuration != configuration_signature:
                    st.session_state["model_results"] = {}
                st.success("Data and settings saved. Forecasting pages are ready.")
                st.rerun()
elif raw_data is not None:
    st.error("The selected file does not contain any rows.")
else:
    st.info("Upload a file or select a bundled sample to continue.")

if "filtered_df" in st.session_state:
    with st.container(border=True):
        st.subheader("Active data")
        active = st.session_state["filtered_df"]
        st.write(
            f"**{st.session_state.get('source_name', 'Saved series')}** · {len(active):,} observations · "
            f"{st.session_state.get('data_frequency') or 'observed'} frequency"
        )
        st.line_chart(
            active.set_index(active.columns[0])[active.columns[1]], height=260
        )
        if st.button("Clear active data and model results"):
            for key in (
                "filtered_df",
                "data_frequency",
                "forecast_period",
                "evaluation_period",
                "seasonal_period",
                "source_name",
                "data_signature",
                "configuration_signature",
                "model_results",
            ):
                st.session_state.pop(key, None)
            st.rerun()
