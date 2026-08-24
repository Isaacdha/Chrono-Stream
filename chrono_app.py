"""Chrono Stream application entry point."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / ".streamlit"

st.set_page_config(
    page_title="Chrono Stream",
    page_icon="⌛",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def _asset_data_uri(path: str) -> str:
    asset = Path(path)
    mime_type = mimetypes.guess_type(asset.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(asset.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


background = ASSETS / "Body Background.jpg"
if background.exists():
    background_uri = _asset_data_uri(str(background))
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background: linear-gradient(rgba(8, 15, 35, .86), rgba(8, 15, 35, .86)),
                        url("{background_uri}") center / cover fixed;
        }}
        [data-testid="stHeader"] {{ background: rgba(0, 0, 0, 0); }}
        [data-testid="stMainBlockContainer"] {{ padding-top: 2rem; }}
        [data-testid="stSidebar"] {{ border-right: 1px solid rgba(142, 202, 230, .35); }}
        </style>
        """,
        unsafe_allow_html=True,
    )

logo = ASSETS / "Logo.png"
small_logo = ASSETS / "Logo_Small.png"
if logo.exists():
    st.logo(str(logo), icon_image=str(small_logo) if small_logo.exists() else None)

pages = {
    "Workflow": [
        st.Page(
            "method/1_App Overview.py", title="App Overview", icon="🚀", default=True
        ),
        st.Page("method/2_Data Input.py", title="Data Input & Settings", icon="📝"),
        st.Page("method/3_Data Exploration.py", title="Data Exploration", icon="🔍"),
        st.Page(
            "method/4_Result Comparison and Forecasting.py",
            title="Compare Results",
            icon="📊",
        ),
    ],
    "Smoothing": [
        st.Page(
            "method/Smoothing Based Methods/1_Moving Average.py",
            title="Moving Average",
            icon="📎",
        ),
        st.Page(
            "method/Smoothing Based Methods/2_Weighted Moving Average.py",
            title="Weighted Moving Average",
            icon="🖇️",
        ),
        st.Page(
            "method/Smoothing Based Methods/3_Single Exponential Smoothing.py",
            title="Single Exponential",
            icon="1️⃣",
        ),
        st.Page(
            "method/Smoothing Based Methods/4_Double Exponential Smoothing.py",
            title="Double Exponential",
            icon="2️⃣",
        ),
        st.Page(
            "method/Smoothing Based Methods/5_Triple Exponential Smoothing.py",
            title="Triple Exponential",
            icon="3️⃣",
        ),
    ],
    "Statistical": [
        st.Page("method/Statistical Models/1_ARIMA.py", title="ARIMA", icon="🌠"),
        st.Page("method/Statistical Models/2_SARIMA.py", title="SARIMA", icon="❄️"),
        st.Page(
            "method/Statistical Models/4_X-11.py",
            title="STL forecast (X-11-inspired)",
            icon="💫",
        ),
    ],
    "Machine Learning": [
        st.Page(
            "method/Machine Learning Models/1_Prophet.py", title="Prophet", icon="🔮"
        ),
        st.Page("method/Machine Learning Models/2_LSTM.py", title="LSTM", icon="🧠"),
        st.Page("method/Machine Learning Models/3_CNN.py", title="1D CNN", icon="🗃️"),
        st.Page(
            "method/Machine Learning Models/4_XGBoost.py", title="XGBoost", icon="🔥"
        ),
    ],
    "Trend Projection": [
        st.Page(
            "method/Deterministic Trend Projection/1_Linear.py",
            title="Linear",
            icon="↗️",
        ),
        st.Page(
            "method/Deterministic Trend Projection/2_Quadratic.py",
            title="Quadratic",
            icon="➿",
        ),
        st.Page(
            "method/Deterministic Trend Projection/3_Exponential.py",
            title="Exponential",
            icon="✴️",
        ),
        st.Page(
            "method/Deterministic Trend Projection/4_Logarithmic.py",
            title="Logarithmic",
            icon="❇️",
        ),
    ],
}

with st.sidebar:
    if "filtered_df" in st.session_state:
        data = st.session_state["filtered_df"]
        st.success(f"{len(data):,} observations loaded")
        st.caption(
            f"{st.session_state.get('data_frequency') or 'Irregular'} frequency · "
            f"{st.session_state.get('forecast_period', 12)}-period forecast"
        )
    else:
        st.info("Start on **Data Input & Settings** to unlock forecasting.")
    with st.expander("About the author"):
        st.markdown(
            "[Email](mailto:isaacazziz@gmail.com) · "
            "[LinkedIn](https://id.linkedin.com/in/isaacdha) · "
            "[GitHub](https://github.com/Isaacdha)"
        )

navigation = st.navigation(pages, expanded=True)
navigation.run()
