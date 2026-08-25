"""Chrono Stream application entry point."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

import streamlit as st

from chrono_stream.registry import NAVIGATION_GROUPS, methods_for_group


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / ".streamlit"


def _method_page(model_id: str):
    """Build a zero-argument Streamlit page for one registered method."""

    def render() -> None:
        from chrono_stream.ui import render_model_page

        render_model_page(model_id)

    render.__name__ = f"render_{model_id}_page"
    return render

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
            "chrono_stream/page_overview.py",
            title="App Overview",
            icon="🚀",
            default=True,
        ),
        st.Page(
            "chrono_stream/page_data_input.py",
            title="Data Input & Settings",
            icon="📝",
        ),
        st.Page(
            "chrono_stream/page_exploration.py",
            title="Data Exploration",
            icon="🔍",
        ),
        st.Page(
            "chrono_stream/page_comparison.py",
            title="Compare Results",
            icon="📊",
        ),
    ]
}

for group in NAVIGATION_GROUPS:
    pages[group] = [
        st.Page(
            _method_page(spec.model_id),
            title=spec.display_name,
            icon=spec.icon,
            url_path=spec.url_path,
        )
        for spec in methods_for_group(group)
    ]

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
