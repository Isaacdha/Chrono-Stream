import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd

# Page Settings
st.set_page_config(
    page_title="Chrono Stream - Deterministic Trend Projection",
    page_icon="🔮",
    layout="wide"
)
st.logo('.streamlit/Logo.png', icon_image='.streamlit/Logo_small.png', size='large')

# Sidebar Settings
with st.sidebar:
    selected = option_menu(
        "Trend Projection",
        ["Linear", "Quadratic", "Exponential", "Logarithmic"],
        icons=["bi bi-arrow-up-right", "bi bi-box-arrow-in-up-right", "bi bi-box-arrow-up", "bi bi-arrow-90deg-up"],
        menu_icon="bi bi-projector-fill",
        default_index=0,
        styles={
            "menu-title": {"font-size": "17px"},
            "nav-link": {"font-size": "14px", "text-align": "left", "margin": "0px", "--hover-color": "#6082B6"},
            "container": {"background-color": "#252526", "border": "1px solid white"},
            "nav-link-selected": {"background-color": "#00203FFF"}
        }
    )

# Check if data is uploaded
if 'filtered_df' not in st.session_state:
    st.error("Please upload a file in the input data page.")
    st.stop()
else:
    data = st.session_state['filtered_df']

value_column = data.iloc[:, 1]

# Metrics Function
def metrics(data, smoothed_data):
    mse = round(np.mean((data - smoothed_data) ** 2), 2)
    mae = round(np.mean(np.abs(data - smoothed_data)), 2)
    mape = round(np.mean(np.abs((data - smoothed_data) / data)) * 100, 2)
    rmse = round(np.sqrt(mse), 2)
    return mse, mae, mape, rmse

if selected == "Linear":
    st.title("📏 Linear Trend Projection")
    st.image(".streamlit/Border_H.png", use_column_width=True)
    st.write("This model is used to project time series data with a linear trend.")

if selected == "Quadratic":
    st.title("➰ Quadratic Trend Projection")
    st.image(".streamlit/Border_H.png", use_column_width=True)
    st.write("This model is used to project time series data with a quadratic trend.")

if selected == "Exponential":
    st.title("💥 Exponential Trend Projection")
    st.image(".streamlit/Border_H.png", use_column_width=True)
    st.write("This model is used to project time series data with an exponential trend.")
    
if selected == "Logarithmic":
    st.title("📚 Logarithmic Trend Projection")
    st.image(".streamlit/Border_H.png", use_column_width=True)
    st.write("This model is used to project time series data with a logarithmic trend.")