import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd

# Page Settings
st.set_page_config(
    page_title="Chrono Stream - Statistical Models",
    page_icon="📐",
    layout="wide"
)
st.logo('.streamlit/Logo.png', icon_image='.streamlit/Logo_small.png', size='large')

# Sidebar Settings
with st.sidebar:
    selected = option_menu(
        "Statistical Models",
        ["ARIMA", "SARIMA", "X-11"],
        icons=["bi bi-puzzle-fill", "bi bi-asterisk", "bi bi-clock-history"],
        menu_icon="bi bi-calculator-fill",
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
def metrics(data, fitted_data):
    mse = round(np.mean((data - fitted_data) ** 2), 2)
    mae = round(np.mean(np.abs(data - fitted_data)), 2)
    mape = round(np.mean(np.abs((data - fitted_data) / data)) * 100, 2)
    rmse = round(np.sqrt(mse), 2)
    return mse, mae, mape, rmse

# ARIMA Model
if selected == "ARIMA":
    st.title("📉 ARIMA Model")
    st.image(".streamlit/Border_H.png", use_column_width=True)
    st.write("This model is used to forecast time series data.")
    
elif selected == "SARIMA":
    st.title("❄️ SARIMA Model")
    st.image(".streamlit/Border_H.png", use_column_width=True)
    st.write("This model is used to forecast time series data with seasonal components.")
    
elif selected == "X-11":
    st.title("💫 X-11 Model")
    st.image(".streamlit/Border_H.png", use_column_width=True)
    st.write("This model is used to decompose time series data into trend, seasonal, and irregular components.")


