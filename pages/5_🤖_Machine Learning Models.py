import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd

# Page Settings
st.set_page_config(
    page_title="Chrono Stream - Machine LearningS Models",
    page_icon="🤖",
    layout="wide"
)
st.logo('.streamlit/Logo.png', icon_image='.streamlit/Logo_small.png', size='large')

# Sidebar Settings
with st.sidebar:
    selected = option_menu(
        "ML Models",
        ["Prophet", "LSTM (Locked)", "CNN (Locked)", "XGBoost (Locked)"],
        icons=["bi bi-brilliance", "bi bi-bullseye", "bi bi-code-square", "bi bi-cpu-fill"],
        menu_icon="bi bi-robot",
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

if selected == "Prophet":
    st.title("🔮 Prophet Model")
    st.image(".streamlit/Border_H.png", use_column_width=True)
    st.write("This model is used to forecast time series data with daily observations that display patterns on different time scales.")

elif selected == "LSTM (Locked)":
    st.title("🧠 Long-Short Term Memory Model")
    st.image(".streamlit/Border_H.png", use_column_width=True)
    st.write("This model is used to forecast time series data with long short-term memory networks.")
    st.warning("This model is locked due to complexity and resource constraint, would be available in the future update.")

elif selected == "CNN (Locked)":
    st.title("📦 Convolutional Neural Network Model")
    st.image(".streamlit/Border_H.png", use_column_width=True)
    st.write("This model is used to forecast time series data with convolutional neural networks.")
    st.warning("This model is locked due to complexity and resource constraint, would be available in the future update.")
    
elif selected == "XGBoost (Locked)":
    st.title("🔥 XGBoost Model")
    st.image(".streamlit/Border_H.png", use_column_width=True)
    st.write("This model is used to forecast time series data with extreme gradient boosting.")
    st.warning("This model is locked due to complexity and resource constraint, would be available in the future update.")
    
