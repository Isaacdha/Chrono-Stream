import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd

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

# Logarithmic trend projection process
st.title("❇️ Logarithmic Trend Projection")
st.image(".streamlit/Border_H.png", use_column_width=True)
st.write("This model is used to project time series data with a logarithmic trend.")