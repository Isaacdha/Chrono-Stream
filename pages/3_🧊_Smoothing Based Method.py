import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd

with st.sidebar:
    selected = option_menu(
        "Smoothing Methods",
        ["Moving Average", "Single Exponential Smoothing", "Double Exponential Smoothing", "Triple Exponential Smoothing"],
        icons=["graph-up", "arrow-up-circle", "arrow-up-right-circle", "arrow-up-right-square"],
        menu_icon="cast",
        default_index=0,
        styles={
            "menu-title": {"font-size": "17px"},
            "nav-link": {"font-size": "14px", "text-align": "left", "margin": "0px", "--hover-color": "#6082B6"},
            "container": {"background-color": "#36454F", "border": "1px solid white"},
            "nav-link-selected": {"background-color": "#00203FFF"}
        }
    )

if 'filtered_df' not in st.session_state:
    st.error("Please upload a file in the previous step.")
    st.stop()
else:
    data = st.session_state['filtered_df']

if selected == "Moving Average":
    st.subheader("📈 Moving Average")
    st.write("This method is used to smooth out the data by calculating the average of a window of data points.")
    value_column = data.iloc[:, 1]
    
    def moving_average(data, n):
        return np.convolve(data, np.ones(n)/n, mode='valid')
    def metrics(data, smoothed_data, n):
        mse = round(np.mean((data[n-1:] - smoothed_data) ** 2), 2)
        mae = round(np.mean(np.abs(data[n-1:] - smoothed_data)), 2)
        mape = round(np.mean(np.abs((data[n-1:] - smoothed_data) / data[n-1:])) * 100, 2)
        rmse = round(np.sqrt(mse),2)
        return mse, mae, mape, rmse
    def find_optimal_n(data):
        min_error = float('inf')
        optimal_n = 1
        results = []
        for n in range(2, int(round(len(data)/2, 0)+1)):  
            ma = moving_average(data, n)
            mse, mae, mape, rmse = metrics(data, ma, n)
            results.append((n, mse, mae, mape, rmse))
            if mse < min_error:
                min_error = mse
                optimal_n = n
        return optimal_n
    
    optimal = st.checkbox("Automatically find optimal window size (using MSE)", value=False)
    
    if optimal == False:
        col1, col2 = st.columns(2)
        with col1:
            n = st.number_input("Window size", min_value=2, max_value=len(data), value=2)
        smoothed_data = moving_average(value_column, n)
        mse, mae, mape, rmse = metrics(value_column, smoothed_data, n)
        with st.container(border=True):
            st.markdown('#### Fitted vs Real Data')
            st.markdown(" ")
            st.line_chart(pd.DataFrame({"Original": value_column[n-1:], "Smoothed": smoothed_data}), x_label="Time Index", y_label="Value", color=["#aef5f1", "#edb682"])
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MSE", mse)
        col2.metric("MAE", mae)
        col3.metric("MAPE", f"{mape}%")
        col4.metric("RMSE", rmse)
    else:
        if len(data.columns) < 2:
            st.error("The data does not contain enough columns.")
        else:
            optimal_n = find_optimal_n(value_column)
            smoothed_data = moving_average(value_column, optimal_n)
            mse, mae, mape, rmse = metrics(value_column, smoothed_data, optimal_n)
            st.write(f"Optimal window size: {optimal_n}")
            with st.container(border=True):
                st.markdown('#### Fitted vs Real Data')
                st.markdown(" ")
                st.line_chart(pd.DataFrame({"Original": value_column[optimal_n-1:], "Smoothed": smoothed_data}), x_label="Time Index", y_label="Value")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("MSE", mse)
            col2.metric("MAE", mae)
            col3.metric("MAPE", f"{mape}%")
            col4.metric("RMSE", rmse)

elif selected == "Single Exponential Smoothing":
    st.subheader("📉 Single Exponential Smoothing")
    # Add your code for Single Exponential Smoothing here

elif selected == "Double Exponential Smoothing":
    st.subheader("📊 Double Exponential Smoothing")
    # Add your code for Double Exponential Smoothing here

elif selected == "Triple Exponential Smoothing":
    st.subheader("📐 Triple Exponential Smoothing")
    # Add your code for Triple Exponential Smoothing here
