import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd

if 'Moving_Average' in st.session_state:
    st.write(st.session_state['Moving_Average'])
if 'Single_Exponential_Smoothing' in st.session_state:
    st.write(st.session_state['Single_Exponential_Smoothing'])
if 'Double_Exponential_Smoothing' in st.session_state:
    st.write(st.session_state['Double_Exponential_Smoothing'])
if 'Triple_Exponential_Smoothing' in st.session_state:
    st.write(st.session_state['Triple_Exponential_Smoothing'])
if 'ARIMA' in st.session_state:
    st.write(st.session_state['ARIMA'])
if 'SARIMA' in st.session_state:
    st.write(st.session_state['SARIMA'])
if 'X-11' in st.session_state:
    st.write(st.session_state['X-11'])
if 'Prophet' in st.session_state:
    st.write(st.session_state['Prophet'])
if 'Linear' in st.session_state:
    st.write(st.session_state['Linear'])
if 'Quadratic' in st.session_state:
    st.write(st.session_state['Quadratic'])
if 'Exponential' in st.session_state:
    st.write(st.session_state['Exponential'])
if 'Logarithmic' in st.session_state:
    st.write(st.session_state['Logarithmic'])
    
# Decompose the time series data
decomposition = pm.arima.decompose(value_column, 'additive', 12)

col1, col2 = st.columns(2)
with st.expander("Decomposition Plot"):
    # Observed
    df_x = pd.DataFrame({
        'Date': date_column[:len(decomposition.x)],
        'Value': decomposition.x
    })
    with st.container(border=True):
        st.markdown("<h8>X Decomposition Plot</h8>", unsafe_allow_html=True)
        st.line_chart(data = df_x, x = "Date", y = "Value", use_container_width=True,
                    height=200, x_label=None, y_label=None)
    # Trend
    df_trend = pd.DataFrame({
        'Date': date_column[:len(decomposition.trend)],
        'Value': decomposition.trend
    })
    with st.container(border=True):
        st.markdown("<h8>Trend Decomposition Plot</h8>", unsafe_allow_html=True)
        st.line_chart(data = df_trend, x = "Date", y = "Value", use_container_width=True,
                    height=200)
    # Seasonal
    df_seasonal = pd.DataFrame({
        'Date': date_column[:len(decomposition.seasonal)],
        'Value': decomposition.seasonal
    })

    with st.container(border=True):
        st.markdown("<h8>Seasonal Decomposition Plot</h8>", unsafe_allow_html=True)
        st.line_chart(data = df_seasonal, x = "Date", y = "Value", use_container_width=True,
                    height=200)
    # Residual
    df_residual = pd.DataFrame({
        'Date': date_column[:len(decomposition.random)],
        'Value': decomposition.random
    })
    with st.container(border=True):
        st.markdown("<h8>Residual Decomposition Plot</h8>", unsafe_allow_html=True)
        st.line_chart(data = df_residual, x = "Date", y = "Value", use_container_width=True,
                    height=200)