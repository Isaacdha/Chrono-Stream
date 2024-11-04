import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd

# Page Settings
st.set_page_config(
    page_title="Chrono Stream - Result Comparison and Forecasting",
    page_icon="📊",
    layout="wide"
)
st.logo('.streamlit/Logo.png', icon_image='.streamlit/Logo_small.png', size='large')

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