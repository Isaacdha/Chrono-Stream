import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from statsmodels.tsa.stattools import adfuller, kpss
import matplotlib.pyplot as plt
import statsmodels.api as sm


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

# Transformation Function
def yeo_lambda_iteration(data, lambda_threshold=[0.95, 1.05], iterations_limit=None):
    """
    Iteratively applies Yeo-Johnson transformation until the lambda value 
    is within the specified threshold range for variance stationarity.

    Parameters:
    - data: numpy array or pandas Series containing the time series data.
    - lambda_threshold: list or tuple with two elements [low, high] specifying 
                        the target range for lambda value to achieve variance stationarity.
    - iterations_limit: Optional integer, maximum number of iterations to prevent infinite loops.

    Returns:
    - final_transformed_data: The final transformed data after achieving variance stationarity.
    - transformers: List of PowerTransformer objects used in each iteration, 
                    excluding the last iteration if the threshold was already met.
    - iteration: Number of iterations performed.
    """
    transformed_data = data.values.reshape(-1, 1)  # Reshape data for PowerTransformer
    threshold_low, threshold_high = lambda_threshold
    transformers = []  # List to store each PowerTransformer instance
    iteration = 0

    while True and (iterations_limit is None or iteration < iterations_limit):
        # Create a new PowerTransformer and fit the data
        yj = PowerTransformer(method='yeo-johnson')
        yj.fit(transformed_data)
        lambda_value = yj.lambdas_[0]  # Extract lambda value
        final_lambda = lambda_value

        # Check if lambda is within the desired threshold for variance stationarity
        if threshold_low <= lambda_value <= threshold_high:
            print(f"Variance stationarity achieved at iteration {iteration + 1}")
            break  # Exit loop if within threshold without transforming

        # Apply transformation only if the threshold is not met
        transformed_data = yj.transform(transformed_data)
        transformers.append(yj)  # Store the transformer after applying transformation
        iteration += 1  # Update iteration counter

    return transformed_data, transformers, iteration, final_lambda

def inverse_yeo_iteration(final_transformed_data, transformers):
    """
    Applies the inverse transformations to convert the final transformed data back to its original form.

    Parameters:
    - final_transformed_data: The final transformed data after applying iterative Yeo-Johnson transformations.
    - transformers: List of PowerTransformer objects used in each iteration in yeo_lambda_iteration.

    Returns:
    - original_data: The data transformed back to its original form.
    """
    original_data = final_transformed_data

    # Apply the inverse transformations in reverse order
    for yj in reversed(transformers):
        original_data = yj.inverse_transform(original_data)

    return original_data

def boxcox_lambda_iteration(data, lambda_threshold=[0.95, 1.05], iterations_limit=None):
    """
    Iteratively applies Box-Cox transformation until the lambda value 
    is within the specified threshold range for variance stationarity.
    If negative values are encountered, raises an error.

    Parameters:
    - data: numpy array or pandas Series containing the time series data.
    - lambda_threshold: list or tuple with two elements [low, high] specifying 
                        the target range for lambda value to achieve variance stationarity.
    - iterations_limit: Optional integer, maximum number of iterations to prevent infinite loops.

    Returns:
    - final_transformed_data: The final transformed data after achieving variance stationarity.
    - transformers: List of PowerTransformer objects used in each iteration, 
                    excluding the last iteration if the threshold was already met.
    - iteration: Number of iterations performed.
    
    Raises:
    - ValueError: If negative values are encountered, as Box-Cox cannot handle negatives.
    """
    transformed_data = data.values.reshape(-1, 1)  # Reshape data for PowerTransformer
    threshold_low, threshold_high = lambda_threshold
    transformers = []  # List to store each PowerTransformer instance
    iteration = 0

    while True and (iterations_limit is None or iteration < iterations_limit):
        # Check for negative values before applying Box-Cox transformation
        if (transformed_data <= 0).any():
            raise ValueError("Box-Cox transformation encountered negative values. "
                             "Consider using Yeo-Johnson transformation instead.")

        # Create a new PowerTransformer with Box-Cox method and fit the data
        bc = PowerTransformer(method='box-cox')
        bc.fit(transformed_data)
        lambda_value = bc.lambdas_[0]  # Extract lambda value
        final_lambda = lambda_value

        # Check if lambda is within the desired threshold for variance stationarity
        if threshold_low <= lambda_value <= threshold_high:
            print(f"Variance stationarity achieved at iteration {iteration + 1}")
            break  # Exit loop if within threshold without transforming

        # Apply transformation only if the threshold is not met
        transformed_data = bc.transform(transformed_data)
        transformers.append(bc)  # Store the transformer after applying transformation
        iteration += 1  # Update iteration counter

    return transformed_data, transformers, iteration, final_lambda

def inverse_boxcox_iteration(final_transformed_data, transformers):
    """
    Applies the inverse transformations to convert the final transformed data back to its original form.

    Parameters:
    - final_transformed_data: The final transformed data after applying iterative Box-Cox transformations.
    - transformers: List of PowerTransformer objects used in each iteration in boxcox_lambda_iteration.

    Returns:
    - original_data: The data transformed back to its original form.
    """
    original_data = final_transformed_data

    # Apply the inverse transformations in reverse order
    for bc in reversed(transformers):
        original_data = bc.inverse_transform(original_data)

    return original_data

def iterative_adf(data, threshold=0.05):
    """
    Iteratively apply ADF test and differencing until stationarity is achieved.
    
    Parameters:
    data (pd.Series or list): The time series data to check for stationarity.
    threshold (float): The significance level for the ADF test to consider the data stationary.
    
    Returns:
    diffed_data (pd.Series): Differenced data that is stationary.
    d (int): The number of differences applied.
    """
    d = 0
    diffed_data = data
    
    while True:
        # Perform the ADF test
        adf_test = adfuller(diffed_data)
        p_value = adf_test[1]
        
        # Check if p-value is below threshold (indicating stationarity)
        if p_value < threshold:
            break  # Data is stationary
        
        # If not stationary, difference the data
        diffed_data = diffed_data.diff().dropna()
        d += 1
    
    return diffed_data, d

def iterative_kpss(data, threshold=0.05):
    """
    Iteratively apply KPSS test and differencing until stationarity is achieved.
    
    Parameters:
    data (pd.Series or list): The time series data to check for stationarity.
    threshold (float): The significance level for the KPSS test to consider the data stationary.
    
    Returns:
    diffed_data (pd.Series): Differenced data that is stationary.
    d (int): The number of differences applied.
    """
    d = 0
    diffed_data = data
    
    while True:
        # Perform the KPSS test (stationarity is indicated if p-value > threshold)
        kpss_test = kpss(diffed_data, nlags="auto")
        p_value = kpss_test[1]
        
        # Check if p-value is above threshold (indicating stationarity)
        if p_value > threshold:
            break  # Data is stationary according to KPSS test
        
        # If not stationary, difference the data
        diffed_data = diffed_data.diff().dropna()
        d += 1
    
    return diffed_data, d

# ARIMA Model
if selected == "ARIMA":
    st.title("📉 ARIMA Model")
    st.image(".streamlit/Border_H.png", use_column_width=True)
    st.write("This model is used to forecast time series data.")
    
    st.markdown(" ") 
    
    # Stationarity Check
    # Check variance stationarity using Box-Cox transformation
    with st.container(border=True):
        st.subheader("📊 ARIMA Stationarity Check")
        st.write("The Box-Cox transformation is used to stabilize the variance of a time series.")
        st.write("Boxcox Transformation and Augmented Dickey-Fuller test is used to check the stationarity of the time series")
        
        st.markdown(" ")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Please select transformation method")
            transformation_method = st.selectbox("Transformation Method", ["Box-Cox", "Yeo-Johnson"])
            if transformation_method == "Box-Cox":
                try:
                    transformed_data, transformers, iterations, final_lambda = boxcox_lambda_iteration(value_column)
                    st.write(f"Iterations: {iterations}")
                    st.write(f"Final Lambda Value: {round(final_lambda, 3)}")           
                except ValueError as e:
                    st.warning(f"{e}")
            elif transformation_method == "Yeo-Johnson":
                transformed_data, transformers, iterations, final_lambda = yeo_lambda_iteration(value_column) 
                st.write(f"Iterations: {iterations}")
                st.write(f"Final Lambda Value: {round(final_lambda, 3)}")
        with col2:
            st.write("Please select stationarity test")
            stationarity_test = st.selectbox("Stationarity Test", ["Augmented Dickey-Fuller", "Kwiatkowski-Phillips-Schmidt-Shin"]) 
            if stationarity_test == "Augmented Dickey-Fuller":
                diffed_data, d = iterative_adf(pd.Series(transformed_data.flatten()))
                st.write(f"Number of Differencing (d): {d}")
            elif stationarity_test == "Kwiatkowski-Phillips-Schmidt-Shin":
                diffed_data, d = iterative_kpss(pd.Series(transformed_data.flatten()))
                st.write(f"Number of Differencing (d): {d}")
    
    st.markdown(" ")
    with st.container(border=True):
        st.subheader("📉 ARIMA Model")
        st.write("ARIMA Order Selection")
      
        # Calculate ACF values
        acf_values = sm.tsa.acf(diffed_data, nlags=20)
        acf_df = pd.DataFrame({'Lag': range(len(acf_values)), 'ACF': acf_values})
   
        # Determine significance (values outside the confidence interval)
        confidence_interval = 1.96 / np.sqrt(len(diffed_data))
        acf_df['Significant'] = np.abs(acf_df['ACF']) > confidence_interval
   
        # Display the table
        st.table(acf_df)
    
        # Highlight significant ACF values
        st.write("Significant ACF values are highlighted in the table above.")
        
        # Plot ACF of differenced data
        fig, ax = plt.subplots()
        sm.graphics.tsa.plot_acf(diffed_data, ax=ax)
        st.pyplot(fig)
    
elif selected == "SARIMA":
    st.title("❄️ SARIMA Model")
    st.image(".streamlit/Border_H.png", use_column_width=True)
    st.write("This model is used to forecast time series data with seasonal components.")
    
elif selected == "X-11":
    st.title("💫 X-11 Model")
    st.image(".streamlit/Border_H.png", use_column_width=True)
    st.write("This model is used to decompose time series data into trend, seasonal, and irregular components.")


