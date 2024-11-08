import numpy as np
import pandas as pd
import pmdarima as pm
import streamlit as st
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

from streamlit_option_menu import option_menu
from streamlit_extras.stateful_button import button

from scipy.stats import norm
from sklearn.preprocessing import PowerTransformer
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

# Check if data is uploaded
if 'filtered_df' not in st.session_state:
    st.error("Please upload a file in the input data page.")
    st.stop()
else:
    data = st.session_state['filtered_df']

# Metrics Function
def metrics(data, fitted_data):
    mse = round(np.mean((data - fitted_data) ** 2), 2)
    mae = round(np.mean(np.abs(data - fitted_data)), 2)
    mape = round(np.mean(np.abs((data - fitted_data) / data)) * 100, 2)
    rmse = round(np.sqrt(mse), 2)
    return mse, mae, mape, rmse

# Variance Stationarity Test
def variance_stationarity_transform(data, method='yeo-johnson', lambda_threshold=[0.95, 1.05], iterations_limit=None):
    """
    Applies specified transformation to data iteratively until variance stationarity is achieved,
    or applies log or square root transformation once.

    Parameters:
    - data: numpy array or pandas Series containing the time series data.
    - method: String specifying the transformation method: 'yeo-johnson', 'box-cox', 'log', or 'sqrt'.
    - lambda_threshold: list or tuple with two elements [low, high] specifying
                        the target range for lambda value to achieve variance stationarity (only for Yeo-Johnson or Box-Cox).
    - iterations_limit: Optional integer, maximum number of iterations to prevent infinite loops (only for Yeo-Johnson or Box-Cox).

    Returns:
    - tuple containing:
      - transformed_data: The transformed data.
      - method: The transformation method used.
      - transformers: List of PowerTransformer objects used (only for iterative methods).
      - iterations: Number of iterations performed (only for iterative methods).
      - final_lambda: The final lambda value achieved (only for iterative methods).
      - lambda_df: DataFrame containing iteration and lambda values (only for iterative methods).
    """
    transformed_data = data.values.reshape(-1, 1) if hasattr(data, 'values') else data.reshape(-1, 1)
    transformers = []
    lambda_values = []
    iteration_values = []
    iteration = 0
    final_lambda = None
    lambda_df = None

    # Define the transformation based on the selected method
    if method in ['yeo-johnson', 'box-cox']:
        threshold_low, threshold_high = lambda_threshold
        while True and (iterations_limit is None or iteration < iterations_limit):
            # Check for negative values if using Box-Cox transformation
            if method == 'box-cox' and (transformed_data < 0).any():
                raise ValueError("Box-Cox transformation encountered negative values. "
                                 "Consider using Yeo-Johnson transformation instead.")
            
            # Apply iterative PowerTransformer
            transformer = PowerTransformer(method=method)
            transformer.fit(transformed_data)
            lambda_value = transformer.lambdas_[0]
            final_lambda = lambda_value
            iteration_values.append(iteration)
            lambda_values.append(lambda_value)

            # Check for variance stationarity threshold
            if threshold_low <= lambda_value <= threshold_high:
                break

            transformed_data = transformer.transform(transformed_data)
            transformers.append(transformer)
            iteration += 1

        lambda_df = pd.DataFrame({
            'Iteration': iteration_values,
            'Lambda': lambda_values
        })

    elif method == 'log':
        transformed_data = np.log(data).values if hasattr(data, 'values') else np.log(data)
    
    elif method == 'sqrt':
        transformed_data = np.sqrt(data).values if hasattr(data, 'values') else np.sqrt(data)
    
    else:
        raise ValueError("Invalid method. Choose from 'yeo-johnson', 'box-cox', 'log', or 'sqrt'.")

    # Return results as a tuple for direct unpacking
    return transformed_data, method, transformers, iteration, final_lambda, lambda_df

# Inverse Transformation
def transform_back(data, result = None):
    """
    Transforms data back to its original form using the inverse transformation 
    specified in the result tuple.

    Parameters:
    - data: The transformed data to revert back to its original form.
    - result: Tuple output from variance_stationarity_transform function,
              containing the transformation method, transformers, and other details.

    Returns:
    - original_data: The data transformed back to its original form.
    
    Raises:
    - ValueError: If the result does not have a recognized method.
    """
    if result is None:
        original_data = data
    else:
        # Unpack the necessary values from the result tuple
        _, method, transformers, _, _, _ = result

        # Initialize original_data with the transformed data
        original_data = data

        if method in ['yeo-johnson', 'box-cox']:
            # Ensure transformers list is present for iterative methods
            if transformers is None:
                raise ValueError("Transformers not found in result; ensure result is from variance_stationarity_transform.")
            
            # Apply inverse transformations in reverse order for iterative methods
            for transformer in reversed(transformers):
                original_data = transformer.inverse_transform(original_data)

        elif method == 'log':
            # Inverse of log transformation is exponentiation
            original_data = np.exp(original_data)

        elif method == 'sqrt':
            # Inverse of square root transformation is squaring
            original_data = np.square(original_data)

        else:
            raise ValueError("Unrecognized method in result. Ensure result is from variance_stationarity_transform.")

    return original_data

# Order Finder
def significant_order_detector(data, nlags=10, conf_level=0.05):
    """
    Identifies significant AR (p) and MA (q) orders based on ACF and PACF with 
    confidence intervals that account for widening in ACF.

    Parameters:
    - data: The transformed time series data.
    - nlags: Number of lags to consider in ACF and PACF (default is 40).
    - conf_level: Confidence level (default is 0.05 for a 95% confidence interval).

    Returns:
    - A tuple with lists of significant p and q values.
    """
    N = len(data)
    z_critical = norm.ppf(1 - conf_level / 2)
    
    # Calculate ACF and PACF values
    acf_vals = acf(data, nlags=nlags, fft=False)
    pacf_vals = pacf(data, nlags=nlags, method='ywm')

    # Calculate confidence intervals for ACF with widening effect
    acf_confint = [z_critical / np.sqrt(N - k) for k in range(len(acf_vals))]

    # Fixed confidence interval for PACF
    pacf_confint = z_critical / np.sqrt(N)
    
    # Identify significant lags for q (ACF) where values exceed confidence intervals
    significant_q = [
        i for i in range(1, len(acf_vals))
        if abs(acf_vals[i]) > acf_confint[i]
    ]
    
    # Default to [0, 1] if no significant lags found for q
    if not significant_q:
        significant_q = [0, 1]
    else:
        significant_q.insert(0, 0)
    
    # Identify significant lags for p (PACF) where values exceed the fixed confidence interval
    significant_p = [
        i for i in range(1, len(pacf_vals))
        if abs(pacf_vals[i]) > pacf_confint
    ]
    
    # Default to [0, 1] if no significant lags found for p
    if not significant_p:
        significant_p = [0, 1]
    else:
        significant_p.insert(0, 0)
    
    return significant_p, significant_q

# Check Stationarity using ACF significance
def acf_stationarity(data, p, conf_level=0.05):
    """
    Checks for seasonal and non-seasonal stationarity using ACF and applies differencing as necessary.

    Parameters:
    - data: The original time series data.
    - p: Seasonal period (e.g., 12 for monthly data).
    - nlags: Number of lags to consider for ACF in order_finder (default is 10).
    - conf_level: Confidence level for determining significant ACF values (default is 0.05 for a 95% confidence interval).

    Returns:
    - final_data: The differenced time series data after achieving stationarity.
    - d: Non-seasonal differencing order.
    - D: Seasonal differencing order.
    """
    # Flatten data to 1D array if not already
    if not isinstance(data, np.ndarray):
        data = np.array(data).flatten()
    else:
        data = data.flatten()
    
    # Initialize differencing orders
    d, D = 0, 0
    final_data = data

    # Seasonal differencing
    while True:
        # Use order_finder to find significant seasonal lags
        _, significant_q = significant_order_detector(final_data, nlags=4 * p, conf_level=conf_level)
        
        # Check if first four seasonal lags (p, 2p, 3p, 4p) are significant
        seasonal_lags = [p, 2 * p, 3 * p, 4 * p]
        if all(lag in significant_q for lag in seasonal_lags):
            final_data = np.diff(final_data, n=p)
            print(f"Final Data: {final_data}")
            D += 1
        else:
            break

    # Non-seasonal differencing
    while True:
        # Use order_finder to find significant non-seasonal lags
        _, significant_q = significant_order_detector(final_data, nlags=4, conf_level=conf_level)
        
        # Check if first four non-seasonal lags (1, 2, 3, 4) are significant
        non_seasonal_lags = [1, 2, 3, 4]
        if all(lag in significant_q for lag in non_seasonal_lags):
            final_data = np.diff(final_data, n=1)
            d += 1
        else:
            break

    return final_data, d, D

date_column = data.iloc[:, 0]
value_column = data.iloc[:, 1]

# =================================================================================================

# SARIMA Process
st.title("❄️ SARIMA Model")
st.image(".streamlit/Border_H.png", use_column_width=True)
st.markdown("""
    <div style="text-align: justify;">
    ARIMA (Auto-Regressive Integrated Moving Average) is a widely-used forecasting method for time series data, focusing on non-seasonal patterns. It combines three components: autoregression (AR), differencing (I), and moving averages (MA) to model temporal dependencies in stationary data.
    </div>
    <br>
    <div style="text-align: justify;">
    An advantage of ARIMA is its flexibility in capturing short-term correlations in the data, making it valuable for series with trends or fluctuations without clear seasonal cycles. However, ARIMA requires careful tuning and assumes linear relationships, which may limit its accuracy in more complex data patterns.
    </div>
    """, unsafe_allow_html=True)  
st.markdown("")

begin = button("Begin", key="begin2", help="Start SARIMA model fitting", icon="🏃‍♂️")

if begin:
    autosarima = st.checkbox("Use AutoSARIMA", value=False, help="Automatically determine optimal SARIMA order parameters based on AIC")
    m = st.number_input("Seasonal Period (m)", min_value=1, value=12, help="Seasonal period for the time series data, Please refer to data exploration page for the seasonal period.")
    if autosarima:
        # Use Auto ARIMA
        with st.container(border=True):
            st.warning("Auto SARIMA will only choose the model based on AIC and will ignore variance stationarity checks, assumptions, component significance, and model diagnostics.")
            auto_sarima_result = pm.auto_arima(value_column, seasonal=True, stepwise=True, suppress_warnings=True)
            auto_sarima_order = auto_sarima_result.order
            best_model_autosarima = f"ARIMA{auto_sarima_result}"
            transformed_data = np.array(value_column)
            transformation_result = None 
            
    else:
        # Chech Variance Stationarity
        with st.container(border=True):
            st.subheader("Variance Stationary Check")
            st.markdown("The first step in fitting an ARIMA model is to ensure the time series data is variance stationary. This is crucial for ARIMA to model the data effectively.")
            
            st.image(".streamlit/Border_H.png", use_column_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### Please Select Transformation Method")
                transformation_method = st.selectbox("Transformation Method", ["Iterative Box-Cox", "Iterative Yeo-Johnson", 
                                                                                "Log", "Square Root"])
                
                # If transformation method is iterative, show additional options
                if transformation_method == "Iterative Box-Cox" or transformation_method == "Iterative Yeo-Johnson":
                    if transformation_method == "Iterative Box-Cox":
                        try:
                            transformation_result = variance_stationarity_transform(value_column, method='box-cox')
                            transformed_data, _, transformers, iterations, final_lambda, lambda_df = transformation_result
                        except ValueError as e:
                            st.warning(f"{e}")
                            st.stop()
                    elif transformation_method == "Iterative Yeo-Johnson":
                        transformation_result = variance_stationarity_transform(value_column, method='yeo-johnson')
                        transformed_data, _, transformers, iterations, final_lambda, lambda_df = transformation_result
                    
                    st.write(f"Iterations: {iterations} | Final Lambda Value: {round(final_lambda, 3)}")
                    st.success("Variance Stationarity Achieved")

                # If transformation method is Log or Square Root, apply transformation once
                if transformation_method == "Log" or transformation_method == "Square Root":
                    if transformation_method == "Log":
                        transformation_result = variance_stationarity_transform(value_column, method='log')
                        transformed_data, _, _, _, _, _ = transformation_result
                        st.success("Log Transformation Successful")
                    elif transformation_method == "Square Root":
                        transformation_result = variance_stationarity_transform(value_column, method='sqrt')
                        transformed_data, _, _, _, _, _ = transformation_result
                        st.success("Square Root Transformation Successful")
            
            with col2:
                st.markdown("##### Variance Stationarity Result")
                # If transformation is not successful, stop the process
                if 'transformed_data' not in locals():
                    st.error("Transformation was not successful. Please check your data and transformation method.")
                    st.stop()
                
                if transformation_method == "Iterative Box-Cox" or transformation_method == "Iterative Yeo-Johnson":
                    with st.expander("Show Transformation History", expanded=False):
                        st.dataframe(lambda_df.set_index('Iteration'), use_container_width=True, width=200)
                # Display transformed data and plot
                with st.expander("Show Transformed Data", expanded=False):
                    st.dataframe(pd.DataFrame({"Date": date_column, "Transformed Data": transformed_data.flatten()}).set_index("Date"), 
                                use_container_width=True, width=200)  
                
                with st.expander("Show Transformed Data Plot", expanded=False):
                    plt.figure(figsize=(10, 5))
                    plt.plot(date_column, transformed_data, label="Transformed Data")
                    plt.xlabel("Date")
                    plt.ylabel("Transformed Data")
                    plt.legend()
                    st.pyplot(plt)      

        
        # Find Seasonal Differencing Order and Regular Differencing Order
        with st.container(border=True):
            st.subheader("Differencing Order Estimation")
            st.markdown("The next step is to estimate the differencing orders for the SARIMA model. This involves determining the number of regular and seasonal differencing terms required to make the data stationary.")
            
            st.image(".streamlit/Border_H.png", use_column_width=True)
            
            st.markdown("##### Select Differencing Order Estimation Method")
            diff_method = st.selectbox("Differencing Order Estimation Method", ["Statistical Test", "ACF Significance", "Manual Input"], 
                                       help="Method to estimate the differencing orders for the SARIMA model.")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### Seasonal Differencing Order")
                if diff_method == "Statistical Test":
                    ndifftest = st.selectbox("Differencing Test", ["OCSB", "CH"], 
                                             help="OCSB = Osborn, Chui, Smith, and Birchenhall Test; CH = Canova and Hansen Test").lower()
                    D = pm.arima.nsdiffs(transformed_data, m, test=ndifftest)
                if diff_method == "ACF Significance":
                    diffed_data, d, D = acf_stationarity(transformed_data, m)
                    st.info(f"4 First seasonal lag (lag {m}, {2*m}, {3*m}, and {4*m}) already not significant at {D}x seasonal differencing") 
                    st.success(f"D set to {D}")
                if diff_method == "Manual Input":
                    D = st.number_input("Seasonal Differencing Order (D)", min_value=0, max_value=2, value=0)
                    current_data = transformed_data.flatten()
                    for i in range(D):
                        current_data = np.diff(current_data, n=m)
                    with st.expander("Show ACF Plot After Seasonal Diff", expanded=False):
                        plt.figure(figsize=(10, 5))
                        plot_acf(current_data, lags=40)
                        plt.xlabel("Lag")
                        plt.ylabel("Autocorrelation")
                        st.pyplot(plt)
                    with st.expander("Show Data Plot After Seasonal Diff", expanded=False):
                        plt.figure(figsize=(10, 5))
                        plt.plot(date_column[-len(current_data):], current_data, label="Seasonal Diffed Data")
                        plt.xlabel("Date")
                        plt.ylabel("Transformed Data")
                        plt.legend()
                        st.pyplot(plt)
                st.write(f"Seasonal Differencing Order (D): {D}")
            
            with col2:
                st.markdown("##### Regular Differencing Order")
                if diff_method == "Statistical Test":
                    difftest = st.selectbox("Differencing Test", ["ADF", "KPSS"],
                                            help="ADF = Augmented Dickey Fuller Test; KPSS = Kwiatkowski–Phillips–Schmidt–Shin Test").lower()
                    d = pm.arima.ndiffs(transformed_data, test=difftest)
                if diff_method == "ACF Significance":
                    st.info(f"4 First non-seasonal lag (lag 1,2,3,4) already not significant at {d}x non-seasonal differencing") 
                    st.success(f"d set to {d}")
                if diff_method == "Manual Input":
                    d = st.number_input("Regular Differencing Order (d)", min_value=0, max_value=2, value=0)
                    for i in range(d):
                        current_data = np.diff(current_data)
                    with st.expander("Show ACF Plot After Regular Diff", expanded=False):
                        plt.figure(figsize=(10, 5))
                        plot_acf(current_data, lags=40)
                        plt.xlabel("Lag")
                        plt.ylabel("Autocorrelation")
                        st.pyplot(plt)
                    with st.expander("Show Data Plot After Regular Diff", expanded=False):
                        plt.figure(figsize=(10, 5))
                        plt.plot(date_column[-len(current_data):], current_data, label="Regular Diffed Data")
                        plt.xlabel("Date")
                        plt.ylabel("Transformed Data")
                        plt.legend()
                        st.pyplot(plt)
                st.write(f"Regular Differencing Order (d): {d}")
            
            


