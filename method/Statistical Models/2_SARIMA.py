import numpy as np
import pandas as pd
import pmdarima as pm
import streamlit as st
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt

from streamlit_option_menu import option_menu
from streamlit_extras.stateful_button import button
from itertools import product

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
def significant_order_detector(data, nlags=10, conf_level=0.05, m=1, for_model_tentative=False):
    """
    Identifies significant AR (p), MA (q), seasonal AR (P), and seasonal MA (Q) orders 
    based on ACF and PACF with confidence intervals that account for widening in ACF.

    Parameters:
    - data: The transformed time series data.
    - nlags: Number of lags to consider in ACF and PACF (default is 10).
    - conf_level: Confidence level (default is 0.05 for a 95% confidence interval).
    - m: Seasonal period.
    - for_model_tentative: If True, return only the longest consecutive sequence of significant lags.

    Returns:
    - A tuple with lists of significant p, q, P, and Q values.
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
    
    # Identify significant lags for q (ACF) and p (PACF)
    significant_q = [
        i for i in range(1, len(acf_vals))
        if abs(acf_vals[i]) > acf_confint[i]
    ]
    significant_p = [
        i for i in range(1, len(pacf_vals))
        if abs(pacf_vals[i]) > pacf_confint
    ]
    
    # Default to [0, 1] if no significant lags found
    if not significant_q:
        significant_q = [0, 1]
    else:
        significant_q.insert(0, 0)
    if not significant_p:
        significant_p = [0, 1]
    else:
        significant_p.insert(0, 0)
    
    # Identify significant seasonal lags for P (PACF) and Q (ACF) at multiples of m
    significant_P = [
        i for i in range(m, len(pacf_vals), m)
        if abs(pacf_vals[i]) > pacf_confint
    ]
    significant_Q = [
        i for i in range(m, len(acf_vals), m)
        if abs(acf_vals[i]) > acf_confint[i]
    ]
    
    # Default to [0, 1] if no significant seasonal lags found
    if not significant_P:
        significant_P = [0, 1]
    else:
        significant_P.insert(0, 0)
    if not significant_Q:
        significant_Q = [0, 1]
    else:
        significant_Q.insert(0, 0)

    # Select the longest consecutive sequence starting at m if for_model_tentative is True
    if for_model_tentative:
        def longest_consecutive_starting_at_zero(sequence):
            longest = []
            for i in sequence:
                if longest and i != longest[-1] + 1:
                    break  # Stop if the sequence breaks
                longest.append(i)
            return longest

        significant_p = [0] + longest_consecutive_starting_at_zero(significant_p[1:])
        significant_q = [0] + longest_consecutive_starting_at_zero(significant_q[1:])
        significant_P = [0] + longest_consecutive_starting_at_zero(significant_P[1:])
        significant_Q = [0] + longest_consecutive_starting_at_zero(significant_Q[1:])
    
    return significant_p, significant_q, significant_P, significant_Q

# Generate Combination
def generate_combination(p, d, q, P, D, Q):
    """
    Generates a list of all possible combinations of p, d, q, P, D, and Q values.

    Parameters:
    - p: List of significant AR (p) values.
    - d: Number of non-seasonal differences applied.
    - q: List of significant MA (q) values.
    - P: List of significant seasonal AR (P) values.
    - D: Number of seasonal differences applied.
    - Q: List of significant seasonal MA (Q) values.

    Returns:
    - combinations: List of all possible combinations of (p, d, q) and (P, D, Q) values.
    """
    # Generate all combinations of (p, d, q) and (P, D, Q)
    combinations = [
        ((p_val, d, q_val), (P_val, D, Q_val))
        for p_val, q_val, P_val, Q_val in product(p, q, P, Q)
    ]
    
    return combinations

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
        _, significant_q, _, _ = significant_order_detector(final_data, nlags=4 * p, conf_level=conf_level)

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
        _, significant_q, _, _ = significant_order_detector(final_data, nlags=4, conf_level=conf_level)
        
        # Check if first four non-seasonal lags (1, 2, 3, 4) are significant
        non_seasonal_lags = [1, 2, 3, 4]
        if all(lag in significant_q for lag in non_seasonal_lags):
            final_data = np.diff(final_data, n=1)
            d += 1
        else:
            break

    return final_data, d, D

# White Noise Test Function for Residuals
def white_noise_residual_test(model, significance=0.05, method="Ljung-Box", lags=None):
    """
    Performs the Ljung-Box or Box-Pierce test on the residuals of a model to check for white noise.

    Parameters:
    - model: A fitted model object with residuals accessible via `model.resid`.
    - significance: The significance level for the test (default is 0.05).
    - method: The method for the test, either "Ljung-Box" or "Box-Pierce".
    - lags: Optional; number of lags to test for autocorrelation. Default is log(T) based on Harvey (1989).

    Returns:
    - DataFrame: Test statistic, p-value, and white noise interpretation for the residuals.
    """
    data = model.resid
    if lags is None:
        lags = int(np.log(len(data)))  # Default to log(T) based on Harvey (1989)

    if method == "Ljung-Box":
        test_results = acorr_ljungbox(data, lags=lags, return_df=True, boxpierce=False)
        q_score = test_results['lb_stat'].iloc[0]  # For the first lag
        q_pvalue = test_results['lb_pvalue'].iloc[0]
    elif method == "Box-Pierce":
        test_results = acorr_ljungbox(data, lags=lags, return_df=True, boxpierce=True)
        q_score = test_results['bp_stat'].iloc[0]  # For the first lag
        q_pvalue = test_results['bp_pvalue'].iloc[0]
    else:
        raise ValueError("Method must be either 'Ljung-Box' or 'Box-Pierce'.")

    result = "White Noise" if q_pvalue > significance else "Not White Noise"
    return pd.DataFrame({
        "Statistic": [q_score],
        "p-value": [q_pvalue],
        "Result": [result]
    })

def arima_extract_parameters(model, significance=0.05):
    # Extract parameters and p-values, omitting the last row
    params = pd.Series(model.params[:-1]) if isinstance(model.params, pd.Series) else pd.Series(model.params[:-1], index=[f"param_{i}" for i in range(len(model.params[:-1]))])
    pvalues = pd.Series(model.pvalues[:-1]) if isinstance(model.pvalues, pd.Series) else pd.Series(model.pvalues[:-1], index=params.index)
    
    # Create a DataFrame for results
    results = pd.DataFrame({
        "Component": params.index,
        "Coefficient": params.values,
        "P-Value": pvalues.values
    })
    
    # Convert component names to user-friendly format dynamically
    def format_component_name(name):
        parts = name.split('.')
        if len(parts) == 2:
            prefix = parts[0].upper()  # Left part (ar, ma, sar, etc.)
            order = parts[1].replace('L', '')  # Right part, removing 'L' and taking the number
            return f"{prefix}({order})"
        return name  # Return as-is if it doesn't match expected format
    
    results["Component"] = results["Component"].apply(format_component_name)
    
    # Determine significance based on the p-value
    results["Significance"] = results["P-Value"].apply(lambda x: "Significant" if x < significance else "Not Significant")
    
    return results

# Normality Test Function for Residuals
def normality_residual_test(model, significance=0.05, method="Jarque-Bera"):
    """
    Performs a normality test on the residuals of a model to check for normality.

    Parameters:
    - model: A fitted model object (e.g., from ARIMA) with residuals accessible via `model.resid`.
    - significance: The significance level for the test (default is 0.05).
    - method: The method for the normality test. Options are 'jb' (Jarque-Bera), 
              'shapiro' (Shapiro-Wilk), 'kstest' (Kolmogorov-Smirnov), 
              'anderson' (Anderson-Darling), or 'lilliefors' (Lilliefors test).

    Returns:
    - DataFrame: Containing the test statistic, p-value (if available), and interpretation.
    """
    data = model.resid
    data = (data - np.mean(data)) / np.std(data)
    test_results = {}
    
    if method == "Jarque-Bera":
        score, p_value = model.test_normality(method='jarquebera')[0, 0], model.test_normality(method='jarquebera')[0, 1]
        result = "Normal" if p_value > significance else "Not Normal"
        test_results = {"Statistic": score, "p-value": p_value, "Result": result}
    
    elif method == "Shapiro-Wilk":
        score, p_value = stats.shapiro(data)
        result = "Normal" if p_value > significance else "Not Normal"
        test_results = {"Statistic": score, "p-value": p_value, "Result": result}
    
    elif method == "Kolmogorov-Smirnov":
        score, p_value = stats.kstest(data, 'norm')
        result = "Normal" if p_value > significance else "Not Normal"
        test_results = {"Statistic": score, "p-value": p_value, "Result": result}
    
    elif method == "Anderson-Darling":
        test_result = stats.anderson(data, dist='norm')
        score = test_result.statistic
        critical_value = test_result.critical_values[2]  # For 5% significance
        result = "Normal" if score < critical_value else "Not Normal"
        test_results = {
            "Statistic": score,
            "Critical Value (5%)": critical_value,
            "Result": result
        }
    
    elif method == "Lilliefors":
        try:
            from statsmodels.stats.diagnostic import lilliefors
            score, p_value = lilliefors(data, dist='norm')
            result = "Normal" if p_value > significance else "Not Normal"
            test_results = {"Statistic": score, "p-value": p_value, "Result": result}
        except ImportError:
            raise ImportError("Lilliefors test requires `statsmodels` version supporting `lilliefors`.")

    else:
        raise ValueError(f"Unsupported method '{method}'. Choose from 'jb', 'shapiro', 'kstest', 'anderson', or 'lilliefors'.")
    
    return pd.DataFrame([test_results])

def arima_model_mass_fit(data, combinations, m=1, white_noise_method="Ljung-Box", normality_method="Jarque-Bera", sign=0.05):
    summary_rows = []

    for (p, d, q), (P, D, Q) in combinations:
        # Fit the SARIMA model
        try:
            model = sm.tsa.SARIMAX(data, order=(p, d, q), seasonal_order=(P, D, Q, m)).fit()
        except Exception as e:
            print(f"Model SARIMA({p},{d},{q})x({P},{D},{Q},{m}) failed to fit: {e}")
            continue  # Skip to the next combination if fitting fails

        # Get model name
        model_name = f"SARIMA({p},{d},{q})x({P},{D},{Q},{m})"
        
        # 1. Component Significance
        components_df = arima_extract_parameters(model, significance=sign)
        if components_df['Significance'].eq("Significant").all():
            component_significance = "All Significant"
        elif components_df['Significance'].eq("Not Significant").all():
            component_significance = "Insignificant"
        else:
            component_significance = "Partially Significant"
        
        # 2. Residual Normality
        normality_df = normality_residual_test(model, significance=sign, method=normality_method)
        residual_normality = normality_df['Result'].iloc[0]
        
        # 3. White Noise Test
        white_noise_df = white_noise_residual_test(model, significance=sign, method=white_noise_method)
        white_noise_result = white_noise_df['Result'].iloc[0]
        
        # 4. Model Selection Criteria (AIC, BIC, HQIC)
        aic = model.aic
        bic = model.bic
        hqic = model.hqic
        
        # Append the row data to the summary
        summary_rows.append({
            "Model": model_name,
            "Component Significance": component_significance,
            "Residual Normality": residual_normality,
            "White Noise": white_noise_result,
            "AIC": aic,
            "BIC": bic,
            "HQIC": hqic
        })
    
    # Convert the list of rows to a DataFrame
    summary_df = pd.DataFrame(summary_rows)
    return summary_df

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
            
            if d is not None and D is not None:
                continue_tentative = button("Continue to ARIMA Tentative Order", key="button3", help="Continue to ARIMA Tentative Order", icon="🌫️")
            passed_data = transformed_data.flatten()
            for i in range (D):
                passed_data = np.diff(passed_data, n=m)
            for i in range (d):
                passed_data = np.diff(passed_data)
            
        if 'continue_tentative':
             with st.container(border=True):
                st.subheader("🌫️ ARIMA Tentative Model")
                st.markdown("""
                    <div style="text-align: justify;">
                    A tentative ARIMA model with order combination grid search involves testing various combinations of ARIMA orders (p, d, q) to identify the best-fitting model for a time series. The goal is to automate the selection of the best parameters by running a search across possible values for p (autoregressive terms), d (differencing), and q (moving average terms), evaluating each model based on a performance criterion like AIC (Akaike Information Criterion). 
                    </div>
                    <br>
                    <div style="text-align: justify;">
                    The advantage is a more systematic, potentially optimal choice of ARIMA parameters, while a disadvantage is the high computational cost, particularly for large datasets or long time series.
                    </div>
                    """, unsafe_allow_html=True)  
                st.markdown("")
                st.image(".streamlit/Border_H.png", use_column_width=True)
                st.markdown("#### ARIMA Order Selection")
                
                autosearch = st.checkbox("Auto Order", value=True, help="Automatically Pass the order of ARIMA component to next phase based on ACF and PACF.")
                
                # If autosearch is enabled, find the AR and MA orders
                if autosearch:
                    p_order, q_order, P_order, Q_order = significant_order_detector(passed_data, for_model_tentative=True, m=m)
                    st.success(f"Found significant AR (p) orders: {p_order}")
                    st.success(f"Found significant MA (q) orders: {q_order}")
                    st.success(f"Found significant Seasonal AR (P) orders: {P_order}")
                    st.success(f"Found significant Seasonal MA (Q) orders: {Q_order}")
                    
                # If autosearch is disabled, allow manual input of AR and MA orders
                else:
                    p, q = None, None
                    
                    # Input for AR order
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown('Please select the AR() order that want to be checked manually. Example input: "1, 2, 3" without quotes')
                        col1a, col2a = st.columns(2)
                        with col1a:
                            p = st.text_input("AR (p)", value="0")
                            p_order = [int(i) for i in p.split(",") if i.strip().isdigit()]
                            if len(p) == 0:
                                p_order = [0, 1]
                                st.warning("No input detected. Defaulting to [0, 1].")
                            else:
                                st.info(f"Order Inputted : {p_order}")
                        with col2a:
                            P = st.text_input("Seasonal AR (P)", value="0")
                            P_order = [int(i) for i in P.split(",") if i.strip().isdigit()]
                            if len(P) == 0:
                                P_order = [0, 1]
                                st.warning("No input detected. Defaulting to [0, 1].")
                            else:
                                st.info(f"Order Inputted : {P_order}")
                            
                    with col2:
                        with st.container(border=True):
                            # Calculate PACF values
                            fig_pacf = plot_pacf(passed_data, lags=4*m+1)
                            fig_pacf.tight_layout()
                            st.pyplot(fig_pacf)
                    
                    # Input for MA order
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown('Please select the MA() order that want to be checked manually. Example input: "1, 2, 3" without quotes')
                        col1b, col2b = st.columns(2)
                        with col1b:
                            q = st.text_input("MA (q)", value="0")
                            q_order = [int(i) for i in q.split(",") if i.strip().isdigit()]
                            if len(q) == 0:
                                q_order = [0, 1]
                                st.warning("No input detected. Defaulting to [0, 1].")
                            else:
                                st.info(f"Order Inputted : {q_order}")
                        with col2b:
                            Q = st.text_input("Seasonal MA (Q)", value="0")
                            Q_order = [int(i) for i in Q.split(",") if i.strip().isdigit()]
                            if len(Q) == 0:
                                Q_order = [0, 1]
                                st.warning("No input detected. Defaulting to [0, 1].")
                            else:
                                st.info(f"Order Inputted : {Q_order}")
                            
                    with col2:
                        with st.container(border=True):
                            # Calculate ACF values
                            fig_acf = plot_acf(passed_data, lags=20)
                            fig_acf.tight_layout()
                            st.pyplot(fig_acf)
                
                # Generate combinations based on AR and MA orders
                if p_order != None and q_order != None and P_order != None and Q_order != None:
                    combinations = generate_combination(p_order, d, q_order, P_order, D, Q_order)
                    st.success(f"Generated {len(combinations)} combinations for ARIMA model")
                    if len(combinations) > 30:
                        st.warning("The number of combinations is large and may take some time to process. Please consider reducing the number of orders using manual input.")
                # Button to continue to ARIMA Model Fit
                continue_modelfit = button("Continue to ARIMA Model Fit", key="button4", help="Continue to ARIMA Model Fit", icon="🌞")
                st.markdown("")
                
        # Continue to ARIMA Model Fit
        if continue_modelfit and continue_tentative:
            with st.container(border=True):
                st.subheader("🌞 ARIMA Model Diagnostic & Mass Fit")
                st.markdown("""
                    <div style="text-align: justify;">
                    ARIMA model diagnostics assess the fit and reliability of a model by examining key criteria. Component significance checks if each parameter (p, d, q) adds value; non-significant terms can be removed to streamline the model. White noise residuals indicate that errors have no patterns and are centered around zero, confirming that the model captures all correlations. Normality of residuals suggests that errors follow a normal distribution, while non-normal residuals may indicate a need for data transformation or model adjustment.
                    </div>
                    <br>
                    <div style="text-align: justify;">
                    Model comparison criteria—AIC, BIC, and HQIC allow for a balance between fit and complexity. Lower values in these criteria indicate a better-fitting model that doesn’t overfit. Together, these diagnostics help ensure the ARIMA model is both parsimonious and accurate. In this section, you can set the diagnostic settings for the ARIMA model, launch mass fit model, and view the results.
                    </div>
                    """, unsafe_allow_html=True)  
                st.markdown("")
                st.image(".streamlit/Border_H.png", use_column_width=True)
                
                # ARIMA Model Diagnostic Settings
                st.markdown("### ⚙️ ARIMA Model Diagnostic Settings")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    # Select Normality Test
                    with st.container(border=True):
                        st.markdown("Select Normality Test")
                        normality = st.selectbox("Normality Test", ["Jarque-Bera", "Shapiro-Wilk", "Kolmogorov-Smirnov", "Anderson-Darling", "Lilliefors"])
                with col2:
                    # Select White Noise Test
                    with st.container(border=True): 
                        st.markdown("Select White Noise Test")
                        white_noise = st.selectbox("White Noise Test", ["Ljung-Box", "Box-Pierce"])
                with col3:
                    # Select Significance Level
                    with st.container(border=True):
                        st.markdown("Select Significance Level")
                        sign = st.number_input("Significance Level", value=0.05, min_value=0.01, max_value=1.0, step=0.01)

                # Button to continue to Mass Fit Model
                continuemassfit = button("Begin Mass Fit Model", key="button5", help="Apply Setting and Run Mass-Fit Model", icon="🔄")
                st.image(".streamlit/Border_H.png", use_column_width=True)
                
                # Continue to Mass Fit Model
                if continuemassfit:
                    st.markdown("### 📋 ARIMA Model List")
                    st.write("The table below shows the summary of ARIMA models fitted with different combinations of AR, I, and MA orders.")
                    with st.spinner(f"Fitting {len(combinations)} ARIMA models..."):
                        model_collection = arima_model_mass_fit(pd.Series(transformed_data.flatten(), index=range(1, len(transformed_data.flatten()) + 1)), 
                                                                combinations, normality_method=normality, white_noise_method=white_noise,
                                                                sign=sign, m=m)
                        st.dataframe(model_collection, use_container_width=True, selection_mode='single-row', key='arima_model_mass_fit')
                    st.image(".streamlit/Border_H.png", use_column_width=True)
        
        st.markdown(" ")
        
        
        

