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

date_column = data.iloc[:, 0]
value_column = data.iloc[:, 1]

# Metrics Function
def metrics(data, fitted_data):
    mse = round(np.mean((data - fitted_data) ** 2), 2)
    mae = round(np.mean(np.abs(data - fitted_data)), 2)
    mape = round(np.mean(np.abs((data - fitted_data) / data)) * 100, 2)
    rmse = round(np.sqrt(mse), 2)
    return mse, mae, mape, rmse

# Yeo-Johnson Transformation Function
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
    lambda_values = []  # List to store lambda values for each iteration
    iteration_values = []  # List to store iteration numbers

    while True and (iterations_limit is None or iteration < iterations_limit):
        # Create a new PowerTransformer and fit the data
        yj = PowerTransformer(method='yeo-johnson')
        yj.fit(transformed_data)
        lambda_value = yj.lambdas_[0]  # Extract lambda value
        final_lambda = lambda_value

        # Store iteration and lambda value
        iteration_values.append(iteration)
        lambda_values.append(lambda_value)

        # Check if lambda is within the desired threshold for variance stationarity
        if threshold_low <= lambda_value <= threshold_high:
            break  # Exit loop if within threshold without transforming

        # Apply transformation only if the threshold is not met
        transformed_data = yj.transform(transformed_data)
        transformers.append(yj)  # Store the transformer after applying transformation
        iteration += 1  # Update iteration counter

    # Create a DataFrame for iteration and lambda values
    lambda_df = pd.DataFrame({
        'Iteration': iteration_values,
        'Lambda': lambda_values
    })

    return transformed_data, transformers, iteration, final_lambda, lambda_df

# Inverse Yeo-Johnson Transformation
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

# Box-Cox Transformation Function
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
    - final_lambda: The final lambda value achieved.
    - lambda_df: DataFrame containing iteration and lambda values.
    
    Raises:
    - ValueError: If negative values are encountered, as Box-Cox cannot handle negatives.
    """
    transformed_data = data.values.reshape(-1, 1)  # Reshape data for PowerTransformer
    threshold_low, threshold_high = lambda_threshold
    transformers = []  # List to store each PowerTransformer instance
    iteration = 0
    lambda_values = []  # List to store lambda values for each iteration
    iteration_values = []  # List to store iteration numbers

    while True and (iterations_limit is None or iteration < iterations_limit):
        # Check for negative values before applying Box-Cox transformation
        if (transformed_data < 0).any():
            raise ValueError("Box-Cox transformation encountered negative values. "
                             "Consider using Yeo-Johnson transformation instead.")

        # Create a new PowerTransformer with Box-Cox method and fit the data
        bc = PowerTransformer(method='box-cox')
        bc.fit(transformed_data)
        lambda_value = bc.lambdas_[0]  # Extract lambda value
        final_lambda = lambda_value

        # Store iteration and lambda value
        iteration_values.append(iteration)
        lambda_values.append(lambda_value)

        # Check if lambda is within the desired threshold for variance stationarity
        if threshold_low <= lambda_value <= threshold_high:
            print(f"Variance stationarity achieved at iteration {iteration + 1}")
            break  # Exit loop if within threshold without transforming

        # Apply transformation only if the threshold is not met
        transformed_data = bc.transform(transformed_data)
        transformers.append(bc)  # Store the transformer after applying transformation
        iteration += 1  # Update iteration counter

    # Create a DataFrame for iteration and lambda values
    lambda_df = pd.DataFrame({
        'Iteration': iteration_values,
        'Lambda': lambda_values
    })

    return transformed_data, transformers, iteration, final_lambda, lambda_df

# Inverse Box-Cox Transformation
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

# Log Transformation Function
def log_transform(data):
    """
    Applies a log transformation to the data.

    Parameters:
    - data: numpy array or pandas Series containing the time series data.

    Returns:
    - transformed_data: The data after applying the log transformation.
    """
    return np.log(data).values

# Square Root Transformation Function
def sqrt_transform(data):
    """
    Applies a square root transformation to the data.

    Parameters:
    - data: numpy array or pandas Series containing the time series data.

    Returns:
    - transformed_data: The data after applying the square root transformation.
    """
    return np.sqrt(data).values

# ADF Test Function
def iterative_adf(data, threshold=0.05):
    """
    Iteratively apply ADF test and differencing until stationarity is achieved.
    
    Parameters:
    data (pd.Series or list): The time series data to check for stationarity.
    threshold (float): The significance level for the ADF test to consider the data stationary.
    
    Returns:
    diffed_data (pd.Series): Differenced data that is stationary.
    d (int): The number of differences applied.
    iteration_df (pd.DataFrame): DataFrame containing each iteration's diff number, p-value, and stationarity result.
    """
    d = 0
    diffed_data = data
    iteration_records = []
    
    while True:
        # Perform the ADF test
        adf_test = adfuller(diffed_data)
        p_value = adf_test[1]
        
        # Determine if the series is stationary
        result = "Stationary" if p_value < threshold else "Not Stationary"
        
        # Append the iteration details to the records
        iteration_records.append({"Diff": d, "p-value": p_value, "Result": result})
        
        # If stationary, break the loop
        if result == "Stationary":
            break
        
        # If not stationary, difference the data and increase d
        diffed_data = diffed_data.diff().dropna()
        d += 1
    
    # Create the DataFrame from records
    iteration_df = pd.DataFrame(iteration_records).set_index("Diff")
    
    return diffed_data, d, iteration_df

# KPSS Test Function
def iterative_kpss(data, threshold=0.05):
    """
    Iteratively apply KPSS test and differencing until stationarity is achieved.
    
    Parameters:
    data (pd.Series or list): The time series data to check for stationarity.
    threshold (float): The significance level for the KPSS test to consider the data stationary.
    
    Returns:
    diffed_data (pd.Series): Differenced data that is stationary.
    d (int): The number of differences applied.
    iteration_df (pd.DataFrame): DataFrame containing each iteration's diff number, p-value, and stationarity result.
    """
    d = 0
    diffed_data = data
    iteration_records = []
    
    while True:
        # Perform the KPSS test (stationarity is indicated if p-value > threshold)
        kpss_test = kpss(diffed_data, nlags="auto")
        p_value = kpss_test[1]
        
        # Determine if the series is stationary
        result = "Stationary" if p_value > threshold else "Not Stationary"
        
        # Append the iteration details to the records
        iteration_records.append({"Diff": d, "p-value": p_value, "Result": result})
        
        # If stationary, break the loop
        if result == "Stationary":
            break
        
        # If not stationary, difference the data and increase d
        diffed_data = diffed_data.diff().dropna()
        d += 1
    
    # Create the DataFrame from records
    iteration_df = pd.DataFrame(iteration_records).set_index("Diff")
    
    return diffed_data, d, iteration_df

# Order Finder Function for ARIMA Model Selection
def order_finder(diffed_data, nlags=10, conf_level=0.05):
    """
    Identifies significant AR (p) and MA (q) orders based on ACF and PACF with 
    confidence intervals that account for widening in ACF.

    Parameters:
    - diffed_data: The differenced time series data.
    - nlags: Number of lags to consider in ACF and PACF (default is 40).
    - conf_level: Confidence level (default is 0.05 for a 95% confidence interval).

    Returns:
    - A dictionary with lists of significant p and q values.
    """
    N = len(diffed_data)
    z_critical = norm.ppf(1 - conf_level / 2)
    
    # Calculate ACF and PACF values
    acf_vals = acf(diffed_data, nlags=nlags, fft=False, bartlett_confint=True)
    pacf_vals = pacf(diffed_data, nlags=nlags, method='ywm')

    # Calculate confidence intervals for ACF with widening effect
    acf_confint = [z_critical / np.sqrt(N - k) for k in range(len(acf_vals))]

    # Fixed confidence interval for PACF
    pacf_confint = z_critical / np.sqrt(N)
    
    # Identify significant lags for q (ACF) where values exceed confidence intervals
    significant_q = [
        i for i in range(1, len(acf_vals))
        if abs(acf_vals[i]) > acf_confint[i]
    ]
    
    # Add 0 to significant_q if it is empty, default to [0, 1]
    if not significant_q:
        significant_q = [0, 1]
    else:
        significant_q.insert(0, 0)
    
    # Identify significant lags for p (PACF) where values exceed the fixed confidence interval
    significant_p = [
        i for i in range(1, len(pacf_vals))
        if abs(pacf_vals[i]) > pacf_confint
    ]
    
    # Add 0 to significant_p if it is empty, default to [0, 1]
    if not significant_p:
        significant_p = [0, 1]
    else:
        significant_p.insert(0, 0)
    
    def filter_continuous_sequence(significant_lags):
        """
        Filters significant lags to include only continuous sequence numbers.

        Parameters:
        - significant_lags: List of significant lags.

        Returns:
        - continuous_lags: List of continuous sequence numbers from significant lags.
        """
        continuous_lags = []
        for i in range(len(significant_lags)):
            if i == 0 or significant_lags[i] == significant_lags[i - 1] + 1:
                continuous_lags.append(significant_lags[i])
            else:
                break
        return continuous_lags

    # Filter significant_q to include only continuous sequence numbers
    order_q = filter_continuous_sequence(significant_q)

    # Filter significant_p to include only continuous sequence numbers
    order_p = filter_continuous_sequence(significant_p)

    return order_p, order_q

# Generate Combinations Function for ARIMA
def generate_combination(p, d, q):
    """
    Generates a list of all possible combinations of p, d, and q values.

    Parameters:
    - p: List of significant AR (p) values.
    - d: Number of differences applied.
    - q: List of significant MA (q) values.

    Returns:
    - combinations: List of all possible combinations of p, d, and q values.
    """
    combinations = []
    for i in range(len(p)):
        for j in range(len(q)):
            combinations.append((p[i], d, q[j]))
    return combinations

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

# Function to take components and components significance from ARIMA model
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

# Function to loop every order combination and fit ARIMA model
def arima_model_mass_fit(data, combination, white_noise_method="Ljung-Box", normality_method="Jarque-Bera", sign=0.05):
    summary_rows = []

    for p, d, q in combination:
        # Fit the ARIMA model
        try:
            model = sm.tsa.ARIMA(data, order=(p, d, q)).fit()
        except Exception as e:
            print(f"Model ARIMA({p},{d},{q}) failed to fit: {e}")
            continue  # Skip to the next combination if fitting fails

        # Get model name
        model_name = f"ARIMA({p},{d},{q})"
        
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

def select_best_arima_model(df):
    # Step 1: Filter models with "All Significant", "Normal", "White Noise", and find the best AIC
    condition_1 = (df['Component Significance'] == 'All Significant') & \
                  (df['Residual Normality'] == 'Normal') & \
                  (df['White Noise'] == 'White Noise')
    df_filtered_1 = df[condition_1]
    if not df_filtered_1.empty:
        return df_filtered_1.loc[df_filtered_1['AIC'].idxmin()]

    # Step 2: If no model fulfills all conditions, omit "Normal" condition and select by best AIC
    condition_2 = (df['Component Significance'] == 'All Significant') & \
                  (df['White Noise'] == 'White Noise')
    df_filtered_2 = df[condition_2]
    if not df_filtered_2.empty:
        return df_filtered_2.loc[df_filtered_2['AIC'].idxmin()]

    # Step 3: If still none, omit the "White Noise" condition as well
    condition_3 = (df['Component Significance'] == 'All Significant')
    df_filtered_3 = df[condition_3]
    if not df_filtered_3.empty:
        return df_filtered_3.loc[df_filtered_3['AIC'].idxmin()]

    # Step 4: If none are "All Significant", select directly the model with the best AIC
    return df.loc[df['AIC'].idxmin()]

# Core Model Code
# ARIMA Model
if selected == "ARIMA":
    st.title("📉 ARIMA Model")
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
    
    begin = button("Begin", key="begin", help="Start ARIMA model fitting", icon="🏃‍♂️")
    
    if begin:
        autoarima = st.checkbox("Use Auto ARIMA", value=False, help="Automatically select the best ARIMA model based on AIC.")
        if autoarima:
            with st.container(border=True):
                st.warning("Auto ARIMA will only choose the model based on AIC and will ignore variance stationarity checks, assumptions, component significance, and model diagnostics.")
                auto_arima_result = pm.auto_arima(value_column, seasonal=False, stepwise=True, suppress_warnings=True)
                auto_arima_order = auto_arima_result.order
                best_model_autoarima = f"ARIMA{auto_arima_order}"
                transformed_data = np.array(value_column)
                transformation_method = None
        else:
            # Check variance stationarity using Box-Cox transformation
            with st.container(border=True):
                st.subheader("📊 ARIMA Stationarity Check")
                st.markdown("""
                    <div style="text-align: justify;">
                    In ARIMA modeling, a stationary time series—one with constant mean and variance over time—is a core requirement to ensure reliable forecasting. Stationarity is typically checked by examining if the series has a constant mean and variance, as well as autocorrelations that decay over time. 
                    </div>
                    <br>
                    <div style="text-align: justify;">
                    If a series is non-stationary, transformations like differencing can stabilize the mean, while techniques such as logarithmic transformations and specialized box-cox or yeo-johnson help stabilize variance. Ensuring stationarity improves model accuracy by aligning the data with ARIMA’s assumptions of temporal consistency.
                    </div>
                    """, unsafe_allow_html=True)  
                st.markdown("")
                st.image(".streamlit/Border_H.png", use_column_width=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("<h4>Variance Stationarity</h4>", unsafe_allow_html=True)
                    st.markdown("Please Select Transformation Method")
                    transformation_method = st.selectbox("Transformation Method", ["Iterative Box-Cox", "Iterative Yeo-Johnson", 
                                                                                "Log", "Square Root"])
                    if transformation_method == "Iterative Box-Cox" or transformation_method == "Iterative Yeo-Johnson":
                        if transformation_method == "Iterative Box-Cox":
                            try:
                                transformed_data, transformers, iterations, final_lambda = boxcox_lambda_iteration(value_column)
                            except ValueError as e:
                                st.warning(f"{e}")
                        elif transformation_method == "Iterative Yeo-Johnson":
                            transformed_data, transformers, iterations, final_lambda, lambda_df = yeo_lambda_iteration(value_column) 
                        if 'transformed_data' not in locals():
                            st.error("Transformation was not successful. Please check your data and transformation method.")
                            st.stop()
                        st.write(f"Iterations: {iterations} | Final Lambda Value: {round(final_lambda, 3)}")
                        st.success("Variance Stationarity Achieved")
                        with st.expander("Show Transformation History", expanded=False):
                            st.dataframe(lambda_df.set_index('Iteration'), use_container_width=True, width=200)
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

                    if transformation_method == "Log":
                        transformed_data = log_transform(value_column)
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
                        st.success("Log Transformation Successful")
                        
                    if transformation_method == "Square Root":
                        transformed_data = sqrt_transform(value_column)
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
                        st.success("Square Root Transformation Successful")
                        
                    if 'transformed_data' not in locals():
                        st.error("Transformation was not successful. Please check your data and transformation method.")
                        st.stop()
                        
                with col2:
                    st.markdown("<h4>Mean Stationarity</h4>", unsafe_allow_html=True)
                    st.markdown("Please Select Stationarity Test")
                    stationarity_test = st.selectbox("Stationarity Test", ["Augmented Dickey-Fuller", "Kwiatkowski-Phillips-Schmidt-Shin"]) 
                    if stationarity_test == "Augmented Dickey-Fuller":
                        diffed_data, d, differencing_history = iterative_adf(pd.Series(transformed_data.flatten()))
                    elif stationarity_test == "Kwiatkowski-Phillips-Schmidt-Shin":
                        diffed_data, d, differencing_history = iterative_kpss(pd.Series(transformed_data.flatten()))
                    st.write(f"Number of Differencing (d): {d}")
                    st.success("Mean Stationarity Achieved")
                    with st.expander("Show Differencing History", expanded=False):
                        st.dataframe(differencing_history, use_container_width=True, width=200)
                    with st.expander("Show Differenced Data", expanded=False):
                        st.dataframe(pd.DataFrame({"Date": date_column[d:], "Differenced Data": diffed_data}).set_index("Date"), 
                                    use_container_width=True, width=200)
                    with st.expander("Show Differenced Data Plot", expanded=False):
                        plt.figure(figsize=(10, 5))
                        plt.plot(date_column[d:], diffed_data, label="Differenced Data")
                        plt.xlabel("Date")
                        plt.ylabel("Differenced Data")
                        plt.legend()
                        st.pyplot(plt)
                
                st.image(".streamlit/Border_H.png", use_column_width=True)
                col1, col2 = st.columns(2)
                with col1:
                    override_d = st.checkbox("Override d", value=False, help="Override the number of differencing if the result is not as expected.")
                    trans_data_pd = pd.Series(transformed_data.flatten())
                    if override_d:
                        d = st.number_input("d (differencing order)", value=d, min_value=0, help="Please input the number of differencing manually.")
                    # Differencing data 'd' times
                    diffed_data = trans_data_pd.copy()
                    for _ in range(int(d)):
                        diffed_data = diffed_data.diff().dropna() 
                    continue_tentative = button("Continue to ARIMA Tentative Order", key="button3", help="Continue to ARIMA Tentative Order", icon="🌫️")
                with col2:
                    if override_d:
                        with st.container(border=True):
                            st.markdown('<div style="text-align: center; font-weight: bold;">ACF Plot to Check d Manually</div>', unsafe_allow_html=True)
                            st.markdown('<div style="font-size: 12px;"></div>', unsafe_allow_html=True)
                            # Calculate ACF values
                            fig_acf = plot_acf(diffed_data, lags=20)
                            fig_acf.tight_layout()
                            st.pyplot(fig_acf)
            
            st.markdown(" ")
            if continue_tentative:
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
                    
                    if autosearch:
                        p_order, q_order = order_finder(diffed_data)
                        st.success(f"Found significant AR (p) orders: {p_order}")
                        st.success(f"Found significant MA (q) orders: {q_order}")
                    else:
                        p, q = None, None
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown('Please select the AR() order that want to be checked manually. Example input: "1, 2, 3" without quotes')
                            p = st.text_input("(p)") #Example input: "1, 2, 3"
                            p_order = [int(i) for i in p.split(",") if i.strip().isdigit()]
                            if len(p) == 0:
                                p_order = [0, 1]
                                st.warning("No input detected. Defaulting to [0, 1].")
                            else:
                                st.info(f"Order Inputted : {p_order}")
                        with col2:
                            with st.container(border=True):
                                # Calculate PACF values
                                fig_pacf = plot_pacf(diffed_data, lags=20)
                                fig_pacf.tight_layout()
                                st.pyplot(fig_pacf)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown('Please select the MA() order that want to be checked manually. Example input: "1, 2, 3" without quotes')
                            q = st.text_input("(q)") #Example input: "1, 2, 3"
                            q_order = [int(i) for i in q.split(",") if i.strip().isdigit()]
                            if len(q) == 0:
                                q_order = [0, 1]
                                st.warning("No input detected. Defaulting to [0, 1].")
                            else:
                                st.info(f"Order Inputted : {q_order}")
                        with col2:
                            with st.container(border=True):
                                # Calculate ACF values
                                fig_acf = plot_acf(diffed_data, lags=20)
                                fig_acf.tight_layout()
                                st.pyplot(fig_acf)
                    
                    if p_order != None and q_order != None:
                        combinations = generate_combination(p_order, d, q_order)
                        st.success(f"Generated {len(combinations)} combinations for ARIMA model : {combinations}")
                    continue_modelfit = button("Continue to ARIMA Model Fit", key="button4", help="Continue to ARIMA Model Fit", icon="🌞")
                    
                st.markdown("")
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
                        st.markdown("### ⚙️ ARIMA Model Diagnostic Settings")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            with st.container(border=True):
                                st.markdown("Select Normality Test")
                                normality = st.selectbox("Normality Test", ["Jarque-Bera", "Shapiro-Wilk", "Kolmogorov-Smirnov", "Anderson-Darling", "Lilliefors"])
                        with col2:
                            with st.container(border=True): 
                                st.markdown("Select White Noise Test")
                                white_noise = st.selectbox("White Noise Test", ["Ljung-Box", "Box-Pierce"])
                        with col3:
                            with st.container(border=True):
                                st.markdown("Select Significance Level")
                                sign = st.number_input("Significance Level", value=0.05, min_value=0.01, max_value=0.1, step=0.01)
                    
                        continuemassfit = button("Begin Mass Fit Model", key="button5", help="Apply Setting and Run Mass-Fit Model", icon="🔄")
                        st.image(".streamlit/Border_H.png", use_column_width=True)
                        if continuemassfit:
                            st.markdown("### 📋 ARIMA Model List")
                            st.write("The table below shows the summary of ARIMA models fitted with different combinations of AR, I, and MA orders.")
                            model_collection = arima_model_mass_fit(pd.Series(transformed_data.flatten(), index=range(1, len(transformed_data.flatten()) + 1)), 
                                                                    combinations, normality_method=normality, white_noise_method=white_noise,
                                                                    sign=sign)
                            st.dataframe(model_collection, use_container_width=True, selection_mode='single-row', key='arima_model_mass_fit')
                            st.image(".streamlit/Border_H.png", use_column_width=True)
                
                st.markdown(" ")
                
        proceed_forecast = False  
        choose_order = None
                 
        if ("model_collection" in globals() or "best_model_autoarima" in globals()) and \
        (("model_collection" in globals() and model_collection is not None and len(model_collection) > 0) or \
            ("best_model_autoarima" in globals() and best_model_autoarima is not None and len(best_model_autoarima) > 0)):

            with st.container(border=True):
                best_model_manual = select_best_arima_model(model_collection) if "model_collection" in globals() else None
                st.subheader("🏆 Best ARIMA Model")
                
                if "best_model_autoarima" in globals() and best_model_autoarima is not None:
                    st.info("A model is detected from auto ARIMA function. You can choose another order if needed.")
                    st.info(f"Auto ARIMA Model: {best_model_autoarima}")
                    choose_order = best_model_autoarima
                    change_order = st.checkbox("Change Order", value=False, help="Change the ARIMA order manually.")
                    if change_order:
                        choose_order = st.text_input("ARIMA Order", value=best_model_autoarima, 
                                help="Please input the ARIMA order manually.")
                        st.write("Current Model Choosed: ", choose_order)
                    confirm = button("Confirm", key="button6", help="Confirm the ARIMA order and show summary", icon="🚀")
                    
                if best_model_manual is not None:
                    col1, col2 = st.columns(2)
                    with col1:
                        with st.container(border=True):
                            st.markdown("""
                            The best ARIMA model are selected from table based on the following criteria (priority order):
                            1. All components are significant.
                            2. Residuals are white noise.
                            3. Residuals are normally distributed.
                            4. Lowest AIC value.
                            """)
                            st.info(f"Current Best Model: {best_model_manual['Model']}")
                    with col2:
                        with st.container(border=True):
                            st.markdown("You can change the ARIMA order if needed in below.")
                            choose_order = st.text_input("ARIMA Order", value=best_model_manual['Model'], 
                                                         help="Please input the ARIMA order manually.")
                            st.write("Current Model Choosed: ", choose_order)
                        confirm = button("Confirm", key="button6", help="Confirm the ARIMA order and show summary", icon="🚀")
                st.image(".streamlit/Border_H.png", use_column_width=True)
                
                if confirm:
                    p_best, d_best, q_best = [int(i) for i in choose_order[6:-1].split(",")]
                    model_best = sm.tsa.ARIMA(pd.Series(transformed_data.flatten(), index=range(1, len(transformed_data.flatten()) + 1)), 
                                              order=(p_best, d_best, q_best)).fit()
                    
                    fitted_val = model_best.fittedvalues.to_numpy().reshape(-1, 1)
                    # Inverse Transformation depends on the transformation method
                    if transformation_method == "Iterative Box-Cox":
                        fitted_val = inverse_boxcox_iteration(fitted_val, transformers)
                    elif transformation_method == "Iterative Yeo-Johnson":
                        fitted_val = inverse_yeo_iteration(fitted_val, transformers)
                    elif transformation_method == "Log":
                        fitted_val = np.exp(fitted_val)
                    elif transformation_method == "Square Root":
                        fitted_val = fitted_val ** 2
                    elif transformation_method == None:
                        fitted_val = fitted_val
                    else:
                        st.error("Transformation method not found.")
                        st.stop()
                    mse, mae, mape, rmse = metrics(value_column[d_best:], fitted_val.flatten()[d_best:]) #the metric calculation omit the first d value
                    
                    st.subheader("📌 Best ARIMA Model Summary")
                    st.markdown("The summary of the best ARIMA model or model you choose is shown below.")
                   
                    with st.container(border=True):
                        st.markdown("##### Metrics")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("MSE", mse)
                        col2.metric("MAE", mae)
                        col3.metric("MAPE", f"{mape}%")
                        col4.metric("RMSE", rmse)
                    with st.container(border=True):
                        st.markdown("##### Model Summary")
                        st.write(model_best.summary())
                        st.write("You can now proceed to the next step to forecast the time series data.")
                    proceed_forecast = button("Proceed to Forecasting", key="button7", help="Proceed to Forecasting", icon="🔮")
                        
        if proceed_forecast:
            with st.container(border=True):
                st.markdown("### 🔮 ARIMA Model Forecast")
                forecast_period = st.session_state['forecast_period']
                forecast = model_best.forecast(forecast_period)
                forecast = forecast.to_numpy().reshape(-1, 1)
                
                # Inverse Transformation depends on the transformation method
                if transformation_method == "Iterative Box-Cox":
                    forecast = inverse_boxcox_iteration(forecast, transformers)
                elif transformation_method == "Iterative Yeo-Johnson":
                    forecast = inverse_yeo_iteration(forecast, transformers)
                elif transformation_method == "Log":
                    forecast = np.exp(forecast)
                elif transformation_method == "Square Root":
                    forecast = forecast ** 2
                elif transformation_method == None:
                    forecast = forecast
                else:
                    st.error("Transformation method not found.")
                    st.stop()
                    
                forecast_df = st.session_state['forecast_template'].copy()
                forecast_df.iloc[:forecast_period, 1] = forecast
                selected_MA_F = option_menu(None, ["Show Forecasted Dataframe", "Show Forecast Plot"], 
                                            icons=["table", "graph-down"], 
                                            menu_icon= "cast", default_index=0, orientation="horizontal",
                                            styles={
                                                    "menu-title": {"font-size": "17px"},
                                                    "nav-link": {"font-size": "14px", "text-align": "center", "margin": "0px", "--hover-color": "#6082B6"},
                                                    "container": {"background-color": "#15173c", "border": "1.5px solid white"},
                                                    })
                      
                if selected_MA_F == "Show Forecasted Dataframe":
                    st.markdown('###### Forecast Dataframe')
                    st.write(f"Forecasted values for the next {forecast_period} periods:")
                    st.dataframe(forecast_df, use_container_width=True, height=280)
                      
                if selected_MA_F == "Show Forecast Plot":
                    st.markdown('###### Forecast Plot')
                    st.markdown(" ")
                    combined_data = pd.concat([value_column, pd.Series(forecast.flatten(), index=range(len(value_column), len(value_column) + forecast_period))])
                    st.line_chart(pd.DataFrame({"Original": combined_data[:len(value_column)], "Forecast": combined_data[len(value_column)-1:]}), 
                                x_label="Time Index", y_label="Value", color=["#aef5f1", "#edb682"],
                                height=300)
                    
            st.markdown("")
            if st.button("Save Model Data", key="save_model_f", icon="💾", help = "Send this model data to result page"):
                if 'forecast_period' in st.session_state:
                    st.session_state['ARIMA'] = {
                        "mae": mae,
                        "mse": mse,
                        "mape": mape,
                        "rmse": rmse,
                        "fitted_data": fitted_val,
                        "forecast_data": forecast_df}
                    st.success("Metric Model & Forecast has been saved to session state.")
                else:
                    st.session_state['ARIMA'] = {
                        "mae": mae,
                        "mse": mse,
                        "mape": mape,
                        "rmse": rmse,
                        "fitted_data": fitted_val}
                    st.success("Metric Model without Forecast has been saved to session state.")
            else:
                pass
            
    with st.expander("ℹ️ More Information"):
        st.markdown("""
        - [Auto Arima](https://alkaline-ml.com/pmdarima/modules/generated/pmdarima.arima.auto_arima.html)
        - [Power Transformer & Lambda](https://scikit-learn.org/dev/modules/generated/sklearn.preprocessing.PowerTransformer.html)
        """)
        st.markdown("Method References:")
        st.markdown("[1] Makridakis, S., Wheelwright, S. C., & Hyndman, R. J. Forecasting: Methods and Applications. John Wiley & Sons, 1998.")
        st.markdown("[2] Hyndman, Rob J., and George Athanasopoulos. Forecasting: Principles and Practice. OTexts, 2018.")
        st.markdown("[3] Box, George EP, Gwilym M. Jenkins, and Gregory C. Reinsel. Time Series Analysis: Forecasting and Control. John Wiley & Sons, 2015.")
        st.markdown("[4] Harvey, Andrew C. Forecasting, Structural Time Series Models and the Kalman Filter. Cambridge University Press, 1990.")
        st.markdown("[5] I.K. Yeo and R.A. Johnson, “A new family of power transformations to improve normality or symmetry.” Biometrika, 87(4), pp.954-959, (2000).")
        st.markdown("[6] Box, G. E. P., & Pierce, D. A. (1970). Distribution of residual autocorrelations in autoregressive-integrated moving average time series models. Journal of the American Statistical Association, 65(332), 1509-1526.")
        
        
elif selected == "SARIMA":
    st.title("❄️ SARIMA Model")
    st.image(".streamlit/Border_H.png", use_column_width=True)
    st.write("This model is used to forecast time series data with seasonal components.")
    
elif selected == "X-11":
    st.title("💫 X-11 Model")
    st.image(".streamlit/Border_H.png", use_column_width=True)
    st.write("This model is used to decompose time series data into trend, seasonal, and irregular components.")


