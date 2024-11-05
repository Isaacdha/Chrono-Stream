import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import norm
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
            print(f"Variance stationarity achieved at iteration {iteration + 1}")
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
        if (transformed_data <= 0).any():
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
def order_finder(diffed_data, nlags=20, conf_level=0.05):
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
    print(z_critical)
    
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

# Ljung-Box Test Function






# Function for iterate Arima using combinations
def iterate_arima(data, combinations, normality_significance=0.05, ):
    """
    Iterates through all combinations of ARIMA models and returns the best model.

    Parameters:
    - data: The time series data.
    - combinations: List of all possible combinations of p, d, and q values.

    Returns:
    - best_model: The best ARIMA model based on AIC.
    - best_order: The order (p, d, q) of the best ARIMA model.
    """
    best_aic = np.inf
    best_order = None
    best_model = None

    for order in combinations:
        try:
            model = sm.tsa.ARIMA(data, order=order).fit()
            aic = model.aic
            if aic < best_aic:
                best_aic = aic
                best_order = order
                best_model = model
        except:
            continue

    return best_model, best_order





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
        
        st.image(".streamlit/Border_H.png", use_column_width=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<h4>Variance Stationarity</h4>", unsafe_allow_html=True)
            st.markdown("Please Select Transformation Method")
            transformation_method = st.selectbox("Transformation Method", ["Box-Cox", "Yeo-Johnson"])
            if transformation_method == "Box-Cox":
                try:
                    transformed_data, transformers, iterations, final_lambda = boxcox_lambda_iteration(value_column)
                except ValueError as e:
                    st.warning(f"{e}")
            elif transformation_method == "Yeo-Johnson":
                transformed_data, transformers, iterations, final_lambda, lambda_df = yeo_lambda_iteration(value_column) 
            if 'transformed_data' not in locals():
                st.error("Transformation was not successful. Please check your data and transformation method.")
                st.stop()
            st.write(f"Iterations: {iterations} | Final Lambda Value: {round(final_lambda, 3)}")
            with st.expander("Show Transformation History", expanded=False):
                st.dataframe(lambda_df.set_index('Iteration'), use_container_width=True, width=200)
            with st.expander("Show Transformed Data", expanded=False):
                st.dataframe(pd.DataFrame({"Date": date_column, "Transformed Data": transformed_data.flatten()}).set_index("Date"), 
                             use_container_width=True, width=200)
        with col2:
            st.markdown("<h4>Mean Stationarity</h4>", unsafe_allow_html=True)
            st.markdown("Please Select Stationarity Test")
            stationarity_test = st.selectbox("Stationarity Test", ["Augmented Dickey-Fuller", "Kwiatkowski-Phillips-Schmidt-Shin"]) 
            if stationarity_test == "Augmented Dickey-Fuller":
                diffed_data, d, differencing_history = iterative_adf(pd.Series(transformed_data.flatten()))
            elif stationarity_test == "Kwiatkowski-Phillips-Schmidt-Shin":
                diffed_data, d, differencing_history = iterative_kpss(pd.Series(transformed_data.flatten()))
            st.write(f"Number of Differencing (d): {d}")
            with st.expander("Show Differencing History", expanded=False):
                st.dataframe(differencing_history, use_container_width=True, width=200)
            with st.expander("Show Differenced Data", expanded=False):
                st.dataframe(pd.DataFrame({"Date": date_column[d:], "Differenced Data": diffed_data}).set_index("Date"), 
                             use_container_width=True, width=200)
            
    
    st.markdown(" ")
    with st.container(border=True):
        st.subheader("📉 ARIMA Model")
        st.write("ARIMA Order Selection")
        
        # Plot ACF of differenced data
        fig, ax = plt.subplots()
        sm.graphics.tsa.plot_acf(diffed_data, ax=ax)
        st.pyplot(fig)
        
        fig, ax = plt.subplots()
        sm.graphics.tsa.plot_pacf(diffed_data, ax=ax)
        st.pyplot(fig)
        
        significant_lags = order_finder(diffed_data)
        st.write(significant_lags)
        p, q = significant_lags
        combinations = generate_combination(p, d, q)
        st.write(combinations)
        
        model = ARIMA(diffed_data, order=(1, 1, 1)).fit()
        residuals = model.resid
        from statsmodels.stats.diagnostic import acorr_ljungbox
        ljung_box_test = acorr_ljungbox(model.resid, lags=[1,2,3,4,5,6,7,8,9,10], return_df=True, boxpierce=False)
        st.write("Ljung-Box Test p-values:")
        for lag, p_value in enumerate(ljung_box_test['lb_pvalue'].values, start=1):
            st.write(f"Lag {lag}: {p_value}")
        st.write(model.summary())
    
elif selected == "SARIMA":
    st.title("❄️ SARIMA Model")
    st.image(".streamlit/Border_H.png", use_column_width=True)
    st.write("This model is used to forecast time series data with seasonal components.")
    
elif selected == "X-11":
    st.title("💫 X-11 Model")
    st.image(".streamlit/Border_H.png", use_column_width=True)
    st.write("This model is used to decompose time series data into trend, seasonal, and irregular components.")


