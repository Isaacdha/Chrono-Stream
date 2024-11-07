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

date_column = data.iloc[:, 0]
value_column = data.iloc[:, 1]

# Metrics Function
def metrics(data, fitted_data):
    mse = round(np.mean((data - fitted_data) ** 2), 2)
    mae = round(np.mean(np.abs(data - fitted_data)), 2)
    mape = round(np.mean(np.abs((data - fitted_data) / data)) * 100, 2)
    rmse = round(np.sqrt(mse), 2)
    return mse, mae, mape, rmse

# SARIMA Process
st.title("❄️ SARIMA Model")
st.image(".streamlit/Border_H.png", use_column_width=True)
st.write("This model is used to forecast time series data with seasonal components.")

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


# SARIMA Process
st.title("❄️ SARIMA Model")
st.image(".streamlit/Border_H.png", use_column_width=True)

# Decompose the time series data
decomposition = pm.arima.decompose(value_column, 'additive', 12)

# Plot the decomposed components
fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

# Observed
axes[0].plot(decomposition.x, label='Observed')
axes[0].legend(loc='upper left')
axes[0].set_title('Observed')

# Trend
axes[1].plot(decomposition.trend, label='Trend', color='orange')
axes[1].legend(loc='upper left')
axes[1].set_title('Trend')

# Seasonal
axes[2].plot(decomposition.seasonal, label='Seasonal', color='green')
axes[2].legend(loc='upper left')
axes[2].set_title('Seasonal')

# Residual
axes[3].plot(decomposition.random, label='Residual', color='red')
axes[3].legend(loc='upper left')
axes[3].set_title('Residual')

plt.tight_layout()
st.pyplot(fig)
