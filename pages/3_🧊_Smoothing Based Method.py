import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt


# Page Settings
st.set_page_config(
    page_title="Chrono Stream - Smoothing",
    page_icon="🧊",
    layout="wide"
)
st.logo('.streamlit/Logo.png', icon_image='.streamlit/Logo_small.png', size='large')

# Sidebar Settings
with st.sidebar:
    selected = option_menu(
        "Smoothing Methods",
        ["Moving Average", "Single Exponential", "Double Exponential", "Triple Exponential"],
        icons=["graph-up", "arrow-up-circle", "arrow-up-right-circle", "arrow-up-right-square"],
        menu_icon="cast",
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
def metrics(data, smoothed_data):
    mse = round(np.mean((data - smoothed_data) ** 2), 2)
    mae = round(np.mean(np.abs(data - smoothed_data)), 2)
    mape = round(np.mean(np.abs((data - smoothed_data) / data)) * 100, 2)
    rmse = round(np.sqrt(mse), 2)
    return mse, mae, mape, rmse

# ====================================================================================================
# Begin Smoothing Methods
if selected == "Moving Average":
    st.header("🧮 Simple Moving Average")
    st.image(".streamlit/Border_H.png", use_column_width=True)
    st.markdown("""
    <div style="text-align: justify;">
    Simple Moving Average (SMA) is a simple yet effective method used to smooth time series data by calculating the average of observations over a specific number of periods. This method helps to reduce noise and highlight trends by averaging out short-term fluctuations. A basic moving average can be centered or trailing, depending on whether it averages data symmetrically or up to the current point. 
    </div>
    <br>
    <div style="text-align: justify;">
    While it provides a straightforward approach to smoothing, it lacks the ability to adapt to changing trends and does not account for seasonality or trend projections unless extended with more complex versions like weighted or exponential moving averages.
    </div>
    """, unsafe_allow_html=True)   
    st.markdown(" ") 
    
    # Moving Average Fuctions
    def moving_average(data, n):
        return np.convolve(data, np.ones(n)/n, mode='valid')

    # Find Optimal Window Size Function
    def find_optimal_n(data):
        min_error = float('inf')
        optimal_n = 1
        results = []
        for n in range(2, int(round(len(data)/2, 0)+1)):  
            ma = moving_average(data, n)
            mse, mae, mape, rmse = metrics(data[-len(ma):], ma)
            results.append((n, mse, mae, mape, rmse))
            if mse < min_error:
                min_error = mse
                optimal_n = n
        return optimal_n
    
    with st.container(border=True):
        st.subheader("⚙️ Smoothing Parameters")
        optimal = st.toggle("Automatically find optimal window size", value=True)
        if optimal == False:
            col1, col2 = st.columns(2)
            with col1:
                windows_size = st.number_input("Set window size", min_value=2, max_value=len(data), value=2)
        else:
            windows_size = find_optimal_n(value_column)
            st.markdown(f"**Optimal window size:** {windows_size}")
        
        # Smoothed Data & Metrics
        smoothed_data = moving_average(value_column, windows_size)
        mse, mae, mape, rmse = metrics(value_column[-len(smoothed_data):], smoothed_data)
        fitted = pd.DataFrame({"Date": data.iloc[windows_size-1:, 0], "Original": value_column[windows_size-1:], "Smoothed": smoothed_data})

    
    with st.container(border = True):
        st.subheader("📋 Results")
        
        st.image(".streamlit/Border_H.png", use_column_width=True)
        
        st.markdown("#### Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MSE", mse)
        col2.metric("MAE", mae)
        col3.metric("MAPE", f"{mape}%")
        col4.metric("RMSE", rmse)
        
        st.image(".streamlit/Border_H.png", use_column_width=True)
        
        st.markdown("#### Fitted Data")
        selected_MA = option_menu(None, ["Show Fitted Dataframe", "Show Fitted Plot"], 
                                icons=["table", "graph-down"], 
                                menu_icon= "cast", default_index=0, orientation="horizontal",
                                styles={
                                        "menu-title": {"font-size": "17px"},
                                        "nav-link": {"font-size": "14px", "text-align": "center", "margin": "0px", "--hover-color": "#6082B6"},
                                        "container": {"background-color": "#15173c", "border": "1px solid white"},
                                        })
    
        if selected_MA == "Show Fitted Plot":
            st.markdown('###### Read vs Fitted Data Plot')
            st.markdown(" ")
            st.line_chart(pd.DataFrame({"Original": value_column[windows_size-1:], "Smoothed": smoothed_data}), 
                          x_label="Time Index", y_label="Value", color=["#aef5f1", "#edb682"],
                          height=300)
        
        if selected_MA == "Show Fitted Dataframe":
            st.markdown('###### Read & Fitted Data Table')
            st.dataframe(fitted, use_container_width=True, height=300)
        
    # Border for Forecasting
    st.image(".streamlit/Border_H.png", use_column_width=True)

    # Forecasting
    if 'forecast_period' in st.session_state:
        with st.container(border=True):
            st.subheader("🔮 Forecasting")
            forecast_period = st.session_state['forecast_period']
            forecast = []
            
            # Start with the initial moving average based on the last 'window_size' values
            current_window = list(value_column[-windows_size:])

            # Forecast each future period iteratively
            for _ in range(forecast_period):
                # Calculate the moving average for the current window
                next_prediction = np.mean(current_window)
                # Append the prediction to the forecast list
                forecast.append(next_prediction)
                # Update the current window by removing the oldest value and adding the new prediction
                current_window.pop(0)
                current_window.append(next_prediction)
            
            forecast_df = st.session_state['forecast_template'].copy()
            forecast_df.iloc[:forecast_period, 1] = forecast
            
            selected_MA_F = option_menu(None, ["Show Forecasted Dataframe", "Show Forecast Plot"], 
                                icons=["table", "graph-down"], 
                                menu_icon= "cast", default_index=0, orientation="horizontal",
                                styles={
                                        "menu-title": {"font-size": "17px"},
                                        "nav-link": {"font-size": "14px", "text-align": "center", "margin": "0px", "--hover-color": "#6082B6"},
                                        "container": {"background-color": "#15173c", "border": "1px solid white"},
                                        })
            
            if selected_MA_F == "Show Forecast Plot":
                st.markdown('###### Forecast Plot')
                st.markdown(" ")
                combined_data = pd.concat([value_column, pd.Series(forecast, index=range(len(value_column), len(value_column) + forecast_period))])
                st.line_chart(pd.DataFrame({"Original": combined_data[:len(value_column)], "Forecast": combined_data[len(value_column)-1:]}), 
                              x_label="Time Index", y_label="Value", color=["#aef5f1", "#edb682"],
                              height=300)
        
            if selected_MA_F == "Show Forecasted Dataframe":
                st.markdown('###### Forecast Dataframe')
                st.write(f"Forecasted values for the next {forecast_period} periods:")
                st.dataframe(forecast_df, use_container_width=True, height=280)

    st.markdown("")
    if st.button("Save Model Data", key="save_model_f"):
        if 'forecast_period' in st.session_state:
            st.session_state['Moving_Average'] = {
                "mae": mae,
                "mse": mse,
                "mape": mape,
                "rmse": rmse,
                "fitted_data": fitted,
                "forecast_data": forecast_df}
            st.success("Metric model & forecast has been saved to session state.")
        else:
            st.session_state['Moving_Average'] = {
                "mae": mae,
                "mse": mse,
                "mape": mape,
                "rmse": rmse,
                "fitted_data": fitted}
            st.success("Metric model without forecast has been saved to session state.")
    else:
        pass

    with st.expander("ℹ️ More Information"):
        st.markdown("Method References:")
        st.markdown("[1] Makridakis, Spyros, Steven C. Wheelwright, and Rob J. Hyndman. Forecasting: Methods and Applications. John Wiley & Sons, 1998.")

# Single Exponential Smoothing
elif selected == "Single Exponential":
    st.header("⚀ Single Exponential Smoothing")
    st.image(".streamlit/Border_H.png", use_column_width=True)
    st.markdown("""
    <div style="text-align: justify;">
    Single Exponential Smoothing (SES) is used for time series data with no trend or seasonal pattern. It calculates a weighted moving average where more recent observations are given higher importance, controlled by the smoothing parameter α.
    </div>
    <br>
    <div style="text-align: justify;">
    The method provides a simple way to smooth data by reducing fluctuations to reveal underlying trends. A higher α makes the model more responsive to recent changes, while a lower α results in a smoother series that considers older data more evenly.
    </div>
    """, unsafe_allow_html=True)
    st.markdown(" ")
    
    with st.container(border=True):
        st.subheader("⚙️ Smoothing Parameters")
        optimal = st.toggle("Automatically find optimal alpha value", value=True)
        if optimal == False:
            col1, col2 = st.columns(2)
            with col1:
                alpha = st.number_input("Set alpha parameter", min_value=0.0, max_value=1.0, value=0.5, step=0.001)
            model = SimpleExpSmoothing(value_column, initialization_method="estimated")
            model_fit = model.fit(smoothing_level=alpha)
            st.markdown(f"**Alpha value:** {alpha}")
        else:
            model = SimpleExpSmoothing(value_column, initialization_method="estimated")
            model_fit = model.fit()
            alpha = model_fit.model.params['smoothing_level']
            st.markdown(f"**Optimal alpha value:** {alpha:.3f}")
        
        # Smoothed Data & Metrics
        smoothed_data = model_fit.fittedvalues
        mse, mae, mape, rmse = metrics(value_column, smoothed_data)
        fitted = pd.DataFrame({"Date": data.iloc[:, 0], "Original": value_column, "Smoothed": smoothed_data})

    with st.container(border = True):
        st.subheader("📋 Results")
        
        st.image(".streamlit/Border_H.png", use_column_width=True)
        
        st.markdown("#### Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MSE", mse)
        col2.metric("MAE", mae)
        col3.metric("MAPE", f"{mape}%")
        col4.metric("RMSE", rmse)
        
        st.image(".streamlit/Border_H.png", use_column_width=True)
        
        st.markdown("#### Fitted Data")
        selected_MA = option_menu(None, ["Show Fitted Dataframe", "Show Fitted Plot"], 
                                icons=["table", "graph-down"], 
                                menu_icon= "cast", default_index=0, orientation="horizontal",
                                styles={
                                        "menu-title": {"font-size": "17px"},
                                        "nav-link": {"font-size": "14px", "text-align": "center", "margin": "0px", "--hover-color": "#6082B6"},
                                        "container": {"background-color": "#15173c", "border": "1px solid white"},
                                        })
        
        if selected_MA == "Show Fitted Dataframe":
            st.markdown('###### Read & Fitted Data Table')
            st.dataframe(fitted, use_container_width=True, height=300)
            
        if selected_MA == "Show Fitted Plot":
            st.markdown('###### Read vs Fitted Data Plot')
            st.markdown(" ")
            st.line_chart(pd.DataFrame({"Original": value_column, "Smoothed": smoothed_data}), 
                          x_label="Time Index", y_label="Value", color=["#aef5f1", "#edb682"],
                          height=300)
        
    # Border for Forecasting
    st.image(".streamlit/Border_H.png", use_column_width=True)

    # Forecasting
    if 'forecast_period' in st.session_state:
        with st.container(border=True):
            forecast_period = st.session_state['forecast_period']
            forecast = model_fit.forecast(forecast_period)
            forecast_df = st.session_state['forecast_template'].copy()
            forecast_df.iloc[:forecast_period, 1] = forecast
            
            st.subheader("🔮 Forecasting")
            selected_MA_F = option_menu(None, ["Show Forecasted Dataframe", "Show Forecast Plot"], 
                                icons=["table", "graph-down"], 
                                menu_icon= "cast", default_index=0, orientation="horizontal",
                                styles={
                                        "menu-title": {"font-size": "17px"},
                                        "nav-link": {"font-size": "14px", "text-align": "center", "margin": "0px", "--hover-color": "#6082B6"},
                                        "container": {"background-color": "#15173c", "border": "1px solid white"},
                                        })
            
            if selected_MA_F == "Show Forecast Plot":
                st.markdown('###### Forecast Plot')
                st.markdown(" ")
                combined_data = pd.concat([value_column, pd.Series(forecast, index=range(len(value_column), len(value_column) + forecast_period))])
                st.line_chart(pd.DataFrame({"Original": combined_data[:len(value_column)], "Forecast": combined_data[len(value_column)-1:]}), 
                              x_label="Time Index", y_label="Value", color=["#aef5f1", "#edb682"],
                              height=300)
        
            if selected_MA_F == "Show Forecasted Dataframe":
                st.markdown('###### Forecast Dataframe')
                st.write(f"Forecasted values for the next {forecast_period} periods:")
                st.dataframe(forecast_df, use_container_width=True, height=280)

    st.markdown("")
    if st.button("Save Model Data", key="save_model_f"):
        if 'forecast_period' in st.session_state:
            st.session_state['Single_Exponential_Smoothing'] = {
                "mae": mae,
                "mse": mse,
                "mape": mape,
                "rmse": rmse,
                "fitted_data": fitted,
                "forecast_data": forecast_df}
            st.success("Metric model & morecast has been saved to session state.")
        else:
            st.session_state['Single_Exponential_Smoothing'] = {
                "mae": mae,
                "mse": mse,
                "mape": mape,
                "rmse": rmse,
                "fitted_data": fitted}
            st.success("metric model without forecast has been saved to session state.")
    else:
        pass
    
    with st.expander("ℹ️ More Information"):
        st.markdown("""
        - [Exponential Smoothing](https://otexts.com/fpp3/expsmooth.html)
        - [Holt-Winters Method](https://otexts.com/fpp3/holt-winters.html)
        """)
        st.markdown("Method References:")
        st.markdown("[2] Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles and practice. OTexts, 2014.")

elif selected == "Double Exponential":
    st.header("⚁ Double Exponential Smoothing")
    st.image(".streamlit/Border_H.png", use_column_width=True)
    st.markdown("""
    <div style="text-align: justify;">
    Double Exponential Smoothing (Holt’s Method) extends SES to handle data with a trend by introducing a second equation that accounts for the trend component. This method uses two smoothing parameters: α for the level and β for the trend. 
    </div>
    <br>
    <div style="text-align: justify;">
    The approach helps model data where there is an upward or downward movement over time, providing forecasts that adapt to changes in both the current level and the rate of trend growth. It’s well-suited for series where trends are present but seasonality is not.
    </div>
    """, unsafe_allow_html=True)   
    st.markdown(" ") 
    with st.container(border=True):
        st.subheader("⚙️ Smoothing Parameters")
        optimal = st.toggle("Automatically find optimal alpha and beta value", value=True)
        if optimal == False:
            col1, col2 = st.columns(2)
            with col1:
                alpha = st.number_input("Set alpha parameter", min_value=0.0, max_value=1.0, value=0.5, step=0.001)
            with col2:
                beta = st.number_input("Set beta parameter", min_value=0.0, max_value=1.0, value=0.5, step=0.001)
            model = Holt(value_column, initialization_method="estimated")
            model_fit = model.fit(smoothing_level=alpha, smoothing_trend=beta)
            st.markdown(f"**Alpha value:** {alpha}      |     **Beta value:** {beta}")
        else:
            model = Holt(value_column, initialization_method="estimated")
            model_fit = model.fit()
            alpha = model_fit.model.params['smoothing_level']
            beta = model_fit.model.params['smoothing_trend']
            st.markdown(f"**Optimal alpha value:** {alpha:.3f}     |     **Optimal beta value:** {beta:.3f}")
        
        # Smoothed Data & Metrics
        smoothed_data = model_fit.fittedvalues
        mse, mae, mape, rmse = metrics(value_column, smoothed_data)
        fitted = pd.DataFrame({"Date": data.iloc[:, 0], "Original": value_column, "Smoothed": smoothed_data})

    with st.container(border = True):
        st.subheader("📋 Results")
        
        st.image(".streamlit/Border_H.png", use_column_width=True)
        
        st.markdown("#### Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MSE", mse)
        col2.metric("MAE", mae)
        col3.metric("MAPE", f"{mape}%")
        col4.metric("RMSE", rmse)
        
        st.image(".streamlit/Border_H.png", use_column_width=True)
        
        st.markdown("#### Fitted Data")
        selected_ST = option_menu(None, ["Show Fitted Dataframe", "Show Fitted Plot"], 
                                icons=["table", "graph-down"], 
                                menu_icon= "cast", default_index=0, orientation="horizontal",
                                styles={
                                        "menu-title": {"font-size": "17px"},
                                        "nav-link": {"font-size": "14px", "text-align": "center", "margin": "0px", "--hover-color": "#6082B6"},
                                        "container": {"background-color": "#15173c", "border": "1px solid white"},
                                        })
        
        if selected_ST == "Show Fitted Dataframe":
            st.markdown('###### Read & Fitted Data Table')
            st.dataframe(fitted, use_container_width=True, height=300)
            
        if selected_ST == "Show Fitted Plot":
            st.markdown('###### Read vs Fitted Data Plot')
            st.markdown(" ")
            st.line_chart(pd.DataFrame({"Original": value_column, "Smoothed": smoothed_data}), 
                          x_label="Time Index", y_label="Value", color=["#aef5f1", "#edb682"],
                          height=300)
        
    # Border for Forecasting
    st.image(".streamlit/Border_H.png", use_column_width=True)

    # Forecasting
    if 'forecast_period' in st.session_state:
        with st.container(border=True):
            forecast_period = st.session_state['forecast_period']
            forecast = model_fit.forecast(forecast_period)
            forecast_df = st.session_state['forecast_template'].copy()
            forecast_df.iloc[:forecast_period, 1] = forecast
            
            st.subheader("🔮 Forecasting")
            selected_MA_F = option_menu(None, ["Show Forecasted Dataframe", "Show Forecast Plot"], 
                                icons=["table", "graph-down"], 
                                menu_icon= "cast", default_index=0, orientation="horizontal",
                                styles={
                                        "menu-title": {"font-size": "17px"},
                                        "nav-link": {"font-size": "14px", "text-align": "center", "margin": "0px", "--hover-color": "#6082B6"},
                                        "container": {"background-color": "#15173c", "border": "1px solid white"},
                                        })
            
            if selected_MA_F == "Show Forecast Plot":
                st.markdown('###### Forecast Plot')
                st.markdown(" ")
                combined_data = pd.concat([value_column, pd.Series(forecast, index=range(len(value_column), len(value_column) + forecast_period))])
                st.line_chart(pd.DataFrame({"Original": combined_data[:len(value_column)], "Forecast": combined_data[len(value_column)-1:]}), 
                              x_label="Time Index", y_label="Value", color=["#aef5f1", "#edb682"],
                              height=300)
        
            if selected_MA_F == "Show Forecasted Dataframe":
                st.markdown('###### Forecast Dataframe')
                st.write(f"Forecasted values for the next {forecast_period} periods:")
                st.dataframe(forecast_df, use_container_width=True, height=280)

    st.markdown("")
    if st.button("Save Model Data", key="save_model_f"):
        if 'forecast_period' in st.session_state:
            st.session_state['Double_Exponential_Smoothing'] = {
                "mae": mae,
                "mse": mse,
                "mape": mape,
                "rmse": rmse,
                "fitted_data": fitted,
                "forecast_data": forecast_df}
            st.success("Metric model & morecast has been saved to session state.")
        else:
            st.session_state['Double_Exponential_Smoothing'] = {
                "mae": mae,
                "mse": mse,
                "mape": mape,
                "rmse": rmse,
                "fitted_data": fitted}
            st.success("metric model without forecast has been saved to session state.")
    else:
        pass
    
    with st.expander("ℹ️ More Information"):
        st.markdown("""
        - [Exponential Smoothing](https://otexts.com/fpp3/expsmooth.html)
        - [Holt-Winters Method](https://otexts.com/fpp3/holt-winters.html)
        """)
        st.markdown("Method References:")
        st.markdown("[2] Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles and practice. OTexts, 2014.")

elif selected == "Triple Exponential":
    st.header("⚂ Triple Exponential Smoothing")
    st.image(".streamlit/Border_H.png", use_column_width=True)
    st.markdown("""
    <div style="text-align: justify;">
    Triple Exponential Smoothing (Holt-Winters Method) builds on double exponential smoothing to accommodate time series data that includes both trend and seasonal variations. It incorporates three smoothing equations for the level, trend, and seasonality, each controlled by α, β, and γ, respectively. 
    </div>
    <br>
    <div style="text-align: justify;">
    This method can model seasonal patterns additively or multiplicatively, making it adaptable for data with consistent or proportionally changing seasonal fluctuations. It is highly effective for time series with clear periodic patterns, such as monthly or quarterly sales data.
    </div>
    """, unsafe_allow_html=True)   
    st.markdown(" ") 
    
    with st.container(border=True):
        st.subheader("⚙️ Smoothing Parameters")
        col1, col2 = st.columns(2)
        with col1:
            trend = st.selectbox("Trend Type", ["additive", "multiplicative"])
            seasonal_period = st.number_input("Set seasonal period", min_value=2, max_value=len(data), value=2)
        with col2:
            seasonal = st.selectbox("Seasonal Type", ["additive", "multiplicative"])
        optimal = st.toggle("Automatically find optimal alpha, beta, and gamma value", value=True)
        if optimal == False:
            col1, col2 = st.columns(2)
            with col1:
                alpha = st.number_input("Set alpha parameter", min_value=0.0, max_value=1.0, value=0.5, step=0.001)
                gamma = st.number_input("Set gamma parameter", min_value=0.0, max_value=1.0, value=0.5, step=0.001)
            with col2:
                beta = st.number_input("Set beta parameter", min_value=0.0, max_value=1.0, value=0.5, step=0.001)
            model = ExponentialSmoothing(value_column, seasonal_periods=seasonal_period, 
                                         initialization_method="estimated", trend=trend, seasonal=seasonal)
            model_fit = model.fit(smoothing_level=alpha, smoothing_trend=beta, smoothing_seasonal=gamma)
            st.markdown(f"**Alpha value:** {alpha}      |     **Beta value:** {beta}     |     **Gamma value:** {gamma}")
        else:
            model = ExponentialSmoothing(value_column, initialization_method="estimated", trend=trend, seasonal=seasonal, seasonal_periods=seasonal_period)
            model_fit = model.fit()
            alpha = model_fit.model.params['smoothing_level']
            beta = model_fit.model.params['smoothing_trend']
            gamma = model_fit.model.params['smoothing_seasonal']
            st.markdown(f"**Optimal alpha value:** {alpha:.3f}     |     **Optimal beta value:** {beta:.3f}    |     **Optimal gamma value:** {gamma:.3f}")
        
        # Smoothed Data & Metrics
        smoothed_data = model_fit.fittedvalues
        mse, mae, mape, rmse = metrics(value_column, smoothed_data)
        fitted = pd.DataFrame({"Date": data.iloc[:, 0], "Original": value_column, "Smoothed": smoothed_data})

    with st.container(border = True):
        st.subheader("📋 Results")
        
        st.image(".streamlit/Border_H.png", use_column_width=True)
        
        st.markdown("#### Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MSE", mse)
        col2.metric("MAE", mae)
        col3.metric("MAPE", f"{mape}%")
        col4.metric("RMSE", rmse)
        
        st.image(".streamlit/Border_H.png", use_column_width=True)
        
        st.markdown("#### Fitted Data")
        selected_ST = option_menu(None, ["Show Fitted Dataframe", "Show Fitted Plot"], 
                                icons=["table", "graph-down"], 
                                menu_icon= "cast", default_index=0, orientation="horizontal",
                                styles={
                                        "menu-title": {"font-size": "17px"},
                                        "nav-link": {"font-size": "14px", "text-align": "center", "margin": "0px", "--hover-color": "#6082B6"},
                                        "container": {"background-color": "#15173c", "border": "1px solid white"},
                                        })
        
        if selected_ST == "Show Fitted Dataframe":
            st.markdown('###### Read & Fitted Data Table')
            st.dataframe(fitted, use_container_width=True, height=300)
            
        if selected_ST == "Show Fitted Plot":
            st.markdown('###### Read vs Fitted Data Plot')
            st.markdown(" ")
            st.line_chart(pd.DataFrame({"Original": value_column, "Smoothed": smoothed_data}), 
                          x_label="Time Index", y_label="Value", color=["#aef5f1", "#edb682"],
                          height=300)
        
    # Border for Forecasting
    st.image(".streamlit/Border_H.png", use_column_width=True)

    # Forecasting
    if 'forecast_period' in st.session_state:
        with st.container(border=True):
            forecast_period = st.session_state['forecast_period']
            forecast = model_fit.forecast(forecast_period)
            forecast_df = st.session_state['forecast_template'].copy()
            forecast_df.iloc[:forecast_period, 1] = forecast
            
            st.subheader("🔮 Forecasting")
            selected_MA_F = option_menu(None, ["Show Forecasted Dataframe", "Show Forecast Plot"], 
                                icons=["table", "graph-down"], 
                                menu_icon= "cast", default_index=0, orientation="horizontal",
                                styles={
                                        "menu-title": {"font-size": "17px"},
                                        "nav-link": {"font-size": "14px", "text-align": "center", "margin": "0px", "--hover-color": "#6082B6"},
                                        "container": {"background-color": "#15173c", "border": "1px solid white"},
                                        })
            
            if selected_MA_F == "Show Forecast Plot":
                st.markdown('###### Forecast Plot')
                st.markdown(" ")
                combined_data = pd.concat([value_column, pd.Series(forecast, index=range(len(value_column), len(value_column) + forecast_period))])
                st.line_chart(pd.DataFrame({"Original": combined_data[:len(value_column)], "Forecast": combined_data[len(value_column)-1:]}), 
                              x_label="Time Index", y_label="Value", color=["#aef5f1", "#edb682"],
                              height=300)
        
            if selected_MA_F == "Show Forecasted Dataframe":
                st.markdown('###### Forecast Dataframe')
                st.write(f"Forecasted values for the next {forecast_period} periods:")
                st.dataframe(forecast_df, use_container_width=True, height=280)

    st.markdown("")
    if st.button("Save Model Data", key="save_model_f"):
        if 'forecast_period' in st.session_state:
            st.session_state['Triple_Exponential_Smoothing'] = {
                "mae": mae,
                "mse": mse,
                "mape": mape,
                "rmse": rmse,
                "fitted_data": fitted,
                "forecast_data": forecast_df}
            st.success("Metric model & morecast has been saved to session state.")
        else:
            st.session_state['Triple_Exponential_Smoothing'] = {
                "mae": mae,
                "mse": mse,
                "mape": mape,
                "rmse": rmse,
                "fitted_data": fitted}
            st.success("metric model without forecast has been saved to session state.")
    else:
        pass
    
    with st.expander("ℹ️ More Information"):
        st.markdown("""
        - [Exponential Smoothing](https://otexts.com/fpp3/expsmooth.html)
        - [Holt-Winters Method](https://otexts.com/fpp3/holt-winters.html)
        """)
        st.markdown("Method References:")
        st.markdown("[2] Hyndman, Rob J., and George Athanasopoulos. Forecasting: principles and practice. OTexts, 2014.")
