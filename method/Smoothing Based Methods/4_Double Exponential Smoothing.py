import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_extras.stateful_button import button
import numpy as np
import pandas as pd
from statsmodels.tsa.api import Holt

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

# Double Exponential Smoothing Process
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

begin = button("Begin", key="begin_ETS2", icon="🏃‍♂️")

if begin:
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
                                        "container": {"background-color": "#15173c", "border": "1.5px solid white"},
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
                                        "container": {"background-color": "#15173c", "border": "1.5px solid white"},
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
    if st.button("Save Model Data", key="save_model_f", icon="💾", help = "Send this model data to result page"):
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