import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Chrono Stream - Data Input",
    page_icon="📝",
    layout="wide"
)
st.logo('.streamlit/Logo.png', icon_image='.streamlit/Logo_small.png', size='large')

st.title('Data Input')

if 'filtered_df' in st.session_state:
    st.markdown("<span style='color:red'>Dataframe already exists in session state, please continue to next page or reupload to change the data.</span>", unsafe_allow_html=True)
 
with st.container(border=True):
    df = pd.DataFrame(None)
    st.markdown("#### Upload file")
    st.markdown("Please upload a CSV or XLSX file with at least one datetime column and one value column.")
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                sheet_names = pd.ExcelFile(uploaded_file).sheet_names
                if len(sheet_names) > 1:
                    col1, col2 = st.columns(2)
                    with col1:
                        sheet_name = st.selectbox("Select a sheet", sheet_names)
                    df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
                else:
                    df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("Please upload a CSV or XLSX file.")

if not df.empty:
    with st.container(border=True):
        st.markdown("#### Select Columns")
        col1, col2 = st.columns(2)
        with col1:
            date_column = st.selectbox("Select the date/time column", df.columns)
            st.write(f"Data type of {date_column}: {df[date_column].dtype}")
        with col2:
            value_column = st.selectbox("Select the value column", df.columns)
            st.write(f"Data type of {value_column}: {df[value_column].dtype}")
        
        st.image(".streamlit/Border_H.png", use_column_width=True)
        
        if df[date_column].dtype == 'datetime64[ns]':
            if date_column and value_column and date_column != value_column:
                st.markdown("#### Data Preview & Save")
                st.markdown(" ")
                col1, col2 = st.columns([5,5], gap = 'small')
                with col1:
                    st.markdown(df[[date_column, value_column]].head().style.set_table_styles(
                        [{'selector': 'th', 'props': [('text-align', 'center')]},
                         {'selector': 'td', 'props': [('text-align', 'center')]}]
                    ).to_html(index=False), unsafe_allow_html=True)
                
                with col2:
                    st.write("Please check the data preview before proceeding.")
                    if st.button("Confirm & Save Data"):
                        new_df = df[[date_column, value_column]].copy()
                        if 'filtered_df' in st.session_state:
                            del st.session_state['filtered_df']
                        st.session_state['filtered_df'] = new_df
                        st.success("Dataframe has been filtered and saved to session state.")
            else:   
                st.error("Please select different columns.")
        else:
            if df[date_column].dtype == 'object':
                try:
                    df[date_column] = pd.to_datetime(df[date_column])
                    if df[date_column].dtype == 'datetime64[ns]':
                        if date_column and value_column and date_column != value_column:
                            st.markdown("#### Data Preview & Save")
                            st.markdown(" ")
                            col1, col2 = st.columns([5,5], gap = 'small')
                            with col1:
                                st.markdown(df[[date_column, value_column]].head().style.set_table_styles(
                                    [{'selector': 'th', 'props': [('text-align', 'center')]},
                                     {'selector': 'td', 'props': [('text-align', 'center')]}]
                                ).to_html(index=False), unsafe_allow_html=True)
                            
                            with col2:
                                st.write("Please check the data preview before proceeding.")
                                if st.button("Confirm & Save Data"):
                                    new_df = df[[date_column, value_column]].copy()
                                    if 'filtered_df' in st.session_state:
                                        del st.session_state['filtered_df']
                                    st.session_state['filtered_df'] = new_df
                                    st.success("Dataframe has been filtered and saved to session state.")
                        else:   
                            st.error("Please select different columns.")
                    else:
                        st.error("Conversion to datetime failed. Please select a column with datetime data type.")
                except Exception as e:
                    st.error(f"Error converting to datetime: {e}")
            else:
                st.error("Please select a column with datetime data type.")
        


with st.container(border = True):
    st.subheader("Forecast Options")
    forecast = st.checkbox("Use Method to Forecast Data?")
    if forecast == True:
        col1, col2 = st.columns(2)
        with col1:
            forecast_number = st.number_input("Enter the number of periods to forecast", min_value=1, value=1, step=1)
        if st.button("Confirm Forecasting"):
            st.session_state['forecast_period'] = forecast_number
            if 'filtered_df' in st.session_state:
                last_date = st.session_state['filtered_df'].iloc[:, 0].max()
                df = st.session_state['filtered_df']
                date_column = df.columns[0]
                value_column = df.columns[1]
                freq = None
                if df.iloc[:, 0].dtype == 'datetime64[ns]':
                    freq = pd.infer_freq(st.session_state['filtered_df'].iloc[:, 0])
                    if freq:
                        last_date = pd.to_datetime(last_date) + pd.tseries.frequencies.to_offset(freq)
                    else:   
                        last_date = pd.to_datetime(last_date) + pd.DateOffset(days=1)
                forecast_dates = pd.date_range(start=last_date, periods=forecast_number, freq=freq)
                forecast_df = pd.DataFrame({date_column: forecast_dates, value_column: [np.nan] * forecast_number})
                st.session_state['forecast_template'] = forecast_df
            else:
                st.error("No data available to forecast.")
            st.write(f"Forecasting enabled and Set for {forecast_number} periods.")
            st.write(forecast_df)
    
    
            
       

            

        


            
            