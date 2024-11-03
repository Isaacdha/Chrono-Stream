import streamlit as st
import pandas as pd

st.title('Data Input')

if 'filtered_df' in st.session_state:
    st.markdown("<span style='color:red'>Dataframe already exists in session state, please continue to next page or reupload to change the data.</span>", unsafe_allow_html=True)
 
with st.container(border=True):
    df = pd.DataFrame(None)
    st.markdown("#### Upload file")
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                sheet_names = pd.ExcelFile(uploaded_file).sheet_names
                if len(sheet_names) > 1:
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
        date_column = st.selectbox("Select the date/time column", df.columns)
        st.write(f"Data type of {date_column}: {df[date_column].dtype}")
        value_column = st.selectbox("Select the value column", df.columns)
        st.write(f"Data type of {value_column}: {df[value_column].dtype}")
        
        if df[date_column].dtype == 'datetime64[ns]':
            if date_column and value_column and date_column != value_column:
                st.markdown("#### Data Preview")
                st.write(df[[date_column, value_column]].head())
            else:   
                st.error("Please select different columns.")
        else:
            st.error("Please select a column with datetime data type.")
        
        if st.button("Confirm"):
            new_df = df[[date_column, value_column]].copy()
            if 'filtered_df' in st.session_state:
                del st.session_state['filtered_df']
            st.session_state['filtered_df'] = new_df
            st.success("Dataframe has been filtered and saved to session state.")
    
    
            
       

            

        


            
            