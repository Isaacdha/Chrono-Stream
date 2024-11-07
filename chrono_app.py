import streamlit as st

# Page Settings
st.set_page_config(
    page_title="Chrono Stream App",
    page_icon="⌛",
    layout="wide"
)

def add_logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                background-image: ".streamlit/image.png;
                background-repeat: no-repeat;
                padding-top: 120px;
                background-position: 20px 20px;
            }
            [data-testid="stSidebarNav"]::before {
                content: "Chrono Stream App";
                margin-left: 20px;
                margin-top: 0px;
                font-size: 30px;
                position: static;
                top: 0px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
add_logo()

st.sidebar.title("Chrono Stream App")
pages = {
    "Core Workflow": [
        st.Page("method/1_App Overview.py", title="App Overview", icon="🚀"),
        st.Page("method/2_Data Input.py", title="Input Data", icon="📝"),
        st.Page("method/3_Data Exploration.py", title="Data Exploration", icon="🔍"),
        st.Page("method/4_Result Comparison and Forecasting.py", title="Result Comparison and Forecasting", icon="📊"),
    ],
    "Smoothing Method": [
        st.Page("method/Smoothing Based Methods/1_Moving Average.py", title="Moving Average", icon = "📎"),
        st.Page("method/Smoothing Based Methods/2_Weighted Moving Average.py", title="Weighted Moving Average", icon = "🖇️"),
        st.Page("method/Smoothing Based Methods/3_Single Exponential Smoothing.py", title="Single Exponential Smoothing", icon = "1️⃣"),
        st.Page("method/Smoothing Based Methods/4_Double Exponential Smoothing.py", title="Double Exponential Smoothing", icon = "2️⃣"),
        st.Page("method/Smoothing Based Methods/5_Triple Exponential Smoothing.py", title="Triple Exponential Smoothing", icon = "3️⃣"),
    ],
    "Statistical Models": [
        st.Page("method/Statistical Models/1_ARIMA.py", title="ARIMA", icon="🌠"),
        st.Page("method/Statistical Models/2_SARIMA.py", title="SARIMA", icon="❄️"),
        st.Page("method/Statistical Models/4_X-11.py", title="X-11", icon="💫"),
    ],
    "Machine Learning Models": [
        st.Page("method/Machine Learning Models/1_Prophet.py", title="Prophet", icon="🔮"),
        st.Page("method/Machine Learning Models/2_LSTM.py", title="LSTM", icon="🧠"),
        st.Page("method/Machine Learning Models/3_CNN.py", title="CNN", icon="🗃️"),
        st.Page("method/Machine Learning Models/4_XGBoost.py", title="XGBoost", icon="🔥"),
    ],
    "Deterministic Trend Projection": [
        st.Page("method/Deterministic Trend Projection/1_Linear.py", title="Linear", icon="↗️"),
        st.Page("method/Deterministic Trend Projection/2_Quadratic.py", title="Quadratic", icon="➿"),
        st.Page("method/Deterministic Trend Projection/3_Exponential.py", title="Exponential", icon="✴️"),
        st.Page("method/Deterministic Trend Projection/4_Logarithmic.py", title="Logarithmic", icon="❇️"),
    ],
}

st.logo('.streamlit/Logo.png', icon_image='.streamlit/Logo_small.png', size='large')
with st.sidebar:
    with st.expander("Coder Profile"):    
        st.markdown(
            """
            **Email:** [Isaac's Email](mailto:isaacazziz@gmail.com)  
            **LinkedIn:** [Isaac's LinkedIn](https://id.linkedin.com/in/isaacdha)  
            **GitHub:** [Isaac's GitHub](https://github.com/Isaacdha)
            """
        )

pg = st.navigation(pages)
pg.run()


