import streamlit as st
import base64

# Page Settings
st.set_page_config(
    page_title="Chrono Stream App",
    page_icon="⌛",
    layout="wide"
)

@st.cache_resource
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

bg_img = get_img_as_base64(".streamlit/Body Background.jpg")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/png;base64,{bg_img}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

[data-testid="stSidebar"] > div:first-child {{
    background-image: url("data:image/png;base64,{bg_img}");
    background-size: cover;
    background-position: center; 
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

[data-testid="stSidebar"] {{
    height: 100vh;
    box-sizing: border-box;
    border-right: 6px ridge #b1d4e3;
    border-top: 6px ridge #b1d4e3;
    border-bottom: 6px ridge #b1d4e3;
    border-top-right-radius: 20px;
    border-bottom-right-radius: 20px;
    overflow: clip;
}}

[data-testid="stMainBlockContainer"] {{
    padding-top: 0px;
    background: rgba(0, 0, 10, 0.5);
}}

[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# Button Style
st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #1c2c54;
}
</style>""", unsafe_allow_html=True)

st.markdown("""
<style>
div.stMainBlockContainer > button:first-child {
    background-color: #1c2c54;
}
</style>""", unsafe_allow_html=True)

def add_title():
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@700&display=swap');
            [data-testid="stSidebarNav"]::before {
                content: "⌛ Chrono Stream App";
                margin-left: 20px;
                margin-top: -10px;
                font-size: 27px;
                font-family: 'Roboto', sans-serif;
                position: static;
                top: 0px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
add_title()

pages_b = {
    "Core Workflow": [
        st.Page("method/1_App Overview.py", title="App Overview", icon="🚀"),
        st.Page("method/2_Data Input.py", title="Data Input & Forecast Settings", icon="📝")
    ]
}

pages_a = {
    "Core Workflow": [
        st.Page("method/1_App Overview.py", title="App Overview", icon="🚀"),
        st.Page("method/2_Data Input.py", title="Data Input & Forecast Settings", icon="📝"),
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
    ]
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

if 'filtered_df' not in st.session_state:
    pages = pages_b
    st.sidebar.warning("Other features will be available after data is inputted.")
else:
    pages = pages_a
    st.session_state['unlocked'] = True
    
pg = st.navigation(pages)
pg.run()


