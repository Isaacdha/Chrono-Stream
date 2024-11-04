import streamlit as st

# Set the title of the app
st.set_page_config(
    page_title="Chrono Stream - Overview",
    page_icon="🚀",
    layout="wide"
)

st.logo('.streamlit/Logo.png', icon_image='.streamlit/Logo_small.png', size='large')

# Add a header
st.header('Welcome to Chrono-Stream')

# Add a subheader
st.subheader('Overview of the Project')

# Add some text
st.text('This is a simple Streamlit app to provide an overview of the Chrono-Stream project.')

# Add a sidebar
st.sidebar.title('Navigation')
st.sidebar.write('Use the sidebar to navigate through the app.')

# Add a footer
st.write('---')
st.write('Created by [Your Name]')