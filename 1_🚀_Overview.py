import streamlit as st

# Set the title of the app
st.title('Chrono-Stream Overview')

# Add a header
st.header('Welcome to Chrono-Stream')

# Add a subheader
st.subheader('Overview of the Project')

# Add some text
st.text('This is a simple Streamlit app to provide an overview of the Chrono-Stream project.')

# Add a sidebar
st.sidebar.title('Navigation')
st.sidebar.write('Use the sidebar to navigate through the app.')

# Add a selectbox in the sidebar
option = st.sidebar.selectbox(
    'Select a section',
    ['Introduction', 'Data', 'Model', 'Results']
)

# Display the selected section
if option == 'Introduction':
    st.write('This section provides an introduction to the project.')
elif option == 'Data':
    st.write('This section provides an overview of the data used in the project.')
elif option == 'Model':
    st.write('This section provides details about the model used in the project.')
elif option == 'Results':
    st.write('This section provides the results of the project.')

# Add a footer
st.write('---')
st.write('Created by [Your Name]')