import streamlit as st
from streamlit_lottie import st_lottie

st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
col1, col2, col3 = st.columns([0.2,0.7,0.1])
with col2:
    st.title('Welcome to ClinicalAI!')
col1, col2, col3 = st.columns([0.1,0.8,0.1])
with col2:
    st_lottie('https://lottie.host/92980b79-f0c4-465c-a4d1-bcab6c27b79d/fh5Z0MxKSN.json')