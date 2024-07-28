import streamlit as st
# st.image('pages/ClinicalAILogo.png')

trial = st.Page('pages/trial.py', title='Estimate your trial duration')
loading = st.Page('pages/loading.py', title='Prediction Results')

pg = st.navigation([trial, loading], position='hidden')
pg.run()