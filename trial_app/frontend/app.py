import streamlit as st

# To run:
# "streamlit run app.py" in terminal

# st.page_link('pages/home.py', label='Home')
# st.page_link('pages/about.py', label='About')
# st.page_link('pages/contact.py', label='Contact us')
# st.page_link('pages/trial.py', label='Try us out')

home = st.Page('pages/home.py', title='Home')
about = st.Page('pages/about.py', title='About')
contact = st.Page('pages/contact.py', title='Contact us')
trial = st.Page('pages/trial.py', title='Try us out')
loading = st.Page('pages/loading.py', title='Loading')


col1, col2, col3, col4 = st.columns(4)
with col1:
    st.page_link('pages/home.py', label='Home')
with col2:
    st.page_link('pages/about.py', label='About')
with col3:
    st.page_link('pages/contact.py', label='Contact us')
with col4:
    st.page_link('pages/trial.py', label='Try us out')

pg = st.navigation([home, about, contact, trial, loading], position='hidden')
pg.run()