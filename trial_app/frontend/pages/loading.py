import streamlit as st
import time

progress_text = "Operation in progress. Please wait."
my_bar = st.progress(0, text=progress_text)

for percent_complete in range(100):
    time.sleep(0.001)
    my_bar.progress(percent_complete + 1, text=progress_text)
time.sleep(1)
my_bar.empty()

st.header('All done!')
st.balloons()

st.write('')
placeholder = 'blah'
st.download_button(
    label='Download my results',
    data=placeholder,
    file_name='placeholder.csv',
    mime='text/csv',
)