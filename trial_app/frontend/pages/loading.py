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

# Check if prediction result is available in session state
if 'prediction_result' in st.session_state:
    prediction = st.session_state['prediction_result']
    st.write('Prediction Result:', prediction)
else:
    st.error('No prediction result available. Please go back to the input page.')

# Add a button to go back to the input page
if st.button('Go Back'):
    st.session_state['submitted'] = False
    # st.experimental_set_query_params(page="trial")
    st.experimental_rerun()

# st.write('')
# placeholder = 'blah'
# st.download_button(
#     label='Download my results',
#     data=placeholder,
#     file_name='placeholder.csv',
#     mime='text/csv',
# )