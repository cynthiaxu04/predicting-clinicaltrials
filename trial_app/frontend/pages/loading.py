import streamlit as st
import time
# import pandas as pd
# from pages.trial import (
#     type, display_location, display_dmc, display_resp_party, display_intervention, display_intervention_types, display_purpose,
#     display_outcome_measures, mask, vol, display_primary_duration, display_secondary_duration, features)

progress_text = "Operation in progress. Please wait."
my_bar = st.progress(0, text=progress_text)

for percent_complete in range(100):
    time.sleep(0.001)
    my_bar.progress(percent_complete + 1, text=progress_text)
time.sleep(1)
my_bar.empty()

st.header('Your prediction result is...')
st.balloons()
st.write('')

# display_features = {'Cancer type': type, 'Phase': features['phase'], 'Locations': display_location,
#                     'Number of sites': features['num_locations'], 'Number of subjects': features['enroll_count'],
#                     'Number of inclusion criteria': features['num_inclusion'], 'Number of exclusion criteria': features['num_exclusion'],
#                     'Data monitoring committee': display_dmc, 'Responsible party': display_resp_party,
#                     'Intervention model': display_intervention, 'Intervention type(s)': display_intervention_types, 
#                     'Primary purpose': display_purpose, 'Number of groups': features['number_of_groups'],
#                     'Outcome measure(s)': display_outcome_measures, 'Masking': mask, 'Healthy volunteers': vol,
#                     'Primary outcome measure duration (from baseline)': display_primary_duration,
#                     'Secondary outcome measure duration (from baseline)': display_secondary_duration}

# df = pd.Series(display_features, name='Inputs')

# Check if prediction result is available in session state
if 'prediction_result' in st.session_state:
    prediction = st.session_state['prediction_result']
    lower = prediction['bin_interval'][1:5]
    upper = prediction['bin_interval'][7:11]
    st.write('Your trial is estimated to take between ', lower, ' and ', upper, " years.")
    # st.write('')
    # st.write('Your inputs were:')
    # st.write('')
    # st.table(df)

else:
    st.error('No prediction result available. Please go back to the input page.')

# Add a button to go back to the input page
st.page_link('pages/trial.py', label='**Go Back**')