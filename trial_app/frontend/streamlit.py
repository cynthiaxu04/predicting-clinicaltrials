import streamlit as st
import requests

st.title('Clinical Trial Duration Prediction')

st.header("Basic Information")

# num_location: integer
st.write("How many locations will your clinial trial take place?")
num_locations = st.slider('Number of Locations', 1.0, 1200.00) # max_value in training set = 1,084

# location: ordinal
st.write("Will your study include the following locations?")
us = st.checkbox("Includes USA location(s)")
non_us = st.checkbox("Includes non-USA location(s)")
if us and non_us:
    location == 2
elif us:
    location == 0
elif non_us:
    location == 1
else: 
    st.error("You must check the appropriate box(es).")

# phase: phase_PHASE2_ PHASE3, phase_PHASE3

# enroll_count: integer
st.write("How many participants will you plan to enroll?")
enroll_count = st.slider('Number of Locations', 1.0, 12000.00) # max_value in test set = 12,000

# resp_party:

# has_dmc: boolean
st.write("Will you have a data monitoring committe?")
has_dmc = st.checkbox("Yes")

# conditions: TEXT


st.header("Participant Criteria")
# healthy_vol: 
# num_inclusion: integer
# num_exclusion: integer

st.header("Study Design")
# PRIMARY PURPOSE: treatment_purpose, diagnostic_purpose, prevention_purpose, ‘'supportive_purpose' (not included in the model)
# allocation: 0,1 
'''     'NON_RANDOMIZED': 0,
        'RANDOMIZED': 1
'''
# intervention_model: 0,1,2
'''
 assign_map = {
        "CROSSOVER": "OTHER",
        "SEQUENTIAL": "OTHER",
        "FACTORIAL": "OTHER"
    }
    assign_map2 = {
        "SINGLE_GROUP": 0,
        "PARALLEL": 1,
        "OTHER": 2
'''
# masking: 0,1,2,3,4
'''
mask_map = {
        "NONE": 0,
        "SINGLE": 1,
        "DOUBLE": 2,
        "TRIPLE": 3,
        "QUADRUPLE": 4
    }
'''
st.header("Interventions")
# number_of_intervention_types: integer
# checkbox
'''
device_intervention
drug_intervention
radiation_intervention
biological_intervention
'''
st.header("Outcome Measures")
# primary_max_days: integer
# secondary_max_days: integer
# checkbox
'''os_outcome_measure
dor_outcome_measure
ae_outcome_measure
'''