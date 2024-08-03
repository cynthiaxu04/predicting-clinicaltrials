import streamlit as st
import requests
import pandas as pd
from pages.helper_functions import shortcut_map, conditions_5yr_survival_map, conditions_max_treatment_duration_map, conditions_min_treatment_duration_map

st.header('Estimate your trial duration')
st.write('Below are a few questions to help us better understand your trial. Your responses will be used to generate the predicted duration.')

features = dict()
display_features = dict()
with st.form('Features'):

    # Cancer type
    st.write('1. What is the category of disease(s) your trial is researching?')
    type = st.selectbox('category of disease', 
                        ('Adenocarcinoma', 'Squamous Cell Carcinoma', 'Transitional Cell Carcinoma', 'Basal Cell Carcinoma',
                        'Ductal Carcinoma', 'Other Carcinoma', 'Brain Cancer', 'Sarcoma', 'Lymphoma', 'Leukemia', 'Melanoma',
                        'Myeloma', 'Pediatric Cancer', 'Pain relating to any disease', 'Other'), index=None, label_visibility='collapsed')
    
    display_features['Cancer type'] = type
    type = shortcut_map.get(type)
    features['survival_5yr_relative'] = conditions_5yr_survival_map.get(type)
    features['max_treatment_duration'] = conditions_max_treatment_duration_map.get(type)
    features['min_treatment_duration'] = conditions_min_treatment_duration_map.get(type)

    # Phase
    st.write('2. What is the phase of your trial?')
    phase = st.radio('phase', [2, 3], index=None, label_visibility='collapsed')
    display_features['Phase'] = phase
    if phase == 2:
       features['phase_PHASE2_PHASE3'] = 1
       features['phase_PHASE3'] = 1
    else:
       features['phase_PHASE2_PHASE3'] = 0
    if phase == 3:
       features['phase_PHASE3'] = 1
       features['phase_PHASE2_PHASE3'] = 0
    else:
       features['phase_PHASE3'] = 0

    # Num locations
    st.write('3. How many sites will there be?')
    num_sites = st.number_input('sites', min_value = 0, max_value=3000, step=1, value=None,
                                label_visibility='collapsed', placeholder='Type a number...')
    if num_sites == 0 or num_sites is None:
        st.error('Your trial must have at least 1 site.')
    st.write('')
    display_features['Number of sites'] = num_sites
    features['num_locations'] = num_sites
    
    # US vs Non US
    st.write('4. Where will your trial take place?')
    us = st.checkbox('Includes USA location(s)')
    non_us = st.checkbox('Includes non-USA location(s)')
    if us and non_us:
        features['location'] = 2
        display_features['Location'] ='USA and non-USA'
    elif us:
        features['location'] = 0
        display_features['Location'] ='USA'
    elif non_us:
        features['location'] = 1
        display_features['Location'] = 'non-USA'
    st.write('')

    # Patients
    st.write('5. How many evaluable subjects does your trial need?')
    num_patients = st.number_input('Subjects', min_value = 0, max_value=10000, step=1, value=None,
                                   label_visibility='collapsed', placeholder='Type a number...')
    if num_patients is None or num_patients == 0:
        st.error('Your trial must have at least 1 subject.')
    display_features['Number of subjects'] = num_patients
    features['enroll_count'] = num_patients

    # Criteria
    st.write('6. What is the number of inclusion criteria?')
    num_inclusion = st.number_input('Inclusion criteria', min_value = 0, max_value=500, step=1, value=None,
                                    label_visibility='collapsed', placeholder='Type a number...')
    if num_inclusion is None or num_inclusion == 0:
        st.error('Your trial must have at least 1 inclusion criteria.')
    display_features['Number of inclusion criteria'] = num_inclusion
    features['num_inclusion'] = num_inclusion
    st.write('')

    st.write('7. What is the number of exclusion criteria?')
    num_exclusion = st.number_input('Exclusion criteria', min_value = 0, max_value=500, step=1, value=None,
                                    label_visibility='collapsed', placeholder='Type a number...')
    if num_exclusion is None or num_exclusion == 0:
        st.error('Your trial must have at least 1 exclusion criteria.')
    display_features['Number of exclusion criteria'] = num_exclusion
    features['num_exclusion'] = num_exclusion

    # DMC
    st.write('8. Will you have a data monitoring committee?')
    has_dmc = st.radio('data monitoring committee', ['Yes', 'No'], index=None, label_visibility='collapsed')
    display_features['Data monitoring committee'] = has_dmc
    if has_dmc == 'Yes':
        features['has_dmc'] = 1
    elif has_dmc == 'No':
        features['has_dmc'] = 0

    # Responsible party
    st.write('9. Who will be the responsible party?')
    resp_party = st.radio('responsible party', ['PI', 'Sponsor', 'PI and Sponsor'], index=None, label_visibility='collapsed')
    display_features['Responsible party'] = resp_party
    if resp_party == 'PI':
        features['resp_party'] = 0
    elif resp_party == 'Sponsor':
        features['resp_party'] = 1
    else:
        features['resp_party'] = 2

    # Intervention model
    st.write('10. What is the intervention model for your trial?')
    intervention = st.radio('Intervention model', ['Single Group', 'Parallel', 'Other'], index=None, label_visibility='collapsed')
    display_features['Intervention model'] = intervention
    if intervention == 'Single Group':
        features['intervention_model'] = 0
    elif intervention == 'Parallel':
        features['intervention_model'] = 1
    elif intervention == 'Other':
        features['intervention_model'] = 2

    # Intervention type
    st.write('11. What is the intervention type(s) for your trial?')
    intervention_type = st.multiselect(
        'Intervention Types',
        ['Procedure', 'Device', 'Behavioral', 'Drug', 'Radiation', 'Biological'], label_visibility='collapsed')
    intervention_str = ', '.join(intervention_type)
    display_features['Intervention type(s)'] = intervention_str
    count = 0
    if 'Procedure' in intervention_type:
        count +=1
    if 'Device' in intervention_type:
        count +=1
    if 'Behavioral' in intervention_type:
        count +=1
    if 'Drug' in intervention_type:
        features['drug_intervention'] = 1
        count +=1
    else:
        features['drug_intervention'] = 0
    if 'Radiation' in intervention_type:
        count +=1
    else:
        features['radiation_intervention'] = 0
    if 'Biological' in intervention_type:
        features['biological_intervention'] = 1
        count +=1
    else:
        features['biological_intervention'] = 0

    features['number_of_intervention_types'] = count

    # Primary purpose
    st.write('12. What is the primary purpose(s) of your trial?')
    treatment_purpose = st.checkbox('Treatment')
    diagnostic_purpose = st.checkbox('Diagnostic')
    prevention_purpose = st.checkbox('Prevention')
    supportive_purpose = st.checkbox('Supportive')
    purpose_list = []
    if treatment_purpose:
        features['treatment_purpose'] = 1
        purpose_list.append('Treatment')
    else: 
       features['treatment_purpose'] = 0
    if diagnostic_purpose:
        features['diagnostic_purpose'] = 1
        purpose_list.append('Diagnostic')
    else:
       features['diagnostic_purpose'] = 0
    if prevention_purpose:
       features['prevention_purpose'] = 1
       purpose_list.append('Prevention')
    else: 
       features['prevention_purpose'] = 0
    
    purpose_str = ', '.join(purpose_list)
    display_features['Primary purpose(s)'] = purpose_str

    # Groups
    st.write('13. How many groups will your trial have?')
    num_groups = st.number_input('Number of groups', min_value = 0, max_value=100, step=1, value=None,
                                 label_visibility='collapsed', placeholder='Type a number...')
    if num_groups is None or num_groups == 0:
        st.error('Your trial must have at least 1 group.')
    features['number_of_groups'] = num_groups
    display_features['Number of groups'] = num_groups

    # Outcome measures
    st.write('14. What are the outcome measures of your trial?')
    outcome_measures = st.multiselect('Outcome Measures', ['Overall Survival', 'Adverse Events', 'Duration of Response', 'Other'], label_visibility='collapsed')
    om_str = ', '.join(outcome_measures)
    display_features['Outcome measure(s)'] = om_str
    if 'Overall Survival' in outcome_measures:
        features['os_outcome_measure'] = 1
    else:
        features['os_outcome_measure'] = 0
    if 'Adverse Events' in outcome_measures:
        features['ae_outcome_measure'] = 1
    else:
        features['ae_outcome_measure'] = 0
    if 'Duration of Response' in outcome_measures:
        features['dor_outcome_measure'] = 1
    else:
        features['dor_outcome_measure'] = 0
    
    # Masking
    st.write('15. What is the masking for your trial?')
    mask = st.radio('Masking', ['Open', 'Single', 'Double', 'Triple', 'Quadruple'], index=None, label_visibility='collapsed')
    display_features['Masking'] = mask
    if mask == 'Open':
        features['masking'] = 0
    elif mask == 'Single':
        features['masking'] = 1
    elif mask == 'Double':
        features['masking'] = 2
    elif mask == 'Triple':
        features['masking'] = 3
    elif mask == 'Quadruple':
        features['masking'] = 4

    # Healthy volunteers
    st.write('16. Will your trial include healthy volunteers?')
    vol = st.radio('Healthy volunteers', ['Yes', 'No'], index=None, label_visibility='collapsed')
    display_features['Healthy volunteers'] = vol
    if vol == 'Yes':
        features['healthy_vol'] = 1
    else:
        features['healthy_vol'] = 0 

    # Outcome measures days
    st.write('17. What is the maximum duration from baseline to the primary outcome measure for one patient?')
    primary_max = st.number_input('Primary outcome measure duration', min_value = 0.0, max_value=10000.0, step=0.5, value=None,
                                  label_visibility='collapsed', placeholder='Type a number...')
    unit = st.radio('Unit', ['Days', 'Months', 'Years'], index=None)
    duration = str(primary_max) + ' ' + str(unit)
    if unit == 'Days':
        features['primary_max_days'] = primary_max
    elif unit == 'Months':
        features['primary_max_days'] = primary_max * 30
    elif unit == 'Years':
        features['primary_max_days'] = primary_max * 365
    display_features['Primary outcome measure duration (from baseline)'] = duration
    
    st.write('18. What is the maximum duration from baseline to the secondary outcome measure for one patient?')
    secondary_max = st.number_input('Secondary outcome measure duration', min_value = 0.0, max_value=10000.0, step=0.5, value=None,
                                    label_visibility='collapsed', placeholder='Type a number or leave blank if no secondary outcome measure...')
    unit_secondary = st.radio('Unit', ['Days', 'Months', 'Years', 'No secondary outcome measure'], key=2, index=None)
    
    if secondary_max is None:
        features['secondary_max_days'] = None
        display_features['Secondary outcome measure duration (from baseline)'] = 'None'
    else:
        sec_duration = str(secondary_max) + ' ' + str(unit_secondary)
        display_features['Secondary outcome measure duration (from baseline)'] = sec_duration 

    if unit_secondary == 'Days':
        features['secondary_max_days'] = secondary_max
    elif unit_secondary == 'Months':
        features['secondary_max_days'] = secondary_max * 30
    elif unit_secondary == 'Years':
        features['secondary_max_days'] = secondary_max * 365  
    
    # Submission
    submitted = st.form_submit_button('Submit')
    if submitted:
        st.write(features)
        if len(features) < 28:
            st.error('Please answer all questions.')
        else:
          response = requests.post('http://backend:8000/predict', json=features)

          if response.status_code == 200:
            prediction = response.json()
            st.session_state['prediction_result'] = prediction
            st.session_state['submitted'] = True
            # st.experimental_rerun()  # Navigate to the results page
            st.switch_page('pages/loading.py')
          else:
            st.error('Prediction failed. Please try again.')