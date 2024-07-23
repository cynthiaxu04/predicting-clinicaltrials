import streamlit as st
from processing.py import conditions_map
from processing.py import conditions_5yr_survival_map

st.header('Try us out!')
st.write('Below are a few questions to help us better understand your trial. Your responses will be used to generate the predicted duration.')

features = dict()
with st.form('Features'):

    # Cancer type
    type = st.selectbox('1\. What is the category of disease(s) your trial is researching?', 
                        ('Adenocarcinoma', 'Squamous Cell Carcinoma', 'Transitional Cell Carcinoma', 'Basal Cell Carcinoma',
                        'Ductal Carcinoma', 'Other Carcinoma', 'Brain Cancer', 'Sarcoma', 'Lymphoma', 'Leukemia', 'Melanoma',
                        'Myeloma', 'Pediatric Cancer', 'Pain relating to any disease', 'Other'), index=None)
    type = conditions_map(type)
    survival = conditions_5yr_survival_map[type]
    survival = survival['5yr_survival']
    features['survival_5yr_relative'] = survival
    features['number_of_conditions'] = 1

    # Phase
    phase = st.radio('2\. What is the phase of your trial?', [1, 2, 3], index=None)
    features['phase'] = phase

    # Num locations
    num_sites = st.number_input('3\. How many sites will there be?', min_value = 0, max_value=3000, step=1, value=None, placeholder='Type a number...')
    if num_sites == 0 or num_sites is None:
        st.error('Your trial must have at least 1 site.')
    st.write('')
    features['num_locations'] = num_sites

    # US vs Non US
    st.write('4. Where will your trial take place?')
    us = st.checkbox('Includes USA location(s)')
    non_us = st.checkbox('Includes non-USA location(s)')
    if us and non_us:
        features['location'] = 2
    elif us:
        features['location'] = 0
    elif non_us:
        features['location'] = 1
    st.write('')

    # Patients
    st.write('5. How many subjects will be enrolled?')
    num_patients = st.number_input('Subjects', min_value = 0, max_value=10000, step=1, value=None, placeholder='Type a number...')
    if num_patients is None or num_patients == 0:
        st.error('Your trial must have at least 1 subject.')
    st.write('')
    features['enroll_count'] = num_patients

    # Criteria
    st.write('6. What is the number of inclusion criteria?')
    num_inclusion = st.number_input('Inclusion criteria', min_value = 0, max_value=500, step=1, value=None, placeholder='Type a number...')
    if num_inclusion is None or num_inclusion == 0:
        st.error('Your trial must have at least 1 inclusion criteria.')
    features['num_inclusion'] = num_inclusion
    st.write('')

    st.write('7. What is the number of exclusion criteria?')
    num_exclusion = st.number_input('Exclusion criteria', min_value = 0, max_value=500, step=1, value=None, placeholder='Type a number...')
    if num_exclusion is None or num_exclusion == 0:
        st.error('Your trial must have at least 1 exclusion criteria.')
    features['num_exclusion'] = num_exclusion
    st.write('')

    # Sponsor type
    sponsor = st.radio('8\. What is the sponsor type for your trial?', ['Industry', 'Other'], index=None)
    features['sponsor_type'] = sponsor
    st.write('')

    # DMC
    has_dmc = st.radio('9\. Will you have a data monitoring committe?', ['Yes', 'No'], index=None)

    if has_dmc == 'Yes':
        has_dmc = 1
    elif has_dmc == 'No':
        has_dmc = 0
    features['has_dmc'] = has_dmc
    st.write('')

    resp_party = st.radio('10\. Who will be the responsible party?', ['PI', 'Sponsor', 'PI and Sponsor'], index=None)
    if resp_party == 'PI':
        features['resp_party'] = 0
    elif resp_party == 'Sponsor':
        features['resp_party'] = 1
    else:
        features['resp_party'] = 2

    # Intervention model
    intervention = st.radio('11\. What is the intervention model for your trial?', ['Single Group', 'Parallel', 'Other'], index=None)
    if intervention == 'Single':
        features['intervention_model'] = 0
    elif intervention == 'Parallel':
        features['intervention_model'] = 1
    elif intervention == 'Other':
        features['intervention_model'] = 2

    # Intervention type
    st.write('12. What is the intervention type(s) for your trial?')
    intervention_type = st.multiselect(
        'Intervention Types',
        ['Procedure', 'Device', 'Behavioral', 'Drug', 'Radiation', 'Biological'])
    count = 0
    if 'Procedure' in intervention_type:
        features['procedure_intervention'] = 1
        count +=1
    else:
        features['procedure_intervention'] = 0
    if 'Device' in intervention_type:
        features['device_intervention'] = 1
        count +=1
    else:
        features['device_intervention'] = 0
    if 'Behavioral' in intervention_type:
        features['behavioral_intervention'] = 1
        count +=1
    else:
        features['behavioral_intervention'] = 0
    if 'Drug' in intervention_type:
        features['drug_intervention'] = 1
        count +=1
    else:
        features['drug_intervention'] = 0
    if 'Radiation' in intervention_type:
        features['radiation_intervention'] = 1
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
    st.write('13. What is the primary purpose of your trial?')
    treatment_purpose = st.checkbox('Treatment')
    diagnostic_purpose = st.checkbox('Diagnostic')
    prevention_purpose = st.checkbox('Prevention')
    supportive_purpose = st.checkbox('Supportive')

    features['treatment_purpose'] = treatment_purpose
    features['diagnostic_purpose'] = diagnostic_purpose
    features['supportive_purpose'] = supportive_purpose
    features['prevention_purpose'] = prevention_purpose

    # Groups
    st.write('14. How many groups will your trial have?')
    num_groups = st.number_input('Number of groups', min_value = 0, max_value=100, step=1, value=None, placeholder='Type a number...')
    if num_groups is None or num_groups == 0:
        st.error('Your trial must have at least 1 group.')
    features['number_of_groups'] = num_groups

    # Age groups
    age_group = st.radio('15\. What is the target age group?', ['Youth', 'Adult', 'All'], index=None)
    if age_group == 'Youth': 
        age_group = 0
    elif age_group == 'Adult':
        age_group = 1
    elif age_group == 'All':
        age_group = 2
    features['age_group'] = age_group

    # Outcome measures
    st.write('16. What are the outcome measures of your trial?')
    outcome_measures = st.multiselect('Outcome Measures', ['Overall Survival', 'Adverse Events', 'Duration of Response', 'Other'])
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

    # Randomized
    rand = st.radio('17\. Is your trial randomized?', ['Yes', 'No'], index=None)
    if rand == 'Yes':
        features['allocation'] = 1
    else:
        features['allocation'] = 0 
    
    # Masking
    mask = st.radio('18\. What is the masking for your trial?', [0, 1, 2, 3, 4], index=None)
    features['masking'] = mask

    # Healthy volunteers
    vol = st.radio('19\. Will your trial include healthy volunteers?', ['Yes', 'No'], index=None)
    if vol == 'Yes':
        features['healthy_vol'] = 1
    else:
        features['healthy_vol'] = 0 

    # Outcome measures days
    st.write('20. What is the maximum duration from baseline to the primary outcome measure for one patient?')
    primary_max = st.number_input('Primary outcome measure duration', min_value = 0, max_value=10000, step=0.5, value=None, placeholder='Type a number...')
    unit = st.radio('Unit', ['Days', 'Months', 'Years'])
    if unit == 'Days':
        features['primary_max_days'] = primary_max
    elif unit == 'Months':
        features['primary_max_days'] = primary_max * 30
    elif unit == 'Years':
        features['primary_max_days'] = primary_max * 365
    
    st.write('21. What is the maximum duration from baseline to the secondary outcome measure for one patient?')
    secondary_max = st.number_input('Secondary outcome measure duration', min_value = 0, max_value=10000, step=0.5, value=None, placeholder='Type a number...')
    unit = st.radio('Unit', ['Days', 'Months', 'Years'])
    if unit == 'Days':
        features['secondary_max_days'] = secondary_max
    elif unit == 'Months':
        features['secondary_max_days'] = secondary_max * 30
    elif unit == 'Years':
        features['secondary_max_days'] = secondary_max * 365  

    # Submission
    submitted = st.form_submit_button('Submit')
    if submitted:
        if None in features:
            st.error('Please answer all questions.')
        else:
            st.switch_page('pages/loading.py')
        