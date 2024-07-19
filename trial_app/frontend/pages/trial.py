import streamlit as st

st.header('Try us out!')
st.write('Below are a few questions to help us better understand your trial. Your responses will be used to generate the predicted duration.')

features = []
with st.form('Features'):

    # Cancer type
    type = st.selectbox('1\. What is the category of disease(s) your trial is researching?', 
                        ('Adenocarcinoma', 'Squamous Cell Carcinoma', 'Transitional Cell Carcinoma', 'Basal Cell Carcinoma',
                        'Ductal Carcinoma', 'Other Carcinoma', 'Brain Cancer', 'Sarcoma', 'Lymphoma', 'Leukemia', 'Melanoma',
                        'Myeloma', 'Pediatric Cancer', 'Pain relating to any disease', 'Other'), index=None)
    features.append(type)

    # Num locations
    num_sites = st.number_input('2\. How many sites will there be?', min_value = 0, max_value=3000, step=1, value=None, placeholder='Type a number...')
    if num_sites == 0 or num_sites is None:
        st.error('Your trial must have at least 1 site.')
    st.write('')
    features.append(num_sites)

    # US vs Non US
    st.write('3. Where will your trial take place?')
    us = st.checkbox('Includes USA location(s)')
    non_us = st.checkbox('Includes non-USA location(s)')
    if us and non_us:
        features.append(2)
    elif us:
        features.append(0)
    elif non_us:
        features.append(1)
    st.write('')

    # Patients
    st.write('4. How many subjects will be enrolled?')
    num_patients = st.number_input('Subjects', min_value = 0, max_value=10000, step=1, value=None, placeholder='Type a number...')
    if num_patients is None or num_patients == 0:
        st.error('Your trial must have at least 1 subject.')
    st.write('')
    features.append(num_patients)

    # Criteria
    st.write('5. What is the number of inclusion criteria?')
    num_inclusion = st.number_input('Inclusion criteria', min_value = 0, max_value=500, step=1, value=None, placeholder='Type a number...')
    if num_inclusion is None or num_inclusion == 0:
        st.error('Your trial must have at least 1 inclusion criteria.')
    features.append(num_inclusion)
    st.write('')
    st.write('6. What is the number of exclusion criteria?')
    num_exclusion = st.number_input('Exclusion criteria', min_value = 0, max_value=500, step=1, value=None, placeholder='Type a number...')
    if num_exclusion is None or num_exclusion == 0:
        st.error('Your trial must have at least 1 exclusion criteria.')
    features.append(num_exclusion)
    st.write('')

    # Sponsor type
    sponsor = st.radio('7\. What is the sponsor type for your trial?', ['Industry', 'Other'], index=None)
    features.append(sponsor)
    st.write('')

    # DMC
    has_dmc = st.radio('8\. Will you have a data monitoring committe?', ['Yes', 'No'], index=None)

    if has_dmc == 'Yes':
        has_dmc = 1
    elif has_dmc == 'No':
        has_dmc = 0
    features.append(has_dmc)
    st.write('')

    # Intervention model
    intervention = st.radio('9\. What is the intervention model for your trial?', ['Single Group', 'Parallel', 'Other'], index=None)
    if intervention == 'Single':
        features.append(0)
    elif intervention == 'Parallel':
        features.append(1)
    elif intervention == 'Other':
        features.append(2)

    # Intervention type
    st.write('10. What is the intervention type(s) for your trial?')
    intervention_type = st.multiselect(
        'Intervention Types',
        ['Procedure', 'Device', 'Behavioral', 'Drug', 'Radiation', 'Biological'])
    features.append(intervention_type)

    # Primary purpose
    st.write('11. What is the primary purpose of your trial?')
    treatment_purpose = st.checkbox('Treatment')
    diagnostic_purpose = st.checkbox('Diagnostic')
    prevention_purpose = st.checkbox('Prevention')
    supportive_purpose = st.checkbox('Supportive')
    features.append([treatment_purpose, diagnostic_purpose, prevention_purpose, supportive_purpose])

    # Groups
    st.write('12. How many groups will your trial have?')
    num_groups = st.number_input('Number of groups', min_value = 0, max_value=100, step=1, value=None, placeholder='Type a number...')
    if num_groups is None or num_groups == 0:
        st.error('Your trial must have at least 1 group.')
    features.append(num_groups)

    # Age groups
    age_group = st.radio('13\. What is the target age group?', ['Youth', 'Adult', 'All'], index=None)
    if age_group == 'Youth': 
        age_group = 0
    elif age_group == 'Adult':
        age_group = 1
    elif age_group == 'All':
        age_group = 2
    features.append(age_group)

    # Outcome measures
    st.write('14. What are the outcome measures of your trial?')
    outcome_measures = st.multiselect('Outcome Measures', ['Overall Survival', 'Adverse Events', 'Duration of Response', 'Other'])
    features.append(outcome_measures)

    # Randomized
    rand = st.radio('15\. Is your trial randomized?', ['Yes', 'No'], index=None)
    features.append(rand)

    # Submission
    submitted = st.form_submit_button('Submit')
    if submitted:
        if None in features:
            st.error('Please answer all questions.')
        else:
            st.switch_page('pages/loading.py')
        