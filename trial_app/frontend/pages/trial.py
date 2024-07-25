import streamlit as st
import pandas as pd

def conditions_map(condition):
  # squamous cell (carcinoma)
  if 'cell lung' in condition:
    return 'squamous cell'
  if 'head and neck' in condition:
    return 'squamous cell'
  if 'squamous cell' in condition:
    return 'squamous cell'
  if 'small cell' in condition:
    return 'squamous cell'
  if 'lung' in condition:
    return 'squamous cell'
  if 'keratosis' in condition:
    return 'squamous cell'
  if 'squamous' in condition:
    return 'squamous cell'
  if 'pelvic' in condition:
    return 'squamous cell'
  if 'uterine' in condition:
    return 'squamous cell'
  if 'ovarian' in condition:
    return 'squamous cell'
  if 'ovary' in condition:
    return 'squamous cell'
  if 'nsclc' in condition:
    return 'squamous cell'
  if 'oral' in condition:
    return 'squamous cell'

  # myeloma (blood)
  if 'myeloma' in condition:
    return 'myeloma'

  # sarcoma (bone/soft tissue)
  if 'sarcoma' in condition:
    return 'sarcoma'
  if 'gastrointestinal' in condition:
    return 'sarcoma'
  if 'bone' in condition:
    return 'sarcoma'

  # lymphoma (lymph system)
  if 'lymphoma' in condition:
    return 'lymphoma'
  if 'lymphoid' in condition:
    return 'lymphoma'
  if 'lymph' in condition:
    return 'lymphoma'

  # brain
  if 'brain cancer' in condition:
    return 'brain'
  if 'brain' in condition:
    return 'brain'
  if 'glioblastoma' in condition:
    return 'brain'
  if 'glioma' in condition:
    return 'brain'

  # melanoma (skin)
  if 'melanoma' in condition:
    return 'melanoma'
  if 'skin' in condition:
    return 'melanoma'

  # adenocarcinoma (carcinoma)
  if 'adenocarcinoma' in condition:
    return 'adeno'
  if 'prostate cancer' in condition:
    return 'adeno'
  if 'rectal' in condition:
    return 'adeno'
  if 'kidney' in condition:
    return 'adeno'
  if 'renal' in condition:
    return 'adeno'
  if 'gastric' in condition:
    return 'adeno'
  if 'gi' in condition:
    return 'adeno'
  if 'digestive' in condition:
    return 'adeno'
  if 'esophageal' in condition:
    return 'adeno'
  if 'cervix' in condition:
    return 'adeno'
  if 'cervical' in condition:
    return 'adeno'
  if 'liver' in condition:
    return 'adeno'
  if 'hepatic' in condition:
    return 'adeno'
  if 'hepatocellular' in condition:
    return 'adeno'
  if 'thyroid' in condition:
    return 'adeno'
  if 'abdomin' in condition:
    return 'adeno'

  # ductal (adenocarcinoma)
  if 'breast' in condition:
    return 'ductal'
  if 'pancreatic' in condition:
    return 'ductal'
  if 'pancreas' in condition:
    return 'ductal'

  # leukemia (blood)
  if 'leukemia' in condition:
    return 'leukemia'
  if 'hematopoietic' in condition:
    return 'leukemia'
  if 'myelofibrosis' in condition:
    return 'leukemia'

  # pediatric
  if 'blastoma' in condition:
    return 'pediatric'
  if 'pediatric' in condition:
    return 'pediatric'
  if 'child' in condition:
    return 'pediatric'

  # transitional cell (carcinoma)
  if 'bladder' in condition:
    return 'transitional cell'

  # neoplasm (non cancer)
  if 'neoplasm' in condition:
    return 'neoplasm'

  # pain and other
  if 'pain' in condition:
    return 'pain'
  elif 'carcinoma' in condition:
    return 'carcinoma'
  return 'other'

# # change root
# root = '/Users/adelinechin/210-capstone-clinicaltrials/data/'
# conditions_5yr_survival_map_df = pd.read_csv(root + 'survival_dict.csv')
# conditions_5yr_survival_map_df = conditions_5yr_survival_map_df.set_index('type')
# conditions_5yr_survival_map = conditions_5yr_survival_map_df.to_dict('index')

st.header('Try us out!')
st.write('Below are a few questions to help us better understand your trial. Your responses will be used to generate the predicted duration.')

features = dict()
with st.form('Features'):

    # Cancer type
    type = st.selectbox('1\. What is the category of disease(s) your trial is researching?', 
                        ('Adenocarcinoma', 'Squamous Cell Carcinoma', 'Transitional Cell Carcinoma', 'Basal Cell Carcinoma',
                        'Ductal Carcinoma', 'Other Carcinoma', 'Brain Cancer', 'Sarcoma', 'Lymphoma', 'Leukemia', 'Melanoma',
                        'Myeloma', 'Pediatric Cancer', 'Pain relating to any disease', 'Other'), index=None)
    
    # type = conditions_map(type)
    # if type in conditions_5yr_survival_map.keys():
    #     survival = conditions_5yr_survival_map[type]
    #     survival = survival['5yr_survival']
    #     features['survival_5yr_relative'] = survival
    # else:
    #    features['survival_5yr_relative'] = 0.5
    
    features['survival_5yr_relative'] = 0.5
    features['number_of_conditions'] = 1
    features['max_treatment_duration'] = 4320
    features['min_treatment_duration'] = 14

    # Phase
    phase = st.radio('2\. What is the phase of your trial?', [1, 2, 3], index=None)
    if phase == 2:
       features['phase_PHASE2_PHASE3'] = 1
    else:
       features['phase_PHASE2_PHASE3'] = 0
    if phase == 3:
       features['phase_PHASE3'] = 1
    else:
       features['phase_PHASE3'] = 0

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

    # DMC
    has_dmc = st.radio('8\. Will you have a data monitoring committe?', ['Yes', 'No'], index=None)

    if has_dmc == 'Yes':
        has_dmc = 1
    elif has_dmc == 'No':
        has_dmc = 0
    features['has_dmc'] = has_dmc
    st.write('')

    # Responsible party
    resp_party = st.radio('9\. Who will be the responsible party?', ['PI', 'Sponsor', 'PI and Sponsor'], index=None)
    if resp_party == 'PI':
        features['resp_party'] = 0
    elif resp_party == 'Sponsor':
        features['resp_party'] = 1
    else:
        features['resp_party'] = 2

    # Intervention model
    intervention = st.radio('10\. What is the intervention model for your trial?', ['Single Group', 'Parallel', 'Other'], index=None)
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
        ['Procedure', 'Device', 'Behavioral', 'Drug', 'Radiation', 'Biological'])
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
    st.write('12. What is the primary purpose of your trial?')
    treatment_purpose = st.checkbox('Treatment')
    diagnostic_purpose = st.checkbox('Diagnostic')
    prevention_purpose = st.checkbox('Prevention')
    supportive_purpose = st.checkbox('Supportive')

    if treatment_purpose:
        features['treatment_purpose'] = 1
    else: 
       features['treatment_purpose'] = 0
    if diagnostic_purpose:
        features['diagnostic_purpose'] = 1
    else:
       features['diagnostic_purpose'] = 0
    if prevention_purpose:
       features['prevention_purpose'] = 1
    else: 
       features['prevention_purpose'] = 0

    # Groups
    st.write('13. How many groups will your trial have?')
    num_groups = st.number_input('Number of groups', min_value = 0, max_value=100, step=1, value=None, placeholder='Type a number...')
    if num_groups is None or num_groups == 0:
        st.error('Your trial must have at least 1 group.')
    features['number_of_groups'] = num_groups

    # Outcome measures
    st.write('15. What are the outcome measures of your trial?')
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
    
    # Masking
    mask = st.radio('17\. What is the masking for your trial?', [0, 1, 2, 3, 4], index=None)
    features['masking'] = mask

    # Healthy volunteers
    vol = st.radio('18\. Will your trial include healthy volunteers?', ['Yes', 'No'], index=None)
    if vol == 'Yes':
        features['healthy_vol'] = 1
    else:
        features['healthy_vol'] = 0 

    # Outcome measures days
    st.write('19. What is the maximum duration from baseline to the primary outcome measure for one patient?')
    primary_max = st.number_input('Primary outcome measure duration', min_value = 0.0, max_value=10000.0, step=0.5, value=None, placeholder='Type a number...')
    unit = st.radio('Unit', ['Days', 'Months', 'Years'])
    if unit == 'Days':
        features['primary_max_days'] = primary_max
    elif unit == 'Months':
        features['primary_max_days'] = primary_max * 30
    elif unit == 'Years':
        features['primary_max_days'] = primary_max * 365
    
    st.write('20. What is the maximum duration from baseline to the secondary outcome measure for one patient?')
    secondary_max = st.number_input('Secondary outcome measure duration', min_value = 0.0, max_value=10000.0, step=0.5, value=None, placeholder='Type a number...')
    unit_secondary = st.radio('Unit', ['Days', 'Months', 'Years'], key=2)
    if unit_secondary == 'Days':
        features['secondary_max_days'] = secondary_max
    elif unit_secondary == 'Months':
        features['secondary_max_days'] = secondary_max * 30
    elif unit_secondary == 'Years':
        features['secondary_max_days'] = secondary_max * 365  

    # Submission
    submitted = st.form_submit_button('Submit')
    if submitted:
        if len(features) < 18:
            st.error('Please answer all questions.')
        else:
            st.switch_page('pages/loading.py')
        