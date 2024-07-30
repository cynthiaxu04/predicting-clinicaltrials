def conditions_map(condition):
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
  if 'myeloma' in condition:
    return 'myeloma'
  if 'sarcoma' in condition:
    return 'sarcoma'
  if 'lymphoma' in condition:
    return 'lymphoma'
  if 'brain cancer' in condition:
    return 'brain'
  if 'melanoma' in condition:
    return 'melanoma'
  if 'adenocarcinoma' in condition:
    return 'adeno'
  if 'prostate cancer' in condition:
    return 'adeno'
  if 'breast' in condition:
    return 'ductal'
  if 'leukemia' in condition:
    return 'leukemia'
  if 'colorectal' in condition:
    return 'adeno'
  if 'glioblastoma' in condition:
    return 'brain'
  if 'kidney' in condition:
    return 'adeno'
  if 'renal' in condition:
    return 'adeno'
  if 'hematopoietic' in condition:
    return 'leukemia'
  if 'lymphoid' in condition:
    return 'lymphoma'
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
  if 'nsclc' in condition:
    return 'squamous cell'
  if 'thyroid' in condition:
    return 'adeno'
  if 'pain' in condition:
    return 'pain'
  elif 'carcinoma' in condition:
    return 'carcinoma'
  return 'other'

conditions_5yr_survival_map = {'myeloma': 0.598, 'squamous cell': 0.99, 
                                'adeno': 0.175, 'carcinoma': 0.99, 'leukemia': 0.65, 
                               'ductal': 0.99, 'sarcoma': 0.65,
                               'lymphoma': 0.83, 'melanoma': 0.94, 
                               'brain': 0.326, 'pain': 0.68, 'other': 0.68}

conditions_max_treatment_duration_map = {'myeloma': 180, 'squamous cell': 49, 'adeno': 1080, 
                                             'carcinoma': 1440, 'leukemia': 1095, 'ductal': 1825, 
                                             'sarcoma': 1825,'lymphoma': 730, 'melanoma': 730, 
                                             'brain': 4320, 'pain': 4320, 'other': 4320}

conditions_min_treatment_duration_map = {'myeloma': 90, 'squamous cell': 14, 'adeno': 360, 
                                             'carcinoma': 360, 'leukemia': 730, 'ductal': 365, 'sarcoma': 240,
                                             'lymphoma': 180, 'melanoma': 150, 'brain': 1080, 'pain': 14, 'other': 14}