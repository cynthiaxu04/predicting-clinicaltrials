shortcut_map = {'Adenocarcinoma':'adeno', 'Squamous Cell Carcinoma':'squamous cell', 'Transitional Cell Carcinoma': 'carcinoma',
                 'Basal Cell Carcinoma': 'carcinoma', 'Ductal Carcinoma': 'ductal', 'Other Carcinoma': 'carcinoma',
                 'Brain Cancer': 'brain', 'Sarcoma': 'sarcoma', 'Lymphoma': 'lymphoma', 'Leukemia': 'leukemia', 'Melanoma':'melanoma',
                  'Myeloma': 'myeloma', 'Pediatric Cancer': 'other', 'Pain relating to any disease': 'pain', 'Other': 'other'}

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