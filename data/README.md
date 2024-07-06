# Data Pipeline


## ```processing.py``` for cleaning and processing the data for model training

The purpose of this script is to validate the correct raw data columns are available (from the clinicaltrials.gov website) and to process the data so that it is ready for modeleling experiments.

1. The raw data file, manually downloaded from CT.gov, should be saved as a ```.csv``` in this ```/data``` folder.
2. To process this data, ensure you are in the ```/data``` directory. Then, run the command:
```
python processing.py <file_name.csv> ../config/default_cols.yaml
```
3. This script returns two data ```csv``` files, in which the cleaned data is split 70:30 (70% for training & validation, 30% for testing):
```
cleaned_data_train.csv
cleaned_data_test.csv
```
### Notes
* The relevant columns for data processing are explicity declared in ```/config/default_cols.yaml```. You can add or remove columns as necessary.
* If any columns are missing, these will be printed in the ```/log/data_validation.log``` file.

## Data Engineering Notes
This section describes each column of the ```cleaned_data.csv```.

| Column Name | Data Type | Description|
| -------- | -------- | -------- |
| 'protocolSection_identificationModule_nctId'| object, str | unique ID for study|
| 'primary_study_duration_days'| float64| duration of study from start until all primary outcomes measured|
| 'study_duration_days'| float64| duration of study from start until all outcomes measured|
| 'primary_eq_bins'| object, interval| quantile binned primary study duration, n_bins=5|
| 'study_eq_bins'| object, interval| quantile binned study duration, n_bins=5|
| 'study_eq_labels'| int, categorical| labels for bins|
| 'primary_eq_labels'| int, categorical| labels for bins|
| 'number_of_conditions'| int| the number of conditions the study examines|
| 'number_of_groups'| int| the number of arm groups the study has; "Arm" means a pre-specified group or subgroup of participant(s) in a clinical trial assigned to receive specific intervention(s) (or no intervention) according to a protocol.|
| 'age_group'| int, categorical| age groups of study participants; includes "youth" (children and adults <20 y.o.), "adult" (adults >20 y.o.), and "all" (all ages)|
| 'num_locations'| int| number of locations where the study is condcuted|
| 'location'| int, cateogrical| whether study location(s) are in the USA, outside of the USA, or both|
| 'num_inclusion'| int| number of patient inclusion criteria|
| 'num_exclusion'| int| number of patient exclusion criteria|
| 'number_of_intervention_types'| int| number of intervention types of the study, e.g. behavioral, drug, device, etc.|
| 'sponsor_type'| int, categorical| whether the sponsor of the study is from industry or other|
| 'intervention_model'| int, categorical| the type of intervention model, e.g. single group, parallel or other|
| 'resp_party'| int, categorical| responsible party of the study, e.g. principal investigator, sponsor, or sponsor investigator|
| 'has_dmc'| bool| whether or not the study has a data monitoring committee|
| 'phase'| int, categorical| phase of the study, e.g. Phase I, II, III, or IV, one-hot encoded later|
| 'allocation'| int, categorical| allocation type, the method by which participants are assigned to an arm, e.g. randomized or non-randomized|
| 'masking'| int, categorical| party or parties involved in the clinical trial who are prevented from having knowledge of the interventions assigned to individual participants, e.g. none, single, double, triple, quadruple|
| 'enroll_count'| int| the number of patients enrolled in the study|
| 'healthy_vol'| bool| whether or not the study includes healthy volunteers|
| 'treatment_purpose'| bool| whether or not the purpose of the study is for treatment|
| 'diagnostic_purpose'| bool| whether or not the purpose of the study is diagnostic|
| 'prevention_purpose'| bool| whether or not the purpose of the study is for prevention|
| 'supportive_purpose'| bool| whether or not the purpose of the study is supportive|
| 'procedure_intervention'| bool| whether or not the type of study intervention is procedural|
| 'device_intervention'| bool| whether or not the type of study intervention is with a device|
| 'behavioral_intervention'| bool| whether or not the type of study intervention is behavioral|
| 'drug_intervention'| bool| whether or not the type of study intervention is with a drug|
| 'radiation_intervention'| bool| whether or not the type of study intervention is with radiation|
| 'biological_intervention'| bool| whether or not the type of study intervention is biological|
| 'os_outcome_measure'| bool | whether or not the outcome measure is overall survival|
| 'dor_outcome_measure'| bool | whether or not the outcome measure is duration of response|
| 'ae_outcome_measure'| bool | whether or not the outcome measure is adverse outcome|
| 'primary_max_days'| float| time measures extracted from primary outcome measures|
| 'secondary_max_days'| float| time measures extracted from secondary outcome measures|
| 'max_treatment_duration'| float| maximum treatment duration in days for a condition type|
| 'min_treatment_duration'| float| minimum treatment duration in days for a condition type|
| 'survival_5yr_relative'| float| relative rate of 5 year survival for a condition|
| 'conditions_category_num'| int, categorical| category/type of condition, e.g. myeloma, adeno, carcinoma, etc.|