# Data Pipeline

## ```fetch_data.py``` for fetching the latest data from clinicaltrials.gov
The purpose of this script is to fetch data from clinicaltrials.gov via their API with user specified search parameters.

Work in progress, TBA


## ```processing.py``` for cleaning and processing the data for model training

The purpose of this script is to validate the correct raw data columns are available (from the clinicaltrials.gov website) and to process the data so that it is ready for modeleling experiments.

1. The raw data file should be saved as a ```.csv``` in this ```/data``` folder.
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
This section describes each column of the ```cleaned_data.csv``` including:
* data type
* what raw column(s) it was derived from
* any other notes

'primary_study_duration_days'
'study_duration_days'
'number_of_conditions'
'number_of_groups'
'age_group'
'num_locations'
'number_of_intervention_types'
'sponsor_type'
'intervention_model'
'resp_party'
'has_dmc'
'protocolSection_designModule_phases'
'allocation'
'masking'
'protocolSection_designModule_enrollmentInfo_count'
'protocolSection_eligibilityModule_healthyVolunteers'
'treatment_purpose'
'diagnostic_purpose'
'prevention_purpose'
'supportive_purpose'
'procedure_intervention'
'device_intervention'
'behavioral_intervention'
'drug_intervention'
'radiation_intervention'
'biological_intervention'
'os_outcome_measure'
'dor_outcome_measure'
'ae_outcome_measure'
'primary_max_days'
'secondary_max_days'