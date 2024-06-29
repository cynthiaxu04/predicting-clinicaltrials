import pandas as pd
import numpy as np
import os
import re
import ast
import argparse
import yaml
import logging
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans


#########################################################################################################################
# Set up logging
log_filename = 'data_validation.log'  # Specify the log file name
log_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'log', log_filename)

# Create a logger instance
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set the logging level (INFO, WARNING, ERROR, etc.)

# Create a file handler for the log file
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.INFO)  # Set the logging level for the file handler
formatter = logging.Formatter('%(asctime)s - %(message)s')  # Specify the log message format
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)

# Functions for data processing

def yaml_file(value):
    if not value.endswith(('.yaml')):
        raise argparse.ArgumentTypeError(f'File must have .yaml extension: {value}')
    return value

def csv_file(value):
    if not value.endswith(('.csv')):
        raise argparse.ArgumentTypeError(f'File must have .csv extension: {value}')
    return value

def get_parser():
    parser = argparse.ArgumentParser(description='Validate CSV file columns using a YAML schema.')
    parser.add_argument('csv_file', type=csv_file, help='Path to the CSV file')
    parser.add_argument('yaml_file', type=yaml_file, help='Path to the YAML schema file')

    return parser.parse_args()

def load_yaml(file_path):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def validate_columns(csv_file, yaml_file):
    # Load CSV file
    df = pd.read_csv(csv_file)

    # Load YAML schema
    config = load_yaml(yaml_file)

    # Validate columns
    expected_columns = config['columns']
    actual_columns = df.columns.tolist()

    missing_columns = [col for col in expected_columns if col not in actual_columns]
    # extra_columns = [col for col in actual_columns if col not in expected_columns]

    if missing_columns:
        missing_msg = f"Missing columns in CSV: {', '.join(missing_columns)}"
        logger.warning(missing_msg)
        print(missing_msg)

    # if not missing_columns and not extra_columns:
    if not missing_columns:
        success_msg = "CSV file columns match the YAML schema."
        logger.info(success_msg)
        print(success_msg)

# Function to convert dates to datetime, handling YYYY-MM format
def convert_to_datetime(date_str):
    try:
        return pd.to_datetime(date_str, format='%Y-%m-%d')
    except ValueError:
        try:
            return pd.to_datetime(date_str, format='%Y-%m') + pd.offsets.MonthBegin(0)  # Assume first day of the month
        except ValueError:
            return np.nan  # If conversion fails, return NaN

def safe_eval(item):
    if isinstance(item, str):  # Check if the item is a string
        return eval(item)
    return item  # Return the item unchanged if it's not a string

# Function to extract 'type' from list of dictionaries
def extract_type(interventions):
    return [d['type'] for d in interventions]

# function to count number of sites
def count_loc(value):
  if pd.isna(value):
    return np.nan
  else:
    new_string = value.split(", ")
    counter = 0
    for i in new_string:
      if "facility" in i:
        counter += 1
    return counter

# function to count number of sites
def trial_loc(value):
  if pd.isna(value):
    return np.nan
  else:
    new_string = value.split(", ")
    temp_list = []
    for i in new_string:
      if "country" in i:
        temp_list.append(i)
    has_us = [i for i in temp_list if "United States" in i]

    if all(i == temp_list[0] for i in temp_list) and has_us:
        loc = 'USA'
    elif not has_us:
        loc = "non-USA"
    else:
        loc = "USA & non-USA"
    return loc

def extract_measures(outcomes):
    # Check if 'outcomes' is iterable (i.e., a list in this context)
    if isinstance(outcomes, list):
        return [item.get('measure', 'No description available') for item in outcomes]
    else:
        return ['No description available']  # Return a list with a default message if not iterable

def extract_timeframes(outcomes):
    # Check if 'outcomes' is iterable (i.e., a list in this context)
    if isinstance(outcomes, list):
        return [item.get('timeFrame', 'No description available') for item in outcomes]
    else:
        return ['No description available']  # Return a list with a default message if not iterable

def extract_time_length_from_list(timeFrames):
    # Function to extract time length from a single timeframe string
    def extract_time_length(timeFrame):
        # Regular expression to find numbers followed by time units
        time_pattern = re.compile(r"(\d+)\s*(hours?|days?|weeks?|months?|years?)", re.IGNORECASE)
        results = time_pattern.findall(timeFrame)

        # Convert the extracted time durations to a more structured format
        time_durations = []
        for amount, unit in results:
            time_durations.append((int(amount), unit.lower()))
        return time_durations  # Make sure this return statement is aligned with the for-loop, not inside it

    # Apply the extract_time_length function to each item in the list and combine results
    combined_durations = []
    for timeFrame in timeFrames:
        if timeFrame:  # This checks if the timeFrame is not empty or None
            combined_durations.extend(extract_time_length(timeFrame))
    return combined_durations

def convert_to_days(amount, unit):
    if unit == 'hours':
        return np.ceil(amount / 24)
    elif unit == 'weeks':
        return amount * 7
    elif unit == 'months':
        return amount * 30  # Simplified conversion, assuming each month has 30 days
    elif unit == 'years':
        return amount * 365  # Simplified conversion, assuming each year has 365 days
    elif unit == 'days':
        return amount
    else:
        return 0

def find_max_duration(durations):
    if not durations:  # Check if the list is empty
        return np.nan # Use numpy Nan into indicate no data
    max_days = max(convert_to_days(amount, unit) for amount, unit in durations)
    return max_days

# Function to remove specified characters
def remove_special_chars(col):
    return col.replace("[", "").replace("]", "").replace("'", "").replace(",", "_")

def count_criteria(criteria):
    # pattern
    # line break, \n, followed by any amount of whitespace
    # followed by an alphanumeric character 1-2 characters in length followed by a period
    # OR an asterisk
    # OR a hyphen
    pattern =  r'\n\s*\w{1,2}\.|[*]|\-'
    if "exclusion criteria" in criteria.lower():
        parts = criteria.lower().split("exclusion criteria")
        inclusion_criteria = parts[0]
        exclusion_criteria = parts[1]

        inclusion_matches = re.findall(pattern, inclusion_criteria)
        exclusion_matches = re.findall(pattern, exclusion_criteria)
        num_inclusion = len(inclusion_matches)
        num_exclusion = len(exclusion_matches)
    else:
        inclusion_matches = re.findall(pattern, criteria)
        num_inclusion = len(inclusion_matches)
        num_exclusion = np.nan
    
    return num_inclusion, num_exclusion

#########################################################################################################################
# Code to do actual data processing
if __name__ == "__main__":
    # get command line arguments
    args = get_parser()
    # validate the data csv file
    # print(args.yaml_file)
    validate_columns(args.csv_file, args.yaml_file)

    #start data processing
    df = pd.read_csv(args.csv_file)
    # len of rows
    rows0 = len(df)
    
    #study duration, float
    df['start_date'] = df['protocolSection_statusModule_startDateStruct_date'].apply(convert_to_datetime)
    df['primary_completion_date'] = df['protocolSection_statusModule_primaryCompletionDateStruct_date'].apply(convert_to_datetime)
    df['completion_date'] = df['protocolSection_statusModule_completionDateStruct_date'].apply(convert_to_datetime)
    df['primary_study_duration_days'] = (df['primary_completion_date'] - df['start_date']).dt.days
    df['study_duration_days'] = (df['completion_date'] - df['start_date']).dt.days

    # cols that are vital for accurate prediction
    need_cols = [
        'primary_study_duration_days',
        'study_duration_days',
        'protocolSection_designModule_enrollmentInfo_count']

    # remove rows with NaNs for primary_study_duration_days, study_duration_days, num_locations, enrollmentinfo_count
    df = df.dropna(subset=need_cols).copy()
    # len of rows after dropping NaNs
    rows1 = len(df)
    msg = f"{rows0 - rows1} rows were dropped due to missing values in one of: {need_cols}"
    logger.info(msg)
    print(msg)

    # make bins with k means clustering
    # Desired number of intervals
    n_intervals = 5

    df['study_eq_bins'] = pd.qcut(df['study_duration_days'], q=n_intervals)
    df['primary_eq_bins'] = pd.qcut(df['primary_study_duration_days'], q=n_intervals)

    df['study_eq_labels'] = df['study_eq_bins'].cat.codes
    df['primary_eq_labels'] = df['primary_eq_bins'].cat.codes

    bins_dict = df.groupby('study_eq_labels')['study_eq_bins'].apply(lambda x: x.unique()[0]).to_dict()
    msg2 = f"Bin labels and their corresponding intervals are: {bins_dict}"
    print(msg2)
    logger.info(msg2)

    # get number of inclusion and exclusion criteria
    df[['num_inclusion', 'num_exclusion']]= df['protocolSection_eligibilityModule_eligibilityCriteria'].apply(count_criteria).apply(pd.Series)

    # sponsor type, categorical
    spon_map = {
        'OTHER_GOV': 'OTHER',
        'NETWORK': 'OTHER',
        'NIH': 'OTHER',
        'FED': 'OTHER',
        'INDIV': 'OTHER'
    }

    spon_map2 = {
        "INDUSTRY": 1,
        "OTHER": 0
    }

    df['sponsor_type0'] = df['protocolSection_sponsorCollaboratorsModule_leadSponsor_class'].replace(spon_map)
    df['sponsor_type'] = df['sponsor_type0'].map(spon_map2)

    # number of conditions, int
    df['protocolSection_conditionsModule_conditions'] = df['protocolSection_conditionsModule_conditions'].apply(ast.literal_eval)
    df['number_of_conditions'] = df['protocolSection_conditionsModule_conditions'].apply(lambda x: len(x))

    # intervention model, categorical (mapped to int/float)
    assign_map = {
        "CROSSOVER": "OTHER",
        "SEQUENTIAL": "OTHER",
        "FACTORIAL": "OTHER"
    }

    assign_map2 = {
        "SINGLE_GROUP": 0,
        "PARALLEL": 1,
        "OTHER": 2
    }

    df['intervention_model0'] = df['protocolSection_designModule_designInfo_interventionModel'].replace(assign_map)
    df['intervention_model'] = df['intervention_model0'].map(assign_map2)

    # primary purpose, bool/int
    # Extract purposes into new columns
    df['protocolSection_designModule_designInfo_primaryPurpose'] = df['protocolSection_designModule_designInfo_primaryPurpose'].fillna('')
    df['treatment_purpose'] = df['protocolSection_designModule_designInfo_primaryPurpose'].apply(lambda x: 1 if 'TREATMENT' in x else 0)
    df['diagnostic_purpose'] = df['protocolSection_designModule_designInfo_primaryPurpose'].apply(lambda x: 1 if 'DIAGNOSTIC' in x else 0)
    df['prevention_purpose'] = df['protocolSection_designModule_designInfo_primaryPurpose'].apply(lambda x: 1 if 'PREVENTION' in x else 0)
    df['supportive_purpose'] = df['protocolSection_designModule_designInfo_primaryPurpose'].apply(lambda x: 1 if 'SUPPORTIVE_CARE' in x else 0)

    # intervention type, bool/int
    # Extract interventions into new columns
    df['procedure_intervention'] = df['protocolSection_armsInterventionsModule_interventions'].apply(lambda x: 1 if 'PROCEDURE' in x else 0)
    df['device_intervention'] = df['protocolSection_armsInterventionsModule_interventions'].apply(lambda x: 1 if 'DEVICE' in x else 0)
    df['behavioral_intervention'] = df['protocolSection_armsInterventionsModule_interventions'].apply(lambda x: 1 if 'BEHAVIORAL' in x else 0)
    df['drug_intervention'] = df['protocolSection_armsInterventionsModule_interventions'].apply(lambda x: 1 if 'DRUG' in x else 0)
    df['radiation_intervention'] = df['protocolSection_armsInterventionsModule_interventions'].apply(lambda x: 1 if 'RADIATION' in x else 0)
    df['biological_intervention'] = df['protocolSection_armsInterventionsModule_interventions'].apply(lambda x: 1 if 'BIOLOGICAL' in x else 0)

    # number of groups, int
    df['protocolSection_armsInterventionsModule_armGroups'] = df['protocolSection_armsInterventionsModule_armGroups'].apply(safe_eval)
    df['number_of_groups'] = df['protocolSection_armsInterventionsModule_armGroups'].apply(lambda x: len(x) if isinstance(x, list) else 0)

    # number of intervention types, int
    df['protocolSection_armsInterventionsModule_interventions'] = df['protocolSection_armsInterventionsModule_interventions'].apply(eval)
    df['intervention_types'] = df['protocolSection_armsInterventionsModule_interventions'].apply(extract_type)
    df['number_of_intervention_types'] = df['intervention_types'].apply(len)

    # age group, categorical (mapped to int/float)
    age_map = {
        "['ADULT', 'OLDER_ADULT']": "adult",
        "['ADULT']": "adult",
        "['OLDER_ADULT']": "adult",
        "['CHILD']": "youth",
        "['CHILD', 'ADULT']": "youth",
        "['CHILD', 'ADULT', 'OLDER_ADULT']": "all"
    }

    age_map2 = {
        "youth": 0,
        "adult": 1,
        "all": 2
    }
    df["age_group0"] = df["protocolSection_eligibilityModule_stdAges"].map(age_map)
    df["age_group"] = df["age_group0"].map(age_map2)

    # number of locations, int
    df["num_locations"] = df["protocolSection_contactsLocationsModule_locations"].apply(count_loc)

    loc_map = {
        "USA": 0,
        "non-USA": 1,
        "USA & non-USA": 2
    }
    #location of trials, categorical
    df['location0'] = df["protocolSection_contactsLocationsModule_locations"].apply(trial_loc)
    df['location'] = df['location0'].map(loc_map)

    # outcome measures, bool/int
    # Combine outcome measures
    # May need to separate this and re-do outcome measures features depending on our y variable
    df['outcome_measures'] = df['protocolSection_outcomesModule_primaryOutcomes'] + df['protocolSection_outcomesModule_secondaryOutcomes']
    df['outcome_measures'] = df['outcome_measures'].astype('str')
    df['outcome_measures'] = df['outcome_measures'].str.lower()

    df['os_outcome_measure'] = df['outcome_measures'].apply(lambda x: 1 if ' OS ' in x else 0) # sometimes abbreviated as OS
    df['os_outcome_measure2'] = df['outcome_measures'].apply(lambda x: 1 if 'overall survival' in x else 0) # sometimes fully spelled out

    # Make sure there's no overlap
    df['os_outcome_measure'] = df['os_outcome_measure'] + df['os_outcome_measure2']
    df = df.drop(columns=['os_outcome_measure2'])
    df['os_outcome_measure'] = df['os_outcome_measure'].apply(lambda x: x if x==0 else 1)

    # adverse outcome, bool/int
    df['ae_outcome_measure'] = df['outcome_measures'].apply(lambda x: 1 if 'adverse event' in x else 0)

    # DOM response, bool/int
    # Duration of Response OM
    df['dor_outcome_measure'] = df['outcome_measures'].apply(lambda x: 1 if ' DOR ' in x else 0) # sometimes abbreviated as DOR
    df['dor_outcome_measure2'] = df['outcome_measures'].apply(lambda x: 1 if 'duration of response' in x else 0) # sometimes fully spelled out

    # Make sure there's no overlap
    df['dor_outcome_measure'] = df['dor_outcome_measure'] + df['dor_outcome_measure2']
    df = df.drop(columns=['dor_outcome_measure2'])
    df['dor_outcome_measure'] = df['dor_outcome_measure'].apply(lambda x: x if x==0 else 1)

    # max timeframe from primary/secondary outcome measures, float
    df['protocolSection_outcomesModule_primaryOutcomes'] = df['protocolSection_outcomesModule_primaryOutcomes'].apply(safe_eval)
    df['protocolSection_outcomesModule_secondaryOutcomes'] = df['protocolSection_outcomesModule_secondaryOutcomes'].apply(safe_eval)

    # Applying the function to create the new column
    df['primary_measure'] = df['protocolSection_outcomesModule_primaryOutcomes'].apply(extract_measures)
    df['secondary_measure'] = df['protocolSection_outcomesModule_secondaryOutcomes'].apply(extract_measures)

    # Applying the function to create the new column
    df['primary_timeFrame'] = df['protocolSection_outcomesModule_primaryOutcomes'].apply(extract_timeframes)
    df['secondary_timeFrame'] = df['protocolSection_outcomesModule_secondaryOutcomes'].apply(extract_timeframes)

    df['primary_duration'] = df['primary_timeFrame'].apply(extract_time_length_from_list)
    df['secondary_duration'] = df['secondary_timeFrame'].apply(extract_time_length_from_list)

    # Applying the function to the dataframe
    df['primary_max_days'] = df['primary_duration'].apply(find_max_duration)
    df['secondary_max_days'] = df['secondary_duration'].apply(find_max_duration)


    # has data monitoring committee, bool/int
    df['protocolSection_oversightModule_oversightHasDmc'] = df['protocolSection_oversightModule_oversightHasDmc'].astype('str')
    df['protocolSection_oversightModule_oversightHasDmc'] = df['protocolSection_oversightModule_oversightHasDmc'].str.lower()
    dmc_map = {
        'true': 1,
        'false': 0,
    }
    df['has_dmc'] = df['protocolSection_oversightModule_oversightHasDmc'].map(dmc_map)

    # responsible party, categorical (mapped to int)
    party_map = {
        "PRINCIPAL_INVESTIGATOR": 0,
        "SPONSOR": 1,
        "SPONSOR_INVESTIGATOR": 2
    }

    df['resp_party'] = df['protocolSection_sponsorCollaboratorsModule_responsibleParty_type'].map(party_map)

    # allocation, categorical (mapped to int)
    allo_map = {
        'NON_RANDOMIZED': 0,
        'RANDOMIZED': 1
    }

    df['allocation'] = df['protocolSection_designModule_designInfo_allocation'].map(allo_map)

    # masking, categorical (mapped to int)
    mask_map = {
        "NONE": 0,
        "SINGLE": 1,
        "DOUBLE": 2,
        "TRIPLE": 3,
        "QUADRUPLE": 4
    }

    df['masking'] = df['protocolSection_designModule_designInfo_maskingInfo_masking'].map(mask_map)

    # rename some columns
    df = df.rename(columns={'protocolSection_designModule_phases': 'phase',
                             'protocolSection_designModule_enrollmentInfo_count': 'enroll_count',
                             'protocolSection_eligibilityModule_healthyVolunteers': 'healthy_vol'})

    #make a dataframe with just the columns of interest
    cols = [
        'protocolSection_identificationModule_nctId',
        'primary_study_duration_days',
        'study_duration_days',
        'primary_eq_bins',
        'study_eq_bins',
        'study_eq_labels',
        'primary_eq_labels',
        'number_of_conditions',
        'number_of_groups',
        'age_group',
        'num_locations',
        'location',
        'num_inclusion',
        'num_exclusion',
        # 'intervention_types',
        'number_of_intervention_types',
        'sponsor_type',
        'intervention_model',
        'resp_party',
        'has_dmc',
        'phase',
        'allocation',
        'masking',
        'enroll_count',
        'healthy_vol',
        'treatment_purpose',
        'diagnostic_purpose',
        'prevention_purpose',
        'supportive_purpose',
        'procedure_intervention',
        'device_intervention',
        'behavioral_intervention',
        'drug_intervention',
        'radiation_intervention',
        'biological_intervention',
        'os_outcome_measure',
        'dor_outcome_measure',
        'ae_outcome_measure',
        'primary_max_days',
        'secondary_max_days'
        # 'protocolSection_outcomesModule_primaryOutcomes',
        # 'protocolSection_outcomesModule_secondaryOutcomes',
        # 'protocolSection_eligibilityModule_eligibilityCriteria'
    ]

    clean_df = df[cols].copy()

    # handle missing values by filling with the mode to preserve data distribution
    nan_counts = clean_df.isna().sum()

    nan_cols = nan_counts[nan_counts > 0].index.tolist()
    # remove mode calculation for 'primary_max_days' and 'secondary_max_days'
    nan_cols.remove('primary_max_days')
    nan_cols.remove('secondary_max_days')
    for column in nan_cols:
        mode_value = clean_df[column].mode()[0]  # Calculate the mode
        clean_df[column] = clean_df[column].fillna(mode_value)

    # one hot encode remaining object columns
    object_columns = list(clean_df.select_dtypes(include=['object']).columns)
    object_columns = [i for i in object_columns if 'nctId' not in i]
    encoded_df = pd.get_dummies(clean_df, columns=object_columns)

    # Apply function to all column names
    encoded_df.columns = encoded_df.columns.map(remove_special_chars)

    # now split the data
    train_df, test_df = train_test_split(encoded_df, test_size=0.3, random_state=42, shuffle=True)

    # save the data file
    train_df.to_csv("cleaned_data_train.csv", index=False)
    test_df.to_csv("cleaned_data_test.csv", index=False)

data_msg = "Data processing completed."
logger.info(data_msg)
print(data_msg)