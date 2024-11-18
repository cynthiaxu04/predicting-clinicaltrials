import pandas as pd
import numpy as np
import os
import re
# import ast
import argparse
import yaml
import logging
from sklearn.model_selection import train_test_split
# from sklearn.cluster import KMeans
import json
import spacy
from sklearn.preprocessing import LabelEncoder




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

def json_file(value):
   if not value.endswith(('.json')):
      raise argparse.ArgumentTypeError(f"File must have extension: {value}")
   return value

def csv_file(value):
    if not value.endswith(('.csv')):
        raise argparse.ArgumentTypeError(f'File must have .csv extension: {value}')
    return value

def get_parser():
    parser = argparse.ArgumentParser(description='Validate CSV file columns using a YAML schema.')

    # parser.add_argument('yaml_file', type=yaml_file, help='Path to the YAML schema file')
    parser.add_argument('--opt_file', type=str, help='Path to an additional CSV or JSON file')


    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--csv_file', type=csv_file, help='Path to the CSV file')
    group.add_argument('--json_file', type=json_file, help='Path to the JSON file')

    parser.add_argument('--bins', type=int, choices=range(2, 6), required=True, help='Number of bins (must be an integer between 2 and 5)')

    return parser.parse_args()

def load_yaml(file_path):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def validate_columns(file, yaml_file):
    
    if '.csv' in file:
        df = pd.read_csv(file)
    elif '.json' in file:
       raw_df = pd.read_json(file)
       df = pd.json_normalize(raw_df['protocolSection'], sep='_')
       df.columns = ['protocolSection_' + col for col in df.columns]

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
    return df

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

# Extracting timeframe in number of days    
def extract_timeframes(outcomes):
    # Check if 'outcomes' is iterable (i.e., a list in this context)
    if isinstance(outcomes, list):
        return [item.get('timeFrame', 'No description available') for item in outcomes]
    else:
        return ['No description available']  # Return a list with a default message if not iterable


def extract_time_length_from_list(timeFrames):
    # Dictionary to convert written numbers to numeric values
    written_numbers = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, 
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
        "nineteen": 19, "twenty": 20
    }

    # Regular expression components
    number_pattern = r"(\d+(\.\d+)?)"
    written_number_pattern = r"(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)"
    unit_pattern = r"(hour|day|week|month|year|minute)s?"
    reversed_pattern = rf"(?P<unit3>{unit_pattern})\s+(?P<number2>\d+(\.\d+)?)"

    # Combine the components into a single regular expression with VERBOSE flag for readability
    time_pattern = re.compile(
        rf"""
        (?:
            (?P<number>{number_pattern})\s*(?P<unit1>{unit_pattern})           # Numeric values followed by unit
            |                         # OR
            (?P<written>{written_number_pattern})\s*(?P<unit2>{unit_pattern})   # Written out numbers followed by unit
            |                         # OR
            {reversed_pattern}         # Reversed order: Day 15
        )
        """,
        re.IGNORECASE | re.VERBOSE
    )

    # Function to extract time length from a single timeframe string
    def extract_time_length(timeFrame):
        results = time_pattern.findall(timeFrame)
        time_durations = []
        
        for result in results:
            if result[0]:  # Numeric values followed by unit
                number = result[0]
                unit = result[3]
            elif result[5]:  # Written out numbers followed by unit
                number = written_numbers[result[5].lower()]
                unit = result[7]
            elif result[10]:  # Reversed order: Day 15
                number = result[11]
                unit = result[9]

            if isinstance(number, str):
                number = float(number)
            time_durations.append((number, unit.lower()))
        return time_durations
    
    combined_durations = []
    for timeFrame in timeFrames:
        if timeFrame:  # This checks if the timeFrame is not empty or None
            combined_durations.extend(extract_time_length(timeFrame))
    return combined_durations

def convert_to_days(amount, unit):
    if unit == 'minutes' or unit == 'minute':
        return np.ceil(amount / 1440)
    elif unit == 'hours' or unit =='hour' :
        return np.ceil(amount / 24)
    elif unit == 'weeks' or unit == 'week':
        return amount * 7
    elif unit == 'months' or unit == 'month':
        return amount * 30  # Simplified conversion, assuming each month has 30 days
    elif unit == 'years' or unit == 'year':
        return amount * 365  # Simplified conversion, assuming each year has 365 days
    elif unit == 'days' or unit =='day':
        return amount
    else:
        return 0

def find_max_duration(durations):
    if not durations:  # Check if the list is empty
        return np.nan  # Use numpy Nan to indicate no data
    max_days = max(convert_to_days(amount, unit) for amount, unit in durations)
    return max_days

# Function to remove specified characters
def remove_special_chars(col):
    return col.replace("[", "").replace("]", "").replace("'", "").replace(", ", "_")

def count_criteria(criteria):
    if pd.isna(criteria):
        return np.nan, np.nan
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

def list_to_lower_string(lst):
    # Ensure the input is a list
    if isinstance(lst, list):
        # Convert each element to a lowercase string and join with spaces
        return ", ".join(map(str, lst)).lower()

def drop_outliers(df, threshold=5):
    # Calculate the mean and standard deviation for each column
    means = df.mean()
    stds = df.std()
    # Identify outliers
    outliers = (np.abs((df - means) / stds) > threshold)
    # Create a DataFrame to store the outliers
    dropped_values = df[outliers]
    # Drop the rows with outliers
    df_cleaned = df.drop(index=dropped_values.dropna(how='all').index)
    return df_cleaned

def interval_to_string(o):
    if isinstance(o, pd.Interval):
        return str(o)
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

def save_dict_to_json(filename, data):
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4, default=interval_to_string)

def convert_dtypes(df):
    # Convert all int64 columns to int
    int_cols = df.select_dtypes(include=['int8', 'int16', 'int32', 'int64']).columns
    df[int_cols] = df[int_cols].astype(int)
    
    # Convert all float64 columns to float
    float_cols = df.select_dtypes(include=['float16', 'float32', 'float64']).columns
    df[float_cols] = df[float_cols].astype(float)
    
    return df

def trial_loc_counter(value):
    # Initialize the counter
    counter = {"USA": 0, "non-USA": 0}
    
    # Check if the value is a list of dictionaries
    if isinstance(value, list):
        for i in value:
            if isinstance(i, dict) and 'country' in i:
                if i['country'] == 'United States':
                    counter["USA"] += 1
                else:
                    counter["non-USA"] += 1
        # Return True if USA has more counts than non-USA
        return counter['USA'] > counter['non-USA']
    else:
        # Return a default or handle the error case as appropriate
        return False
    
def check_crossover(item):
    if isinstance(item, str):  # Ensure item is a string
        # Check if "CROSSOVER" is in the string
        return "CROSSOVER" in item.upper()  # Convert to uppercase for case-insensitive check
    return False  # Return False if item is not a string

def create_all_terms(value):
    if isinstance(value, (list, dict)):
        all_terms = []
        for i in value:
            text = i["measure"].lower()
            all_terms.append(text)
        return all_terms
    else:
        return np.nan

def concatenate_strings(series:list):
    # Filter out np.nan and flatten the lists
    all_strings = []
    for i in series:
        all_strings.extend(i.dropna().sum())  # Concatenate lists while dropping NaN

    joined = ' '.join(all_strings)
    if len(joined) < 1000000:
       return joined
    else:
        return joined[:1000000] 

def spacy_list(keyword:str, ref_text):
    # load the spacy model
    nlp = spacy.load("en_core_web_md")  # Load the medium-sized model
    # Process the keyword and the large text
    keyword_doc = nlp(keyword)
    text_doc = nlp(ref_text)

    # Set a threshold for similarity
    similarity_threshold = 0.8

    high_similarity_phrases = []
    seen_phrases = set()  # Set to track unique words or phrases

    # Check individual tokens
    for token in text_doc:
        # Skip stop words and punctuation
        if not token.is_stop and not token.is_punct:
            # Compute similarity
            similarity_score = token.similarity(keyword_doc)
            if similarity_score >= similarity_threshold and token.text not in seen_phrases:
                high_similarity_phrases.append((token.text, similarity_score))
                seen_phrases.add(token.text)  # Add to the set of seen phrases

    # Check multi-word phrases
    for chunk in text_doc.noun_chunks:
        # Compute similarity with the keyword
        similarity_score = chunk.similarity(keyword_doc)
        if similarity_score >= similarity_threshold and chunk.text not in seen_phrases:
            high_similarity_phrases.append((chunk.text, similarity_score))
            seen_phrases.add(chunk.text)  # Add to the set of seen phrases
    return high_similarity_phrases

def check_terms_in_outcomes(outcomes, terms):
    if isinstance(outcomes, list):  # Only proceed if 'outcomes' is a list
        for outcome in outcomes:
            if isinstance(outcome, dict):  # Ensure each item is a dictionary
                for value in outcome.values():  # Check each value in the dictionary
                    if isinstance(value, str) and any(term in value for term, _ in terms):
                        return True
    return False

def check_tmax_in_outcomes(outcomes, terms):
    if isinstance(outcomes, list):  # Only proceed if 'outcomes' is a list
        for outcome in outcomes:
            if isinstance(outcome, dict):  # Ensure each item is a dictionary
                for value in outcome.keys():  # Check each value in the dictionary
                    if value == terms:
                        return True
    return False

def process_data(file):
    # validate the data file
    current = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(current)
    # print(root)
    path = 'config'
    filename = 'default_cols.yaml'
    yaml_file = os.path.join(root,path,filename)
    if not os.path.isfile(yaml_file):
        raise OSError(f"YAML file {yaml_file} does not exist.")
    
    logger.info(f"Data file being processed: {file}")
    # logger.info(f"Number of bins: {args.bins}")

    # print(args.yaml_file)
    df = validate_columns(file=file, yaml_file=yaml_file)

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
        # 'primary_study_duration_days',
        'study_duration_days',
        'protocolSection_designModule_enrollmentInfo_count']

    # remove rows with NaNs for primary_study_duration_days, study_duration_days, num_locations, enrollmentinfo_count
    df = df.dropna(subset=need_cols).copy()
    # len of rows after dropping NaNs
    rows1 = len(df)
    msg = f"{rows0 - rows1} rows were dropped due to missing values in one of: {need_cols}"
    logger.info(msg)
    print(msg)

    # make bins with average duration
    avg_dur = np.mean(df['study_duration_days'])
    df['study_eq_bins'] = pd.cut(df['study_duration_days'], bins=[df['study_duration_days'].min(), avg_dur, df['study_duration_days'].max()], labels=['Low', 'High'], right=False)

    df['study_eq_labels'] = df['study_eq_bins'].cat.codes
    le = LabelEncoder()
    df['study_eq_labels'] = le.fit_transform(df['study_eq_labels'])

    bins_dict = df.groupby('study_eq_labels')['study_eq_bins'].apply(lambda x: x.unique()[0]).to_dict()
    msg2 = f"Bin labels and their corresponding intervals are: {bins_dict}"
    print(msg2)
    logger.info(msg2)

    # sponsor type, categorical
    spon_map = {
        'OTHER_GOV': 'OTHER',
        'NETWORK': 'OTHER',
        'NIH': 'OTHER',
        'FED': 'OTHER',
        'INDIV': 'OTHER'
    }

    spon_map2 = {
        "INDUSTRY": True,
        "OTHER": False
    }

    df['sponsor_type0'] = df['protocolSection_sponsorCollaboratorsModule_leadSponsor_class'].replace(spon_map)
    df['sponsor_type'] = df['sponsor_type0'].map(spon_map2)

    # responsible party, categorical (mapped to int)
    party_map = {
        "PRINCIPAL_INVESTIGATOR": 0,
        "SPONSOR": 1,
        "SPONSOR_INVESTIGATOR": 2
    }

    df['resp_party'] = df['protocolSection_sponsorCollaboratorsModule_responsibleParty_type'].map(party_map)

    # allocation, categorical (mapped to int)
    allo_map = {
        'NON_RANDOMIZED': False,
        'RANDOMIZED': True
    }

    df['allocation'] = df['protocolSection_designModule_designInfo_allocation'].map(allo_map)

    df['masking'] = df['protocolSection_designModule_designInfo_maskingInfo_masking'].apply(lambda x: False if x == "NONE" else True)


    # number of conditions, int
    df['protocolSection_conditionsModule_conditions'] = df['protocolSection_conditionsModule_conditions'].apply(safe_eval)
    df['number_of_conditions'] = df['protocolSection_conditionsModule_conditions'].apply(lambda x: len(x))

    # number of outcome measures, int
    oms = [col for col in df.columns if "outcome" in col]
    for i in oms:
        df[i].apply(list_to_lower_string)
        df[f"num_{i}"] = df[i].apply(lambda x: len(x) if isinstance(x, list) else 0)
    num_om_cols = [col for col in df.columns if "num" in col and "outcome" in col]
    df['num_oms'] = df[num_om_cols].sum(axis=1)

    # primary purpose, bool/int
    # Extract purposes into new columns
    df['protocolSection_designModule_designInfo_primaryPurpose'] = df['protocolSection_designModule_designInfo_primaryPurpose'].fillna('')
    df['treatment_purpose'] = df['protocolSection_designModule_designInfo_primaryPurpose'].apply(lambda x: 1 if 'TREATMENT' in x else 0)
    df['prevention_purpose'] = df['protocolSection_designModule_designInfo_primaryPurpose'].apply(lambda x: 1 if 'PREVENTION' in x else 0)

    # intervention type, bool/int
    # Extract interventions into new columns
    df['procedure_intervention'] = df['protocolSection_armsInterventionsModule_interventions'].apply(lambda x: True if 'PROCEDURE' in x else False)
    df['drug_intervention'] = df['protocolSection_armsInterventionsModule_interventions'].apply(lambda x: True if 'DRUG' in x else False)
    df['radiation_intervention'] = df['protocolSection_armsInterventionsModule_interventions'].apply(lambda x: True if 'RADIATION' in x else False)
    df['biological_intervention'] = df['protocolSection_armsInterventionsModule_interventions'].apply(lambda x: True if 'BIOLOGICAL' in x else False)

    # number of intervention types, int
    df['protocolSection_armsInterventionsModule_interventions'] = df['protocolSection_armsInterventionsModule_interventions'].apply(safe_eval)
    df['intervention_types'] = df['protocolSection_armsInterventionsModule_interventions'].apply(extract_type)
    df['number_of_intervention_types'] = df['intervention_types'].apply(len)

    # number of locations, int
    df["num_locations"] = df["protocolSection_contactsLocationsModule_locations"].astype(str).apply(count_loc)

    # loc_map = {
    #     "USA": 0,
    #     "non-USA": 1,
    #     "USA & non-USA": 2
    # }
    # location of trials, US-led, bool
    # "Trial primarily led in the US"
    # I'm interpreting this to mean there are more trial sites in the USA than outside the USA
    df['us_led'] = df["protocolSection_contactsLocationsModule_locations"].apply(lambda x: trial_loc_counter(x))
    # location of trials, US-included, bool
    df['us_included'] = df["protocolSection_contactsLocationsModule_locations"].astype(str).apply(trial_loc).apply(lambda x: True if x == "USA" or x == "USA & non-USA" else False)

    # nic sponsorship, bool
    df['nci_sponsor'] = df['protocolSection_sponsorCollaboratorsModule_leadSponsor_name'].apply(lambda x: True if "National Cancer Institute (NCI)" in x else False)

    # crossover assignment, bool
    df['crossover'] = df['protocolSection_designModule_designInfo_interventionModel'].apply(check_crossover)

    # outcome measures, bool/int
    # Combine outcome measures
    df['all_primary_terms'] = df['protocolSection_outcomesModule_primaryOutcomes'].apply(lambda x: create_all_terms(x))
    df["all_secondary_terms"] = df['protocolSection_outcomesModule_secondaryOutcomes'].apply(lambda x: create_all_terms(x))
    # create a string all text from outcome measures in it for spacy perusal
    all_terms = concatenate_strings([df['all_primary_terms'],df["all_secondary_terms"]]) # text from both primary and secondary measures
    primary_terms = concatenate_strings([df['all_primary_terms']]) # test from primary measures only

    # initiate dictionaries to hold the list of terms with the highest similarity values as calculated by spacy
    spacy_dict = {}
    lead_spacy_dict = {}

    # create a list of the terms of intererst (for overall outcome measures)
    terms = {
        "aes": "adverse events",
        "osr": "overall survival rate",
        "mtd": "maximally tolerated dose",
        "dor": "duration of response",
        "dlt": "dose limiting toxicity",
        "cmax": "maximum measured concentration"
    }
    # for primary outcome measures only
    lead_terms = {
        "aes": "adverse events",
        "mtd": "maximally tolerated dose",
        "dlt": "dose limiting toxicity",
    }

    # fill the dictionaries
    for i in terms:
        spacy_dict[i] = spacy_list(keyword=terms[i], ref_text=all_terms)
    for i in lead_terms:
        lead_spacy_dict[i] = spacy_list(keyword=terms[i], ref_text=primary_terms)

    #create the boolean outcome measure features
    df['outcome_measures'] = df['protocolSection_outcomesModule_primaryOutcomes'].apply(lambda x: x if isinstance(x, list) else [x]).fillna('') + \
                         df['protocolSection_outcomesModule_secondaryOutcomes'].apply(lambda x: x if isinstance(x, list) else [x]).fillna('')

    for i in terms:
        df[f"spacy_{i}_outcome"] = df['outcome_measures'].apply(
        lambda x: check_terms_in_outcomes(x, spacy_dict[i])
    )

    for i in lead_terms:
        df[f"spacy_{i}_lead_outcome"] = df['protocolSection_outcomesModule_primaryOutcomes'].apply(
        lambda x: check_terms_in_outcomes(x, lead_spacy_dict[i])
    )

    # tmax measure, bool
        # if the cmax is included in outcome measure and the timeframe of the cmax is included
    df['spacy_tmax_outcome'] = df.loc[df["spacy_cmax_outcome"] == True, 'outcome_measures'].apply(
        lambda x: check_tmax_in_outcomes(x, "timeFrame"))
    
    # rename some columns
    df = df.rename(columns={'protocolSection_designModule_enrollmentInfo_count': 'enroll_count'})
    
    # df['phase'] = df['phase'].astype(str)

    #make a dataframe with just the columns of interest
    cols = [
        'study_duration_days',
        'study_eq_bins',
        'study_eq_labels',
        'num_oms',
        'number_of_intervention_types',
        "num_locations",
        'number_of_conditions',
        'allocation',
        'masking',
        'enroll_count',
        'biological_intervention',
        'procedure_intervention',
        'drug_intervention',
        'radiation_intervention',
        'treatment_purpose',
        'prevention_purpose',
        'us_led',
        'us_included',
        'nci_sponsor',
        'spacy_aes_outcome',
        'spacy_osr_outcome',
        'spacy_mtd_outcome',
        'spacy_dor_outcome',
        'spacy_dlt_outcome',
        'spacy_cmax_outcome',
        'spacy_aes_lead_outcome',
        'spacy_dlt_lead_outcome',
        'spacy_mtd_lead_outcome',
        'spacy_tmax_outcome'
    ]

    clean_df = df[cols].copy()

    # treat NaN values as separate category
    # Select only the categorical columns (object type)
    categorical_columns = clean_df.select_dtypes(include=['object']).columns

    # Replace NaN with "missing" in categorical columns
    clean_df[categorical_columns] = clean_df[categorical_columns].fillna(-1)
    clean_df[categorical_columns] = clean_df[categorical_columns].astype(int)

    # convert boolean columns to int
    bool_columns = clean_df.select_dtypes(include=['bool']).columns
    clean_df[bool_columns] = clean_df[bool_columns].astype(int)

    # impute the mean for missing enrollment count
    clean_df['enroll_count'] = clean_df['enroll_count'].fillna(clean_df["enroll_count"].mean())
    return clean_df, root, bins_dict

def main():
   # get command line arguments
    args = get_parser()

    if args.csv_file:
        file = args.csv_file
    elif args.json_file:
        file = args.json_file

    if not args.opt_file:
        clean_data, root, bins_dict = process_data(file)
        
        # now split the data
        train_df, test_df = train_test_split(clean_data, test_size=0.3, random_state=42, shuffle=True)

        # save the data file
        train_df.to_csv(os.path.join(root,"data", f"long_cleaned_data_{file[:12]}_{args.bins}bin_train.csv"), index=False)
        test_df.to_csv(os.path.join(root, "data", f"long_cleaned_data_{file[:12]}_{args.bins}bin_test.csv"), index=False)

        # save the meta data, i.e. data file name and the number of bins
        meta_dict = {"data_file": file, "phase": file[:6], "num_bins": args.bins, "bins": bins_dict}
        save_dict_to_json(os.path.join(root,"data","metadata.json"), meta_dict)
    elif args.opt_file:
        print(f"Processing additional file: {args.opt_file}")
        clean_train_data, root, bins_dict = process_data(file)
        clean_test_data, _, _ = process_data(args.opt_file)

        clean_train_data.to_csv(os.path.join(root,"data", f"long_cleaned_data_{file[:12]}_{args.bins}bin_train.csv"), index=False)
        clean_test_data.to_csv(os.path.join(root, "data", f"long_cleaned_data_{args.opt_file[:12]}_{args.bins}bin_test.csv"), index=False)
        # save the meta data, i.e. data file name and the number of bins
        meta_dict = {"data_file": file, "phase": file[:6], "num_bins": args.bins, "bins": bins_dict}
        save_dict_to_json(os.path.join(root,"data","long_metadata.json"), meta_dict)

#########################################################################################################################
# Code to do actual data processing
if __name__ == "__main__":
   main()
data_msg = "Data processing completed."
logger.info(data_msg)
print(data_msg)