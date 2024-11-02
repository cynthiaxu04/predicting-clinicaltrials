import pandas as pd
import numpy as np
import os
import re
# import ast
import argparse
import yaml
import logging
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import json


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

'''def extract_timeframes(outcomes):
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
    return max_days'''

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

def contains_any(string, substrings):
    return any(sub in string for sub in substrings)

def conditions_map(condition):
    # pediatric
    if contains_any(condition, ['peds pilocytic astrocytoma', 'pediatric pilocytic astrocytoma', 'ped pilocytic astrocytoma', 'pilocytic astrocytoma, pediatric', 'pediatric PA', 
                               'PA, pediatric', 'pilocytic astrocytoma']):
        return 'peds pilocytic astrocytoma'
    if contains_any(condition, ['peds diffuse astrocytoma', 'pediatric diffuse astrocytoma', 'ped diffuse astrocytoma', 'diffuse astrocytoma, pediatric', 'pediatric low-grade glioma', 
                               'low-grade glioma, pediatric', 'low grade glioma, pediatric', 'pediatric low grade glioma']):
        return 'peds diffuse astrocytoma'
    if contains_any(condition, ['peds anaplastic astrocytoma', 'pediatric anaplastic astrocytoma', 'ped anaplastic astrocytoma', 'anaplastic astrocytoma, pediatric', 'pediatric AA', 
                               'AA, pediatric']):
        return 'peds anaplastic astrocytoma'
    if contains_any(condition, ['peds glioblastoma', 'pediatric glioblastoma', 'ped glioblastoma', 'glioblastoma, pediatric']):
        return 'peds glioblastoma'
    if contains_any(condition, ['peds oligodendroglioma', 'pediatric oligodendroglioma', 'ped oligodendroglioma', 'oligodendroglioma, pediatric']):
        return 'peds oligodendroglioma'
    if contains_any(condition, ['peds ependymoma', 'pediatric ependymoma', 'ped ependymoma', 'ependymoma, pediatric']):
        return 'peds ependymoma'
    if contains_any(condition, ['peds embryonal tumors', 'pediatric embryonal tumors', 'ped embryonal tumors', 'embryonal tumors, pediatric', 'embryonal tumors']):
        return 'peds embryonal tumors'
    if contains_any(condition, ['peds brain', 'pediatric brain', 'ped brain', 'brain cancer, pediatric', 'peds brain cancer', 'pediatric brain cancer, ped brain cancer']):
        return 'peds brain'
    if contains_any(condition, ['peds acute lymphoblastic leukemia', 'pediatric acute lymphoblastic leukemia', 'ped acute lymphoblastic leukemia', 'acute lymphoblastic leukemia, pediatric',
                                'pediatric ALL', 'ALL, pediatric']):
        return 'peds all'
    if contains_any(condition, ['peds acute myelogenous leukemia', 'pediatric acute myelogenous leukemia', 'ped acute myelogenous leukemia', 'acute myelogenous leukemia, pediatric',
                                'pediatric AML', 'AML, pediatric']):
        return 'peds aml'
    if contains_any(condition, ['peds juvenile myelomonocytic leukemia', 'pediatric juvenile myelomonocytic leukemia', 'ped juvenile myelomonocytic leukemia', 'juvenile myelomonocytic leukemia, pediatric',
                                'pediatric JMML', 'JMML, pediatric']):
        return 'peds jmml'
    if contains_any(condition, ['peds chronic myeloid leukemia', 'pediatric chronic myeloid leukemia', 'ped chronic myeloid leukemia', 'chronic myeloid leukemia, pediatric',
                                'pediatric CML', 'CML, pediatric']):
        return 'peds cml'
    if contains_any(condition, ['peds leukemia', 'pediatric leukemia', 'ped leukemia', 'leukemia, pediatric']):
        return 'peds leukemia'
    if contains_any(condition, ['peds ', 'pediatric', 'ped ', 'childhood', 'neuroblastoma']):
        return 'pediatric'
    
    # head and neck
    if contains_any(condition, ['oral cavity', 'pharynx', 'oral', 'oralpharyngeal', 'nasopharyngeal']):
        return 'oral cavity and pharynx'
    if contains_any(condition, ['lip']):
        return 'lip'
    if contains_any(condition, ['tongue']):
        return 'tongue'
    if contains_any(condition, ['laryngeal']):
        return 'laryngeal'
    if contains_any(condition, ['thyroid']):
        return 'thyroid'
    if contains_any(condition, ['head and neck', 'head', 'neck', 'mouth']):
        return 'head and neck'

    # lung
    if contains_any(condition, ['non small cell lung', 'nsclc', 'non-small cell lung', 'non-small lung', 'non small lung', 'non-small cell', 'non small cell']):
        return 'nsclc'
    if contains_any(condition, ['small cell lung', 'sclc', 'small cell']):
        return 'sclc'
    if contains_any(condition, ['lung']):
        return 'lung'
    
    # ovarian 
    if contains_any(condition, ['invasive epithelial ovarian', 'epithelial ovarian']):
        return 'invasive epithelial ovarian'
    if contains_any(condition, ['ovarian stromal', 'stromal, ovarian']):
        return 'ovarian stromal'
    if contains_any(condition, ['germ cell tumors of ovary', 'germ cell tumors of ovaries', 'germ cell tumor of ovary', 'germ cell tumor of ovaries', 'germ cell, ovary']):
        return 'germ cell tumors of ovary'
    if contains_any(condition, ['fallopian tube', 'fallopian']):
        return 'fallopian tube'
    if contains_any(condition, ['ovarian', 'ovary']):
        return 'ovarian'
    
    if contains_any(condition, ['anal cancer', 'anal carcinoma']):
        return 'anal cancer'
    
    # bile duct
    if contains_any(condition, ['intrahepatic bile duct']):
        return 'intrahepatic bile duct'
    if contains_any(condition, ['extrahepatic bile duct']):
        return 'extrahepatic bile duct'
    if contains_any(condition, ['bile duct', 'bile']):
        return 'bile duct'
    
    if contains_any(condition, ['colorectal', 'rectum cancer', 'rectal carcinoma', 'rectal', 'rectum', 'colon']):
        return 'colorectal'
    
    if contains_any(condition, ['esophagus', 'esophageal']):
        return 'esophagus'
    
    if contains_any(condition, ['gallbladder', 'gall bladder', 'gall']):
        return 'gallbladder'
    
    if contains_any(condition, ['gastrointestinal stromal tumor', 'gist', 'gastrointestinal']):
        return 'gist'
    
    if contains_any(condition, ['liver', 'hcc', 'hepatocellular', 'hepatocellular carcinoma']):
        return 'liver'
    
    if contains_any(condition, ['pancreatic', 'pancreas']):
        return 'pancreatic'
    
    if contains_any(condition, ['small intestine']):
        return 'small intestine'
    
    if contains_any(condition, ['stomach', 'gastric']):
        return 'stomach'
    
    if contains_any(condition, ['bladder']):
        return 'bladder'
    
    if contains_any(condition, ['kidney', 'kidneys', 'renal']):
        return 'kidney'
    
    if contains_any(condition, ['malignant mesothelioma', 'mesothelioma']):
        return 'malignant mesothelioma'
    
    # reproductive
    if contains_any(condition, ['breast']):
        return 'breast'
    if contains_any(condition, ['penile', 'penis']): 
        return 'penile'
    if contains_any(condition, ['prostate']):
        return 'prostate'
    if contains_any(condition, ['testicular', 'testicle', 'testicles']):
        return 'testicular'
    if contains_any(condition, ['vagina', 'vaginal']):
        return 'vaginal'
    
    # uterine sarcoma
    if contains_any(condition, ['leiomyosarcoma']):
        return 'leiomysarcoma'
    if contains_any(condition, ['undifferentiated sarcoma']):
        return 'undifferentiated sarcoma'
    if contains_any(condition, ['endometrial stromal sarcoma']):
        return 'endometrial stromal sarcoma'
    if contains_any(condition, ['uterine sarcoma', 'uterine']):
        return 'uterine sarcoma'
    
    if contains_any(condition, ['cervix', 'cervical', 'pelvic']):
        return 'cervical'
    if contains_any(condition, ['endometrial', 'endometrius']):
        return 'endometrial'
    if contains_any(condition, ['vulvar', 'vulva']):
        return 'vulvar'
    
    if contains_any(condition, ['adrenal']):
        return 'adrenal'
    
    if contains_any(condition, ['melanoma', 'skin']):
        return 'melanoma'
    
    if contains_any(condition, ['eye', 'ocular']):
        return 'eye'
    
    if contains_any(condition, ['soft tissue sarcoma']):
        return 'soft tissue sarcoma'
    
    if contains_any(condition, ['osteosarcoma']):
        return 'osteosarcoma'
    
    # bone
    if contains_any(condition, ['chondrosarcoma']):
        return 'chondrosarcoma'
    if contains_any(condition, ['chordoma']):
        return 'chordoma'
    if contains_any(condition, ['giant cell tumor of bone']):
        return 'giant cell tumor of bone'
    if contains_any(condition, ['bone']):
        return 'bone'
    
    # brain
    if contains_any(condition, ['low-grade astrocytoma', 'low grade astrocytoma', 'diffuse astrocytoma']):
        return 'diffuse astrocytoma'
    if contains_any(condition, ['anaplastic astrocytoma', 'aa']):
        return 'anaplastic astrocytoma'
    if contains_any(condition, ['glioblastoma']):
        return 'glioblastoma'
    if contains_any(condition, ['oligodendroglioma', 'oligodendrogliomal']):
        return 'oligodendroglioma'
    if contains_any(condition, ['anaplastic oligodendroglioma']):
        return 'anaplastic oligodendroglioma'
    if contains_any(condition, ['ependymoma', 'anaplastic ependymoma']):
        return 'ependymoma'
    if contains_any(condition, ['meningioma']):
        return 'meningioma'
    if contains_any(condition, ['brain', 'glioma', 'gioma']):
        return 'brain'

    if contains_any(condition, ['myeloma']):
        return 'myeloma'
    
    # leukemia
    if contains_any(condition, ['acute myeloid leukemia', 'aml', 'mds', 'myelodysplastic syndrome', 'myelodysplastic']):
        return 'acute myeloid leukemia'
    if contains_any(condition, ['lymphocytic leukemia']):
        return 'lymphocytic leukemia'
    if contains_any(condition, ['leukemia']):
        return 'leukemia'
    
    # lymphoma
    if contains_any(condition, ['hodgkin lymphoma', 'hodgkin', 'hodgkins lymphoma', 'hodgkins']):
        return 'hodgkin lymphoma'
    if contains_any(condition, ['non-hodgkin lymphoma', 'non-hodgkins lymphoma', 'non hodgkin lymphoma', 'non hodgkins lymphoma', 'non-hodgkins', 'non-hodgkin',
                                'non hodgkins, non hodgkin']):
        return 'non hodgkin lymphoma'
    if contains_any(condition, ['lymphoma']):
        return 'lymphoma'
 
    #carcinoma
    if contains_any(condition, ['adenocarcinoma', 'adeno']):
        return 'adenocarcinoma'
    if contains_any(condition, ['basal cell', ' basal']):
        return 'basal cell'
    if contains_any(condition, ['squamous cell', 'squamous']):
        return 'squamous cell'
    if contains_any(condition, ['transitional cell', 'transitional']):
        return 'transitional cell'
    if contains_any(condition, ['carcinoma']):
        return 'carcinoma'
    
    if contains_any(condition, ['sarcoma']):
        return 'sarcoma'
    
    if contains_any(condition, ['cancer', 'malignant neoplasm', 'malignant neoplasms', 'neoplasms, malignant', 'neoplasm, malignant', 'neoplasm', 'neoplasms',
                                'benign neoplasm', 'benign neoplasms', 'neoplasm, benign', 'neoplasms, benign']):
        return 'cancer'
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


#########################################################################################################################
# Code to do actual data processing
if __name__ == "__main__":
    # get command line arguments
    args = get_parser()

    if args.csv_file:
        file = args.csv_file
    elif args.json_file:
        file = args.json_file

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
    logger.info(f"Number of bins: {args.bins}")

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
    n_intervals = args.bins

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
    df['protocolSection_conditionsModule_conditions'] = df['protocolSection_conditionsModule_conditions'].apply(safe_eval)
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
    df['protocolSection_armsInterventionsModule_interventions'] = df['protocolSection_armsInterventionsModule_interventions'].apply(safe_eval)
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
    df["age_group0"] = df["protocolSection_eligibilityModule_stdAges"].astype(str).map(age_map)
    df["age_group"] = df["age_group0"].map(age_map2)

    # number of locations, int
    df["num_locations"] = df["protocolSection_contactsLocationsModule_locations"].astype(str).apply(count_loc)

    loc_map = {
        "USA": 0,
        "non-USA": 1,
        "USA & non-USA": 2
    }
    #location of trials, categorical
    df['location0'] = df["protocolSection_contactsLocationsModule_locations"].astype(str).apply(trial_loc)
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

    df['conditions'] = df['protocolSection_conditionsModule_conditions'].apply(list_to_lower_string)
    df['conditions_category'] = df['conditions'].apply(lambda x: conditions_map(x))

    # 5 year survival dict
    conditions_5yr_survival_map = {'peds pilocytic astrocytoma': 0.95,
                                    'peds diffuse astrocytoma': 0.825,
                                    'peds anaplastic astrocytoma': 0.25, 
                                    'peds glioblastoma': 0.2,
                                    'peds oligodendroglioma': 0.9,
                                    'peds ependymoma': 0.75,
                                    'peds embryonal tumors': 0.625,
                                    'peds brain': 0.6428571429,
                                    'peds all': 0.9,
                                    'peds aml': 0.675,
                                    'peds jmml': 0.5,
                                    'peds cml': 0.7,
                                    'peds leukemia': 0.69375,
                                    'pediatric': 0.6624313187,
                                    'oral cavity and pharynx': 0.68,
                                    'lip': 0.914,
                                    'tongue': 0.688,
                                    'laryngeal': 0.61,
                                    'thyroid': 0.984,
                                    'head and neck': 0.7752,
                                    'nsclc': 0.28,
                                    'sclc': 0.07,
                                    'lung': 0.175,
                                    'cervical': 0.67,
                                    'endometrial': 0.81,
                                    'invasive epithelial ovarian': 0.5,
                                    'ovarian stromal': 0.89,
                                    'germ cell tumors of ovary': 0.92,
                                    'fallopian tube': 0.55,
                                    'ovarian': 0.715,
                                    'anal cancer': 0.7,
                                    'intrahepatic bile duct': 0.09,
                                    'extrahepatic bile duct': 0.11,
                                    'bile duct': 0.1,
                                    'colorectal': 0.63,
                                    'esophagus': 0.22,
                                    'gallbladder': 0.2,
                                    'gist': 0.82,
                                    'liver': 0.22,
                                    'pancreatic': 0.13,
                                    'small intestine': 0.69,
                                    'stomach': 0.36,
                                    'bladder': 0.78,
                                    'kidney': 0.78,
                                    'malignant mesothelioma': 0.12,
                                    'breast': 0.91,
                                    'penile': 0.65,
                                    'prostate': 0.97,
                                    'testicular': 0.95,
                                    'vaginal': 0.51,
                                    'leiomyosarcoma': 0.38,
                                    'undifferentiated sarcoma': 0.43,
                                    'endometrial stromal sarcoma': 0.96,
                                    'uterine sarcoma': 0.59,
                                    'vulvar': 0.71,
                                    'adrenal': 0.5,
                                    'melanoma': 0.94,
                                    'eye': 0.81,
                                    'soft tissue sarcoma': 0.65,
                                    'osteosarcoma': 0.59,
                                    'chondrosarcoma': 0.79,
                                    'chordoma': 0.84,
                                    'giant cell tumor of bone': 0.78,
                                    'bone': 0.8033333333,
                                    'diffuse astrocytoma': 0.48,
                                    'anaplastic astrocytoma': 0.34,
                                    'glioblastoma': 0.12,
                                    'oligodendroglioma': 0.8,
                                    'anaplastic oligodendroglioma': 0.63,
                                    'ependymoma': 0.9,
                                    'meningioma': 0.79,
                                    'brain': 0.58,
                                    'myeloma': 0.58,
                                    'hodgkin lymphoma': 0.89,
                                    'non hodgkin lymphoma': 0.78,
                                    'lymphoma': 0.835,
                                    'leukemia': 0.67,
                                    'acute myeloid leukemia': 0.295,
                                    'lymphocytic leukemia': 0.681,
                                    'adenocarcinoma': 0.606,
                                    'basal cell': 1.0,
                                    'squamous cell': 0.95, 
                                    'transitional cell': 0.543,
                                    'carcinoma': 0.775,
                                    'sarcoma': 0.65,
                                    'cancer': 0.717,
                                    'other': 0.717}

    conditions_distant_survival_map = {'peds pilocytic astrocytoma': 0.95,
                                        'peds diffuse astrocytoma': 0.825,
                                        'peds anaplastic astrocytoma': 0.25, 
                                        'peds glioblastoma': 0.2,
                                        'peds oligodendroglioma': 0.9,
                                        'peds ependymoma': 0.75,
                                        'peds embryonal tumors': 0.625,
                                        'peds brain': 0.6428571429,
                                        'peds all': 0.9,
                                        'peds aml': 0.675,
                                        'peds jmml': 0.5,
                                        'peds cml': 0.7,
                                        'peds leukemia': 0.69375,
                                        'pediatric': 0.6624313187,
                                        'oral cavity and pharynx': 0.404,
                                        'lip': 0.381,
                                        'tongue': 0.407,
                                        'laryngeal': 0.339,
                                        'thyroid': 0.533,
                                        'head and neck': 0.4128,
                                        'nsclc': 0.09,
                                        'sclc': 0.03,
                                        'lung': 0.06,
                                        'cervical': 0.19,
                                        'endometrial': 0.18,
                                        'invasive epithelial ovarian': 0.31,
                                        'ovarian stromal': 0.7,
                                        'germ cell tumors of ovary': 0.71,
                                        'fallopian tube': 0.44,
                                        'ovarian': 0.54,
                                        'anal cancer': 0.36,
                                        'intrahepatic bile duct': 0.03,
                                        'extrahepatic bile duct': 0.02,
                                        'bile duct': 0.025,
                                        'colorectal': 0.13,
                                        'esophagus': 0.06,
                                        'gallbladder': 0.03,
                                        'gist': 0.52,
                                        'liver': 0.04,
                                        'pancreatic': 0.03,
                                        'small intestine': 0.42,
                                        'stomach': 0.07,
                                        'bladder': 0.08,
                                        'kidney': 0.17,
                                        'malignant mesothelioma': 0.07,
                                        'breast': 0.31,
                                        'penile': 0.09,
                                        'prostate': 0.34,
                                        'testicular': 0.95,
                                        'vaginal': 0.26,
                                        'leiomyosarcoma': 0.12,
                                        'undifferentiated sarcoma': 0.18,
                                        'endometrial stromal sarcoma': 0.08,
                                        'uterine sarcoma': 0.366666667,
                                        'vulvar': 0.19,
                                        'adrenal': 0.38,
                                        'melanoma': 0.35,
                                        'eye': 0.16,
                                        'soft tissue sarcoma': 0.15,
                                        'osteosarcoma': 0.24,
                                        'chondrosarcoma': 0.17,
                                        'chordoma': 0.69,
                                        'giant cell tumor of bone': 0.36,
                                        'bone': 0.40666666667,
                                        'diffuse astrocytoma': 0.48,
                                        'anaplastic astrocytoma': 0.34,
                                        'glioblastoma': 0.12,
                                        'oligodendroglioma': 0.8,
                                        'anaplastic oligodendroglioma': 0.63,
                                        'ependymoma': 0.9,
                                        'meningioma': 0.79,
                                        'brain': 0.58,
                                        'myeloma': 0.57,
                                        'hodgkin lymphoma': 0.83,
                                        'non hodgkin lymphoma': 0.73,
                                        'lymphoma': 0.78,
                                        'leukemia': 0.67,
                                        'acute myeloid leukemia': 0.295,
                                        'lymphocytic leukemia': 0.681,
                                        'adenocarcinoma': 0.606,
                                        'basal cell': 1.0,
                                        'squamous cell': 0.95, 
                                        'transitional cell': 0.543,
                                        'carcinoma': 0.775,
                                        'sarcoma': 0.15,
                                        'cancer': 0.717,
                                        'other': 0.717
                                        }

    conditions_regional_survival_map = {'peds pilocytic astrocytoma': 0.95,
                                        'peds diffuse astrocytoma': 0.825,
                                        'peds anaplastic astrocytoma': 0.25, 
                                        'peds glioblastoma': 0.2,
                                        'peds oligodendroglioma': 0.9,
                                        'peds ependymoma': 0.75,
                                        'peds embryonal tumors': 0.625,
                                        'peds brain': 0.6428571429,
                                        'peds all': 0.9,
                                        'peds aml': 0.675,
                                        'peds jmml': 0.5,
                                        'peds cml': 0.7,
                                        'peds leukemia': 0.69375,
                                        'pediatric': 0.6624313187,
                                        'oral cavity and pharynx': 0.69,
                                        'lip': 0.634,
                                        'tongue': 0.698,
                                        'laryngeal': 0.462,
                                        'thyroid': 0.983,
                                        'head and neck': 0.6934,
                                        'nsclc': 0.37,
                                        'sclc': 0.18,
                                        'lung': 0.275,
                                        'cervical': 0.6,
                                        'endometrial': 0.7,
                                        'invasive epithelial ovarian': 0.75,
                                        'ovarian stromal': 0.86,
                                        'germ cell tumors of ovary': 0.94,
                                        'fallopian tube': 0.53,
                                        'ovarian': 0.77,
                                        'anal cancer': 0.67,
                                        'intrahepatic bile duct': 0.09,
                                        'extrahepatic bile duct': 0.18,
                                        'bile duct': 0.135,
                                        'colorectal': 0.73,
                                        'esophagus': 0.28,
                                        'gallbladder': 0.28,
                                        'gist': 0.84,
                                        'liver': 0.14,
                                        'pancreatic': 0.16,
                                        'small intestine': 0.78,
                                        'stomach': 0.35,
                                        'bladder': 0.39,
                                        'kidney': 0.74,
                                        'malignant mesothelioma': 0.16,
                                        'breast': 0.86,
                                        'penile': 0.51,
                                        'prostate': 0.99,
                                        'testicular': 0.95,
                                        'vaginal': 0.57,
                                        'leiomyosarcoma': 0.37,
                                        'undifferentiated sarcoma': 0.37,
                                        'endometrial stromal sarcoma': 0.94,
                                        'uterine sarcoma': 0.56,
                                        'vulvar': 0.53,
                                        'adrenal': 0.53,
                                        'melanoma': 0.74,
                                        'eye': 0.67,
                                        'soft tissue sarcoma': 0.56,
                                        'osteosarcoma': 0.64,
                                        'chondrosarcoma': 0.76,
                                        'chordoma': 0.84,
                                        'giant cell tumor of bone': 0.77,
                                        'bone': 0.79,
                                        'diffuse astrocytoma': 0.48,
                                        'anaplastic astrocytoma': 0.34,
                                        'glioblastoma': 0.12,
                                        'oligodendroglioma': 0.8,
                                        'anaplastic oligodendroglioma': 0.63,
                                        'ependymoma': 0.9,
                                        'meningioma': 0.79,
                                        'brain': 0.58,
                                        'myeloma': 0.58,
                                        'hodgkin lymphoma': 0.95,
                                        'non hodgkin lymphoma': 0.83,
                                        'lymphoma': 0.89,
                                        'leukemia': 0.67,
                                        'acute myeloid leukemia': 0.295,
                                        'lymphocytic leukemia': 0.681,
                                        'adenocarcinoma': 0.606,
                                        'basal cell': 1.0,
                                        'squamous cell': 0.95, 
                                        'transitional cell': 0.543,
                                        'carcinoma': 0.775,
                                        'sarcoma': 0.56,
                                        'cancer': 0.717,
                                        'other': 0.717
                                        }

    conditions_local_survival_map = {'peds pilocytic astrocytoma': 0.95,
                                    'peds diffuse astrocytoma': 0.825,
                                    'peds anaplastic astrocytoma': 0.25, 
                                    'peds glioblastoma': 0.2,
                                    'peds oligodendroglioma': 0.9,
                                    'peds ependymoma': 0.75,
                                    'peds embryonal tumors': 0.625,
                                    'peds brain': 0.6428571429,
                                    'peds all': 0.9,
                                    'peds aml': 0.675,
                                    'peds jmml': 0.5,
                                    'peds cml': 0.7,
                                    'peds leukemia': 0.69375,
                                    'pediatric': 0.6624313187,
                                    'oral cavity and pharynx': 0.863,
                                    'lip': 0.941,
                                    'tongue': 0.842,
                                    'laryngeal': 0.783,
                                    'thyroid': 0.999,
                                    'head and neck': 0.8856,
                                    'nsclc': 0.65,
                                    'sclc': 0.30,
                                    'lung': 0.475,
                                    'cervical': 0.91,
                                    'endometrial': 0.95,
                                    'invasive epithelial ovarian': 0.93,
                                    'ovarian stromal': 0.97,
                                    'germ cell tumors of ovary': 0.97,
                                    'fallopian tube': 0.94,
                                    'ovarian': 0.9525,
                                    'anal cancer': 0.83,
                                    'intrahepatic bile duct': 0.23,
                                    'extrahepatic bile duct': 0.18,
                                    'bile duct': 0.205,
                                    'colorectal': 0.91,
                                    'esophagus': 0.49,
                                    'gallbladder': 0.69,
                                    'gist': 0.95,
                                    'liver': 0.37,
                                    'pancreatic': 0.44,
                                    'small intestine': 0.84,
                                    'stomach': 0.75,
                                    'bladder': 0.71,
                                    'kidney': 0.93,
                                    'malignant mesothelioma': 0.24,
                                    'breast': 0.99,
                                    'penile': 0.79,
                                    'prostate': 0.99,
                                    'testicular': 0.95,
                                    'vaginal': 0.69,
                                    'leiomyosarcoma': 0.6,
                                    'undifferentiated sarcoma': 0.71,
                                    'endometrial stromal sarcoma': 0.995,
                                    'uterine sarcoma': 0.7683333333,
                                    'vulvar': 0.86,
                                    'adrenal': 0.73,
                                    'melanoma': 0.99,
                                    'eye': 0.85,
                                    'soft tissue sarcoma': 0.81,
                                    'osteosarcoma': 0.76,
                                    'chondrosarcoma': 0.91,
                                    'chordoma': 0.87,
                                    'giant cell tumor of bone': 0.9,
                                    'bone': 0.8933333333,
                                    'diffuse astrocytoma': 0.48,
                                    'anaplastic astrocytoma': 0.34,
                                    'glioblastoma': 0.12,
                                    'oligodendroglioma': 0.8,
                                    'anaplastic oligodendroglioma': 0.63,
                                    'ependymoma': 0.9,
                                    'meningioma': 0.79,
                                    'brain': 0.58,
                                    'myeloma': 0.79,
                                    'hodgkin lymphoma': 0.93,
                                    'non hodgkin lymphoma': 0.85,
                                    'lymphoma': 0.89,
                                    'leukemia': 0.67,
                                    'acute myeloid leukemia': 0.295,
                                    'lymphocytic leukemia': 0.681,
                                    'adenocarcinoma': 0.606,
                                    'basal cell': 1.0,
                                    'squamous cell': 0.95, 
                                    'transitional cell': 0.543,
                                    'carcinoma': 0.775,
                                    'sarcoma': 0.81,
                                    'cancer': 0.717,
                                    'other': 0.717
                                    }
    
    # Create 5 year survival columns
    df['survival_5yr'] = df['conditions_category'].map(conditions_5yr_survival_map)
    df['survival_5yr_local'] = df['conditions_category'].map(conditions_local_survival_map)
    df['survival_5yr_regional'] = df['conditions_category'].map(conditions_regional_survival_map)
    df['survival_5yr_distant'] = df['conditions_category'].map(conditions_distant_survival_map)
    
    # Treatment maps

    conditions_primary_radiation_map = {'peds pilocytic astrocytoma': 0,
                                        'peds diffuse astrocytoma': 1,
                                        'peds anaplastic astrocytoma': 1, 
                                        'peds glioblastoma': 1,
                                        'peds oligodendroglioma': 1,
                                        'peds ependymoma': 1,
                                        'peds embryonal tumors': 1,
                                        'peds brain': 1,
                                        'peds all': 0,
                                        'peds aml': 1,
                                        'peds jmml': 0,
                                        'peds cml': 0,
                                        'peds leukemia': 1,
                                        'pediatric': 1,
                                        'oral cavity and pharynx': 1,
                                        'lip': 1,
                                        'tongue': 1,
                                        'laryngeal': 1,
                                        'thyroid': 1,
                                        'head and neck': 1,
                                        'nsclc': 0,
                                        'sclc': 1,
                                        'lung': 1,
                                        'cervical': 1,
                                        'endometrial': 1,
                                        'invasive epithelial ovarian': 0,
                                        'ovarian stromal': 0,
                                        'germ cell tumors of ovary': 1,
                                        'fallopian tube': 0,
                                        'ovarian': 1,
                                        'anal cancer': 0,
                                        'intrahepatic bile duct': 1,
                                        'extrahepatic bile duct': 1,
                                        'bile duct': 1,
                                        'colorectal': 1,
                                        'esophagus': 0,
                                        'gallbladder': 1,
                                        'gist': 0,
                                        'liver': 0,
                                        'pancreatic': 1,
                                        'small intestine': 1,
                                        'stomach': 1,
                                        'bladder': 0,
                                        'kidney': 1,
                                        'malignant mesothelioma': 1,
                                        'breast': 1,
                                        'penile': 1,
                                        'prostate': 1,
                                        'testicular': 1,
                                        'vaginal': 1,
                                        'leiomyosarcoma': 1,
                                        'undifferentiated sarcoma': 1,
                                        'endometrial stromal sarcoma': 0,
                                        'uterine sarcoma': 1,
                                        'vulvar': 1,
                                        'adrenal': 0,
                                        'melanoma': 0,
                                        'eye': 1,
                                        'soft tissue sarcoma': 1,
                                        'osteosarcoma': 1,
                                        'chondrosarcoma': 0,
                                        'chordoma': 1,
                                        'giant cell tumor of bone': 0,
                                        'bone': 1,
                                        'diffuse astrocytoma': 1,
                                        'anaplastic astrocytoma': 1,
                                        'glioblastoma': 1,
                                        'oligodendroglioma': 1,
                                        'anaplastic oligodendroglioma': 1,
                                        'ependymoma': 0,
                                        'meningioma': 1,
                                        'brain': 1,
                                        'myeloma': 0,
                                        'hodgkin lymphoma': 1,
                                        'non hodgkin lymphoma': 1,
                                        'lymphoma': 1,
                                        'leukemia': 1,
                                        'acute myeloid leukemia': 1,
                                        'lymphocytic leukemia': 1,
                                        'adenocarcinoma': 1,
                                        'basal cell': 0,
                                        'squamous cell': 0, 
                                        'transitional cell': 1,
                                        'carcinoma': 1,
                                        'sarcoma': 1,
                                        'cancer': 1,
                                        'other': 1
                                        }
    
    conditions_primary_surgery_map = {'peds pilocytic astrocytoma': 1,
                                        'peds diffuse astrocytoma': 1,
                                        'peds anaplastic astrocytoma': 1, 
                                        'peds glioblastoma': 1,
                                        'peds oligodendroglioma': 1,
                                        'peds ependymoma': 1,
                                        'peds embryonal tumors': 1,
                                        'peds brain': 1,
                                        'peds all': 0,
                                        'peds aml': 0,
                                        'peds jmml': 0,
                                        'peds cml': 0,
                                        'peds leukemia': 1,
                                        'pediatric': 1,
                                        'oral cavity and pharynx': 1,
                                        'lip': 1,
                                        'tongue': 1,
                                        'laryngeal': 1,
                                        'thyroid': 1,
                                        'head and neck': 1,
                                        'nsclc': 1,
                                        'sclc': 0,
                                        'lung': 1,
                                        'cervical': 1,
                                        'endometrial': 0,
                                        'invasive epithelial ovarian': 1,
                                        'ovarian stromal': 1,
                                        'germ cell tumors of ovary': 1,
                                        'fallopian tube': 1,
                                        'ovarian': 1,
                                        'anal cancer': 1,
                                        'intrahepatic bile duct': 1,
                                        'extrahepatic bile duct': 1,
                                        'bile duct': 1,
                                        'colorectal': 1,
                                        'esophagus': 1,
                                        'gallbladder': 1,
                                        'gist': 1,
                                        'liver': 1,
                                        'pancreatic': 1,
                                        'small intestine': 1,
                                        'stomach': 1,
                                        'bladder': 1,
                                        'kidney': 1,
                                        'malignant mesothelioma': 1,
                                        'breast': 1,
                                        'penile': 1,
                                        'prostate': 1,
                                        'testicular': 0,
                                        'vaginal': 1,
                                        'leiomyosarcoma': 1,
                                        'undifferentiated sarcoma': 1,
                                        'endometrial stromal sarcoma': 1,
                                        'uterine sarcoma': 1,
                                        'vulvar': 1,
                                        'adrenal': 1,
                                        'melanoma': 1,
                                        'eye': 1,
                                        'soft tissue sarcoma': 1,
                                        'osteosarcoma': 1,
                                        'chondrosarcoma': 1,
                                        'chordoma': 1,
                                        'giant cell tumor of bone': 1,
                                        'bone': 1,
                                        'diffuse astrocytoma': 0,
                                        'anaplastic astrocytoma': 0,
                                        'glioblastoma': 1,
                                        'oligodendroglioma': 0,
                                        'anaplastic oligodendroglioma': 0,
                                        'ependymoma': 1,
                                        'meningioma': 1,
                                        'brain': 1,
                                        'myeloma': 0,
                                        'hodgkin lymphoma': 0,
                                        'non hodgkin lymphoma': 10,
                                        'lymphoma': 0,
                                        'leukemia': 0,
                                        'acute myeloid leukemia': 0,
                                        'lymphocytic leukemia': 0,
                                        'adenocarcinoma': 1,
                                        'basal cell': 1,
                                        'squamous cell': 1, 
                                        'transitional cell': 1,
                                        'carcinoma': 1,
                                        'sarcoma': 1,
                                        'cancer': 1,
                                        'other': 1
                                        }

    conditions_primary_chemo_map = {'peds pilocytic astrocytoma': 0,
                                    'peds diffuse astrocytoma': 1,
                                    'peds anaplastic astrocytoma': 0,
                                    'peds glioblastoma': 0,
                                    'peds oligodendroglioma': 1,
                                    'peds ependymoma': 0,
                                    'peds embryonal tumors': 1,
                                    'peds brain': 1,
                                    'peds all': 1,
                                    'peds aml': 1,
                                    'peds jmml': 0,
                                    'peds cml': 0,
                                    'peds leukemia': 1,
                                    'pediatric': 1,
                                    'oral cavity and pharynx': 1,
                                    'lip': 0,
                                    'tongue': 0,
                                    'laryngeal': 1,
                                    'thyroid': 1,
                                    'head and neck': 1,
                                    'nsclc': 1,
                                    'sclc': 1,
                                    'lung': 1,
                                    'cervical': 1,
                                    'endometrial': 0,
                                    'invasive epithelial ovarian': 1,
                                    'ovarian stromal': 0,
                                    'germ cell tumors of ovary': 1,
                                    'fallopian tube': 1,
                                    'ovarian': 1,
                                    'anal cancer': 1,
                                    'intrahepatic bile duct': 1,
                                    'extrahepatic bile duct': 1,
                                    'bile duct': 1,
                                    'colorectal': 1,
                                    'esophagus ': 1,
                                    'gallbladder': 1,
                                    'gist': 1,
                                    'liver': 0,
                                    'pancreatic': 1,
                                    'small intestine': 1,
                                    'stomach': 1,
                                    'bladder': 1,
                                    'kidney': 0,
                                    'malignant mesothelioma': 1,
                                    'breast': 0,
                                    'penile': 1,
                                    'prostate': 1,
                                    'testicular': 1,
                                    'vaginal': 1,
                                    'leiomyosarcoma': 1,
                                    'undifferentiated sarcoma': 1,
                                    'endometrial stromal sarcoma': 0,
                                    'uterine sarcoma': 1,
                                    'vulvar': 1,
                                    'adrenal': 1,
                                    'melanoma': 1,
                                    'eye': 1,
                                    'soft tissue sarcoma': 1,
                                    'osteosarcoma': 1,
                                    'chondrosarcoma': 0,
                                    'chordoma': 0,
                                    'giant cell tumor of bone': 0,
                                    'bone': 0,
                                    'diffuse astrocytoma': 1,
                                    'anaplastic astrocytoma': 1,
                                    'glioblastoma': 1,
                                    'oligodendroglioma': 1,
                                    'anaplastic oligodendroglioma': 1,
                                    'ependymoma': 0,
                                    'meningioma': 0,
                                    'brain ': 1,
                                    'myeloma': 1,
                                    'hodgkin lymphoma': 1,
                                    'non hodgkin lymphoma': 1,
                                    'lymphoma': 1,
                                    'leukemia': 1,
                                    'acute myeloid leukemia': 1,
                                    'lymphocytic leukemia': 1,
                                    'adenocarcinoma': 1,
                                    'basal cell': 0,
                                    'squamous cell': 0,
                                    'transitional cell': 0,
                                    'carcinoma': 1,
                                    'sarcoma': 1,
                                    'cancer': 1,
                                    'other': 1}
    
    conditions_primary_immuno_map = {'peds pilocytic astrocytoma': 0,
                                    'peds diffuse astrocytoma': 0,
                                    'peds anaplastic astrocytoma': 0,
                                    'peds glioblastoma': 0,
                                    'peds oligodendroglioma': 0,
                                    'peds ependymoma': 0,
                                    'peds embryonal tumors': 0,
                                    'peds brain': 0,
                                    'peds all': 0,
                                    'peds aml': 0,
                                    'peds jmml': 0,
                                    'peds cml': 0,
                                    'peds leukemia': 1,
                                    'pediatric': 1,
                                    'oral cavity and pharynx': 0,
                                    'lip': 0,
                                    'tongue': 0,
                                    'laryngeal': 1,
                                    'thyroid': 0,
                                    'head and neck': 1,
                                    'non small cell lung': 0,
                                    'small cell lung': 0,
                                    'lung': 0,
                                    'cervical': 0,
                                    'endometrial': 0,
                                    'invasive epithelial ovarian': 0,
                                    'ovarian stromal': 0,
                                    'germ cell tumors of ovary': 0,
                                    'fallopian tube': 0,
                                    'ovarian': 0,
                                    'anal cancer': 0,
                                    'intrahepatic bile duct': 0,
                                    'extrahepatic bile duct': 0,
                                    'bile duct': 0,
                                    'colorectal': 0,
                                    'esophagus ': 1,
                                    'gallbladder': 0,
                                    'gist': 0,
                                    'liver': 0,
                                    'pancreatic': 0,
                                    'small intestine': 0,
                                    'stomach': 0,
                                    'bladder': 0,
                                    'kidney': 0,
                                    'malignant mesothelioma': 0,
                                    'breast': 0,
                                    'penile': 1,
                                    'prostate': 1,
                                    'testicular': 0,
                                    'vaginal': 0,
                                    'leiomyosarcoma': 0,
                                    'undifferentiated sarcoma': 0,
                                    'endometrial stromal sarcoma': 0,
                                    'uterine sarcoma': 0,
                                    'vulvar': 0,
                                    'adrenal': 0,
                                    'melanoma': 1,
                                    'eye': 1,
                                    'soft tissue sarcoma': 1,
                                    'osteosarcoma': 0,
                                    'chondrosarcoma': 0,
                                    'chordoma': 0,
                                    'giant cell tumor of bone': 0,
                                    'bone': 0,
                                    'diffuse astrocytoma': 0,
                                    'anaplastic astrocytoma': 0,
                                    'glioblastoma': 0,
                                    'oligodendroglioma': 0,
                                    'anaplastic oligodendroglioma': 0,
                                    'ependymoma': 0,
                                    'meningioma': 0,
                                    'brain ': 0,
                                    'myeloma': 0,
                                    'hodgkin lymphoma': 1,
                                    'non hodgkin lymphoma': 0,
                                    'lymphoma': 1,
                                    'leukemia': 1,
                                    'acute myeloid leukemia': 1,
                                    'lymphocytic leukemia': 1,
                                    'adenocarcinoma': 0,
                                    'basal cell': 0,
                                    'squamous cell': 0,
                                    'transitional cell': 0,
                                    'carcinoma': 1,
                                    'sarcoma': 0,
                                    'cancer': 1,
                                    'other': 1}
 
    conditions_primary_stemcell_map = {'peds pilocytic astrocytoma': 0,
                                        'peds diffuse astrocytoma': 0,
                                        'peds anaplastic astrocytoma': 0,
                                        'peds glioblastoma': 0,
                                        'peds oligodendroglioma': 0,
                                        'peds ependymoma': 0,
                                        'peds embryonal tumors': 0,
                                        'peds brain': 0,
                                        'peds all': 0,
                                        'peds aml': 0,
                                        'peds jmml': 1,
                                        'peds cml': 0,
                                        'peds leukemia': 1,
                                        'pediatric': 1,
                                        'oral cavity and pharynx': 0,
                                        'lip': 0,
                                        'tongue': 0,
                                        'laryngeal': 0,
                                        'thyroid': 0,
                                        'head and neck': 0,
                                        'nsclc': 0,
                                        'sclc': 0,
                                        'lung': 0,
                                        'cervical': 0,
                                        'endometrial': 0,
                                        'invasive epithelial ovarian': 0,
                                        'ovarian stromal': 0,
                                        'germ cell tumors of ovary': 0,
                                        'fallopian tube': 0,
                                        'ovarian': 0,
                                        'anal cancer': 0,
                                        'intrahepatic bile duct': 0,
                                        'extrahepatic bile duct': 0,
                                        'bile duct': 0,
                                        'colorectal': 0,
                                        'esophagus ': 0,
                                        'gallbladder': 0,
                                        'gist': 0,
                                        'liver': 0,
                                        'pancreatic': 0,
                                        'small intestine': 0,
                                        'stomach': 0,
                                        'bladder': 0,
                                        'kidney': 0,
                                        'malignant mesothelioma': 0,
                                        'breast': 0,
                                        'penile': 0,
                                        'prostate': 0,
                                        'testicular': 1,
                                        'vaginal': 0,
                                        'leiomyosarcoma': 0,
                                        'undifferentiated sarcoma': 0,
                                        'endometrial stromal sarcoma': 0,
                                        'uterine sarcoma': 0,
                                        'vulvar': 0,
                                        'adrenal': 0,
                                        'melanoma': 0,
                                        'eye': 0,
                                        'soft tissue sarcoma': 0,
                                        'osteosarcoma': 0,
                                        'chondrosarcoma': 0,
                                        'chordoma': 0,
                                        'giant cell tumor of bone': 0,
                                        'bone': 0,
                                        'diffuse astrocytoma': 0,
                                        'anaplastic astrocytoma': 0,
                                        'glioblastoma': 0,
                                        'oligodendroglioma': 0,
                                        'anaplastic oligodendroglioma': 0,
                                        'ependymoma': 0,
                                        'meningioma': 0,
                                        'brain ': 0,
                                        'myeloma': 1,
                                        'hodgkin lymphoma': 1,
                                        'non hodgkin lymphoma': 0,
                                        'lymphoma': 1,
                                        'leukemia': 1,
                                        'acute myeloid leukemia': 1,
                                        'lymphocytic leukemia': 0,
                                        'adenocarcinoma': 0,
                                        'basal cell': 0,
                                        'squamous cell': 0,
                                        'transitional cell': 0,
                                        'carcinoma': 1,
                                        'sarcoma': 0,
                                        'cancer': 1,
                                        'other': 1}
    
    conditions_primary_targeted_map = {'peds pilocytic astrocytoma': 0,
                                        'peds diffuse astrocytoma': 0,
                                        'peds anaplastic astrocytoma': 0,
                                        'peds glioblastoma': 0,
                                        'peds oligodendroglioma': 0,
                                        'peds ependymoma': 0,
                                        'peds embryonal tumors': 0,
                                        'peds brain': 1,
                                        'peds all': 0,
                                        'peds aml': 1,
                                        'peds jmml': 0,
                                        'peds cml': 1,
                                        'peds leukemia': 1,
                                        'pediatric': 1,
                                        'oral cavity and pharynx': 0,
                                        'lip': 0,
                                        'tongue': 0,
                                        'laryngeal': 0,
                                        'thyroid': 1,
                                        'head and neck': 1,
                                        'nsclc': 0,
                                        'sclc': 0,
                                        'lung': 0,
                                        'cervical': 0,
                                        'endometrial': 1,
                                        'invasive epithelial ovarian': 1,
                                        'ovarian stromal': 0,
                                        'germ cell tumors of ovary': 0,
                                        'fallopian tube': 1,
                                        'ovarian': 1,
                                        'anal cancer': 0,
                                        'intrahepatic bile duct': 0,
                                        'extrahepatic bile duct': 0,
                                        'bile duct': 0,
                                        'colorectal': 0,
                                        'esophagus ': 0,
                                        'gallbladder': 0,
                                        'gist': 0,
                                        'liver': 0,
                                        'pancreatic': 1,
                                        'small intestine': 0,
                                        'stomach': 0,
                                        'bladder': 0,
                                        'kidney': 0,
                                        'malignant mesothelioma': 0,
                                        'breast': 1,
                                        'penile': 0,
                                        'prostate': 1,
                                        'testicular': 0,
                                        'vaginal': 0,
                                        'leiomyosarcoma': 0,
                                        'undifferentiated sarcoma': 0,
                                        'endometrial stromal sarcoma': 0,
                                        'uterine sarcoma': 0,
                                        'vulvar': 0,
                                        'adrenal': 0,
                                        'melanoma': 1,
                                        'eye': 1,
                                        'soft tissue sarcoma': 1,
                                        'osteosarcoma': 1,
                                        'chondrosarcoma': 0,
                                        'chordoma': 0,
                                        'giant cell tumor of bone': 0,
                                        'bone': 0,
                                        'diffuse astrocytoma': 0,
                                        'anaplastic astrocytoma': 0,
                                        'glioblastoma': 0,
                                        'oligodendroglioma': 0,
                                        'anaplastic oligodendroglioma': 0,
                                        'ependymoma': 0,
                                        'meningioma': 0,
                                        'brain ': 0,
                                        'myeloma': 0,
                                        'hodgkin lymphoma': 1,
                                        'non hodgkin lymphoma': 0,
                                        'lymphoma': 1,
                                        'leukemia': 1,
                                        'acute myeloid leukemia': 1,
                                        'lymphocytic leukemia': 1,
                                        'adenocarcinoma': 0,
                                        'basal cell': 0,
                                        'squamous cell': 0,
                                        'transitional cell': 0,
                                        'carcinoma': 1,
                                        'sarcoma': 0,
                                        'cancer': 1,
                                        'other': 1}
    
    conditions_recurrent_radiation_map = {'peds pilocytic astrocytoma': 1,
                                            'peds diffuse astrocytoma': 1,
                                            'peds anaplastic astrocytoma': 1,
                                            'peds glioblastoma': 1,
                                            'peds oligodendroglioma': 1,
                                            'peds ependymoma': 1,
                                            'peds embryonal tumors': 1,
                                            'peds brain': 1,
                                            'peds all': 0,
                                            'peds aml': 1,
                                            'peds jmml': 0,
                                            'peds cml': 0,
                                            'peds leukemia': 1,
                                            'pediatric': 1,
                                            'oral cavity and pharynx': 1,
                                            'lip': 1,
                                            'tongue': 0,
                                            'laryngeal': 1,
                                            'thyroid': 1,
                                            'head and neck': 1,
                                            'nsclc': 0,
                                            'sclc': 0,
                                            'lung': 0,
                                            'cervical': 1,
                                            'endometrial': 1,
                                            'invasive epithelial ovarian': 0,
                                            'ovarian stromal': 0,
                                            'germ cell tumors of ovary': 1,
                                            'fallopian tube': 0,
                                            'ovarian': 1,
                                            'anal cancer': 1,
                                            'intrahepatic bile duct': 1,
                                            'extrahepatic bile duct': 1,
                                            'bile duct': 1,
                                            'colorectal': 0,
                                            'esophagus ': 0,
                                            'gallbladder': 1,
                                            'gist': 0,
                                            'liver': 0,
                                            'pancreatic': 0,
                                            'small intestine': 1,
                                            'stomach': 0,
                                            'bladder': 0,
                                            'kidney': 0,
                                            'malignant mesothelioma': 0,
                                            'breast': 1,
                                            'penile': 1,
                                            'prostate': 1,
                                            'testicular': 0,
                                            'vaginal': 1,
                                            'leiomyosarcoma': 1,
                                            'undifferentiated sarcoma': 1,
                                            'endometrial stromal sarcoma': 0,
                                            'uterine sarcoma': 1,
                                            'vulvar': 1,
                                            'adrenal': 1,
                                            'melanoma': 0,
                                            'eye': 1,
                                            'soft tissue sarcoma': 1,
                                            'osteosarcoma': 1,
                                            'chondrosarcoma': 0,
                                            'chordoma': 1,
                                            'giant cell tumor of bone': 0,
                                            'bone': 1,
                                            'diffuse astrocytoma': 1,
                                            'anaplastic astrocytoma': 1,
                                            'glioblastoma': 1,
                                            'oligodendroglioma': 1,
                                            'anaplastic oligodendroglioma': 1,
                                            'ependymoma': 1,
                                            'meningioma': 1,
                                            'brain ': 1,
                                            'myeloma': 0,
                                            'hodgkin lymphoma': 1,
                                            'non hodgkin lymphoma': 0,
                                            'lymphoma': 1,
                                            'leukemia': 1,
                                            'acute myeloid leukemia': 0,
                                            'lymphocytic leukemia': 1,
                                            'adenocarcinoma': 1,
                                            'basal cell': 0,
                                            'squamous cell': 0,
                                            'transitional cell': 0,
                                            'carcinoma': 1,
                                            'sarcoma': 1,
                                            'cancer': 1,
                                            'other': 1}
    
    conditions_recurrent_surgery_map = {'peds pilocytic astrocytoma': 1,
                                        'peds diffuse astrocytoma': 1,
                                        'peds anaplastic astrocytoma': 1,
                                        'peds glioblastoma': 1,
                                        'peds oligodendroglioma': 1,
                                        'peds ependymoma': 1,
                                        'peds embryonal tumors': 1,
                                        'peds brain': 1,
                                        'peds all': 0,
                                        'peds aml': 0,
                                        'peds jmml': 0,
                                        'peds cml': 0,
                                        'peds leukemia': 1,
                                        'pediatric': 1,
                                        'oral cavity and pharynx': 1,
                                        'lip': 1,
                                        'tongue': 1,
                                        'laryngeal': 1,
                                        'thyroid': 1,
                                        'head and neck': 1,
                                        'nsclc': 0,
                                        'sclc': 0,
                                        'lung': 0,
                                        'cervical': 0,
                                        'endometrial': 1,
                                        'invasive epithelial ovarian': 0,
                                        'ovarian stromal': 1,
                                        'germ cell tumors of ovary': 0,
                                        'fallopian tube': 0,
                                        'ovarian': 1,
                                        'anal cancer': 1,
                                        'intrahepatic bile duct': 1,
                                        'extrahepatic bile duct': 1,
                                        'bile duct': 0,
                                        'colorectal': 1,
                                        'esophagus ': 0,
                                        'gallbladder': 0,
                                        'gist': 0,
                                        'liver': 1,
                                        'pancreatic': 0,
                                        'small intestine': 1,
                                        'stomach': 1,
                                        'bladder': 1,
                                        'kidney': 0,
                                        'malignant mesothelioma': 0,
                                        'breast': 1,
                                        'penile': 1,
                                        'prostate': 0,
                                        'testicular': 1,
                                        'vaginal': 1,
                                        'leiomyosarcoma': 1,
                                        'undifferentiated sarcoma': 1,
                                        'endometrial stromal sarcoma': 0,
                                        'uterine sarcoma': 1,
                                        'vulvar': 1,
                                        'adrenal': 1,
                                        'melanoma': 0,
                                        'eye': 0,
                                        'soft tissue sarcoma': 1,
                                        'osteosarcoma': 1,
                                        'chondrosarcoma': 1,
                                        'chordoma': 1,
                                        'giant cell tumor of bone': 1,
                                        'bone': 1,
                                        'diffuse astrocytoma': 0,
                                        'anaplastic astrocytoma': 0,
                                        'glioblastoma': 1,
                                        'oligodendroglioma': 0,
                                        'anaplastic oligodendroglioma': 0,
                                        'ependymoma': 0,
                                        'meningioma': 1,
                                        'brain ': 1,
                                        'myeloma': 0,
                                        'hodgkin lymphoma': 0,
                                        'non hodgkin lymphoma': 0,
                                        'lymphoma': 0,
                                        'leukemia': 0,
                                        'acute myeloid leukemia': 0,
                                        'lymphocytic leukemia': 0,
                                        'adenocarcinoma': 1,
                                        'basal cell': 1,
                                        'squamous cell': 1,
                                        'transitional cell': 0,
                                        'carcinoma': 1,
                                        'sarcoma': 1,
                                        'cancer': 1,
                                        'other': 1}
    
    conditions_recurrent_chemo_map = {'peds pilocytic astrocytoma': 0,
                                        'peds diffuse astrocytoma': 1,
                                        'peds anaplastic astrocytoma': 0,
                                        'peds glioblastoma': 0,
                                        'peds oligodendroglioma': 1,
                                        'peds ependymoma': 1,
                                        'peds embryonal tumors': 1,
                                        'peds brain': 1,
                                        'peds all': 1,
                                        'peds aml': 1,
                                        'peds jmml': 1,
                                        'peds cml': 0,
                                        'peds leukemia': 1,
                                        'pediatric': 1,
                                        'oral cavity and pharynx': 1,
                                        'lip': 1,
                                        'tongue': 1,
                                        'laryngeal': 1,
                                        'thyroid': 1,
                                        'head and neck': 1,
                                        'nsclc': 1,
                                        'sclc': 1,
                                        'lung': 1,
                                        'cervical': 1,
                                        'endometrial': 1,
                                        'invasive epithelial ovarian': 1,
                                        'ovarian stromal': 1,
                                        'germ cell tumors of ovary': 1,
                                        'fallopian tube': 1,
                                        'ovarian': 1,
                                        'anal cancer': 1,
                                        'intrahepatic bile duct': 1,
                                        'extrahepatic bile duct': 1,
                                        'bile duct': 1,
                                        'colorectal': 1,
                                        'esophagus ': 1,
                                        'gallbladder': 1,
                                        'gist': 1,
                                        'liver': 0,
                                        'pancreatic': 1,
                                        'small intestine': 1,
                                        'stomach': 0,
                                        'bladder': 1,
                                        'kidney': 1,
                                        'malignant mesothelioma': 1,
                                        'breast': 1,
                                        'penile': 0,
                                        'prostate': 1,
                                        'testicular': 1,
                                        'vaginal': 0,
                                        'leiomyosarcoma': 1,
                                        'undifferentiated sarcoma': 1,
                                        'endometrial stromal sarcoma': 0,
                                        'uterine sarcoma': 1,
                                        'vulvar': 1,
                                        'adrenal': 0,
                                        'melanoma': 1,
                                        'eye': 1,
                                        'soft tissue sarcoma': 1,
                                        'osteosarcoma': 1,
                                        'chondrosarcoma': 0,
                                        'chordoma': 0,
                                        'giant cell tumor of bone': 0,
                                        'bone (avg)': 0,
                                        'diffuse astrocytoma': 1,
                                        'anaplastic astrocytoma': 1,
                                        'glioblastoma': 1,
                                        'oligodendroglioma': 1,
                                        'anaplastic oligodendroglioma': 1,
                                        'ependymoma': 1,
                                        'meningioma': 0,
                                        'brain ': 1,
                                        'myeloma': 1,
                                        'hodgkin lymphoma': 1,
                                        'non hodgkin lymphoma': 0,
                                        'lymphoma': 1,
                                        'leukemia': 1,
                                        'acute myeloid leukemia': 1,
                                        'lymphocytic leukemia': 1,
                                        'adenocarcinoma': 1,
                                        'basal cell': 0,
                                        'squamous cell': 0,
                                        'transitional cell': 1,
                                        'carcinoma': 1,
                                        'sarcoma': 1,
                                        'cancer': 1,
                                        'other': 1}
    
    conditions_recurrent_immuno_map = {'peds pilocytic astrocytoma': 0,
                                        'peds diffuse astrocytoma': 0,
                                        'peds anaplastic astrocytoma': 0,
                                        'peds glioblastoma': 0,
                                        'peds oligodendroglioma': 0,
                                        'peds ependymoma': 0,
                                        'peds embryonal tumors': 0,
                                        'peds brain': 0,
                                        'peds all': 0,
                                        'peds aml': 0,
                                        'peds jmml': 0,
                                        'peds cml': 0,
                                        'peds leukemia': 1,
                                        'pediatric': 1,
                                        'oral cavity and pharynx': 1,
                                        'lip': 0,
                                        'tongue': 0,
                                        'laryngeal': 1,
                                        'thyroid': 0,
                                        'head and neck': 1,
                                        'nsclc': 0,
                                        'sclc': 1,
                                        'lung': 1,
                                        'cervical': 1,
                                        'endometrial': 1,
                                        'invasive epithelial ovarian': 0,
                                        'ovarian stromal': 0,
                                        'germ cell tumors of ovary': 0,
                                        'fallopian tube': 0,
                                        'ovarian': 0,
                                        'anal cancer': 0,
                                        'intrahepatic bile duct': 0,
                                        'extrahepatic bile duct': 0,
                                        'bile duct': 1,
                                        'colorectal': 1,
                                        'esophagus ': 1,
                                        'gallbladder': 0,
                                        'gist': 0,
                                        'liver': 0,
                                        'pancreatic': 0,
                                        'small intestine': 0,
                                        'stomach': 1,
                                        'bladder': 1,
                                        'kidney': 1,
                                        'malignant mesothelioma': 1,
                                        'breast': 0,
                                        'penile': 0,
                                        'prostate': 0,
                                        'testicular': 0,
                                        'vaginal': 0,
                                        'leiomyosarcoma': 0,
                                        'undifferentiated sarcoma': 0,
                                        'endometrial stromal sarcoma': 0,
                                        'uterine sarcoma': 0,
                                        'vulvar': 0,
                                        'adrenal': 0,
                                        'melanoma': 1,
                                        'eye': 1,
                                        'soft tissue sarcoma': 0,
                                        'osteosarcoma': 0,
                                        'chondrosarcoma': 0,
                                        'chordoma': 0,
                                        'giant cell tumor of bone': 0,
                                        'bone': 0,
                                        'diffuse astrocytoma': 0,
                                        'anaplastic astrocytoma': 0,
                                        'glioblastoma': 0,
                                        'oligodendroglioma': 0,
                                        'anaplastic oligodendroglioma': 0,
                                        'ependymoma': 0,
                                        'meningioma': 0,
                                        'brain ': 0,
                                        'myeloma': 1,
                                        'hodgkin lymphoma': 1,
                                        'non hodgkin lymphoma': 0,
                                        'lymphoma': 1,
                                        'leukemia': 1,
                                        'acute myeloid leukemia': 0,
                                        'lymphocytic leukemia': 1,
                                        'adenocarcinoma': 0,
                                        'basal cell': 0,
                                        'squamous cell': 0,
                                        'transitional cell': 1,
                                        'carcinoma': 1,
                                        'sarcoma': 0,
                                        'cancer': 1,
                                        'other': 1}
    
    conditions_recurrent_stemcell_map = {'peds pilocytic astrocytoma': 0,
                                        'peds diffuse astrocytoma': 0,
                                        'peds anaplastic astrocytoma': 0,
                                        'peds glioblastoma': 0,
                                        'peds oligodendroglioma': 0,
                                        'peds ependymoma': 0,
                                        'peds embryonal tumors': 0,
                                        'peds brain': 0,
                                        'peds all': 1,
                                        'peds aml': 0,
                                        'peds jmml': 1,
                                        'peds cml': 1,
                                        'peds leukemia': 1,
                                        'pediatric': 1,
                                        'oral cavity and pharynx': 0,
                                        'lip': 0,
                                        'tongue': 0,
                                        'laryngeal': 0,
                                        'thyroid': 0,
                                        'head and neck': 0,
                                        'nsclc': 0,
                                        'sclc': 0,
                                        'lung': 0,
                                        'cervical': 0,
                                        'endometrial': 1,
                                        'invasive epithelial ovarian': 0,
                                        'ovarian stromal': 0,
                                        'germ cell tumors of ovary': 0,
                                        'fallopian tube': 0,
                                        'ovarian': 0,
                                        'anal cancer': 0,
                                        'intrahepatic bile duct': 0,
                                        'extrahepatic bile duct': 0,
                                        'bile duct': 1,
                                        'colorectal': 0,
                                        'esophagus ': 0,
                                        'gallbladder': 0,
                                        'gist': 0,
                                        'liver': 0,
                                        'pancreatic': 0,
                                        'small intestine': 0,
                                        'stomach': 0,
                                        'bladder': 0,
                                        'kidney': 0,
                                        'malignant mesothelioma': 0,
                                        'breast': 0,
                                        'penile': 0,
                                        'prostate': 0,
                                        'testicular': 1,
                                        'vaginal': 0,
                                        'leiomyosarcoma': 0,
                                        'undifferentiated sarcoma': 0,
                                        'endometrial stromal sarcoma': 0,
                                        'uterine sarcoma': 0,
                                        'vulvar': 0,
                                        'adrenal': 0,
                                        'melanoma': 0,
                                        'eye': 0,
                                        'soft tissue sarcoma': 0,
                                        'osteosarcoma': 0,
                                        'chondrosarcoma': 0,
                                        'chordoma': 0,
                                        'giant cell tumor of bone': 0,
                                        'bone': 0,
                                        'diffuse astrocytoma': 0,
                                        'anaplastic astrocytoma': 0,
                                        'glioblastoma': 0,
                                        'oligodendroglioma': 0,
                                        'anaplastic oligodendroglioma': 0,
                                        'ependymoma': 0,
                                        'meningioma': 0,
                                        'brain ': 0,
                                        'myeloma': 0,
                                        'hodgkin lymphoma': 1,
                                        'non hodgkin lymphoma': 1,
                                        'lymphoma': 1,
                                        'leukemia': 1,
                                        'acute myeloid leukemia': 1,
                                        'lymphocytic leukemia': 1,
                                        'adenocarcinoma': 0,
                                        'basal cell': 0,
                                        'squamous cell': 0,
                                        'transitional cell': 0,
                                        'carcinoma': 1,
                                        'sarcoma': 0,
                                        'cancer': 1,
                                        'other': 1}
    
    conditions_recurrent_targeted_map = {'peds pilocytic astrocytoma': 0,
                                        'peds diffuse astrocytoma': 0,
                                        'peds anaplastic astrocytoma': 0,
                                        'peds glioblastoma': 0,
                                        'peds oligodendroglioma': 0,
                                        'peds ependymoma': 0,
                                        'peds embryonal tumors': 0,
                                        'peds brain': 1,
                                        'peds all': 0,
                                        'peds aml': 1,
                                        'peds jmml': 0,
                                        'peds cml': 1,
                                        'peds leukemia': 1,
                                        'pediatric': 1,
                                        'oral cavity and pharynx': 1,
                                        'lip': 0,
                                        'tongue': 0,
                                        'laryngeal': 0,
                                        'thyroid': 1,
                                        'head and neck': 1,
                                        'nsclc': 1,
                                        'sclc': 0,
                                        'lung': 1,
                                        'cervical': 0,
                                        'endometrial': 1,
                                        'invasive epithelial ovarian': 1,
                                        'ovarian stromal': 0,
                                        'germ cell tumors of ovary': 0,
                                        'fallopian tube': 1,
                                        'ovarian': 1,
                                        'anal cancer': 0,
                                        'intrahepatic bile duct': 0,
                                        'extrahepatic bile duct': 0,
                                        'bile duct': 1,
                                        'colorectal': 0,
                                        'esophagus ': 0,
                                        'gallbladder': 0,
                                        'gist': 0,
                                        'liver': 0,
                                        'pancreatic': 1,
                                        'small intestine': 0,
                                        'stomach': 1,
                                        'bladder': 1,
                                        'kidney': 1,
                                        'malignant mesothelioma': 1,
                                        'breast': 1,
                                        'penile': 0,
                                        'prostate': 1,
                                        'testicular': 0,
                                        'vaginal': 0,
                                        'leiomyosarcoma': 0,
                                        'undifferentiated sarcoma': 0,
                                        'endometrial stromal sarcoma': 0,
                                        'uterine sarcoma': 0,
                                        'vulvar': 0,
                                        'adrenal': 0,
                                        'melanoma': 1,
                                        'eye': 1,
                                        'soft tissue sarcoma': 1,
                                        'osteosarcoma': 1,
                                        'chondrosarcoma': 0,
                                        'chordoma': 0,
                                        'giant cell tumor of bone': 0,
                                        'bone (avg)': 0,
                                        'diffuse astrocytoma': 0,
                                        'anaplastic astrocytoma': 0,
                                        'glioblastoma': 0,
                                        'oligodendroglioma': 0,
                                        'anaplastic oligodendroglioma': 0,
                                        'ependymoma': 0,
                                        'meningioma': 0,
                                        'brain ': 0,
                                        'myeloma': 0,
                                        'hodgkin lymphoma': 1,
                                        'non hodgkin lymphoma': 1,
                                        'lymphoma': 1,
                                        'leukemia': 1,
                                        'acute myeloid leukemia': 0,
                                        'lymphocytic leukemia': 1,
                                        'adenocarcinoma': 1,
                                        'basal cell': 0,
                                        'squamous cell': 0,
                                        'transitional cell': 1,
                                        'carcinoma': 1,
                                        'sarcoma': 0,
                                        'cancer': 1,
                                        'other': 1}

    # Create treatment type columns
    df['primary_radiation'] = df['conditions_category'].map(conditions_primary_radiation_map)
    df['primary_surgery'] = df['conditions_category'].map(conditions_primary_surgery_map)
    df['primary_chemo'] = df['conditions_category'].map(conditions_primary_chemo_map)
    df['primary_immuno'] = df['conditions_category'].map(conditions_primary_immuno_map)
    df['primary_stemcell'] = df['conditions_category'].map(conditions_primary_stemcell_map)
    df['primary_targeted'] = df['conditions_category'].map(conditions_primary_targeted_map)
    df['recurrent_radiation'] = df['conditions_category'].map(conditions_recurrent_radiation_map)
    df['recurrent_surgery'] = df['conditions_category'].map(conditions_recurrent_surgery_map)
    df['recurrent_chemo'] = df['conditions_category'].map(conditions_recurrent_chemo_map)
    df['recurrent_immuno'] = df['conditions_category'].map(conditions_recurrent_immuno_map)
    df['recurrent_stemcell'] = df['conditions_category'].map(conditions_recurrent_stemcell_map)
    df['recurrent_targeted'] = df['conditions_category'].map(conditions_recurrent_targeted_map)

    # rename some columns
    df = df.rename(columns={'protocolSection_designModule_phases': 'phase',
                             'protocolSection_designModule_enrollmentInfo_count': 'enroll_count',
                             'protocolSection_eligibilityModule_healthyVolunteers': 'healthy_vol'})
    
    df['phase'] = df['phase'].astype(str)

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
        'secondary_max_days',
        'primary_radiation',
        'primary_surgery',
        'primary_chemo',
        'primary_immuno',
        'primary_stemcell',
        'primary_targeted',
        'recurrent_radiation',
        'recurrent_surgery',
        'recurrent_chemo',
        'recurrent_immuno',
        'recurrent_stemcell',
        'recurrent_targeted',      
        'survival_5yr',
        'survival_5yr_distant',
        'survival_5yr_regional',
        'survival_5yr_local'
        # 'conditions_category_num'
        # 'protocolSection_outcomesModule_primaryOutcomes',
        # 'protocolSection_outcomesModule_secondaryOutcomes',
        # 'protocolSection_eligibilityModule_eligibilityCriteria'
    ]

    clean_df = df[cols].copy()

    # first, remove outliers
    # get the numeric columns only
    numeric_cols = df.select_dtypes(include=['float16', 'float32', 'float64', 'int', 'int32', 'int64', 'bool']).columns
    temp = df[numeric_cols]
    temp2 = drop_outliers(temp)

    # get a cleaned df with the indices df with the outliers dropped
    clean_df2 = clean_df.loc[temp2.index]

    clean_df2 = convert_dtypes(clean_df2)

    msg3 = f"The number of dropped due to outliers (greater than 5 stdevs from the mean) is {len(clean_df) - len(clean_df2)}"
    logger.info(msg3)
    print(msg3)

    # handle missing values by filling with the mode to preserve data distribution
    nan_counts = clean_df2.isna().sum()

    # print(nan_counts)

    nan_cols = nan_counts[nan_counts > 0].index.tolist()
    # remove mode calculation for 'primary_max_days' and 'secondary_max_days'
    nan_cols.remove('primary_max_days')
    nan_cols.remove('secondary_max_days')
    for column in nan_cols:
        mode_value = clean_df[column].mode()[0]  # Calculate the mode
        clean_df2[column] = clean_df[column].fillna(mode_value)
        # clean_df = clean_df.infer_objects(copy=False) # idk this was supposed to silence a warning but it didn't

    # clean_df.to_csv("temp_data.csv", index=False)

    # one hot encode remaining object columns
    object_columns = list(clean_df2.select_dtypes(include=['object']).columns)
    object_columns = [i for i in object_columns if 'nctId' not in i]
    # print(f"these are the column types: {clean_df.dtypes}")
    # print(f"these are the object_columns: {object_columns}")
    encoded_df = pd.get_dummies(clean_df2, columns=object_columns)

    # Apply function to all column names
    encoded_df.columns = encoded_df.columns.map(remove_special_chars)

    # now split the data
    train_df, test_df = train_test_split(encoded_df, test_size=0.3, random_state=42, shuffle=True)

    # save the data file
    train_df.to_csv(os.path.join(root,"data", "cleaned_data_train.csv"), index=False)
    test_df.to_csv(os.path.join(root, "data", "cleaned_data_test.csv"), index=False)

    # save the meta data, i.e. data file name and the number of bins
    meta_dict = {"data_file": file, "num_bins": args.bins, "bins": bins_dict}
    save_dict_to_json(os.path.join(root,"data","metadata.json"), meta_dict)

data_msg = "Data processing completed."
logger.info(data_msg)
print(data_msg)