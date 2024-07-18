import pandas as pd
import os
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score
import argparse
import logging
import json

#########################################################################################################################
# Set up logging
log_filename = 'model.log'  # Specify the log file name
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

def csv_file(value):
    if not value.endswith(('.csv')):
        raise argparse.ArgumentTypeError(f'File must have .csv extension: {value}')
    return value

def get_parser():
    parser = argparse.ArgumentParser(description='Train & save a model.')
    # parser.add_argument('csv_file', type=csv_file, help='Path to the CSV file')
    parser.add_argument('model', type=str, choices=['lgb', 'xgb', 'rf'], help='Choose model type to run')

    return parser.parse_args()

def get_model_files(root, type:str):
    model_file = os.path.join(root, "model", f"model_{type}.pkl")

    if os.path.exists(model_file):
        with open(model_file, 'rb') as file:
            model = pickle.load(file)
    else:
        raise ValueError(f"Pickle file: {model_file} does not exist.")
    
    cols_file = os.path.join(root, "model", f'model_columns_{type}.txt')

    # Initialize an empty list to store the read lines
    cols_list = []

    if os.path.exists(cols_file):
        # Open the file in read mode
        with open(cols_file, 'r') as file:
            # Read all lines from the file
            lines = file.readlines()
            cols_list = [line.strip() for line in lines]
    else:
        raise ValueError(f"Pickle file: {cols_file} does not exist.")
    
    return model, cols_list

def append_to_json(data, filename):
    # Read existing JSON data
    with open(filename, 'r') as file:
        try:
            json_data = json.load(file)
        except json.JSONDecodeError:
            json_data = []

    # Append new data to the existing JSON data
    json_data.append(data)

    # Write back to the JSON file
    with open(filename, 'w') as file:
        json.dump(json_data, file, indent=4)

#########################################################################################################################
# Code to do actual modeling
if __name__ == "__main__":
    # get data file path
    current = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(current)
    path = 'data'
    filename = 'cleaned_data_test.csv'
    meta_file = 'metadata.json'
    
    file = os.path.join(root,path,filename)

    # do some data file validation
    if os.path.isfile(file):
        print(f"Test data file is: {file}")
    else:
        raise OSError("File does not exist.")
    
    logger.info(f"Test data file is: {file}")
    # get command line arguments
    args = get_parser()
    # validate the data csv file
    csv_file(file)

    # read in the data
    df = pd.read_csv(file)

    model, cols_list = get_model_files(type=args.model, root=root)

    x_cols = cols_list
    y_cols = ['study_eq_labels']
    X = df[x_cols]
    y = df[y_cols]

    y_pred = model.predict(X)

    mse = mean_squared_error(y, y_pred)
    acc = accuracy_score(y, y_pred)
    pre = precision_score(y, y_pred, average='weighted')
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    row = {
        'model': args.model,
        'accuracy': acc,
        'precision': pre,
        'r_squared': r2,
        'mean_squared_error': mse,
        'mean_absolute_error': mae
    }

    json_path = os.path.join(root,path,meta_file)
    if not os.path.isfile(json_path):
        raise OSError("File does not exist.")
    
    with open(json_path, 'r') as file:
        meta_data = json.load(file)

    row.update(meta_data)
    json_filename = "modeling_test_results.json"
    append_to_json(row, filename=os.path.join(root,'model',json_filename))

data_msg = "Model evaluation complete."
logger.info(data_msg)
print(data_msg)