import os
import logging
import pandas as pd
import numpy as np
import argparse
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score
import pickle
import json
import shutil

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
    parser.add_argument('--prod', action='store_true', help="Use --prod flag if this model is for production.")

    return parser.parse_args()

def train_model(type:str, X_train, y_train, X_test, y_test, root, phase, bins, save_model:bool, prod:bool):
    """Train a model and save it.

    Args:
        type (str): type of model, set by user and is one of ['lgb', 'xgb', 'rf']
        X_train (pd.DataFrame): X training dataset
        y_train (pd.Series): y training labels
        X_test (pd.DataFrame): X test dataset
        y_test (pd.Series): y test labels

    Raises:
        ValueError: Must choose a valid model, one of ['lgb', 'xgb', 'rf']

    Returns:
        dict: model metrics
    """
    # parameters are the result of gridsearch in model_experiments.ipynb
    if type=='lgb':
        model = lgb.LGBMClassifier(learning_rate=0.05,
                                   n_estimators=50,
                                   num_leaves=20)
    elif type=='xgb':
        model = xgb.XGBClassifier(colsample_bytree=0.8,
                                  gamma=0.1,
                                  learning_rate=0.1,
                                  max_depth=5,
                                  min_child_weight=1,
                                  n_estimators=50)
    elif type=='rf':
        model =  RandomForestClassifier(max_depth=None,
                                        min_samples_leaf=1,
                                        min_samples_split=2,
                                        n_estimators=100)
    else:
        raise ValueError("Must choose a valid model, one of ['lgb', 'xgb', 'rf'].")
        # Initialize Stratified K-Fold cross-validator
    
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Perform k-fold cross-validation
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')

    model.fit(X_train, y_train)

    # Predict probabilities for each class
    y_pred_prob = model.predict_proba(X_test)
    # Convert probabilities to class labels
    y_pred = np.argmax(y_pred_prob, axis=1) + 1

    mse = mean_squared_error(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, average='weighted')
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # return model results
    row = {
        'model': type,
        'phase': phase,
        'accuracy': acc,
        'mean_accuracy_kfold': cv_results.mean(),
        'precision': pre,
        'r_squared': r2,
        'mean_squared_error': mse,
        'mean_absolute_error': mae,
    }

    if save_model==True:
        with open(os.path.join(root, "model", f'model_{type}_{phase}_{bins}bin.pkl'), 'wb') as file:
            pickle.dump(model, file)

    if prod==True:
        with open(os.path.join(root, "trial_app", "backend", 'model.pkl'), 'wb') as file:
            pickle.dump(model, file)

    # filename = "modeling_results.json"
    # with open(filename, 'a') as file:
    #     json.dump(row, file)

    return row

def append_to_json(data, filename):
    # Check if the file exists
    if os.path.exists(filename):
        # Read existing JSON data
        with open(filename, 'r') as file:
            try:
                json_data = json.load(file)
            except json.JSONDecodeError:
                json_data = []
    else:
        # print(f"File {filename} does not exist. Creating it.")
        json_data = []

    # Append new data to the existing JSON data
    json_data.append(data)

    # Write back to the JSON file
    with open(filename, 'w') as file:
        json.dump(json_data, file, indent=4)

def copy_json_file(src_file, dest_dir):
    # Ensure the destination directory exists
    os.makedirs(dest_dir, exist_ok=True)
    
    # Construct the destination file path
    dest_file = os.path.join(dest_dir, os.path.basename(src_file))
    
    # Copy the file
    shutil.copy2(src_file, dest_file)
    
    # print(f"Copied {src_file} to {dest_file}")

#########################################################################################################################
# Code to do actual modeling
if __name__ == "__main__":
    # get data file path
    current = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(current)
    path = 'data'
    # filename = 'cleaned_data_train.csv'
    meta_file = 'metadata.json'

    # path to metadata.json contains info on bins and data source
    metadata_path = os.path.join(root, path, meta_file)

    with open(metadata_path, 'r') as file:
        data = json.load(file)
    bins = data.get('num_bins')
    phase = data.get('phase')

    file = os.path.join(root,path,f"cleaned_data_{phase}_{bins}bin_train.csv")

    # do some data file validation
    if os.path.isfile(file):
        print(f"Train data file is: {file}")
    else:
        raise OSError("File does not exist.")
    
    logger.info(f"Train data file is: {file}")
    
    # get command line arguments
    args = get_parser()
    
    # validate the data csv file
    csv_file(file)

    # read in the data
    df = pd.read_csv(file)

    # get only the numerical columns
    temp_df = df.select_dtypes(exclude='object')
    # get correlations
    corrmat = temp_df.corr()
    # exclude y variable columns as possible X variables
    exclude_columns = ['study_eq_labels', 'study_duration_days', 'primary_eq_labels', 'primary_study_duration_days']
    filtered = corrmat.loc['study_duration_days'][(abs(corrmat.loc['study_duration_days']) > 0.05) & (~corrmat.columns.isin(exclude_columns))]
    corr_cols = filtered.index.to_list()

    # save the list of column names and datatypes used for training
    # File path
    file_path = os.path.join(root, "model", f'model_columns_{args.model}.txt')
    prod_path = os.path.join(root, "trial_app", "backend", f'model_columns.txt')
    cols_dtypes = [(col, df[col].dtype.name) for col in corr_cols if col in df.columns]


    # Open the file in write mode
    with open(file_path, 'w') as file:
        # Write each element of the list to the file
        for column, dtype in cols_dtypes:
            file.write(f"{column}: {dtype}\n")

    if args.prod:
        with open(prod_path, 'w') as file:
            # Write each element of the list to the file
            for column, dtype in cols_dtypes:
                file.write(f"{column}: {dtype}\n")

    # split the data
    x_cols = corr_cols
    y_cols = ['study_eq_labels']
    X = df[x_cols]
    y = df[y_cols].values.ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # train and save a model
    results = train_model(type=args.model, X_train=X_train, y_train=y_train,
                          X_test=X_test, y_test=y_test, phase=phase, bins=bins, save_model=True, root=root, prod=args.prod)
    print(f"Model results: {results}")

    # # save a copy of the model's metadata in the trial_app folder too
    if args.prod == True:
        dest_dir = os.path.join(root, "trial_app", "backend", meta_file)

        with open(metadata_path, 'r') as file:
            data = json.load(file)
        
        data["model"] = f"{args.model}"

        with open(dest_dir, 'w') as file:
            json.dump(data, file, indent=4)

        # copy_json_file(dest_dir=dest_dir, src_file=metadata_path)


    json_path = os.path.join(root,path,meta_file)
    if not os.path.isfile(json_path):
        raise OSError("File does not exist.")
    
    with open(json_path, 'r') as file:
        meta_data = json.load(file)

    results.update(meta_data)
    json_filename = "modeling_results.json"
    append_to_json(results, filename=os.path.join(root, "model", json_filename))


data_msg = "Model training complete."
logger.info(data_msg)
print(data_msg)