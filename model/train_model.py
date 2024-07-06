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
    parser.add_argument('csv_file', type=csv_file, help='Path to the CSV file')
    parser.add_argument('model', type=str, choices=['lgb', 'xgb', 'rf'], help='Choose model type to run')

    return parser.parse_args()

def train_model(type:str, X_train, y_train, X_test, y_test, save_model:bool):
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
        'accuracy': acc,
        'mean_accuracy_kfold': cv_results.mean(),
        'precision': pre,
        'r_squared': r2,
        'mean_squared_error': mse,
        'mean_absolute_error': mae,
    }

    if save_model==True:
        with open(f'model_{type}.pkl', 'wb') as file:
            pickle.dump(model, file)

    # filename = "modeling_results.json"
    # with open(filename, 'a') as file:
    #     json.dump(row, file)

    return row

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
    # get command line arguments
    args = get_parser()
    # validate the data csv file
    csv_file(args.csv_file)

    # read in the data
    df = pd.read_csv(args.csv_file)

    # get only the numerical columns
    temp_df = df.select_dtypes(exclude='object')
    # get correlations
    corrmat = temp_df.corr()
    # exclude y variable columns as possible X variables
    exclude_columns = ['study_eq_labels', 'study_duration_days', 'primary_eq_labels', 'primary_study_duration_days']
    filtered = corrmat.loc['study_duration_days'][(abs(corrmat.loc['study_duration_days']) > 0.05) & (~corrmat.columns.isin(exclude_columns))]
    corr_cols = filtered.index.to_list()

    # split the data
    x_cols = corr_cols
    y_cols = ['study_eq_labels']
    X = df[x_cols]
    y = df[y_cols]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # train and save a model
    results = train_model(type=args.model, X_train=X_train, y_train=y_train,
                          X_test=X_test, y_test=y_test, save_model=False)
    print(f"Model results: {results}")

    filename = "modeling_results.json"
    append_to_json(results, filename=filename)



data_msg = "Model training complete."
logger.info(data_msg)
print(data_msg)