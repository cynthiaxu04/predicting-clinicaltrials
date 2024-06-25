import os
import logging
import pandas as pd
import numpy as np
import argparse

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
    parser = argparse.ArgumentParser(description='Validate CSV file columns using a YAML schema.')
    parser.add_argument('csv_file', type=csv_file, help='Path to the CSV file')

    return parser.parse_args()

#########################################################################################################################
# Code to do actual modeling
if __name__ == "__main__":
    pass

data_msg = "Model training complete."
logger.info(data_msg)
print(data_msg)