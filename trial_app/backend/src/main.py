import os
import numpy as np
# import pandas as pd
import json
import joblib
from pydantic import BaseModel, create_model
# from datetime import datetime
from fastapi import FastAPI
from typing import Union, Dict, Type


app = FastAPI()

model_name = 'model.pkl'
meta_file = 'metadata.json'
cols_file = 'model_columns.txt'

# get current dir
current_dir = os.path.dirname(os.path.realpath(__file__))
# get root dir
root_dir = os.path.dirname(os.path.dirname(current_dir))

# load model
model_path = os.path.join(root_dir, "backend", model_name)
# load metadata
meta_path = os.path.join(root_dir, "backend", meta_file)
# load column names & dtypes
col_path = os.path.join(root_dir, "backend", cols_file)

if os.path.isfile(model_path):
    model = joblib.load(model_path)
else:
    raise OSError(f"Model pkl file not found: {model_path}")

if os.path.isfile(meta_path):
    with open(meta_path, 'r') as file:
        meta_data = json.load(file)

def read_column_types(file_path: str):
    column_types = {}
    with open(file_path, 'r') as file:
        for line in file:
            column_name, dtype = line.strip().split(':')
            column_types[column_name.strip()] = dtype.strip()
    return column_types

if os.path.isfile(col_path):
    cols = read_column_types(col_path)

# print(cols)
# print(meta_data)

# a function to convert the bin intervals from days to years
def convert_interval_to_years(interval: str) -> str:
    # print(f"here is the interval being operated on: {interval}")
    start, end = interval.strip('(').strip(']').split(',')
    start_days = float(start)
    end_days = float(end)
    start_years = start_days / 365
    end_years = end_days / 365
    return f"({start_years:.2f}, {end_years:.2f}]"

def create_pydantic_model(column_types: Dict[str, str]) -> Type[BaseModel]:
    # Mapping from file dtypes to Python types for Pydantic
    type_mapping = {
        'int64': (int, ...),
        'float64': (float, ...),
        'bool': (bool, ...)
    }

    # Create a dictionary to hold the fields for the Pydantic model
    fields = {col: type_mapping[dtype] for col, dtype in column_types.items()}

    # Create the Pydantic model
    return create_model('UserInput', **fields)

#create the dynamic Pydantic model
UserInput = create_pydantic_model(cols)

# print(UserInput.schema())

# # validate against model/model_columns_{model}.txt
# class UserInput(BaseModel):
#     num_locations: Union[int, float]
#     location: Union[int, float]
#     num_inclusion: Union[int, float]
#     num_exclusion: Union[int, float]
#     number_of_intervention_types: Union[int, float]
#     intervention_model: Union[int, float]
#     resp_party: Union[int, float]
#     has_dmc: Union[int, float]
#     allocation: Union[int, float]
#     masking: Union[int, float]
#     enroll_count: Union[int, float]
#     healthy_vol: bool
#     treatment_purpose: int
#     diagnostic_purpose: int
#     prevention_purpose: int
#     device_intervention: int
#     drug_intervention: int
#     radiation_intervention: int
#     biological_intervention: int
#     os_outcome_measure: int
#     dor_outcome_measure: int
#     ae_outcome_measure: int
#     primary_max_days: Union[int, float]
#     secondary_max_days: Union[int, float]
#     max_treatment_duration: Union[int, float]
#     min_treatment_duration: Union[int, float]
#     survival_5yr_relative: Union[int, float]
#     phase_PHASE2_PHASE3: bool
#     phase_PHASE3: bool

#     # def to_series(self):
#     #     return pd.read_json(self)
    
#     def to_np(self):
#         return np.array([list(vars(self).values())])

class Output(BaseModel):
    bin_label: int
    bin_interval: str

    @classmethod
    def from_label(cls, bin_label: int, bins: dict):
        bin_interval_days = bins[str(bin_label)]
        bin_interval_years = convert_interval_to_years(bin_interval_days)
        return cls(bin_label=bin_label, bin_interval=bin_interval_years)

# predict endpoint
@app.post("/predict", response_model=Output)
async def predict(values: UserInput):
    # Convert the Pydantic model to a dictionary
    data_dict = values.dict()
    # Convert the dictionary to a list of values (assuming the order is correct)
    data_list = [data_dict[col] for col in cols]  # `column_order` should match the order expected by the model
    # Convert list to a NumPy array if necessary
    data_array = np.array([data_list])
    # data = values
    prediction = model.predict(data_array)[0] #predict returns a list, get just the first (and only) element
    return Output.from_label(prediction, meta_data['bins'])