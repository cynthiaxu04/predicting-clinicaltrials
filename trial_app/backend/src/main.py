import os
import numpy as np
import pandas as pd
import json
import joblib
from datetime import datetime
from fastapi import FastAPI

app = FastAPI()

# ML model for API
model = joblib.load("../../../model/model_rf.pkl")

# root
@app.get("/")
async def root():
    return {"message": "Welcome to the model API!"}

# health endpoint returning the current time in the ISO format
@app.get("/health")
async def health_check():
    current_time = datetime.now()
    current_time = current_time.isoformat()
    return {"time": f"{current_time}"}

# predict endpoint
@app.post("/predict", response_model=HousePredictions)
async def predict(
    model_inputs = np.array([list(house_dict.values()) for house_dict in houses.dict()['houses']])
    model_results = model.predict(model_inputs)
    return {"predictions": list(model_results)}