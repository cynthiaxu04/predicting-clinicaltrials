# ClinicalAI App

## Application Objective
The purpose of this application is to provide a user friendly interface for clinical trial conducters to enter study protocol information and get a predictied time interval for how long the trial will take.

## How to build the Application
Within the ```/trial_app``` folder, build the application with the command:
```
docker-compose build
```
Allow at least a couple of minutes for both the frontend and backend images to build.
## How to run the Application
Once the images are built, run the application with the commmand:
```
docker-compose up
```
And enter the provided localhost URL into your web browser of choice, e.g.
```
http://localhost:8501
```

## Developer Notes
### Backend
The backend of the application consists of a ```POST``` endpoint that will:
1. load the model from ```model.pkl```
2. load the input data fields and their associated data types to the model from ```model_columns.txt``` into a pydantic model
3. load the mapping from bin labels to time intervals from ```metadata.json```

The model outputs the bin label (an integer) as prediction. This is mapped back to the explicit time interval (in days). The final output to the user is this time interval converted into years for readability.

To test the backend separately, while in the directory ```/trial_app```, run the command:
```
docker build -f Dockerfile.fastapi -t backend .
```
Once the image is built, run the backend with the command:
```
docker run -p 8000:8000 backend
```
In a new terminal window, test the ```POST``` endpoint with the curl command:
```
curl -X 'POST' \
  'http://0.0.0.0:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
"number_of_conditions": 1,
    "number_of_groups": 2,
    "num_locations": 10,
    "location": 1,
    "num_inclusion": 5.0,
    "num_exclusion": 3.0,
    "number_of_intervention_types": 4,
    "intervention_model": 1.5,
    "resp_party": 1,
    "has_dmc": 1.0,
    "masking": 1.0,
    "enroll_count": 100.0,
    "healthy_vol": true,
    "treatment_purpose": 1,
    "diagnostic_purpose": 0,
    "prevention_purpose": 0,
    "drug_intervention": 1,
    "biological_intervention": 0,
    "os_outcome_measure": 1,
    "dor_outcome_measure": 0,
    "ae_outcome_measure": 0,
    "primary_max_days": 365.0,
    "secondary_max_days": 730.0,
    "max_treatment_duration": 365,
    "min_treatment_duration": 30,
    "survival_5yr_relative": 0.8,
    "phase_PHASE2_PHASE3": false,
    "phase_PHASE3": true
}'
```
**Note**: this curl command is specific to the RandomForest model with 3 bins trained on Phase 3 oncology data from post-2011.

### Frontend
To test the backend separately, while in the directory ```/trial_app```, run the command:
```
docker build -f Dockerfile.streamlit -t frontend .
```
Once the image is built, launch the frontend with the command:
```
docker run -p 8501:8501 frontend
```
Copy and paste the provided localhost links into your web browser, e.g.
```
http://localhost:8501
```