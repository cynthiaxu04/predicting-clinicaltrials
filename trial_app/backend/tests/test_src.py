from fastapi.testclient import TestClient
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from main import app
from numpy.testing import assert_almost_equal

client = TestClient(app)


# DOCS - should work tests
def test_docs():
    response = client.get("/docs")
    assert response.status_code == 200
    assert "Swagger UI" in response.text


def test_docs_content():
    response = client.get("/docs")
    assert response.headers["content-type"] == "text/html; charset=utf-8"


# OPENAPI tests


def test_openapi_json():
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"

    assert "info" in response.json()
    assert "openapi" in response.json()


# Predict tests
# commands need to be updated if prediction columns change
def test_predict():
    data = {
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
        "healthy_vol": True,
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
        "phase_PHASE2_PHASE3": False,
        "phase_PHASE3": True,
    }

    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert_almost_equal(response.json()["bin_label"], 1)

# missing 1 data field
def test_predict_missing():
    data = {
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
        "healthy_vol": True,
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
        "phase_PHASE2_PHASE3": False
    }

    response = client.post("/predict", json=data)
    assert response.status_code == 422

def test_predict_extra():
    data = {
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
        "healthy_vol": True,
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
        "phase_PHASE2_PHASE3": False,
        "phase_PHASE3": True,
        "pooper": 2
    }

    response = client.post("/predict", json=data)
    assert response.status_code == 422
