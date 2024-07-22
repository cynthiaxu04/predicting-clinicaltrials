from fastapi.testclient import TestClient
from datetime import datetime
from src.main import app
from numpy.testing import assert_almost_equal

client = TestClient(app)

#DOCS - should work tests
def test_docs():
	response = client.get("/docs")
	assert response.status_code == 200
	assert "Swagger UI" in response.text

def test_docs_content():
	response = client.get("/docs")
	assert response.headers["content-type"] == "text/html; charset=utf-8"

#OPENAPI tests

def test_openapi_json():
	response = client.get("/openapi.json")
	assert response.status_code == 200
	assert response.headers["content-type"] == "application/json"

	assert "info" in response.json()
	assert "openapi" in response.json()


#Predict tests

def test_predict():
	data = {"MedInc": 0.3252, "HouseAge": 41.0, "AveRooms": 6.98412698, 
		"AveBedrms": 1.02380952, "Population": 322.0, "AveOccup": 2.55555556, 
		"Latitude": 37.88, "Longitude": -122.23}

	response = client.post("/predict", json=data)
	assert response.status_code == 200
	assert_almost_equal( response.json()["prediction"], 1.969, decimal=3)

def test_predict_long():
	data = {"MedInc": 0.3252, "HouseAge": 41.0, "AveRooms": 6.98412698,
		"AveBedrms": 1.02380952, "Population": 322.0, "AveOccup": 2.55555556,
		"Latitude": 37.88, "Longitude": -190.23}
	
	response = client.post("/predict", json=data)
	assert response.status_code == 422

def test_predict_lat():
	data = {"MedInc": 0.3252, "HouseAge": 41.0, "AveRooms": 6.98412698,
		"AveBedrms": 1.02380952, "Population": 322.0, "AveOccup": 2.55555556,
		"Latitude": 100.0, "Longitude": -122.23}
	
	response = client.post("/predict", json=data)
	assert response.status_code == 422

def test_predict_missing():
	data = {"MedInc": 0.3252}

	response = client.post("/predict", json=data)
	assert response.status_code == 422

def test_predict_extra():
	data = {"MedInc": 0.3252, "HouseAge": 41.0, "AveRooms": 6.98412698, 
		"AveBedrms": 1.02380952, "Population": 322.0, "AveOccup": 2.55555556, 
		"Latitude": 37.88, "Longitude": -122.23, "Extra": 2}

	response = client.post("/predict", json=data)
	assert response.status_code == 422

