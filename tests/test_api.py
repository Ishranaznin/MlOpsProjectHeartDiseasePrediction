"""
API Tests using FastAPI TestClient
Tests: health check, single prediction, batch prediction, invalid input
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from src.app import app

VALID_PATIENT = {
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
}


@pytest.fixture(scope="module")
def client():
    """Use context manager so lifespan (model loading) runs correctly."""
    with TestClient(app) as c:
        yield c


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_valid_input(client):
    response = client.post("/predict", json=VALID_PATIENT)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in [0, 1]
    assert "prediction_label" in data
    assert 0.0 <= data["probability_disease"] <= 1.0
    assert 0.0 <= data["probability_no_disease"] <= 1.0
    assert abs(data["probability_disease"] + data["probability_no_disease"] - 1.0) < 1e-4


def test_predict_batch(client):
    response = client.post("/predict_batch", json={"patients": [VALID_PATIENT, VALID_PATIENT]})
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert len(data["predictions"]) == 2


def test_predict_invalid_age(client):
    bad = {**VALID_PATIENT, "age": 200}
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_missing_field(client):
    incomplete = {k: v for k, v in VALID_PATIENT.items() if k != "chol"}
    assert client.post("/predict", json=incomplete).status_code == 422


def test_model_info(client):
    response = client.get("/model_info")
    assert response.status_code == 200
    assert "champion" in response.json()
