from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_api_health_endpoint():
    """
    Verify /health endpoint returns HTTP 200 and healthy status.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_api_metrics_endpoint():
    """
    Verify /metrics endpoint returns HTTP 200 and plain text Prometheus stats.
    """
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "api_requests_total" in response.text


def test_api_predict_success():
    """
    Verify hitting /predict with a valid payload returns HTTP 200 with prediction results.
    """
    payload = {
        "age": 63,
        "sex": 1,
        "cp": 3,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 0,
        "ca": 0,
        "thal": 1,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "prediction" in data
    assert "label" in data
    assert "confidence" in data
    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["confidence"] <= 1.0


def test_api_predict_invalid_payload():
    """
    Verify hitting /predict with a missing parameter payload returns HTTP 422 validation error.
    """
    # Missing 'age' parameter
    payload = {
        "sex": 1,
        "cp": 3,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 0,
        "ca": 0,
        "thal": 1,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
