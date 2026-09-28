import pytest
import pandas as pd
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["sgcc_model"] == "loaded"
    assert data["forecast_model"] == "loaded"

def test_models_info_endpoint():
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "sgcc" in data
    assert "forecasting" in data
    assert data["sgcc"]["feature_count"] == 18
    assert data["forecasting"]["feature_count"] == 29

def test_detection_analyze_endpoint():
    payload = {
        "meter_id": "CONS_API_001",
        "dates": ["2014-01-01", "2014-01-02", "2014-01-03", "2014-01-04", "2014-01-05"],
        "consumption": [12.5, 13.0, 11.8, 14.2, 12.0]
    }
    response = client.post("/api/v1/detection/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["meter_id"] == "CONS_API_001"
    assert data["prediction"] in ["normal", "potential_tampering"]
    assert "confirmed_theft" not in data["prediction"]
    assert 0.0 <= data["probability"] <= 1.0

def test_detection_invalid_dates():
    payload = {
        "meter_id": "CONS_INVALID",
        "dates": ["invalid-date-string"],
        "consumption": [10.0]
    }
    response = client.post("/api/v1/detection/analyze", json=payload)
    assert response.status_code == 400

def test_forecasting_endpoint():
    dates = pd.date_range("2014-01-01 00:00:00", periods=180, freq="h")
    history = [{"timestamp": str(dt), "load_kwh": 4000.0 + (i % 24) * 50.0} for i, dt in enumerate(dates)]
    payload = {"history": history}

    response = client.post("/api/v1/forecast/next-hour", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["forecast_horizon"] == "next_hour"
    assert "target_timestamp" in data
    assert isinstance(data["predicted_load_kwh"], float)

def test_forecasting_insufficient_history():
    payload = {
        "history": [
            {"timestamp": "2014-01-01 00:00:00", "load_kwh": 100.0}
        ]
    }
    response = client.post("/api/v1/forecast/next-hour", json=payload)
    assert response.status_code == 422 # Pydantic validation error for history length < 168
