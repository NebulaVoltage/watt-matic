import pytest
import numpy as np
from fastapi.testclient import TestClient
from backend.main import app
from backend.inference.sgcc_predictor import SGCCPredictor
from backend.inference.uci_forecaster import UCIForecaster

client = TestClient(app)

@pytest.fixture
def sgcc_predictor():
    return SGCCPredictor()

@pytest.fixture
def uci_forecaster():
    return UCIForecaster()

# =====================================================================
# 1. MODEL LOADING TESTS
# =====================================================================
def test_phase3_model_loading(sgcc_predictor, uci_forecaster):
    # SGCC models
    assert sgcc_predictor.is_loaded() is True
    assert sgcc_predictor.is_experimental_loaded() is True
    assert len(sgcc_predictor.feature_names) == 18
    
    # UCI models
    assert uci_forecaster.is_loaded() is True
    assert uci_forecaster.is_experimental_loaded() is True
    assert len(uci_forecaster.feature_names) == 29
    assert len(uci_forecaster.feature_names_33) == 33

# =====================================================================
# 2. MODEL SELECTION & INFERENCE CONSISTENCY TESTS
# =====================================================================
def test_phase3_sgcc_model_selection(sgcc_predictor):
    dates = [f"2014-01-{i:02d}" for i in range(1, 31)]
    consumption = [12.0 + (i % 4) * 0.5 for i in range(1, 31)]

    # Default model (XGBoost)
    res_default = sgcc_predictor.predict(dates=dates, consumption=consumption, meter_id="M_DEFAULT")
    assert res_default["model_type"] == "xgboost"
    assert 0.0 <= res_default["probability"] <= 1.0

    # Explicit XGBoost
    res_xgb = sgcc_predictor.predict(dates=dates, consumption=consumption, meter_id="M_XGB", model_type="xgboost")
    assert res_xgb["model_type"] == "xgboost"
    assert res_xgb["probability"] == res_default["probability"]

    # Explicit LightGBM Focal Loss
    res_lgb = sgcc_predictor.predict(dates=dates, consumption=consumption, meter_id="M_LGB", model_type="lightgbm_focal")
    assert res_lgb["model_type"] == "lightgbm_focal"
    assert 0.0 <= res_lgb["probability"] <= 1.0
    
    # Both models must obey strict non-theft terminology
    assert res_xgb["prediction"] in ["potential_tampering", "normal"]
    assert res_xgb["prediction"] != "confirmed_theft"
    assert res_lgb["prediction"] in ["potential_tampering", "normal"]
    assert res_lgb["prediction"] != "confirmed_theft"

# =====================================================================
# 3. SGCC OPERATING MODES & THRESHOLD TESTS
# =====================================================================
def test_phase3_sgcc_operating_modes(sgcc_predictor):
    dates = [f"2014-01-{i:02d}" for i in range(1, 31)]
    # Suspicious pattern: 15 days high load followed by 15 days near zero
    consumption = [25.0] * 15 + [0.5] * 15

    # 1. Balanced mode (0.50)
    res_balanced = sgcc_predictor.predict(dates=dates, consumption=consumption, meter_id="M_BAL", operating_mode="balanced")
    assert res_balanced["threshold"] == 0.50
    assert res_balanced["operating_mode"] == "balanced"

    # 2. High-recall mode (0.40)
    res_recall = sgcc_predictor.predict(dates=dates, consumption=consumption, meter_id="M_REC", operating_mode="high_recall")
    assert res_recall["threshold"] == 0.40
    assert res_recall["operating_mode"] == "high_recall"

    # 3. Targeted mode (0.55)
    res_target = sgcc_predictor.predict(dates=dates, consumption=consumption, meter_id="M_TGT", operating_mode="targeted")
    assert res_target["threshold"] == 0.55
    assert res_target["operating_mode"] == "targeted"

    # 4. Custom threshold override
    res_custom = sgcc_predictor.predict(dates=dates, consumption=consumption, meter_id="M_CUST", threshold=0.45)
    assert res_custom["threshold"] == 0.45

# =====================================================================
# 4. API INTEGRATION TESTS
# =====================================================================
def test_phase3_api_analyze_with_model_selection():
    readings = [{"date": f"2014-01-{i:02d}", "consumption": 10.0} for i in range(1, 31)]
    
    # Call with LightGBM and High-Recall mode
    payload = {
        "meter_id": "API_TEST_LGB",
        "readings": readings,
        "model_type": "lightgbm_focal",
        "operating_mode": "high_recall"
    }
    response = client.post("/api/v1/detection/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["meter_id"] == "API_TEST_LGB"
    assert data["model_type"] == "lightgbm_focal"
    assert data["operating_mode"] == "high_recall"
    assert data["threshold"] == 0.40
    assert data["prediction"] in ["potential_tampering", "normal"]

def test_phase3_api_models_endpoint():
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "sgcc" in data
    assert "forecasting" in data
    assert "sgcc_experimental" in data
    assert "forecasting_experimental" in data
    assert data["sgcc_experimental"]["model_name"] == "LightGBM Classifier (Focal Loss alpha=0.85, gamma=2.0)"
    assert data["forecasting_experimental"]["required_history_hours"] == 336

# =====================================================================
# 5. UCI FORECASTING & EXPERIMENTAL EMA TESTS
# =====================================================================
def test_phase3_uci_experimental_ema_endpoint():
    import pandas as pd
    
    # 1. Test rejection if history < 336 hours (e.g. 168 hours)
    dates_168 = pd.date_range("2014-01-01", periods=168, freq="h")
    short_history = [
        {"timestamp": str(dt), "load_kwh": 200000.0 + (i % 20) * 1000}
        for i, dt in enumerate(dates_168)
    ]
    resp_short = client.post("/api/v1/forecast/next-hour-experimental-ema", json={"history": short_history})
    assert resp_short.status_code == 400
    assert "minimum 336 hours required" in resp_short.json()["detail"]

    # 2. Test success with 336 hours history
    dates_336 = pd.date_range("2014-01-01", periods=336, freq="h")
    full_history = [
        {"timestamp": str(dt), "load_kwh": 200000.0 + (i % 20) * 1000}
        for i, dt in enumerate(dates_336)
    ]
    resp_full = client.post("/api/v1/forecast/next-hour-experimental-ema", json={"history": full_history})
    assert resp_full.status_code == 200
    data = resp_full.json()
    assert data["model_type"] == "lightgbm_ema_33"
    assert data["features_used"] == 33
    assert data["history_length_hours"] == 336
    assert data["warmup_error_status"] == "ZERO_BIAS_VERIFIED_336H"
    assert data["predicted_load_kwh"] > 0

# =====================================================================
# 6. INVALID INPUT TESTS
# =====================================================================
def test_phase3_invalid_inputs(sgcc_predictor):
    dates = [f"2014-01-{i:02d}" for i in range(1, 10)]
    consumption = [10.0] * 9

    # Unknown model_type
    with pytest.raises(ValueError, match="Unknown model_type"):
        sgcc_predictor.predict(dates=dates, consumption=consumption, meter_id="M_ERR", model_type="catboost")

    # Unknown operating_mode
    with pytest.raises(ValueError, match="Unknown operating_mode"):
        sgcc_predictor.predict(dates=dates, consumption=consumption, meter_id="M_ERR", operating_mode="ultra_strict")
