import pytest
import numpy as np
from backend.inference.sgcc_predictor import SGCCPredictor

@pytest.fixture
def predictor():
    return SGCCPredictor()

def test_chronological_sorting(predictor):
    # Unsorted dates (lexicographical format)
    dates = ["2014-01-10", "2014-01-01", "2014-01-05"]
    consumption = [10.0, 5.0, 7.5]

    X_df, quality, history = predictor.preprocess_and_extract_features(dates, consumption)

    # Verify history is sorted chronologically
    hist_dates = [h['date'] for h in history]
    assert hist_dates == ["2014-01-01", "2014-01-05", "2014-01-10"]
    assert quality["observation_count"] == 3

def test_sgcc_inference_output(predictor):
    dates = [f"2014-01-{i:02d}" for i in range(1, 31)]
    # Normal consumption pattern
    consumption = [10.0 + (i % 3) * 0.5 for i in range(1, 31)]

    res = predictor.predict(dates=dates, consumption=consumption, meter_id="TEST_001")

    assert res["meter_id"] == "TEST_001"
    assert res["prediction"] in ["normal", "potential_tampering"]
    assert res["prediction"] != "confirmed_theft" # Enforce strict terminology rule
    assert 0.0 <= res["probability"] <= 1.0
    assert res["threshold"] == 0.50
    assert res["risk_level"] in ["High", "Medium", "Low"]
    assert len(res["top_features"]) <= 5
    assert len(res["consumption_history"]) == 30

def test_sgcc_tampering_pattern(predictor):
    dates = [f"2014-01-{i:02d}" for i in range(1, 31)]
    # Sudden drop to 0.0 (simulated tampering / outage pattern)
    consumption = [15.0]*10 + [0.0]*20

    res = predictor.predict(dates=dates, consumption=consumption, meter_id="TEST_TAMPER")
    assert res["prediction"] in ["potential_tampering", "normal"]
    assert res["data_quality"]["longest_missing_streak"] >= 0
