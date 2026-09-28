import pytest
import pandas as pd
from backend.inference.uci_forecaster import UCIForecaster

@pytest.fixture
def forecaster():
    return UCIForecaster()

def test_uci_insufficient_history(forecaster):
    # Only 50 hours (less than required 168)
    dates = pd.date_range("2014-01-01 00:00:00", periods=50, freq="h")
    history = [{"timestamp": str(dt), "load_kwh": 100.0 + i} for i, dt in enumerate(dates)]

    with pytest.raises(ValueError, match="Insufficient history"):
        forecaster.predict_next_hour(history)

def test_uci_forecasting_valid(forecaster):
    # 200 hours (> 168 hours)
    dates = pd.date_range("2014-01-01 00:00:00", periods=200, freq="h")
    history = [{"timestamp": str(dt), "load_kwh": 5000.0 + (i % 24) * 100.0} for i, dt in enumerate(dates)]

    res = forecaster.predict_next_hour(history)

    assert res["forecast_horizon"] == "next_hour"
    assert res["target_timestamp"] == "2014-01-09 08:00:00"
    assert isinstance(res["predicted_load_kwh"], float)
    assert res["predicted_load_kwh"] > 0
    assert "model_info" in res
