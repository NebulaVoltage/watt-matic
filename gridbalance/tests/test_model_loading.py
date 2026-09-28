import pytest
from backend.services.sgcc_service import get_sgcc_predictor
from backend.services.forecasting_service import get_uci_forecaster

def test_sgcc_model_loading():
    predictor = get_sgcc_predictor()
    assert predictor.is_loaded() is True
    assert len(predictor.feature_names) == 18

def test_uci_model_loading():
    forecaster = get_uci_forecaster()
    assert forecaster.is_loaded() is True
    assert len(forecaster.feature_names) == 29
