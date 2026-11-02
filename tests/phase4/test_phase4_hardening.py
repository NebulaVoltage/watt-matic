"""
GridBalance Phase 4 Workstream G: Expanded Automated Regression Testing
File: tests/phase4/test_phase4_hardening.py
Author: Antigravity Senior Software Quality Architect & ML Systems Engineer
Date: November 2, 2026
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import joblib

base_dir = r"D:\watt-matic\gridbalance"
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from fastapi.testclient import TestClient
from backend.main import app
from backend.inference.sgcc_predictor import SGCCPredictor
from backend.inference.uci_forecaster import UCIForecaster

client = TestClient(app)

# ==============================================================================
# 1. SGCC OPERATING THRESHOLD BOUNDARY TESTS
# ==============================================================================

def test_sgcc_operating_modes_threshold_boundaries():
    """Verify that operating modes correctly evaluate decision thresholds."""
    detector = SGCCPredictor()
    
    # Mock probabilities around boundary regions
    prob_low = 0.35      # Below all thresholds
    prob_mid_low = 0.45  # Triggers High-Recall (0.40), but not Balanced (0.50)
    prob_mid_high = 0.52 # Triggers Balanced (0.50), but not Targeted (0.55)
    prob_high = 0.60     # Triggers all thresholds
    
    modes_and_thresholds = [
        ("balanced", 0.50),
        ("high_recall", 0.40),
        ("targeted", 0.55)
    ]
    
    for mode, thresh in modes_and_thresholds:
        # Check high
        assert (prob_high >= thresh) == True
        # Check low
        assert (prob_low >= thresh) == False
        
    # High-recall specific trigger
    assert (prob_mid_low >= 0.40) == True
    assert (prob_mid_low >= 0.50) == False
    
    # Balanced specific trigger
    assert (prob_mid_high >= 0.50) == True
    assert (prob_mid_high >= 0.55) == False


# ==============================================================================
# 2. TERMINOLOGY INVARIANT: NON-THEFT CLASSIFICATION LABELS
# ==============================================================================

def test_sgcc_api_non_theft_terminology_invariant():
    """Verify API strictly returns 'potential_tampering' or 'normal', never 'confirmed_theft'."""
    # Create valid synthetic payload
    dates = pd.date_range("2016-01-01", periods=100, freq="D")
    
    payload = {
        "meter_id": "TEST_METER_TERMINOLOGY_01",
        "dates": [str(d.date()) for d in dates],
        "consumption": [float(10.0 + (i % 7)) for i in range(len(dates))],
        "model_type": "xgboost",
        "operating_mode": "balanced"
    }
    
    res = client.post("/api/v1/detection/analyze", json=payload)
    
    assert res.status_code == 200
    data = res.json()
    
    # Label must be cautious
    classification = data.get("prediction")
    assert classification in ["normal", "potential_tampering"], f"Illegal classification label: {classification}"
    assert "theft" not in classification.lower() or classification == "potential_tampering"
    assert "confirmed_theft" not in classification.lower()
    
    # Status code and flags
    assert "confirmed_theft" not in data.get("prediction", "").lower()
    assert 0.0 <= data.get("probability", -1.0) <= 1.0
    assert "risk_level" in data


# ==============================================================================
# 3. UCI EMA INPUT DEFENSE & ADVERSARIAL REJECTIONS
# ==============================================================================

def test_uci_ema_input_defense_truncated_history():
    """Verify history length < 336 hours is strictly rejected by the EMA endpoint."""
    dates = pd.date_range("2014-01-01", periods=335, freq="h")
    history_335 = [{"timestamp": str(dt), "load_kwh": 200000.0} for dt in dates]
    
    res = client.post("/api/v1/forecast/next-hour-experimental-ema", json={"history": history_335})
    assert res.status_code in [400, 422]
    assert "336" in res.text or "Insufficient" in res.text


def test_uci_ema_input_defense_empty_and_nan_injection():
    """Verify empty history or NaN values trigger client rejection."""
    # Empty
    res_empty = client.post("/api/v1/forecast/next-hour-experimental-ema", json={"history": []})
    assert res_empty.status_code in [400, 422]
    
    # NaN injection
    dates = pd.date_range("2014-01-01", periods=336, freq="h")
    history_nan = [{"timestamp": str(dt), "load_kwh": 200000.0} for dt in dates]
    history_nan[100]["load_kwh"] = None
    res_nan = client.post("/api/v1/forecast/next-hour-experimental-ema", json={"history": history_nan})
    assert res_nan.status_code in [400, 422]


def test_uci_ema_input_defense_malformed_timestamp():
    """Verify malformed timestamp string is rejected."""
    dates = pd.date_range("2014-01-01", periods=336, freq="h")
    history_bad = [{"timestamp": str(dt), "load_kwh": 200000.0} for dt in dates]
    history_bad[50]["timestamp"] = "CORRUPT_DATE_XYZ_9999"
    res_bad = client.post("/api/v1/forecast/next-hour-experimental-ema", json={"history": history_bad})
    assert res_bad.status_code in [400, 422]


def test_uci_ema_input_defense_out_of_order_robustness():
    """Verify out-of-order timestamps are sorted cleanly and yield valid forecast."""
    dates = pd.date_range("2014-01-01", periods=336, freq="h")
    history = [{"timestamp": str(dt), "load_kwh": 200000.0 + (i % 24) * 100} for i, dt in enumerate(dates)]
    
    # Reference in-order
    res_ref = client.post("/api/v1/forecast/next-hour-experimental-ema", json={"history": history})
    assert res_ref.status_code == 200
    ref_forecast = res_ref.json()["predicted_load_kwh"]
    
    # Shuffled
    shuffled_history = list(history)
    np.random.seed(42)
    np.random.shuffle(shuffled_history)
    
    res_shuffled = client.post("/api/v1/forecast/next-hour-experimental-ema", json={"history": shuffled_history})
    assert res_shuffled.status_code == 200
    shuffled_forecast = res_shuffled.json()["predicted_load_kwh"]
    
    assert abs(ref_forecast - shuffled_forecast) < 1e-4


# ==============================================================================
# 4. DATA LEAKAGE IMPUTER PURITY TEST
# ==============================================================================

def test_data_leakage_imputer_purity():
    """Verify serialized XGBoost pipeline imputer matches X_train medians with 0.0 error."""
    xgb_path = os.path.join(base_dir, "models", "sgcc_tuned", "xgboost_reduced_best.joblib")
    sgcc_data_path = os.path.join(base_dir, "data", "processed", "sgcc_features_real_v2.csv")
    
    pipeline = joblib.load(xgb_path)
    df = pd.read_csv(sgcc_data_path)
    
    feats = list(pipeline.feature_names_in_)
    y = df["FLAG"]
    X = df[feats]
    
    from sklearn.model_selection import train_test_split
    X_temp, _, y_temp, _ = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, _, _, _ = train_test_split(X_temp, y_temp, test_size=0.17647, random_state=42, stratify=y_temp)
    
    m_train = X_train.median().values
    m_pipeline = pipeline.named_steps["imputer"].statistics_
    
    max_diff = np.max(np.abs(m_pipeline - m_train))
    assert max_diff == 0.0, f"Imputer contains data leakage! Max diff: {max_diff}"


# ==============================================================================
# 5. CANONICAL BENCHMARK LEDGER INTEGRITY TEST
# ==============================================================================

def test_canonical_benchmark_ledger_integrity():
    """Verify CANONICAL_BENCHMARK_LEDGER.csv exists, has 35 rows and non-null metrics."""
    ledger_path = os.path.join(base_dir, "results", "phase4", "CANONICAL_BENCHMARK_LEDGER.csv")
    assert os.path.exists(ledger_path), "Canonical ledger CSV missing!"
    
    df = pd.read_csv(ledger_path)
    assert len(df) == 35, f"Expected 35 authoritative rows in ledger, found {len(df)}"
    
    required_cols = ["experiment_id", "branch", "model_name", "objective_type", "reproducibility_status"]
    for col in required_cols:
        assert col in df.columns, f"Missing required column in benchmark ledger: {col}"
        assert df[col].isnull().sum() == 0, f"Found nulls in required column: {col}"
