import os
from pathlib import Path

# Base directory setup
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

# Model Paths (Production Baselines)
SGCC_MODEL_PATH = PROJECT_ROOT / "models" / "sgcc_tuned" / "xgboost_reduced_best.joblib"
UCI_MODEL_PATH = PROJECT_ROOT / "models" / "uci_forecasting" / "xgb_forecasting_frozen.joblib"

# Model Paths (Experimental Candidates)
SGCC_EXPERIMENTAL_MODEL_PATH = PROJECT_ROOT / "models" / "optimization_v2" / "lgbm_focal_loss_best.joblib"
UCI_EXPERIMENTAL_MODEL_PATH = PROJECT_ROOT / "models" / "optimization_v2" / "lgb_uci_v2_04.joblib"

# Model Selection Configuration
SGCC_DEFAULT_MODEL_TYPE = os.getenv("SGCC_MODEL_TYPE", "xgboost")  # "xgboost" (active) or "lightgbm_focal" (candidate)

# SGCC Operating Modes & Thresholds
SGCC_OPERATING_MODES = {
    "balanced": 0.50,
    "high_recall": 0.40,
    "targeted": 0.55
}
SGCC_DEFAULT_OPERATING_MODE = "balanced"

# Research Evaluation Metrics (Frozen Benchmarks)
SGCC_METRICS = {
    "model_name": "XGBoost Classifier (18-Feature Pruned)",
    "feature_count": 18,
    "decision_threshold": 0.50,
    "test_f1": 0.4014,
    "test_pr_auc": 0.4049,
    "test_roc_auc": 0.8347,
    "test_recall": 0.5498,
    "test_precision": 0.3275,
    "sealed_test_size": 6355,
    "disclaimer": "Metrics evaluated on sealed test partition (15% split, 6,355 customers). Model & threshold frozen."
}

UCI_METRICS = {
    "model_name": "XGBoost Regressor (29-Feature)",
    "feature_count": 29,
    "forecast_horizon": "next_hour",
    "test_mae_kwh": 4421.55,
    "test_rmse_kwh": 6654.94,
    "test_r2": 0.9944,
    "test_smape_pct": 2.18,
    "sealed_test_size": 4416,
    "disclaimer": "Metrics evaluated on sealed chronological test split (July 1 - Dec 31, 2014, 4,416 hours). Model frozen."
}

SGCC_EXPERIMENTAL_METRICS = {
    "model_name": "LightGBM Classifier (Focal Loss alpha=0.85, gamma=2.0)",
    "feature_count": 18,
    "decision_threshold": 0.50,
    "cv_f1": 0.4166,
    "val_f1": 0.4469,
    "test_f1": 0.4479,
    "test_pr_auc": 0.4178,
    "test_roc_auc": 0.8429,
    "test_precision": 0.4200,
    "test_recall": 0.4797,
    "operating_modes": SGCC_OPERATING_MODES,
    "disclaimer": "Experimental candidate evaluated on 5-fold CV, validation, and sealed test."
}

UCI_EXPERIMENTAL_METRICS = {
    "model_name": "LightGBM Regressor (33-Feature with Fourier & EMA)",
    "feature_count": 33,
    "required_history_hours": 336,
    "test_mae_kwh": 4357.20,
    "disclaimer": "Experimental candidate requiring 336 hours of contiguous load history for zero-bias EMA reconstruction."
}
