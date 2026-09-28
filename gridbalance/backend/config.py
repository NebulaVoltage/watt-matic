import os
from pathlib import Path

# Base directory setup
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

# Model Paths
SGCC_MODEL_PATH = PROJECT_ROOT / "models" / "sgcc_tuned" / "xgboost_reduced_best.joblib"
UCI_MODEL_PATH = PROJECT_ROOT / "models" / "uci_forecasting" / "xgb_forecasting_frozen.joblib"

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
