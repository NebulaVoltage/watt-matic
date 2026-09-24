import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SGCC_MODEL_PATH = BASE_DIR / "models" / "sgcc_tuned" / "xgboost_reduced_best.joblib"
UCI_MODEL_PATH = BASE_DIR / "models" / "uci_forecasting" / "xgb_forecasting_frozen.joblib"

SGCC_THRESHOLD = 0.50
SGCC_FEATURE_COUNT = 18
UCI_FEATURE_COUNT = 29

SGCC_METRICS = {
    "model": "XGBoost Classifier",
    "features": 18,
    "threshold": 0.50,
    "test_f1": 0.4014,
    "test_pr_auc": 0.4049,
    "test_roc_auc": 0.8347,
    "test_recall": 0.5498,
    "test_precision": 0.3275
}

UCI_METRICS = {
    "model": "XGBoost Regressor",
    "features": 29,
    "forecast_horizon": "next_hour",
    "test_mae": 4421.55,
    "test_rmse": 6654.94,
    "test_r2": 0.9944,
    "test_smape": 2.18
}
