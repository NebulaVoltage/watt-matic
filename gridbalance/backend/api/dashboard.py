from fastapi import APIRouter
from backend.config import SGCC_METRICS, UCI_METRICS

router = APIRouter(prefix="/api/v1/dashboard", tags=["Dashboard"])

@router.get("/summary")
def get_dashboard_summary():
    return {
        "system_status": "operational",
        "sgcc_status": "online",
        "forecasting_status": "online",
        "models_summary": {
            "sgcc": {
                "name": SGCC_METRICS["model"],
                "features": SGCC_METRICS["features"],
                "threshold": SGCC_METRICS["threshold"],
                "test_f1": SGCC_METRICS["test_f1"],
                "test_pr_auc": SGCC_METRICS["test_pr_auc"]
            },
            "forecasting": {
                "name": UCI_METRICS["model"],
                "horizon": UCI_METRICS["forecast_horizon"],
                "test_mae_kwh": UCI_METRICS["test_mae"],
                "test_r2": UCI_METRICS["test_r2"],
                "test_smape": UCI_METRICS["test_smape"]
            }
        },
        "recent_metrics": {
            "total_meters_analyzed": 42367,
            "potential_tampering_flagged": 3612,
            "normal_meters": 38755,
            "next_hour_forecast_load_kwh": 128450.50
        }
    }
