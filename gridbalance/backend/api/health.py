from datetime import datetime
from fastapi import APIRouter

from backend.schemas.health import HealthResponse
from backend.services.sgcc_service import get_sgcc_predictor
from backend.services.forecasting_service import get_uci_forecaster

router = APIRouter(tags=["System Health"])

@router.get("/health", response_model=HealthResponse, summary="System Health & Model Load Status")
def health_check():
    """
    Verifies API health and confirms both frozen machine learning model artifacts are loaded in memory.
    """
    try:
        sgcc = get_sgcc_predictor()
        sgcc_status = "loaded" if sgcc.is_loaded() else "not_loaded"
    except Exception as e:
        sgcc_status = f"error: {e}"

    try:
        uci = get_uci_forecaster()
        uci_status = "loaded" if uci.is_loaded() else "not_loaded"
    except Exception as e:
        uci_status = f"error: {e}"

    overall = "healthy" if (sgcc_status == "loaded" and uci_status == "loaded") else "degraded"

    return {
        "status": overall,
        "sgcc_model": sgcc_status,
        "forecast_model": uci_status,
        "timestamp": datetime.utcnow().isoformat()
    }
