from fastapi import APIRouter
from datetime import datetime
from backend.config import SGCC_MODEL_PATH, UCI_MODEL_PATH
import os

router = APIRouter(prefix="/api/v1", tags=["Health"])

@router.get("/health")
def health_check():
    sgcc_loaded = os.path.exists(SGCC_MODEL_PATH)
    uci_loaded = os.path.exists(UCI_MODEL_PATH)
    
    return {
        "status": "healthy" if sgcc_loaded and uci_loaded else "degraded",
        "sgcc_model": "loaded" if sgcc_loaded else "missing",
        "forecast_model": "loaded" if uci_loaded else "missing",
        "timestamp": datetime.utcnow().isoformat()
    }
