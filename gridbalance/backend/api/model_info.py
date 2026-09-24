from fastapi import APIRouter
from backend.config import SGCC_METRICS, UCI_METRICS

router = APIRouter(prefix="/api/v1", tags=["Models"])

@router.get("/models")
def get_model_info():
    return {
        "sgcc": SGCC_METRICS,
        "forecasting": UCI_METRICS,
        "disclaimer": "These evaluation metrics reflect rigorous research evaluations on sealed benchmark test sets."
    }
