from fastapi import APIRouter

from backend.config import SGCC_METRICS, UCI_METRICS
from backend.schemas.models import ModelInfoResponse

router = APIRouter(tags=["Model Metadata"])

@router.get("/models", response_model=ModelInfoResponse, summary="Get Frozen Model Metadata & Research Benchmarks")
def get_model_info():
    """
    Returns frozen model metadata, feature counts, decision thresholds, and sealed test evaluation metrics
    for both the SGCC Meter Tampering Classifier and UCI Electricity Load Forecaster.
    """
    return {
        "sgcc": SGCC_METRICS,
        "forecasting": UCI_METRICS,
        "disclaimer": "All performance metrics reflect rigorous evaluations on sealed benchmark test sets. Models and thresholds are strictly frozen."
    }
