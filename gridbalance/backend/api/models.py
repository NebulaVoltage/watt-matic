from fastapi import APIRouter

from backend.config import (
    SGCC_METRICS,
    UCI_METRICS,
    SGCC_EXPERIMENTAL_METRICS,
    UCI_EXPERIMENTAL_METRICS
)
from backend.schemas.models import ModelInfoResponse

router = APIRouter(tags=["Model Metadata"])

@router.get("/models", response_model=ModelInfoResponse, summary="Get Frozen Model Metadata & Research Benchmarks")
def get_model_info():
    """
    Returns model metadata, feature counts, decision thresholds, and sealed test evaluation metrics
    for both the SGCC Meter Tampering Classifier and UCI Electricity Load Forecaster,
    including production baselines and experimental research candidates.
    """
    return {
        "sgcc": SGCC_METRICS,
        "forecasting": UCI_METRICS,
        "sgcc_experimental": SGCC_EXPERIMENTAL_METRICS,
        "forecasting_experimental": UCI_EXPERIMENTAL_METRICS,
        "disclaimer": "All performance metrics reflect rigorous evaluations on sealed benchmark test sets. Production models remain frozen while experimental candidates are available behind controlled feature flags."
    }
