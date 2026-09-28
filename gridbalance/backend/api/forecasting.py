from fastapi import APIRouter, HTTPException, status

from backend.schemas.forecasting import ForecastInput, ForecastResponse
from backend.services.forecasting_service import ForecastingService

router = APIRouter(prefix="/forecast", tags=["Electricity Load Forecasting"])

@router.post("/next-hour", response_model=ForecastResponse, summary="Predict Next-Hour Grid Electricity Load")
def forecast_next_hour(payload: ForecastInput):
    """
    Predicts next-hour (t+1) aggregate electricity load in kWh given sequential historical load history.
    Extracts 29 lag, rolling, and calendar features and runs inference using the frozen XGBoost Regressor.
    """
    history_dicts = [{"timestamp": item.timestamp, "load_kwh": item.load_kwh} for item in payload.history]

    service = ForecastingService()
    try:
        res = service.predict_next_hour(history_dicts)
        return res
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Forecasting error: {e}")
