from typing import List, Dict, Any
from pydantic import BaseModel, Field, field_validator

class LoadPointInput(BaseModel):
    timestamp: str = Field(..., description="Timestamp string (e.g. '2014-06-30 23:00:00')")
    load_kwh: float = Field(..., description="Hourly aggregate electricity load in kWh")

class ForecastInput(BaseModel):
    history: List[LoadPointInput] = Field(..., description="Sequential hourly load history (minimum 168 hours recommended)")

    @field_validator('history')
    def check_min_history(cls, v):
        if not v or len(v) < 168:
            raise ValueError(f"Insufficient historical observations. Minimum 168 hours required to calculate lag/rolling features, got {len(v) if v else 0}.")
        return v

class ForecastResponse(BaseModel):
    forecast_horizon: str = Field("next_hour", description="Forecast horizon step")
    target_timestamp: str = Field(..., description="Predicted hour timestamp (t+1)")
    predicted_load_kwh: float = Field(..., description="Predicted next-hour load in kWh")
    model_info: Dict[str, Any] = Field(..., description="Frozen model metadata & metrics")
