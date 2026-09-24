from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class LoadPointInput(BaseModel):
    timestamp: str
    load_kwh: float

class ForecastInput(BaseModel):
    history: Optional[List[LoadPointInput]] = None
    timestamps: Optional[List[str]] = None
    load_kwh: Optional[List[float]] = None

class FeatureContributionItem(BaseModel):
    feature: str
    importance: float

class ForecastResponse(BaseModel):
    forecast_horizon: str = "next_hour"
    predicted_load_kwh: float
    current_load_kwh: float
    change_from_current_kwh: float
    change_percentage: float
    model: str = "XGBoost Regressor"
    historical_chart: List[Dict[str, Any]]
    top_features: List[FeatureContributionItem]
