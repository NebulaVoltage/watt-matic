from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ReadingInput(BaseModel):
    date: str
    consumption: float

class MeterAnalyzeInput(BaseModel):
    meter_id: str
    readings: Optional[List[ReadingInput]] = None
    dates: Optional[List[str]] = None
    consumption: Optional[List[float]] = None

class FeatureImportanceItem(BaseModel):
    feature: str
    value: float
    importance: float
    contribution: Optional[float] = 0.0

class DataQualitySchema(BaseModel):
    observation_count: int
    missing_count: int
    missing_ratio: float
    longest_missing_streak: int
    missing_streak_count: int

class SingleMeterResponse(BaseModel):
    meter_id: str
    prediction: str  # "potential_tampering" | "normal"
    probability: float
    threshold: float = 0.50
    risk_level: str  # "Low" | "Moderate" | "High" | "Very High"
    data_quality: DataQualitySchema
    top_features: List[FeatureImportanceItem]
    consumption_history: List[Dict[str, Any]]

class BatchMeterItem(BaseModel):
    meter_id: str
    prediction: str
    probability: float
    risk_level: str
    observation_count: int
    missing_count: int
    missing_ratio: float

class BatchMeterResponse(BaseModel):
    total_meters: int
    normal_count: int
    potential_tampering_count: int
    high_risk_count: int
    results: List[BatchMeterItem]
