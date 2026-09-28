from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field, field_validator

class ReadingInput(BaseModel):
    date: str = Field(..., description="Date string (e.g., '2014-01-01' or '2014/1/1')")
    consumption: float = Field(..., description="Daily electricity consumption in kWh")

class MeterAnalyzeInput(BaseModel):
    meter_id: str = Field(..., description="Unique customer or meter identifier")
    readings: Optional[List[ReadingInput]] = Field(None, description="List of date-consumption reading pairs")
    dates: Optional[List[str]] = Field(None, description="Parallel list of dates")
    consumption: Optional[List[float]] = Field(None, description="Parallel list of consumption values")

    @field_validator('meter_id')
    def validate_meter_id(cls, v):
        if not v or not str(v).strip():
            raise ValueError("meter_id cannot be empty")
        return str(v).strip()

class TopFeatureSchema(BaseModel):
    feature: str = Field(..., description="Feature name")
    value: float = Field(..., description="Computed feature value")
    importance: float = Field(..., description="Feature importance weight")
    contribution: float = Field(0.0, description="Estimated contribution to risk score")

class DataQualitySchema(BaseModel):
    observation_count: int = Field(..., description="Total days evaluated")
    missing_count: int = Field(..., description="Number of missing observations")
    missing_ratio: float = Field(..., description="Ratio of missing observations")
    longest_missing_streak: int = Field(..., description="Longest consecutive missing gap in days")
    missing_streak_count: int = Field(..., description="Number of distinct missing gaps")

class ConsumptionReadingSchema(BaseModel):
    date: str
    consumption: float

class SingleMeterResponse(BaseModel):
    meter_id: str
    prediction: str = Field(..., description="'potential_tampering' or 'normal'")
    probability: float = Field(..., description="Tampering probability score [0.0 - 1.0]")
    threshold: float = Field(0.50, description="Decision threshold used for classification")
    risk_level: str = Field(..., description="'High', 'Medium', or 'Low'")
    data_quality: DataQualitySchema
    top_features: List[TopFeatureSchema]
    consumption_history: List[ConsumptionReadingSchema]

class BatchMeterResponse(BaseModel):
    total_meters: int
    flagged_meters: int
    normal_meters: int
    results: List[SingleMeterResponse]
