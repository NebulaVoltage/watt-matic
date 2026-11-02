from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class ModelInfoResponse(BaseModel):
    sgcc: Dict[str, Any] = Field(..., description="SGCC Meter Tampering Detection active model metadata")
    forecasting: Dict[str, Any] = Field(..., description="UCI Electricity Load Forecasting active model metadata")
    sgcc_experimental: Optional[Dict[str, Any]] = Field(None, description="SGCC LightGBM Focal Loss experimental candidate metadata")
    forecasting_experimental: Optional[Dict[str, Any]] = Field(None, description="UCI LightGBM 33-feature EMA experimental candidate metadata")
    disclaimer: str = Field(..., description="Research disclaimer and evaluation integrity statement")
