from typing import Dict, Any
from pydantic import BaseModel, Field

class ModelInfoResponse(BaseModel):
    sgcc: Dict[str, Any] = Field(..., description="SGCC Meter Tampering Detection model metadata")
    forecasting: Dict[str, Any] = Field(..., description="UCI Electricity Load Forecasting model metadata")
    disclaimer: str = Field(..., description="Research disclaimer and evaluation integrity statement")
