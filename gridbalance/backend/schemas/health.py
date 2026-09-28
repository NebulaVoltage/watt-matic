from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str = Field("healthy", description="Overall API health status")
    sgcc_model: str = Field(..., description="SGCC model load state ('loaded' or 'error')")
    forecast_model: str = Field(..., description="UCI forecasting model load state ('loaded' or 'error')")
    timestamp: str = Field(..., description="Current ISO timestamp")
