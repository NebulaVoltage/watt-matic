from fastapi import APIRouter, HTTPException, UploadFile, File
import pandas as pd
import numpy as np
import io
from backend.schemas.forecasting import ForecastInput, ForecastResponse
from backend.services.forecasting_service import forecasting_service

router = APIRouter(prefix="/api/v1/forecast", tags=["Forecasting"])

@router.post("/next-hour", response_model=ForecastResponse)
def forecast_next_hour(data: ForecastInput):
    try:
        if data.history and len(data.history) > 0:
            timestamps = [pd.to_datetime(h.timestamp) for h in data.history]
            loads = [h.load_kwh for h in data.history]
        elif data.timestamps and data.load_kwh:
            timestamps = [pd.to_datetime(t) for t in data.timestamps]
            loads = data.load_kwh
        else:
            # Sample fallback 168 hours (1 week) of realistic load if no data sent
            date_range = pd.date_range('2014-12-24 00:00:00', periods=168, freq='1h')
            timestamps = list(date_range)
            base_curve = 120000.0 + 30000.0 * np.sin(2 * np.pi * np.arange(168) / 24.0)
            loads = list(base_curve + np.random.normal(0, 2000, 168))
            
        df_load = pd.DataFrame({'load_kwh': loads}, index=timestamps)
        result = forecasting_service.predict_next_hour(df_load)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error running forecast: {str(e)}")

@router.post("/upload-csv", response_model=ForecastResponse)
async def forecast_from_csv(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files supported.")
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        time_col = 'timestamp' if 'timestamp' in df.columns else df.columns[0]
        load_col = 'aggregate_load_kwh' if 'aggregate_load_kwh' in df.columns else ('load_kwh' if 'load_kwh' in df.columns else df.columns[1])
        
        timestamps = pd.to_datetime(df[time_col])
        loads = df[load_col].astype(float).values
        
        df_load = pd.DataFrame({'load_kwh': loads}, index=timestamps)
        result = forecasting_service.predict_next_hour(df_load)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing forecast CSV: {str(e)}")
