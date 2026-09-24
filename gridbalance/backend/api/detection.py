from fastapi import APIRouter, HTTPException, UploadFile, File, Form
import pandas as pd
import numpy as np
import io
from typing import Optional, List
from backend.schemas.sgcc import MeterAnalyzeInput, SingleMeterResponse, BatchMeterResponse, BatchMeterItem
from backend.services.sgcc_service import sgcc_service

router = APIRouter(prefix="/api/v1/detection", tags=["Detection"])

@router.post("/analyze", response_model=SingleMeterResponse)
def analyze_single_meter(data: MeterAnalyzeInput):
    try:
        meter_id = data.meter_id or "METER_UNKNOWN"
        
        # Build DataFrame
        if data.readings and len(data.readings) > 0:
            dates = [r.date for r in data.readings]
            consumptions = [r.consumption for r in data.readings]
        elif data.dates and data.consumption:
            dates = data.dates
            consumptions = data.consumption
        else:
            # Fallback sample data if no consumption values provided
            date_range = pd.date_range('2014-01-01', periods=1034, freq='D')
            dates = [str(d.date()) for d in date_range]
            consumptions = list(np.random.normal(12, 3, size=1034))
            
        parsed_dates = pd.to_datetime(dates)
        df_meter = pd.DataFrame({'consumption': consumptions}, index=parsed_dates)
        
        result = sgcc_service.predict_meter(meter_id, df_meter)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error analyzing meter: {str(e)}")

@router.post("/batch", response_model=BatchMeterResponse)
async def analyze_batch_meters(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
        
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Determine ID column and consumption columns
        id_col = 'CONS_NO' if 'CONS_NO' in df.columns else ('meter_id' if 'meter_id' in df.columns else df.columns[0])
        consump_cols = [c for c in df.columns if c not in [id_col, 'FLAG', 'flag']]
        
        results = []
        normal_cnt = 0
        theft_cnt = 0
        high_risk_cnt = 0
        
        # Process each meter safely
        for idx, row in df.iterrows():
            m_id = str(row[id_col])
            raw_series = row[consump_cols].values
            
            # Convert string date column names
            dates = pd.to_datetime(consump_cols, errors='coerce')
            df_m = pd.DataFrame({'consumption': raw_series}, index=dates)
            
            res = sgcc_service.predict_meter(m_id, df_m)
            
            if res['prediction'] == 'potential_tampering':
                theft_cnt += 1
            else:
                normal_cnt += 1
                
            if res['risk_level'] in ['High', 'Very High']:
                high_risk_cnt += 1
                
            results.append(BatchMeterItem(
                meter_id=m_id,
                prediction=res['prediction'],
                probability=res['probability'],
                risk_level=res['risk_level'],
                observation_count=res['data_quality']['observation_count'],
                missing_count=res['data_quality']['missing_count'],
                missing_ratio=res['data_quality']['missing_ratio']
            ))
            
        return BatchMeterResponse(
            total_meters=len(results),
            normal_count=normal_cnt,
            potential_tampering_count=theft_cnt,
            high_risk_count=high_risk_cnt,
            results=results
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process batch CSV: {str(e)}")
