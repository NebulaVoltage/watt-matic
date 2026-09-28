import io
import pandas as pd
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Query, status
from fastapi.responses import StreamingResponse

from backend.schemas.sgcc import MeterAnalyzeInput, SingleMeterResponse, BatchMeterResponse
from backend.services.sgcc_service import SGCCService

router = APIRouter(prefix="/detection", tags=["Meter Tampering Detection"])

@router.post("/analyze", response_model=SingleMeterResponse, summary="Analyze Single Meter Consumption Series")
def analyze_meter(payload: MeterAnalyzeInput):
    """
    Analyzes a single electricity meter's daily consumption series to detect potential meter tampering.
    Applies chronological date sorting, bounded ffill, exact 18-feature extraction, median imputation,
    and frozen XGBoost inference at decision threshold 0.50.
    """
    dates = []
    consumption = []

    # Extract dates and consumption from either format
    if payload.readings:
        for r in payload.readings:
            dates.append(r.date)
            consumption.append(r.consumption)
    elif payload.dates is not None and payload.consumption is not None:
        dates = payload.dates
        consumption = payload.consumption
    else:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Must provide either 'readings' list or parallel 'dates' and 'consumption' lists."
        )

    if len(dates) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Consumption series cannot be empty."
        )

    if len(dates) != len(consumption):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Length mismatch between dates ({len(dates)}) and consumption values ({len(consumption)})."
        )

    service = SGCCService()
    try:
        res = service.analyze_single_meter(meter_id=payload.meter_id, dates=dates, consumption=consumption)
        return res
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inference error: {e}")


@router.post("/batch", summary="Batch Audit Multiple Meters via CSV")
def analyze_batch(file: UploadFile = File(...), export_csv: bool = Query(False, description="Set to True to download results as CSV")):
    """
    Processes a bulk CSV file containing meter IDs and daily consumption columns.
    Analyzes each meter independently and returns aggregate risk counts and individual assessments.
    Optionally exports the complete results table as a downloadable CSV.
    """
    if not file.filename.endswith(('.csv', '.txt')):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Uploaded file must be a CSV."
        )

    try:
        content = file.file.read()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to read uploaded file: {e}")

    service = SGCCService()
    try:
        batch_res = service.analyze_batch_csv(content)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Batch processing error: {e}")

    if export_csv:
        flat_rows = []
        for r in batch_res["results"]:
            flat_rows.append({
                "meter_id": r["meter_id"],
                "prediction": r["prediction"],
                "probability": r["probability"],
                "threshold": r["threshold"],
                "risk_level": r["risk_level"],
                "observation_count": r["data_quality"]["observation_count"],
                "missing_count": r["data_quality"]["missing_count"],
                "missing_ratio": r["data_quality"]["missing_ratio"],
                "longest_missing_streak": r["data_quality"]["longest_missing_streak"]
            })
        out_df = pd.DataFrame(flat_rows)
        stream = io.StringIO()
        out_df.to_csv(stream, index=False)
        return StreamingResponse(
            io.BytesIO(stream.getvalue().encode('utf-8')),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=GridBalance_Batch_Detection_Results.csv"}
        )

    return batch_res
