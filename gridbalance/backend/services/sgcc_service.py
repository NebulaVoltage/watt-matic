import io
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging

from backend.inference.sgcc_predictor import SGCCPredictor

logger = logging.getLogger("gridbalance.service.sgcc")

_sgcc_predictor: Optional[SGCCPredictor] = None

def get_sgcc_predictor() -> SGCCPredictor:
    global _sgcc_predictor
    if _sgcc_predictor is None:
        _sgcc_predictor = SGCCPredictor()
    return _sgcc_predictor

class SGCCService:
    def __init__(self):
        self.predictor = get_sgcc_predictor()

    def analyze_single_meter(self, meter_id: str, dates: List[str], consumption: List[float]) -> Dict[str, Any]:
        return self.predictor.predict(dates=dates, consumption=consumption, meter_id=meter_id)

    def analyze_batch_csv(self, csv_content: bytes) -> Dict[str, Any]:
        """
        Parses CSV input containing CONS_NO / meter_id and date consumption columns,
        executes inference per meter, and returns batch aggregate analysis.
        """
        try:
            df = pd.read_csv(io.BytesIO(csv_content))
        except Exception as e:
            raise ValueError(f"Invalid CSV file format: {e}")

        if df.empty:
            raise ValueError("Uploaded CSV file is empty.")

        # Identify ID column
        id_col = None
        for candidate in ['meter_id', 'CONS_NO', 'id', 'Customer_ID']:
            if candidate in df.columns:
                id_col = candidate
                break

        if not id_col:
            raise ValueError(f"Missing required meter identifier column. Expected one of ['meter_id', 'CONS_NO'], found: {list(df.columns[:5])}")

        # Date columns are all other columns except id_col and FLAG/target if present
        ignore_cols = {id_col, 'FLAG', 'flag', 'target', 'label'}
        date_cols = [c for c in df.columns if c not in ignore_cols]

        if not date_cols:
            raise ValueError("CSV file does not contain any date consumption columns.")

        results = []
        flagged_count = 0
        normal_count = 0

        for idx, row in df.iterrows():
            m_id = str(row[id_col])
            raw_vals = [row[col] for col in date_cols]

            try:
                res = self.analyze_single_meter(meter_id=m_id, dates=date_cols, consumption=raw_vals)
                results.append(res)
                if res['prediction'] == 'potential_tampering':
                    flagged_count += 1
                else:
                    normal_count += 1
            except Exception as e:
                logger.warning(f"Error processing meter {m_id}: {e}")
                # Provide fallback entry for unprocessable row
                results.append({
                    "meter_id": m_id,
                    "prediction": "normal",
                    "probability": 0.0,
                    "threshold": 0.50,
                    "risk_level": "Low",
                    "data_quality": {
                        "observation_count": len(date_cols),
                        "missing_count": len(date_cols),
                        "missing_ratio": 1.0,
                        "longest_missing_streak": len(date_cols),
                        "missing_streak_count": 1
                    },
                    "top_features": [],
                    "consumption_history": []
                })
                normal_count += 1

        return {
            "total_meters": len(df),
            "flagged_meters": flagged_count,
            "normal_meters": normal_count,
            "results": results
        }
