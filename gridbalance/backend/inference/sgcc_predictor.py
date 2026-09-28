import logging
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from pathlib import Path

from backend.config import SGCC_MODEL_PATH

logger = logging.getLogger("gridbalance.inference.sgcc")

class SGCCPredictor:
    """
    Inference Engine for SGCC Electricity Theft / Tampering Classification.
    Loads the frozen 18-feature XGBoost Pipeline ONCE at initialization.
    """
    def __init__(self, model_path: Path = SGCC_MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.feature_names = [
            'missing_streak_count', 'monthly_std_4', 'monthly_std_3', 
            'mean_abs_daily_change', 'missing_count', 'monthly_mean_10', 
            'longest_missing_streak', 'peak_to_average_ratio', 'monthly_std_5', 
            'monthly_std_10', 'monthly_std_11', 'cv', 'monthly_mean_8', 
            'max', 'monthly_std_8', 'monthly_mean_11', 'monthly_std_1', 'monthly_mean_2'
        ]
        self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Frozen SGCC model artifact not found at {self.model_path}")
        logger.info(f"Loading frozen SGCC model from {self.model_path}")
        self.model = joblib.load(self.model_path)
        logger.info("SGCC model loaded successfully.")

    def is_loaded(self) -> bool:
        return self.model is not None

    def _compute_gap_metrics(self, bool_mask: np.ndarray) -> Tuple[int, int]:
        """Calculates longest missing streak and count of streaks."""
        padded = np.pad(bool_mask.astype(int), (1, 1), mode='constant', constant_values=0)
        edges = np.diff(padded)
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0]
        streaks = ends - starts
        if len(streaks) == 0:
            return 0, 0
        return int(np.max(streaks)), int(len(streaks))

    def preprocess_and_extract_features(self, dates: List[str], consumption: List[float]) -> Tuple[pd.DataFrame, Dict[str, Any], List[Dict[str, float]]]:
        """
        Parses dates chronologically, handles duplicates/missing values, applies bounded ffill,
        and computes the exact 18 required features for SGCC inference.
        """
        if len(dates) != len(consumption):
            raise ValueError(f"Length mismatch: dates ({len(dates)}) vs consumption ({len(consumption)})")

        if len(dates) == 0:
            raise ValueError("Consumption series cannot be empty")

        # 1. Parse and sort chronologically
        try:
            parsed_dates = pd.to_datetime(dates)
        except Exception as e:
            raise ValueError(f"Failed to parse dates into datetime format: {e}")

        # Clean numeric consumption
        numeric_cons = []
        for v in consumption:
            if v is None or pd.isna(v):
                numeric_cons.append(np.nan)
            else:
                try:
                    numeric_cons.append(float(v))
                except (ValueError, TypeError):
                    raise ValueError(f"Non-numeric consumption value encountered: {v}")

        s_df = pd.DataFrame({'date': parsed_dates, 'val': numeric_cons})
        # Handle duplicate dates by taking mean
        s_df = s_df.groupby('date', as_index=False).mean().sort_values('date').reset_index(drop=True)

        raw_vals = s_df['val'].values
        obs_count = len(raw_vals)

        # 2. Audit missingness on raw data
        missing_mask = np.isnan(raw_vals)
        missing_count = int(np.sum(missing_mask))
        missing_ratio = float(missing_count / obs_count) if obs_count > 0 else 0.0
        longest_streak, streak_count = self._compute_gap_metrics(missing_mask)

        data_quality = {
            "observation_count": obs_count,
            "missing_count": missing_count,
            "missing_ratio": round(missing_ratio, 4),
            "longest_missing_streak": longest_streak,
            "missing_streak_count": streak_count
        }

        # 3. Bounded Imputation (ffill limit=7)
        s_filled = pd.Series(raw_vals).ffill(limit=7).values

        # Clean history for response
        history = [
            {"date": str(d.date()) if hasattr(d, 'date') else str(d), "consumption": float(v) if not np.isnan(v) else 0.0}
            for d, v in zip(s_df['date'], s_df['val'])
        ]

        # 4. Extract required domain features
        mean_val = float(np.nanmean(s_filled)) if not np.all(np.isnan(s_filled)) else 0.0
        max_val = float(np.nanmax(s_filled)) if not np.all(np.isnan(s_filled)) else 0.0
        std_val = float(np.nanstd(s_filled)) if not np.all(np.isnan(s_filled)) else 0.0
        cv_val = float(std_val / (mean_val + 1e-6))
        peak_to_avg = float(max_val / (mean_val + 1e-6))

        # Daily changes
        diffs = np.diff(s_filled)
        abs_diffs = np.abs(diffs)
        mean_abs_change = float(np.nanmean(abs_diffs)) if len(abs_diffs) > 0 and not np.all(np.isnan(abs_diffs)) else 0.0

        # Monthly aggregation
        s_df['filled'] = s_filled
        s_df['month'] = s_df['date'].dt.month
        monthly_means = s_df.groupby('month')['filled'].mean()
        monthly_stds = s_df.groupby('month')['filled'].std()

        feat_dict = {
            'missing_streak_count': float(streak_count),
            'monthly_std_4': float(monthly_stds.get(4, np.nan)),
            'monthly_std_3': float(monthly_stds.get(3, np.nan)),
            'mean_abs_daily_change': mean_abs_change,
            'missing_count': float(missing_count),
            'monthly_mean_10': float(monthly_means.get(10, np.nan)),
            'longest_missing_streak': float(longest_streak),
            'peak_to_average_ratio': peak_to_avg,
            'monthly_std_5': float(monthly_stds.get(5, np.nan)),
            'monthly_std_10': float(monthly_stds.get(10, np.nan)),
            'monthly_std_11': float(monthly_stds.get(11, np.nan)),
            'cv': cv_val,
            'monthly_mean_8': float(monthly_means.get(8, np.nan)),
            'max': max_val,
            'monthly_std_8': float(monthly_stds.get(8, np.nan)),
            'monthly_mean_11': float(monthly_means.get(11, np.nan)),
            'monthly_std_1': float(monthly_stds.get(1, np.nan)),
            'monthly_mean_2': float(monthly_means.get(2, np.nan)),
        }

        # Build single-row DataFrame in exact feature_names order
        X_df = pd.DataFrame([feat_dict], columns=self.feature_names)
        return X_df, data_quality, history

    def predict(self, dates: List[str], consumption: List[float], meter_id: str) -> Dict[str, Any]:
        """Executes full SGCC inference pipeline for a single meter."""
        X_df, data_quality, history = self.preprocess_and_extract_features(dates, consumption)

        # Run pipeline (SimpleImputer + XGBClassifier)
        probs = self.model.predict_proba(X_df)[0]
        prob_tampering = float(probs[1])

        threshold = 0.50
        # Requirement: Use "potential_tampering", NEVER "confirmed_theft"
        prediction = "potential_tampering" if prob_tampering >= threshold else "normal"

        if prob_tampering >= 0.70:
            risk_level = "High"
        elif prob_tampering >= 0.50:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        # Feature importance / top features
        xgb_step = self.model.named_steps.get('xgb', self.model)
        importances = getattr(xgb_step, 'feature_importances_', np.zeros(len(self.feature_names)))

        top_features = []
        for feat, val, imp in zip(self.feature_names, X_df.iloc[0].values, importances):
            val_clean = float(val) if not np.isnan(val) else 0.0
            top_features.append({
                "feature": feat,
                "value": round(val_clean, 4),
                "importance": round(float(imp), 4),
                "contribution": round(float(imp * val_clean), 4)
            })

        # Sort top features by importance descending
        top_features.sort(key=lambda x: x["importance"], reverse=True)

        return {
            "meter_id": meter_id,
            "prediction": prediction,
            "probability": round(prob_tampering, 4),
            "threshold": threshold,
            "risk_level": risk_level,
            "data_quality": data_quality,
            "top_features": top_features[:5], # Return top 5 influential features
            "consumption_history": history
        }
