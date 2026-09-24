import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from backend.config import SGCC_MODEL_PATH, SGCC_THRESHOLD

class SGCCService:
    def __init__(self):
        self.model_pipeline = None
        self.feature_names_18 = [
            'missing_streak_count', 'monthly_std_4', 'monthly_std_3', 'mean_abs_daily_change', 
            'missing_count', 'monthly_mean_10', 'longest_missing_streak', 'peak_to_average_ratio', 
            'monthly_std_5', 'monthly_std_10', 'monthly_std_11', 'cv', 'monthly_mean_8', 
            'max', 'monthly_std_8', 'monthly_mean_11', 'monthly_std_1', 'monthly_mean_2'
        ]
        self._load_model()
        
    def _load_model(self):
        if os.path.exists(SGCC_MODEL_PATH):
            print(f"[SGCC Service] Loading frozen model from {SGCC_MODEL_PATH}...")
            self.model_pipeline = joblib.load(SGCC_MODEL_PATH)
            print("[SGCC Service] Model loaded successfully.")
        else:
            print(f"[SGCC Service] WARNING: Model file not found at {SGCC_MODEL_PATH}")

    def _analyze_gaps(self, bool_arr: np.ndarray):
        padded = np.pad(bool_arr, (1, 1), mode='constant', constant_values=False)
        edges = np.diff(padded.astype(int))
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0]
        streaks = ends - starts
        if len(streaks) == 0:
            return 0, 0
        return int(np.max(streaks)), len(streaks)

    def extract_features(self, df_meter: pd.DataFrame):
        # df_meter has index=dates (datetime), column 'consumption' (float)
        df_meter = df_meter.sort_index()
        raw_vals = df_meter['consumption'].values
        missing_mask = np.isnan(raw_vals)
        
        missing_count = int(missing_mask.sum())
        total_obs = len(raw_vals)
        missing_ratio = float(missing_count / total_obs) if total_obs > 0 else 0.0
        
        max_gap, streak_cnt = self._analyze_gaps(missing_mask)
        
        # Bounded fill (limit=7)
        s_filled = df_meter['consumption'].ffill(limit=7)
        vals = s_filled.values
        
        # Statistical features
        mean_val = float(np.nanmean(vals)) if not np.isnan(vals).all() else 0.0
        std_val = float(np.nanstd(vals)) if not np.isnan(vals).all() else 0.0
        cv_val = float(std_val / (mean_val + 1e-6))
        max_val = float(np.nanmax(vals)) if not np.isnan(vals).all() else 0.0
        peak_to_avg = float(max_val / (mean_val + 1e-6))
        
        diffs = np.diff(vals)
        abs_diffs = np.abs(diffs)
        mean_abs_daily_change = float(np.nanmean(abs_diffs)) if len(abs_diffs) > 0 and not np.isnan(abs_diffs).all() else 0.0
        
        # Monthly periodic features
        df_temp = pd.DataFrame({'val': vals, 'month': df_meter.index.month})
        monthly_means = df_temp.groupby('month')['val'].mean().to_dict()
        monthly_stds = df_temp.groupby('month')['val'].std().to_dict()
        
        feat_dict = {
            'missing_streak_count': float(streak_cnt),
            'monthly_std_4': float(monthly_stds.get(4, std_val) if not np.isnan(monthly_stds.get(4, np.nan)) else std_val),
            'monthly_std_3': float(monthly_stds.get(3, std_val) if not np.isnan(monthly_stds.get(3, np.nan)) else std_val),
            'mean_abs_daily_change': mean_abs_daily_change,
            'missing_count': float(missing_count),
            'monthly_mean_10': float(monthly_means.get(10, mean_val) if not np.isnan(monthly_means.get(10, np.nan)) else mean_val),
            'longest_missing_streak': float(max_gap),
            'peak_to_average_ratio': peak_to_avg,
            'monthly_std_5': float(monthly_stds.get(5, std_val) if not np.isnan(monthly_stds.get(5, np.nan)) else std_val),
            'monthly_std_10': float(monthly_stds.get(10, std_val) if not np.isnan(monthly_stds.get(10, np.nan)) else std_val),
            'monthly_std_11': float(monthly_stds.get(11, std_val) if not np.isnan(monthly_stds.get(11, np.nan)) else std_val),
            'cv': cv_val,
            'monthly_mean_8': float(monthly_means.get(8, mean_val) if not np.isnan(monthly_means.get(8, np.nan)) else mean_val),
            'max': max_val,
            'monthly_std_8': float(monthly_stds.get(8, std_val) if not np.isnan(monthly_stds.get(8, np.nan)) else std_val),
            'monthly_mean_11': float(monthly_means.get(11, mean_val) if not np.isnan(monthly_means.get(11, np.nan)) else mean_val),
            'monthly_std_1': float(monthly_stds.get(1, std_val) if not np.isnan(monthly_stds.get(1, np.nan)) else std_val),
            'monthly_mean_2': float(monthly_means.get(2, mean_val) if not np.isnan(monthly_means.get(2, np.nan)) else mean_val)
        }
        
        quality_info = {
            "observation_count": total_obs,
            "missing_count": missing_count,
            "missing_ratio": round(missing_ratio, 4),
            "longest_missing_streak": max_gap,
            "missing_streak_count": streak_cnt
        }
        
        return feat_dict, quality_info

    def predict_meter(self, meter_id: str, df_meter: pd.DataFrame) -> Dict[str, Any]:
        feat_dict, quality_info = self.extract_features(df_meter)
        
        # Build 1-row DataFrame in exact 18 feature order
        df_feat = pd.DataFrame([feat_dict], columns=self.feature_names_18)
        
        prob = 0.0
        if self.model_pipeline is not None:
            probs = self.model_pipeline.predict_proba(df_feat)
            prob = float(probs[0, 1])
        else:
            # Fallback for dev if model not loaded
            prob = 0.15
            
        prediction = "potential_tampering" if prob >= SGCC_THRESHOLD else "normal"
        
        if prob < 0.35:
            risk_level = "Low"
        elif prob < 0.50:
            risk_level = "Moderate"
        elif prob < 0.75:
            risk_level = "High"
        else:
            risk_level = "Very High"
            
        # Top 5 feature contributions (based on global feature importance weights)
        importances_map = {
            'missing_streak_count': 0.5945,
            'monthly_std_4': 0.3279,
            'monthly_std_3': 0.2901,
            'mean_abs_daily_change': 0.2597,
            'missing_count': 0.2378,
            'monthly_mean_10': 0.2330,
            'longest_missing_streak': 0.2256,
            'peak_to_average_ratio': 0.2128,
            'monthly_std_5': 0.2074,
            'monthly_std_10': 0.1929,
            'monthly_std_11': 0.1678,
            'cv': 0.1501,
            'monthly_mean_8': 0.1487,
            'max': 0.1397,
            'monthly_std_8': 0.1350,
            'monthly_mean_11': 0.1200,
            'monthly_std_1': 0.1150,
            'monthly_mean_2': 0.1100
        }
        
        top_feats = []
        for k in self.feature_names_18[:5]:
            top_feats.append({
                "feature": k,
                "value": round(feat_dict[k], 4),
                "importance": importances_map.get(k, 0.1)
            })
            
        # Consumption history for chart
        history = []
        for idx, row in df_meter.reset_index().iterrows():
            d_str = str(row['index'].date()) if hasattr(row['index'], 'date') else str(row['index'])
            c_val = float(row['consumption']) if not np.isnan(row['consumption']) else 0.0
            history.append({"date": d_str, "consumption": c_val})
            
        return {
            "meter_id": meter_id,
            "prediction": prediction,
            "probability": round(prob, 4),
            "threshold": SGCC_THRESHOLD,
            "risk_level": risk_level,
            "data_quality": quality_info,
            "top_features": top_feats,
            "consumption_history": history
        }

sgcc_service = SGCCService()
