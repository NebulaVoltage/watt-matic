import logging
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from pathlib import Path

from backend.config import UCI_MODEL_PATH, UCI_METRICS

logger = logging.getLogger("gridbalance.inference.uci")

class UCIForecaster:
    """
    Inference Engine for UCI Next-Hour Electricity Load Forecasting.
    Loads the frozen 29-feature XGBoost Regressor ONCE at initialization.
    """
    def __init__(self, model_path: Path = UCI_MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.feature_names = [
            'load_t', 'load_t_minus_1', 'load_t_minus_2', 'load_t_minus_3',
            'load_t_minus_6', 'load_t_minus_12', 'load_t_minus_24', 'load_t_minus_48',
            'load_t_minus_72', 'load_t_minus_168', 'rolling_mean_3', 'rolling_mean_6',
            'rolling_mean_12', 'rolling_mean_24', 'rolling_mean_48', 'rolling_mean_168',
            'rolling_std_24', 'rolling_std_168', 'rolling_min_24', 'rolling_max_24',
            'hour', 'day_of_week', 'day_of_month', 'month', 'week_of_year',
            'sin_hour', 'cos_hour', 'sin_day_of_week', 'cos_day_of_week'
        ]
        self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Frozen UCI forecasting model artifact not found at {self.model_path}")
        logger.info(f"Loading frozen UCI forecasting model from {self.model_path}")
        self.model = joblib.load(self.model_path)
        logger.info("UCI forecasting model loaded successfully.")

    def is_loaded(self) -> bool:
        return self.model is not None

    def preprocess_and_extract_features(self, history: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.Timestamp]:
        """
        Validates sequential load history and extracts the 29 lag, rolling, and calendar features.
        """
        if len(history) < 168:
            raise ValueError(f"Insufficient history: minimum 168 hours required, got {len(history)}")

        timestamps = []
        loads = []
        for point in history:
            ts_val = point.get('timestamp')
            load_val = point.get('load_kwh')

            if not ts_val:
                raise ValueError("Missing 'timestamp' field in load history")
            if load_val is None or pd.isna(load_val):
                raise ValueError("Missing or NaN 'load_kwh' field in load history")

            try:
                dt = pd.to_datetime(ts_val)
            except Exception as e:
                raise ValueError(f"Failed to parse timestamp '{ts_val}': {e}")

            try:
                flt_load = float(load_val)
            except (ValueError, TypeError):
                raise ValueError(f"Non-numeric load_kwh value encountered: '{load_val}'")

            timestamps.append(dt)
            loads.append(flt_load)

        df = pd.DataFrame({'timestamp': timestamps, 'aggregate_load_kwh': loads})
        df = df.groupby('timestamp', as_index=False).mean().sort_values('timestamp').reset_index(drop=True)
        df.set_index('timestamp', inplace=True)

        if len(df) < 168:
            raise ValueError(f"Insufficient unique hourly history after deduplication: got {len(df)}, need >= 168")

        s = df['aggregate_load_kwh']
        final_idx = s.index[-1]
        target_timestamp = final_idx + pd.Timedelta(hours=1)

        # Lags at time t
        lags = {
            'load_t': s.iloc[-1],
            'load_t_minus_1': s.iloc[-2],
            'load_t_minus_2': s.iloc[-3],
            'load_t_minus_3': s.iloc[-4],
            'load_t_minus_6': s.iloc[-7],
            'load_t_minus_12': s.iloc[-13],
            'load_t_minus_24': s.iloc[-25],
            'load_t_minus_48': s.iloc[-49],
            'load_t_minus_72': s.iloc[-73],
            'load_t_minus_168': s.iloc[-169] if len(s) >= 169 else s.iloc[0],
        }

        # Rolling windows up to time t
        rolling = {
            'rolling_mean_3': s.iloc[-3:].mean(),
            'rolling_mean_6': s.iloc[-6:].mean(),
            'rolling_mean_12': s.iloc[-12:].mean(),
            'rolling_mean_24': s.iloc[-24:].mean(),
            'rolling_mean_48': s.iloc[-48:].mean(),
            'rolling_mean_168': s.iloc[-168:].mean(),
            'rolling_std_24': s.iloc[-24:].std(),
            'rolling_std_168': s.iloc[-168:].std(),
            'rolling_min_24': s.iloc[-24:].min(),
            'rolling_max_24': s.iloc[-24:].max(),
        }

        # Calendar features at time t
        hour = final_idx.hour
        day_of_week = final_idx.dayofweek
        day_of_month = final_idx.day
        month = final_idx.month
        week_of_year = int(final_idx.isocalendar().week)

        calendar = {
            'hour': hour,
            'day_of_week': day_of_week,
            'day_of_month': day_of_month,
            'month': month,
            'week_of_year': week_of_year,
            'sin_hour': np.sin(2 * np.pi * hour / 24.0),
            'cos_hour': np.cos(2 * np.pi * hour / 24.0),
            'sin_day_of_week': np.sin(2 * np.pi * day_of_week / 7.0),
            'cos_day_of_week': np.cos(2 * np.pi * day_of_week / 7.0),
        }

        all_feats = {**lags, **rolling, **calendar}
        X_df = pd.DataFrame([all_feats], columns=self.feature_names)
        return X_df, target_timestamp

    def predict_next_hour(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Executes next-hour load forecast for input load series."""
        X_df, target_timestamp = self.preprocess_and_extract_features(history)

        pred_val = float(self.model.predict(X_df)[0])

        return {
            "forecast_horizon": "next_hour",
            "target_timestamp": str(target_timestamp),
            "predicted_load_kwh": round(pred_val, 2),
            "model_info": UCI_METRICS
        }
