# GridBalance REST API Reference

Base URL: `http://127.0.0.1:8000/api/v1`

---

### 1. Health Endpoint
- **URL**: `GET /api/v1/health`
- **Response**:
```json
{
  "status": "healthy",
  "sgcc_model": "loaded",
  "forecast_model": "loaded",
  "timestamp": "2026-09-24T08:25:23.102580"
}
```

---

### 2. Model Metadata Endpoint
- **URL**: `GET /api/v1/models`
- **Response**:
```json
{
  "sgcc": {
    "model": "XGBoost Classifier",
    "features": 18,
    "threshold": 0.50,
    "test_f1": 0.4014,
    "test_pr_auc": 0.4049,
    "test_roc_auc": 0.8347
  },
  "forecasting": {
    "model": "XGBoost Regressor",
    "features": 29,
    "forecast_horizon": "next_hour",
    "test_mae": 4421.55,
    "test_rmse": 6654.94,
    "test_r2": 0.9944,
    "test_smape": 2.18
  }
}
```

---

### 3. Single Meter Tampering Analysis
- **URL**: `POST /api/v1/detection/analyze`
- **Content-Type**: `application/json`
- **Request Body**:
```json
{
  "meter_id": "CONS_42001",
  "readings": [
    {"date": "2014-01-01", "consumption": 14.2},
    {"date": "2014-01-02", "consumption": 12.8}
  ]
}
```
- **Response**:
```json
{
  "meter_id": "CONS_42001",
  "prediction": "normal",
  "probability": 0.0584,
  "threshold": 0.50,
  "risk_level": "Low",
  "data_quality": {
    "observation_count": 1034,
    "missing_count": 0,
    "missing_ratio": 0.0,
    "longest_missing_streak": 0,
    "missing_streak_count": 0
  },
  "top_features": [...],
  "consumption_history": [...]
}
```

---

### 4. Batch Meter Tampering Analysis
- **URL**: `POST /api/v1/detection/batch`
- **Content-Type**: `multipart/form-data` (`file`: CSV file containing multiple meter rows)
- **Response**:
```json
{
  "total_meters": 10,
  "normal_count": 9,
  "potential_tampering_count": 1,
  "high_risk_count": 1,
  "results": [...]
}
```

---

### 5. Next-Hour Load Forecasting
- **URL**: `POST /api/v1/forecast/next-hour`
- **Content-Type**: `application/json`
- **Response**:
```json
{
  "forecast_horizon": "next_hour",
  "predicted_load_kwh": 94349.49,
  "current_load_kwh": 92100.00,
  "change_from_current_kwh": 2249.49,
  "change_percentage": 2.44,
  "model": "XGBoost Regressor",
  "historical_chart": [...],
  "top_features": [...]
}
```
