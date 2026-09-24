# GridBalance Platform Integration Audit Report

## 1. Executive Summary
This document outlines the architecture, existing research artifacts, data pipelines, and integration specifications for the **GridBalance Smart Grid Intelligence Web Platform**. Both research branches are complete and frozen. This phase connects the frozen ML engines to a production-grade FastAPI backend and an interactive React/TypeScript dashboard.

---

## 2. Existing Research Artifacts & Models
- **SGCC Theft Detection Engine**:
  - **Model File**: `gridbalance/models/sgcc_tuned/xgboost_reduced_best.joblib`
  - **Feature Count**: 18 non-redundant engineered features
  - **Decision Threshold**: Fixed at `0.50`
  - **Performance Metrics**: Test F1 = 0.4014 | PR-AUC = 0.4049 | ROC-AUC = 0.8347
  - **Feature Extraction Logic**: Defined in `run_shap_ablation.py` and `audit_and_fix_missingness.py` (chronological date sorting, missingness metrics, bounded `ffill(limit=7)`, NaN-aware statistics).

- **UCI Electricity Load Forecasting Engine**:
  - **Model File**: `gridbalance/models/uci_forecasting/xgb_forecasting_frozen.joblib`
  - **Feature Count**: 29 time-series features (lags, causal rolling windows, cyclic time encodings)
  - **Performance Metrics**: Test MAE = 4,421.55 kWh | RMSE = 6,654.94 kWh | R² = 0.9944 | sMAPE = 2.18%
  - **Feature Extraction Logic**: Defined in `run_uci_forecasting_pipeline.py`.

---

## 3. Platform Architecture & Data Flow

```
                              GRIDBALANCE PLATFORM
                                       │
                      ┌────────────────┴────────────────┐
                      │                                 │
           SGCC Detection Engine              UCI Forecasting Engine
             (18-feat XGBoost)                   (XGBoost Regressor)
                      │                                 │
                      └────────────────┬────────────────┘
                                       │
                              FastAPI REST Backend
                                       │
                            React + TypeScript + Vite
                                       │
                         GridBalance Control Dashboard
```

---

## 4. API Endpoints Specification
1. `GET /api/v1/health`: Application health status & loaded model verification.
2. `GET /api/v1/models`: Research evaluation metrics and model configurations.
3. `POST /api/v1/detection/analyze`: Single meter time-series analysis (returns prediction, probability, risk level, data quality, SHAP contributions).
4. `POST /api/v1/detection/batch`: CSV upload for batch meter analysis with pagination, filtering, and export.
5. `POST /api/v1/forecast/next-hour`: Historical load input -> Next-hour aggregate load prediction ($y_{t+1}$).
6. `GET /api/v1/dashboard/summary`: High-level system status and recent analysis metrics.

---

## 5. UI/UX Page Hierarchy
1. `/` — **Overview Dashboard**: High-level system metrics, model online indicators, load trends, quick launch.
2. `/detection` — **Meter Tampering Detection**: Single meter CSV/manual upload, interactive load trajectory, risk breakdown, SHAP explanations.
3. `/detection/batch` — **Batch Detection**: Multi-meter CSV processing, search/filter table, risk scoring, CSV export.
4. `/detection/:meterId` — **Meter Intelligence**: Deep-dive individual customer profile, consumption history, quality metrics, research disclaimers.
5. `/forecast` — **Load Forecasting**: Upload historical demand, forecast next-hour load, actual vs predicted comparison.
6. `/models` — **Model Intelligence**: Research evaluation metrics, SHAP feature rankings, robustness curves.
7. `/research` — **Methodology & Research**: Dataset audits, sealed test protocols, missingness strategy, literature context.
