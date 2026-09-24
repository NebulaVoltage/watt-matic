# GridBalance — Smart Grid Intelligence Platform

> **Optimizing Smart Grids through Regression-Based Load Forecasting and Meter Tampering Classification using Machine Learning.**

GridBalance is a dual-engine smart grid intelligence platform that combines **meter tampering detection** (classification) and **next-hour electricity load forecasting** (regression) into a unified, research-validated web application.

---

## Architecture Overview

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
                             (Port 8000 / Python 3.11)
                                       │
                            React + TypeScript + Vite
                             (Port 5173 / Tailwind CSS)
                                       │
                         GridBalance Control Dashboard
```

---

## Key Machine Learning Engines

### 1. SGCC Meter Tampering Classification Engine
- **Model**: Frozen 18-feature XGBoost Classifier
- **Dataset**: State Grid Corporation of China (SGCC) — 42,367 customers across 1,034 daily readings
- **Threshold**: Fixed at `0.50`
- **Sealed Test Metrics**:
  - **F1-Score**: `0.4014`
  - **PR-AUC**: `0.4049`
  - **ROC-AUC**: `0.8347`

### 2. UCI Electricity Load Forecasting Engine
- **Model**: Frozen XGBoost Regressor (Next-Hour Horizon $y_{t+1}$)
- **Dataset**: UCI Electricity Load Diagrams — 370 customer meters resampled to 35,065 aggregate hourly readings
- **Sealed Test Metrics**:
  - **MAE**: `4,421.55 kWh` (**78.75% improvement** over Daily Seasonal Naive baseline)
  - **RMSE**: `6,654.94 kWh`
  - **R²**: `0.9944`
  - **sMAPE**: `2.18%`

---

## Quick Start Guide

### 1. Start FastAPI Backend (Port 8000)
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

### 2. Start React Frontend (Port 5173)
```bash
cd frontend
npm install
npm run dev
```

Open browser at **`http://127.0.0.1:5173`**.
