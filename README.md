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
- **Features**: Explicit missingness indicators (`missing_streak_count`, `missing_count`), volatility (`mean_abs_daily_change`), and monthly load profile standard deviations.

### 2. UCI Electricity Load Forecasting Engine
- **Model**: Frozen XGBoost Regressor (Next-Hour Horizon $y_{t+1}$)
- **Dataset**: UCI Electricity Load Diagrams — 370 customer meters resampled to 35,065 aggregate hourly readings
- **Sealed Test Metrics**:
  - **MAE**: `4,421.55 kWh` (**78.75% improvement** over Daily Seasonal Naive baseline)
  - **RMSE**: `6,654.94 kWh`
  - **R²**: `0.9944`
  - **sMAPE**: `2.18%`
- **Features**: 29 causal lag features, causal rolling window statistics (`shift(1)`), and cyclic calendar encodings.

---

## Quick Start Guide

### Prerequisites
- Python 3.11+
- Node.js v18+ and npm

### 1. Start FastAPI Backend (Port 8000)
```bash
cd gridbalance
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

### 2. Start React Frontend (Port 5173)
```bash
cd gridbalance/frontend
npm install
npm run dev
```

Open browser at **`http://127.0.0.1:5173`**.

---

## Key Platform Features & Routes
- **Overview Dashboard (`/`)**: Real-time engine status, aggregate load trends, detection class distributions, and high-level platform metrics.
- **Single Meter Tampering Detection (`/detection`)**: Analyze single meter CSV / time series with real probability scores, risk ratings, data quality metrics, and SHAP feature explanations.
- **Batch Tampering Analysis (`/detection/batch`)**: Multi-meter CSV upload with real-time search, risk filtering, pagination, and CSV export.
- **Meter Intelligence (`/detection/:meterId`)**: Deep-dive individual customer profiles with complete load history and diagnostic disclaimers.
- **Next-Hour Load Forecasting (`/forecast`)**: Predict next-hour aggregate demand ($y_{t+1}$) with interactive actual vs predicted charts.
- **Model Intelligence (`/models`)**: Research evaluation metrics, SHAP feature rankings, and robustness sensitivity curves.
- **Research Methodology (`/research`)**: Documentation on sealed test protocols, temporal splitting, missingness strategy, and data audits.

---

## Supported Input CSV Formats

### Meter Tampering Detection Input:
```csv
date,consumption
2014-01-01,14.2
2014-01-02,12.8
2014-01-03,0.0
2014-01-04,15.1
```

### Load Forecasting Input:
```csv
timestamp,load_kwh
2014-12-24 18:00:00,128500.4
2014-12-24 19:00:00,131200.0
2014-12-24 20:00:00,129800.5
```

---

## Research & Safety Disclaimers
- **Potential Tampering Terminology**: Classification labels represent statistical model predictions (`potential_tampering`). They do NOT independently prove physical meter tampering or illegal theft.
- **Research Benchmarks**: Evaluation metrics represent scientific benchmarks on sealed datasets (`SGCC` & `UCI`) and do not represent production deployment certifications.
- **Frozen Inferences**: Models are loaded strictly from joblib artifacts. No retraining or refitting occurs during API requests.

---

## Repository Structure
```
gridbalance/
├── backend/                  # FastAPI Backend API & Services
│   ├── api/                  # REST endpoints (health, detection, forecast, models)
│   ├── services/             # SGCC & UCI Inference Services
│   ├── schemas/              # Pydantic Schemas
│   └── main.py               # FastAPI App Entrypoint
├── frontend/                 # React + TypeScript + Vite + Tailwind CSS App
│   ├── src/pages/            # Dashboard Pages & Intelligence Views
│   ├── src/components/       # Navbar & Footer Components
│   └── src/services/api.ts   # REST API Client
├── models/                   # Frozen Model Joblib Artifacts
│   ├── sgcc_tuned/           # Frozen SGCC 18-Feature XGBoost Model
│   └── uci_forecasting/      # Frozen UCI XGBoost Regressor Model
├── reports/                  # Detailed Research Reports & Audit CSVs
├── docs/                     # Platform Architecture, API Reference & User Guide
└── tests/                    # Unit Tests
```
