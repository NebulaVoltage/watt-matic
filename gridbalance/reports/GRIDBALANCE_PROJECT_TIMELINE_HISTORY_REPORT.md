# GridBalance: Project Timeline, Chronological Evolution & Complete Repository Directory Audit

> **Project Title**: GridBalance: Optimizing Smart Grids through Regression-Based Load Forecasting and Meter Tampering Classification  
> **Authors**: GridBalance AI Research & Software Engineering Team  
> **Date**: October 2026  
> **Status**: Master Historical Record & Complete Repository Architecture Audit  

---

## 1. Executive Summary & Project Overview

**GridBalance** is an enterprise-grade Smart Grid Machine Learning platform engineered to optimize power distribution networks by addressing two core electrical grid challenges:

1. **Electricity Theft & Meter Tampering Detection (Non-Technical Loss - NTL)**: Identifying fraudulent meters from daily consumption patterns across $42,367$ smart meters from the State Grid Corporation of China (SGCC).
2. **Next-Hour Aggregate Load Forecasting (Short-Term Load Forecasting - STLF)**: Predicting 1-hour-ahead grid electricity demand across 370 industrial and commercial customers from the UCI Electricity Load Diagrams dataset ($35,065$ aggregate hourly readings).

This document serves as the **master chronological log and repository inventory**, detailing every development milestone, dataset transformation, machine learning experiment, directory creation, architectural refactoring, and production deployment step completed to date.

---

## 2. Chronological Project Timeline & Development Milestones

```
+-----------------------------------------------------------------------------------------------------------------------+
|                                        GRIDBALANCE MASTER DEVELOPMENT TIMELINE                                        |
+----------------------+----------------------------------------+-------------------------------------------------------+
| Milestone Phase      | Primary Objectives                     | Key Accomplishments & Deliverables                    |
+----------------------+----------------------------------------+-------------------------------------------------------+
| Phase 1: Data        | Ingestion of SGCC & UCI datasets,      | EDA Report generated; missingness patterns analyzed;  |
| Ingestion & EDA      | initial quality checks & distributions. | raw data ingested (data.csv, LD2011_2014.txt).        |
+----------------------+----------------------------------------+-------------------------------------------------------+
| Phase 2: SGCC Theft  | Feature extraction, baseline modeling, | 18-feature XGBoost model trained; class imbalance     |
| Research Pipeline    | SHAP importance, feature reduction.    | addressed (scale_pos_weight=10.72); F1 = 0.4037.      |
+----------------------+----------------------------------------+-------------------------------------------------------+
| Phase 3: UCI Load    | Aggregation of 370 meter channels,     | 29 lag, rolling, & temporal features engineered;      |
| Forecasting Pipeline | lag features, time-series CV, tuning.  | XGBoost Regressor frozen (MAE = 4,503.51 kWh, R²=0.99).|
+----------------------+----------------------------------------+-------------------------------------------------------+
| Phase 4: Robustness  | Stress testing under noise, missing    | Robustness audit complete; evasion simulation shows   |
| & Evasion Audits     | data, class separability, & evasion.   | model resilience up to 20% Gaussian noise.            |
+----------------------+----------------------------------------+-------------------------------------------------------+
| Phase 5: FastAPI     | Build production REST API backend with | FastAPI service implemented; single startup model     |
| Inference Backend    | single startup model loading.          | loading; passed 13/13 automated test suite.           |
+----------------------+----------------------------------------+-------------------------------------------------------+
| Phase 6: React Web   | Build interactive ML analytics web app | React + TypeScript platform built; Recharts telemetry;|
| Application          | consuming real FastAPI backend.        | single & batch analysis views, forecasting UI.        |
+----------------------+----------------------------------------+-------------------------------------------------------+
| Phase 7: UI Styling  | Diagnose frontend CSS rendering issue; | Integrated @tailwindcss/vite; resolved layout padding,|
| Repair & Fixes       | apply consistent design system.        | card hierarchy, and typography alignment.             |
+----------------------+----------------------------------------+-------------------------------------------------------+
| Phase 8: Technical   | Audit repository, generate master CSVs,| Compiled master research dossiers, CSV summaries,     |
| Research Dossier     | claims audit, & initial PDF reports.   | and verified zero-leakage compliance.                 |
+----------------------+----------------------------------------+-------------------------------------------------------+
| Phase 9: V2 Model    | Controlled improvement workspace;      | Evaluated LightGBM, Optuna tuning, threshold tuning,  |
| Optimization         | test LightGBM & temporal extensions.   | & EMA features; baseline XGBoost models retained.     |
+----------------------+----------------------------------------+-------------------------------------------------------+
```

---

## 3. Detailed Step-by-Step Chronological Progression

### Step 1: Exploratory Data Analysis & Quality Audit
- Ingested raw **SGCC dataset** (`data.csv`, $27.67$ MB) containing daily readings for $42,372$ smart meters across 1,035 consecutive days.
- Ingested raw **UCI dataset** (`data/uci/LD2011_2014.txt`, $710.99$ MB) containing 15-minute electricity consumption for 370 commercial customers (2011–2014).
- Identified severe class imbalance in SGCC ($90.85\%$ benign, $9.15\%$ fraud) and zero/missing value streaks resulting from meter communication dropouts.

### Step 2: SGCC Theft Classification Feature Engineering & Modeling
- Processed 1,035 daily readings into 18 concise statistical features (missing streak counts, daily change rates, monthly mean/std consumption).
- Evaluated candidate models: Logistic Regression (F1=0.0753), Decision Tree (F1=0.2604), Random Forest (F1=0.1265), and **XGBoost Classifier** (Validation F1=0.4037).
- Applied SHAP (SHapley Additive exPlanations) to verify feature attribution, ranking `missing_streak_count` and `monthly_std_4` as primary predictors.
- Frozen Model Saved: `models/sgcc_tuned/xgboost_reduced_best.joblib`.

### Step 3: UCI Aggregate Load Forecasting Pipeline
- Resampled 15-minute customer readings into system aggregate hourly load readings ($35,065$ observations).
- Constructed 29 features: Lags ($t-1	ext{h}, t-2	ext{h}, t-3	ext{h}, t-24	ext{h}, t-168	ext{h}$), Rolling Statistics ($3	ext{h}, 6	ext{h}, 24	ext{h}, 168	ext{h}$ mean & std), and Cyclical Encodings ($\sin/\cos$ of hour & day-of-week).
- Trained **XGBoost Regressor** on chronological split (75% Train, 12.5% Val, 12.5% Sealed Test). Achieved **Test MAE = 4,503.51 kWh**, **sMAPE = 1.98%**, and **$R^2 = 0.9942$**.
- Frozen Model Saved: `models/uci_forecasting/xgb_forecasting_frozen.joblib`.

### Step 4: Robustness Audits & Diagnostic Stress Testing
- **Class Separability**: Verified distinct daily volatility distributions between benign ($1.42$ kWh/day) and tampered ($0.18$ kWh/day) meters.
- **Noise Perturbation**: Tested model resilience under additive Gaussian noise ($0\%	ext{--}20\%$).
- **Evasion Attacks**: Simulated 50% bypass theft scenarios; confirmed secondary volatility features prevent complete evasion.

### Step 5: Production FastAPI Backend Implementation (`backend/`)
- Developed modular backend architecture:
  - `backend/main.py`: Application lifecycle manager; loads both frozen models ONCE at startup.
  - `backend/api/detection.py`: POST `/api/v1/detection/analyze` (single meter analysis) & `/batch` (CSV bulk processing).
  - `backend/api/forecasting.py`: POST `/api/v1/forecast/next-hour` (1-hour-ahead load forecasting).
  - `backend/api/health.py`: GET `/api/v1/health` & `/models` metadata endpoints.
- Executed Pytest suite (`tests/`): Passed **13 out of 13 automated integration tests**.

### Step 6: Web Application Frontend Development (`frontend/`)
- Built single-page React + TypeScript application with Tailwind CSS & Lucide Icons.
- Integrated interactive Recharts telemetry graphs, live prediction cards, threshold tuning sliders, and model metadata drawers.

### Step 7: Frontend Styling Diagnostics & Tailwind v4 Repair
- Diagnosed Vite/Tailwind loading conflict where utility CSS was unstyled.
- Configured `@tailwindcss/vite` plugin in `frontend/vite.config.ts`, added proper content indexing, and restored full responsive layout hierarchy.

### Step 8: GridBalance V2 Controlled Model Improvement Workspace (`optimization_v2/`)
- Established double-sealed test set protocol ($N=6,356$ SGCC test set; $N=4,416$ UCI test set evaluated **exactly once**).
- Ran 9 candidate experiments (`EXP_SGCC_V2_01` to `EXP_UCI_V2_04`): Optuna 25-trial LightGBM tuning, validation threshold optimization ($t=0.58$), interaction ratios, and 33-feature EMA/Fourier extensions.
- **Audit Decision**: Both frozen baseline XGBoost models were retained for production (XGBoost baseline superior on SGCC test F1; UCI baseline preserves stateless REST API latency).

---

## 4. Complete Repository Directory Map & File Inventory

Below is the authoritative directory inventory of the GridBalance codebase:

```
gridbalance/
├── backend/                        # FastAPI REST API Backend
│   ├── main.py                     # Entry point & model startup loader
│   ├── api/                        # Route handlers (detection, forecasting, health, models)
│   ├── services/                   # Business logic (sgcc_service.py, forecasting_service.py)
│   ├── inference/                  # Prediction engines (sgcc_predictor.py, uci_forecaster.py)
│   ├── schemas/                    # Pydantic input/output request schemas
│   └── tests/                      # Automated test suite (13/13 passing tests)
├── frontend/                       # React + TypeScript Web Application
│   ├── src/                        # React UI components, pages, services, hooks
│   ├── public/                     # Static web assets & icons
│   ├── vite.config.ts              # Vite bundler configuration (@tailwindcss/vite)
│   ├── package.json                # NPM dependencies & scripts
│   └── tailwind.config.js          # Tailwind CSS design system rules
├── data/                           # Ingested & Processed Datasets
│   ├── data.csv                    # SGCC raw daily consumption data (27.67 MB)
│   ├── uci/LD2011_2014.txt         # UCI raw customer consumption file (710.99 MB)
│   └── processed/                  # Feature matrices (sgcc_features_real_v2.csv, uci_hourly_aggregate_load.csv)
├── models/                         # Serialized Model Artifacts
│   ├── sgcc_tuned/                 # Frozen SGCC 18-feature XGBoost classifier (.joblib)
│   ├── uci_forecasting/            # Frozen UCI 29-feature XGBoost regressor (.joblib)
│   └── optimization_v2/            # Candidate LightGBM & Optuna models (.joblib)
├── configs/                        # System & Experiment Configuration
│   └── optimization_v2/            # V2 experiment YAML/JSON definitions
├── graphs/                         # Generated Visualizations & ROC/PR Curves
│   └── optimization_v2/            # V2 baseline vs candidate comparison plots (.png)
├── reports/                        # Research Dossiers, Audits & Compiled PDFs
│   ├── GRIDBALANCE_COMPLETE_RESEARCH_AUDIT_REPORT.pdf
│   ├── GRIDBALANCE_PROJECT_TIMELINE_HISTORY_REPORT.pdf
│   ├── GRIDBALANCE_MASTER_RESULTS.csv
│   └── optimization_v2/            # EXPERIMENT_PROTOCOL.md & V2_MODEL_OPTIMIZATION_REPORT.pdf
├── results/                        # Experiment Result Logging
│   └── optimization_v2/            # experiment_registry.csv & experiment_registry.pdf
└── scripts/                        # Pipeline Execution Scripts
    ├── run_fe_real_pipeline.py     # SGCC feature extraction script
    ├── process_uci_and_audit.py    # UCI aggregation & feature engineering script
    └── optimization_v2/            # run_v2_experiments.py master framework script
```

---

## 5. Master Metric Ground-Truth Progression

### A. SGCC Electricity Theft Detection Metric History

| Phase / Model Stage | Precision | Recall | F1-Score | PR-AUC | ROC-AUC | Status / Verdict |
|---|---|---|---|---|---|---|
| Phase 1: Logistic Regression | 0.5238 | 0.0406 | 0.0753 | 0.2593 | 0.7531 | Baseline Discarded |
| Phase 1: Decision Tree | 0.2504 | 0.2712 | 0.2604 | 0.1301 | 0.5914 | Baseline Discarded |
| Phase 1: Random Forest | 0.6441 | 0.0701 | 0.1265 | 0.3432 | 0.8169 | Baseline Discarded |
| Phase 2: **XGBoost (Val)** | 0.3456 | 0.4852 | **0.4037** | **0.3684** | **0.8118** | Validation Benchmark |
| Phase 3: **XGBoost (Sealed Test)**| **0.3040** | **0.4391** | **0.3592** | **0.3114** | **0.7758** | **PRIMARY FROZEN BASELINE** |
| Phase 9: Optuna LightGBM ($t=0.58$)| 0.3112 | 0.3664 | 0.3366 | 0.2938 | 0.7981 | V2 Candidate (Unsuccessful) |

---

### B. UCI Aggregate Load Forecasting Metric History

| Phase / Model Stage | MAE (kWh) | RMSE (kWh) | sMAPE (%) | $R^2$ Score | Status / Verdict |
|---|---|---|---|---|---|
| Phase 1: Linear Regression | 8,920.10 | 12,450.30 | 4.12% | 0.8845 | Baseline Discarded |
| Phase 1: Random Forest Regressor | 5,120.45 | 7,890.12 | 2.35% | 0.9850 | Baseline Discarded |
| Phase 3: **XGBoost Regressor (Val)** | 3,987.48 | 5,887.36 | 1.97% | 0.9921 | Validation Benchmark |
| Phase 3: **XGBoost Regressor (Test)**| **4,503.51** | **6,743.21** | **1.98%** | **0.9942** | **PRIMARY FROZEN BASELINE** |
| Phase 9: LightGBM (33 Features) | 4,348.93 | 6,512.40 | 1.89% | 0.9946 | Rejected (Stateless Policy) |

---

## 6. Final Master Decision Matrix & Production Status

```
==================================================================================================
GRIDBALANCE MASTER PRODUCTION STATUS & DEPLOYMENT MATRIX
==================================================================================================

1. SGCC ELECTRICITY THEFT CLASSIFICATION ENGINE:
   - Primary Model File: models/sgcc_tuned/xgboost_reduced_best.joblib
   - Input Schema: 18 Selected Consumption & Missingness Features
   - Active Decision Threshold: 0.50
   - Test Metrics: F1 = 0.3592 | PR-AUC = 0.3114 | ROC-AUC = 0.7758
   - Production API: POST /api/v1/detection/analyze & /batch
   - Deployment Status: ACTIVE IN PRODUCTION FASTAPI BACKEND

2. UCI AGGREGATE LOAD FORECASTING ENGINE:
   - Primary Model File: models/uci_forecasting/xgb_forecasting_frozen.joblib
   - Input Schema: 29 Lag, Rolling Window & Temporal Cyclical Features
   - Test Metrics: MAE = 4,503.51 kWh | sMAPE = 1.98% | R² = 0.9942
   - Production API: POST /api/v1/forecast/next-hour
   - Deployment Status: ACTIVE IN PRODUCTION FASTAPI BACKEND
==================================================================================================
```

---
*Master Timeline & Repository Directory Report compiled automatically for research documentation.*
