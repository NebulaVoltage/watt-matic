# MODEL CARD: UCI ELECTRICITY NEXT-HOUR LOAD FORECASTING MODELS
## GridBalance Smart Grid Research & Deployment Platform

**Document Version:** 1.0.0-FINAL  
**Date:** November 2, 2026  
**Auditor / Author:** GridBalance Senior ML Research Scientist, Time Series Systems Architect  
**Governance Framework:** Phase 4 Research Hardening & Scientific Verification  

---

### 1. MODEL DETAILS

#### 1.1 Overview
This Model Card documents the regression models deployed and evaluated within the UCI Electricity Load Diagrams next-hour aggregate demand forecasting branch of GridBalance. The models forecast short-term regional grid load ($t+1\text{ hour}$) using autoregressive lag structures, trailing window statistics, calendar indicators, and optional dynamic Exponential Moving Averages (EMA).

#### 1.2 Model Architectures

##### A. Production Model: XGBoost Direct Level Regressor (Frozen Default Production)
- **Model ID**: `UCI_XGB_DIRECT`
- **Artifact File**: `models/uci_forecasting/xgb_forecasting_frozen.joblib`
- **Artifact SHA-256**: `1e1279932ee5dbe94553fa65fc62ef4fefaa18c5040e34c382218760074e5088`
- **Model Architecture**: `XGBRegressor` (200 estimators, max depth 6, learning rate 0.05, colsample_bytree 0.8, subsample 0.8)
- **Input Feature Set**: 29 causal Finite Impulse Response (FIR) features (lags, trailing rolling means/stds/extremes, cyclical calendar signals).
- **History Requirement**: $168\text{ hours}$ (due to weekly lag `load_t_minus_168`).
- **Serving Route**: `/api/v1/forecast/next-hour` (Default production endpoint).
- **Deployment Status**: **FROZEN PRODUCTION DEFAULT**.

##### B. Experimental Candidate: LightGBM 33-Feature EMA Regressor (Opt-In Route)
- **Model ID**: `UCI_LGBM_EMA`
- **Artifact File**: `models/optimization_v2/lgb_uci_v2_04.joblib`
- **Artifact SHA-256**: `1e850b556b6aa99b45ea4e410f913d9ddfa9a9b19f6a70e7040ad058e39f3796`
- **Model Architecture**: LightGBM Booster (300 estimators, 31 leaves, learning rate 0.03, feature fraction 0.8)
- **Input Feature Set**: 33 features (29 base FIR features + `ema_6`, `ema_24`, `sin_month`, `cos_month`).
- **History Requirement**: **Strictly $\ge 336\text{ hours}$** ($14\text{ days}$) of contiguous hourly observations to eliminate initial condition warmup bias ($(1 - \alpha)^{336} < 10^{-12}$).
- **Serving Route**: `/api/v1/forecast/next-hour-experimental-ema` (Specialized route).
- **Deployment Status**: **EXPERIMENTAL CANDIDATE** (Dedicated guarded endpoint).

##### C. Research Benchmark: XGBoost Difference Forecasting Regressor
- **Model ID**: `UCI_XGB_DIFFERENCE`
- **Artifact File**: `models/uci_forecasting/xgb_forecasting_diff.joblib`
- **Model Architecture**: `XGBRegressor` predicting first-order difference $\Delta y_{t+1} = y_{t+1} - y_t$.
- **Reconstruction**: Absolute load is reconstructed causally via $\hat{y}_{t+1} = y_t + \widehat{\Delta y}_{t+1}$.
- **Deployment Status**: **RESEARCH BENCHMARK ASSET**.

---

### 2. INTENDED USE & OPERATIONAL CONTRACT

#### 2.1 Intended Use Cases
- Next-hour aggregate regional demand prediction for grid balance optimization.
- Spinning reserve scheduling and economic dispatch planning.
- Real-time ramp rate alerting and peak load warning.

#### 2.2 API Operational Invariants & Input Validation
1. **Production Endpoint (`/api/v1/forecast/next-hour`)**:
   - Ingests minimum 168 hours of historical readings.
   - Guaranteed immunity against IIR state warmup bias (uses strictly FIR features).
2. **Experimental EMA Endpoint (`/api/v1/forecast/next-hour-experimental-ema`)**:
   - Enforces a **strict precondition of $\ge 336$ contiguous hourly readings**.
   - Truncated payloads ($w < 336$) are rejected with `HTTP 400 Bad Request` or `HTTP 422 Unprocessable Entity`.
   - Automatic chronological re-sorting and deduplication guard against asynchronous telemetry delivery.

---

### 3. FACTORS, INPUT FEATURES & DATASETS

#### 3.1 Input Data
- **Dataset**: UCI Electricity Load Diagrams (370 aggregated European clients).
- **Resolution**: 1-hour intervals ($\Delta t = 3,600\text{ s}$).
- **Total Duration**: $35,065\text{ hours}$ (January 1, 2011 00:00:00 to January 1, 2015 00:00:00).
- **Chronological Block Splits**:
  - Training Partition: `2011-01-01` to `2013-12-31 23:00:00` ($26,304\text{ hours}$, 3 calendar years).
  - Validation Partition: `2014-01-01` to `2014-06-30 23:00:00` ($4,344\text{ hours}$, H1 2014).
  - Test Partition: `2014-07-01` to `2014-12-31 23:00:00` ($4,416\text{ hours}$, H2 2014).

#### 3.2 Feature Definitions (29 FIR Base + 4 Dynamic EMA)
- **Autoregressive Lags (10)**: `load_t`, `load_t_minus_1`, `load_t_minus_2`, `load_t_minus_3`, `load_t_minus_6`, `load_t_minus_12`, `load_t_minus_24`, `load_t_minus_48`, `load_t_minus_72`, `load_t_minus_168`.
- **Rolling Windows (10)**: Trailing means (`rolling_mean_3`, `6`, `12`, `24`, `48`, `168`), standard deviations (`rolling_std_24`, `168`), and extremes (`rolling_min_24`, `rolling_max_24`).
- **Calendar & Harmonic Signals (9)**: `hour`, `day_of_week`, `day_of_month`, `month`, `week_of_year`, $\sin(\text{hour})$, $\cos(\text{hour})$, $\sin(\text{day})$, $\cos(\text{day})$.
- **EMA & Annual Harmonics (4, Experimental LGBM only)**: `ema_6` ($\alpha \approx 0.286$), `ema_24` ($\alpha = 0.080$), $\sin(\text{month})$, $\cos(\text{month})$.

---

### 4. QUANTITATIVE PERFORMANCE & BENCHMARK COMPARISON

**Table 1: Authoritative Performance Ledger on Aligned H2 2014 Test Set ($4,416\text{ hours}$)**
*(Authoritative Source: `results/phase4/CANONICAL_BENCHMARK_LEDGER.csv`)*

| Model Candidate | Feature Count | History Required | Test MAE (kWh) | Test RMSE (kWh) | Test MAPE (%) | Test $R^2$ | Relative Error vs Production |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Production XGBoost Direct** | 29 | 168 h | **4,440.05** | $6,192.44$ | $1.87\%$ | $0.9882$ | Baseline ($0.0\%$) |
| **Research XGBoost Difference** | 29 | 168 h | **4,350.13** | $\mathbf{6,045.88}$ | $\mathbf{1.83\%}$ | $\mathbf{0.9888}$ | **$-2.02\%$ (Improvement)** |
| **Experimental LightGBM EMA** | 33 | 336 h | **4,357.54** | $6,094.21$ | $1.84\%$ | $0.9886$ | **$-1.86\%$ (Improvement)** |

#### Key Evaluation Findings:
1. **Benchmark Alignment Resolution**: Unaligned evaluation on raw test slices yields artificial MAEs ($4,503.51\text{ kWh}$ or $4,421.55\text{ kWh}$). On the strictly aligned, causal H2 2014 test window ($4,416\text{ hours}$), the authoritative production baseline achieves $\mathbf{4,440.05\text{ kWh}}$.
2. **Difference Forecasting Superiority**: The difference forecasting architecture achieves the lowest overall MAE ($4,350.13\text{ kWh}$) and RMSE ($6,045.88\text{ kWh}$) without requiring 336 hours of historical telemetry or IIR state warmup, making it a compelling candidate for future production promotion.
3. **EMA Candidate Parity**: The LightGBM EMA model achieves $\mathbf{4,357.54\text{ kWh}}$ MAE, matching the difference model within $7.41\text{ kWh}$ ($0.17\%$). However, it requires double the history payload ($336\text{h}$ vs $168\text{h}$) and strict input defense.

---

### 5. INPUT DEFENSE & NUMERICAL STABILITY SUMMARY

- **Warmup Degradation Risk**: If an uninformed client supplies only 24 hours of data, `ema_24` incurs a mean bias of $902.51\text{ kWh}$ (max error $10,517.73\text{ kWh}$).
- **Defense Verification**: 11 out of 11 adversarial stress tests passed (`DEF_01` through `DEF_11`).
- **Defect Remediation**: Telemetry timestamps are now automatically sorted and deduplicated prior to computing EMA features, ensuring that out-of-order API packets yield identical forecasts to sequential data.
