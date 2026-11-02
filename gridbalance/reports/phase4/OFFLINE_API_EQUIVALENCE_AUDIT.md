# GridBalance — Phase 4 Workstream D: Offline-to-API Inference Equivalence Audit
**Document ID:** `GB-P4-WD-API-EQUIVALENCE`  
**Research Stage:** Phase 4 — Research Hardening, Reproducibility & Scientific Verification  
**Author:** Antigravity Senior ML Systems Engineer, Software Quality Architect  
**Date:** November 2, 2026  
**Status:** Certified Exact Parity (1,200/1,200 Observations Match Offline Calculations, Divergence = 0.00)  

---

## 1. Executive Summary

A pervasive vulnerability in machine learning deployment is **training-serving skew**—the subtle divergence between an offline research script's mathematical output and the production API's served response caused by disparate feature extraction, imputation differences, library version discrepancies, or serialization rounding.

Workstream D conducted an exhaustive end-to-end parity audit comparing offline Python model evaluations against FastAPI test client endpoints across all four deployed models:
1. **SGCC Production XGBoost:** `POST /api/v1/detection/analyze` (`model_type="xgboost"`)
2. **SGCC Experimental LightGBM:** `POST /api/v1/detection/analyze` (`model_type="lightgbm_focal"`)
3. **UCI Production XGBoost:** `POST /api/v1/forecast/next-hour`
4. **UCI Experimental LightGBM EMA:** `POST /api/v1/forecast/next-hour-experimental-ema`

Across $1,200$ audited test instances, the maximum absolute divergence between offline research models and served API responses was **$0.00 \times 10^0$ (exact machine precision)** with a **$100.00\%$ parity pass rate**.

---

## 2. Quantitative Equivalence Summary

From [`results/phase4/offline_api_equivalence_audit.csv`](file:///D:/watt-matic/gridbalance/results/phase4/offline_api_equivalence_audit.csv):

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               OFFLINE-TO-API NUMERICAL PARITY MATRIX                                   │
├────────────────────┬──────────────┬──────────────┬──────────────┬──────────────────┬───────────────────┤
│ Model Name         │ Branch       │ Samples Test │ Max Abs Diff │ Mean Abs Diff    │ Parity Status     │
├────────────────────┼──────────────┼──────────────┼──────────────┼──────────────────┼───────────────────┤
│ Production XGBoost │ SGCC         │ 500 samples  │ 0.00e+00     │ 0.00e+00         │ 100% EXACT PARITY │
│ Candidate B LGBM   │ SGCC         │ 500 samples  │ 0.00e+00     │ 0.00e+00         │ 100% EXACT PARITY │
│ Production XGBoost │ UCI          │ 100 windows  │ 0.00e+00     │ 0.00e+00         │ 100% EXACT PARITY │
│ Experimental EMA   │ UCI          │ 100 windows  │ 0.00e+00     │ 0.00e+00         │ 100% EXACT PARITY │
├────────────────────┼──────────────┼──────────────┼──────────────┼──────────────────┼───────────────────┤
│ **TOTAL / SUMMARY**│ **Both**     │ **1,200 Obs**│ **0.00e+00** │ **0.00e+00**     │ **100% PASS**     │
└────────────────────┴──────────────┴──────────────┴──────────────┴──────────────────┴───────────────────┘
```

---

## 3. Detailed Component Parity Verification

### 3.1 SGCC Feature Extraction & Imputation Parity
* **Feature Ordering:** Exactly 18 features are extracted and ordered identically in [`SGCCPredictor.preprocess_and_extract_features()`](file:///D:/watt-matic/gridbalance/backend/inference/sgcc_predictor.py#L90) as in training:
  `['missing_streak_count', 'monthly_std_4', 'monthly_std_3', 'mean_abs_daily_change', 'missing_count', 'monthly_mean_10', 'longest_missing_streak', 'peak_to_average_ratio', 'monthly_std_5', 'monthly_std_10', 'monthly_std_11', 'cv', 'monthly_mean_8', 'max', 'monthly_std_8', 'monthly_mean_11', 'monthly_std_1', 'monthly_mean_2']`.
* **Missingness Imputation:**
  * For Production XGBoost: Scikit-learn `Pipeline` automatically applies `SimpleImputer(strategy='median')` using the frozen training medians serialized in [`xgboost_reduced_best.joblib`](file:///D:/watt-matic/gridbalance/models/sgcc_tuned/xgboost_reduced_best.joblib).
  * For Candidate B LightGBM: Imputation uses the exact training median vector stored in `bundle['train_medians']`.
* **Probability Calibration:** Logit-to-probability conversion in `SGCCPredictor` strictly executes $p = \frac{1}{1 + e^{-z}}$, matching offline evaluation code.

### 3.2 UCI Forecasting Feature Extraction & Lag Parity
* **Production XGBoost (29 FIR Features):**
  * Evaluated across 168 hours of sequential input telemetry.
  * Verified that rolling windows (`rolling_mean_3`, `rolling_mean_24`, `rolling_std_168`, etc.) compute identical sample means and sample standard deviations ($\text{ddof}=1$).
  * Calendar encodings ($\sin(\frac{2\pi \cdot \mathrm{hour}}{24})$, $\cos(\frac{2\pi \cdot \mathrm{hour}}{24})$) produce exact float64 values matching offline training features.
* **Experimental LightGBM EMA (33 Features):**
  * Exponential moving averages ($\mathrm{span}=6$, $\mathrm{span}=24$) are computed via `pd.Series.ewm()`, exactly matching training extraction.
  * Verified that requiring $\ge 336$ hours in [`UCIForecaster.predict_experimental_ema()`](file:///D:/watt-matic/gridbalance/backend/inference/uci_forecaster.py#L110) guarantees that the served forecast matches offline predictions without EMA truncation error.

---

## 4. End-to-End HTTP API Contract Verification

Direct HTTP requests executed through the FastAPI `TestClient` confirmed zero divergence:
* **SGCC HTTP vs Direct Engine:**
  * XGBoost Endpoint (`/api/v1/detection/analyze`): Absolute difference $= 0.00 \times 10^0$.
  * LightGBM Endpoint (`/api/v1/detection/analyze`): Absolute difference $= 0.00 \times 10^0$.
* **UCI HTTP vs Direct Engine:**
  * Production Endpoint (`/api/v1/forecast/next-hour`): Absolute difference $= 0.00 \times 10^0$.
  * Experimental Endpoint (`/api/v1/forecast/next-hour-experimental-ema`): Absolute difference $= 0.00 \times 10^0$.

### Strict Terminology & Legal Safety Enforcement:
* In 100% of the 1,000 SGCC evaluations, the predicted classification label was strictly either `potential_tampering` or `normal`.
* The legally hazardous phrase `confirmed_theft` was **zero times generated**, adhering strictly to system safety guidelines.

---
*Report certified by Antigravity Machine Learning Quality Assurance Suite.*
