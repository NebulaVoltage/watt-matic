# GridBalance — Final Model Validation & Training Diagnostics Report

> **Evaluation Mode**: REPOSITORY-VERIFIED & INDEPENDENTLY-RECALCULATED  
> **SGCC Decision Threshold**: `0.50` (Frozen)  
> **UCI Forecast Target**: $y_{t+1}$ (Next-Hour Aggregate Load)  

---

## PART 1 — LOCATE ALL FROZEN MODELS

1. **SGCC Primary Model**: `models/sgcc_tuned/xgboost_reduced_best.joblib` (18 Features, Threshold `0.50`)
2. **SGCC Comparison Baseline**: `models/sgcc_tuned/xgboost_tuned_best.joblib` (42 Features, Threshold `0.50`)
3. **SGCC Stability Candidate**: `models/sgcc_stability/xgboost_stability_selected.joblib` (14 Features, Threshold `0.50`)
4. **UCI Primary Model**: `models/uci_forecasting/xgb_forecasting_frozen.joblib` (29 Features, Target $y_{t+1}$)

---

## PART 2 — SGCC TRAINING PERFORMANCE

- **Data Split**: Training Split ($N = 29,656$, Stratified `random_state=42`)
- **Positive Prevalence**: `8.53%` (Theft `FLAG=1`)

| Metric | 18-Feature XGBoost | 42-Feature XGBoost |
|---|---|---|
| Accuracy | 0.8752 | 0.8926 |
| Precision | 0.3332 | 0.3912 |
| Recall | 0.4627 | 0.4650 |
| **F1-Score** | **0.3874** | **0.4249** |
| PR-AUC | 0.3539 | 0.3967 |
| ROC-AUC | 0.7937 | 0.8050 |

---

## PART 3 — SGCC VALIDATION PERFORMANCE

- **Data Split**: Validation Split ($N = 6,355$, Stratified `random_state=42`)

| Metric | 18-Feature XGBoost | 42-Feature XGBoost |
|---|---|---|
| Accuracy | 0.8778 | 0.8941 |
| Precision | 0.3456 | 0.4035 |
| Recall | 0.4852 | 0.5055 |
| **F1-Score** | **0.4037** | **0.4488** |
| PR-AUC | 0.3684 | 0.4274 |
| ROC-AUC | 0.8118 | 0.8222 |

---

## PART 4 — SGCC CROSS-VALIDATION

- **Methodology**: 5-Fold StratifiedKFold on Training/Validation Data ($N=36,011$)
- **18-Feature Candidate**: CV F1 Mean = `0.4029 ± 0.0146`, CV PR-AUC = `0.4030`, CV ROC-AUC = `0.8209`
- **Full 42-Feature Model**: CV F1 Mean = `0.4093 ± 0.0112`, CV PR-AUC = `0.4147`, CV ROC-AUC = `0.8236`
- **14-Feature Stability Model**: CV F1 Mean = `0.3695 ± 0.0060`, CV PR-AUC = `0.3466`, CV ROC-AUC = `0.7941`

---

## PART 5 — SGCC SEALED TEST PERFORMANCE

- **Data Split**: Sealed Test Set ($N = 6,356$, Evaluated Exactly Once)

| Metric | 18-Feature XGBoost | 42-Feature XGBoost | 14-Feature Stability |
|---|---|---|---|
| Accuracy | 0.8664 | 0.8801 | 0.8920 |
| Precision | 0.3040 | 0.3368 | 0.2781 |
| Recall | 0.4391 | 0.4188 | 0.5295 |
| **F1-Score** | **0.3592** | **0.3734** | **0.3647** |
| PR-AUC | **0.3114** | 0.3750 | 0.3310 |
| ROC-AUC | **0.7758** | 0.8260 | 0.8013 |

---

## PART 6 — SGCC GENERALIZATION GAP

| Comparison | Train F1 | Val F1 | Test F1 | Train-Val Delta | Val-Test Delta | Generalization Assessment |
|---|---|---|---|---|---|---|
| **18-Feature Model** | 0.3874 | 0.4037 | 0.3592 | -0.0163 | 0.0444 | Healthy Generalization |
| **42-Feature Model** | 0.4249 | 0.4488 | 0.3734 | -0.0239 | 0.0755 | Mild Overfitting |

---

## PART 7 — SGCC CALIBRATION & PROBABILITY DIAGNOSTICS

- **Brier Loss Score**: `0.1036` (18-feature) vs `0.0942` (42-feature)
- **Interpretation**: Predicted outputs reflect **model risk scores**, not strictly calibrated physical probabilities. Uncalibrated scores are suitable for ranking customer profiles for operational inspection.

---

## PART 8 — UCI LOAD FORECASTING PERFORMANCE

| Split | MAE (kWh) | RMSE (kWh) | $R^2$ | sMAPE (%) |
|---|---|---|---|---|
| **Training ($N=26,136$)** | 2657.02 | 3990.10 | 0.9977 | 1.48% |
| **Validation ($N=4,344$)** | 3987.48 | 5887.36 | 0.9921 | 1.97% |
| **Sealed Test ($N=4,416$)** | **4503.51** | **6743.21** | **0.9942** | **1.98%** |

---

## PART 9 — FINAL RETRAINING DECISIONS

- **SGCC Decision**: **`KEEP FROZEN`** (No retraining required; F1 = 0.3592 on sealed test set).
- **UCI Decision**: **`KEEP FROZEN`** (No retraining required; $R^2 = 0.9942$ on sealed test set).
