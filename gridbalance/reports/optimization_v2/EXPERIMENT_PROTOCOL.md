# GridBalance V2 — Controlled Model Improvement Research Protocol

> **Protocol Version**: 2.0.0  
> **Workspace**: `research/optimization_v2/`  
> **Baseline Policy**: **ORIGINAL FROZEN MODELS REMAIN UNTOUCHED AND AUTHORITATIVE.**  
> **Primary Rule**: No V2 candidate replaces a frozen model unless it demonstrates statistically significant improvement on Cross-Validation & Validation, and survives ONE sealed-test evaluation.

---

## 1. Experimental Methodology & Rules

1. **Leakage Prevention**: All imputation, scaling, and feature transformations are fitted strictly inside training folds.
2. **Double-Sealed Test Protocol**: Sealed test sets ($N=6,356$ for SGCC, $N=4,416$ for UCI) are evaluated EXACTLY ONCE per candidate model after model freezing.
3. **Threshold Selection**: Optimal decision thresholds are selected strictly on Validation set predictions; no threshold tuning on sealed test data is permitted.
4. **Reproducibility**: All experiments use seed `42` across data splitting and model initialization.

---

## 2. Evaluation Metrics

### SGCC Electricity Theft Classification
- **Primary Metrics**: F1-Score, Precision-Recall AUC (PR-AUC)
- **Secondary Metrics**: Precision, Recall, ROC-AUC, Matthews Correlation Coefficient (MCC), Brier Score
- **Supplementary Metric**: Accuracy

### UCI Electricity Load Forecasting
- **Primary Metrics**: Mean Absolute Error (MAE in kWh), Root Mean Squared Error (RMSE in kWh)
- **Secondary Metrics**: Symmetric Mean Absolute Percentage Error (sMAPE %), Coefficient of Determination ($R^2$)

---

## 3. Candidate Experiments Summary

| Experiment ID | Dataset | Model Architecture | Key Feature / Tuning Hypothesis |
|---|---|---|---|
| `EXP_SGCC_V2_01` | SGCC | XGBoost (18 Features) | Re-evaluate frozen baseline benchmark |
| `EXP_SGCC_V2_02` | SGCC | LightGBM Classifier (18 Features) | Test GBDT histogram binning & class weighting |
| `EXP_SGCC_V2_03` | SGCC | Optuna LightGBM (18 Features) | 25-trial Optuna hyperparameter optimization on 5-fold CV |
| `EXP_SGCC_V2_04` | SGCC | Optuna LightGBM (Val Threshold Tuned) | Optimal decision threshold $t \in [0.10, 0.90]$ on Validation |
| `EXP_SGCC_V2_05` | SGCC | LightGBM (20 Interaction Features) | Add interaction features (`volatility_ratio`, `missing_density`) |
| `EXP_UCI_V2_01` | UCI | XGBoost Regressor (29 Features) | Re-evaluate frozen baseline benchmark |
| `EXP_UCI_V2_02` | UCI | LightGBM Regressor (29 Features) | Test LightGBM Regressor on 29 lag/rolling features |
| `EXP_UCI_V2_03` | UCI | Optuna LightGBM Regressor (29 Features) | 25-trial Optuna hyperparameter tuning on expanding CV |
| `EXP_UCI_V2_04` | UCI | LightGBM (33 Enhanced Features) | Add Fourier 12h/168h encodings + EMA-6/EMA-24 features |
| `EXP_UCI_V2_05` | UCI | Ridge + XGB + LGBM Stacking Ensemble | Weighted blend of top regressors optimized on validation |
