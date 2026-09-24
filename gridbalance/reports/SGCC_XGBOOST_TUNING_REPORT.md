# SGCC XGBoost Hyperparameter Tuning & Threshold Optimization Report

## 1. Tuning Methodology
- **Search Space**: `n_estimators`, `max_depth`, `learning_rate`, `min_child_weight`, `subsample`, `colsample_bytree`, `gamma`, `reg_alpha`, `reg_lambda`
- **CV Strategy**: 5-fold StratifiedKFold on the 29,656-row Training split (Random Seed 42). 20 randomized iterations.
- **Data Scaling/Imputation**: `SimpleImputer(strategy='median')` applied internally within each CV fold to strictly prevent leakage.
- **Optimization Metric**: Macro F1 (refit).

## 2. Selected Hyperparameters
```json
{'xgb__subsample': 0.7, 'xgb__reg_lambda': 100, 'xgb__reg_alpha': 1, 'xgb__n_estimators': 200, 'xgb__min_child_weight': 1, 'xgb__max_depth': 7, 'xgb__learning_rate': 0.2, 'xgb__gamma': 0.5, 'xgb__colsample_bytree': 0.7}
```

## 3. CV & Validation Performance
- **Cross-Validation Mean F1**: 0.409
- **Validation F1 (0.50 Threshold)**: 0.457

## 4. Threshold Analysis
- Evaluated threshold grid: `[0.05, 0.95]` strictly on Validation predictions.
- **Optimal F1 Threshold**: 0.50
- The validation F1 score at this optimal threshold is **0.457**.
- A stable threshold region was observed in the precision-recall plots (see `graphs/xgboost_tuning/`).

## 5. Final Sealed Test Evaluation
| Configuration | Threshold | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---------------|-----------|-----------|--------|----|--------|---------|
| Baseline XGB (Weighted) | 0.50 | 0.353 | 0.411 | 0.380 | 0.344 | 0.803 |
| Tuned XGB | 0.50 | 0.345 | 0.483 | 0.402 | 0.375 | 0.826 |
| Tuned XGB (Optimal) | 0.50 | 0.345 | 0.483 | 0.402 | 0.375 | 0.826 |

## 6. Interpretation
- **Baseline vs Tuned**: Hyperparameter tuning slightly shifted the model's capacity. 
- **Validation-to-Test Gap**: The test performance is closely aligned with validation, indicating no severe overfitting.
- **Precision/Recall Tradeoff**: Operating at the optimal threshold balances identifying theft cases (Recall) against overwhelming investigators with false positives (Precision).
- **Note**: Missingness features demonstrated predictive association in the baseline phase; this tuning phase incorporates them but does not assert causality.

## 7. Limitations
- We have not yet pruned correlated features (Feature Selection), which tree models can sometimes struggle with when highly collinear variables exist.
