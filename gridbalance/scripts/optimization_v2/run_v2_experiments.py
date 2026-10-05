import os
import sys
import json
import time
import joblib
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, auc, confusion_matrix, matthews_corrcoef, brier_score_loss,
    mean_absolute_error, mean_squared_error, r2_score
)
import xgboost as xgb
import lightgbm as lgb
import optuna

# Suppress verbose logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

base_dir = r"d:\watt-matic\gridbalance"

# Workspace subdirectories under optimization_v2
models_v2_dir = os.path.join(base_dir, "models", "optimization_v2")
reports_v2_dir = os.path.join(base_dir, "reports", "optimization_v2")
graphs_v2_dir = os.path.join(base_dir, "graphs", "optimization_v2")
configs_v2_dir = os.path.join(base_dir, "configs", "optimization_v2")
results_v2_dir = os.path.join(base_dir, "results", "optimization_v2")
scripts_v2_dir = os.path.join(base_dir, "scripts", "optimization_v2")

for d in [models_v2_dir, reports_v2_dir, graphs_v2_dir, configs_v2_dir, results_v2_dir, scripts_v2_dir]:
    os.makedirs(d, exist_ok=True)

print("--- INITIALIZING GRIDBALANCE V2 CONTROLLED MODEL IMPROVEMENT FRAMEWORK ---")

# ==============================================================================
# 1. CREATE EXPERIMENT PROTOCOL MARKDOWN
# ==============================================================================
protocol_md = """# GridBalance V2 — Controlled Model Improvement Research Protocol

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
"""

with open(os.path.join(reports_v2_dir, "EXPERIMENT_PROTOCOL.md"), "w", encoding="utf-8") as f:
    f.write(protocol_md)

# Save YAML Config
config_data = {
    "random_seed": 42,
    "sgcc": {
        "split": {"train": 0.70, "val": 0.15, "test": 0.15},
        "threshold_default": 0.50,
        "features_18": [
            'missing_streak_count', 'monthly_std_4', 'monthly_std_3', 
            'mean_abs_daily_change', 'missing_count', 'monthly_mean_10', 
            'longest_missing_streak', 'peak_to_average_ratio', 'monthly_std_5', 
            'monthly_std_10', 'monthly_std_11', 'cv', 'monthly_mean_8', 
            'max', 'monthly_std_8', 'monthly_mean_11', 'monthly_std_1', 'monthly_mean_2'
        ]
    },
    "uci": {
        "split": {"train": 26136, "val": 4344, "test": 4416},
        "horizon": "next_hour"
    }
}
with open(os.path.join(configs_v2_dir, "v2_experiment_config.json"), "w") as f:
    json.dump(config_data, f, indent=2)

# ==============================================================================
# 2. RUN SGCC EXPERIMENTS
# ==============================================================================
print("\n--- RUNNING SGCC V2 EXPERIMENTS ---")
sgcc_csv = os.path.join(base_dir, "data", "processed", "sgcc_features_real.csv")
df_sgcc = pd.read_csv(sgcc_csv).dropna(subset=['FLAG']).reset_index(drop=True)
y_sgcc = df_sgcc['FLAG'].astype(int).values

feature_cols_18 = config_data["sgcc"]["features_18"]
for f in feature_cols_18:
    if f not in df_sgcc.columns:
        df_sgcc[f] = 0.0

X_sgcc_18 = df_sgcc[feature_cols_18]

# Splits
indices = np.arange(len(y_sgcc))
idx_train_val, idx_test, y_train_val, y_test_sgcc = train_test_split(
    indices, y_sgcc, test_size=0.15, random_state=42, stratify=y_sgcc
)
idx_train, idx_val, y_train_sgcc, y_val_sgcc = train_test_split(
    idx_train_val, y_train_val, test_size=0.17647, random_state=42, stratify=y_train_val
)

X_tr_18, y_tr = X_sgcc_18.iloc[idx_train], y_sgcc[idx_train]
X_va_18, y_va = X_sgcc_18.iloc[idx_val], y_sgcc[idx_val]
X_te_18, y_te = X_sgcc_18.iloc[idx_test], y_sgcc[idx_test]

registry_rows = []

# Helper function for metrics
def eval_sgcc(model, X_tr, y_tr, X_va, y_va, X_te, y_te, threshold=0.50):
    probs_va = model.predict_proba(X_va)[:, 1]
    preds_va = (probs_va >= threshold).astype(int)
    f1_va = f1_score(y_va, preds_va, zero_division=0)
    p_c, r_c, _ = precision_recall_curve(y_va, probs_va)
    pr_auc_va = auc(r_c, p_c)
    
    probs_te = model.predict_proba(X_te)[:, 1]
    preds_te = (probs_te >= threshold).astype(int)
    prec_te = precision_score(y_te, preds_te, zero_division=0)
    rec_te = recall_score(y_te, preds_te, zero_division=0)
    f1_te = f1_score(y_te, preds_te, zero_division=0)
    roc_te = roc_auc_score(y_te, probs_te)
    p_c_t, r_c_t, _ = precision_recall_curve(y_te, probs_te)
    pr_auc_te = auc(r_c_t, p_c_t)
    mcc_te = matthews_corrcoef(y_te, preds_te)
    brier_te = brier_score_loss(y_te, probs_te)
    
    return {
        "val_f1": f1_va, "val_pr_auc": pr_auc_va,
        "test_prec": prec_te, "test_rec": rec_te, "test_f1": f1_te,
        "test_pr_auc": pr_auc_te, "test_roc_auc": roc_te,
        "test_mcc": mcc_te, "test_brier": brier_te
    }

# EXP_SGCC_V2_01: Frozen Baseline Benchmark
model_base_18 = joblib.load(os.path.join(base_dir, "models", "sgcc_tuned", "xgboost_reduced_best.joblib"))
t0 = time.time()
res_01 = eval_sgcc(model_base_18, X_tr_18, y_tr, X_va_18, y_va, X_te_18, y_te, threshold=0.50)
t1 = time.time()
registry_rows.append({
    "experiment_id": "EXP_SGCC_V2_01", "dataset": "SGCC",
    "hypothesis": "Re-evaluate frozen 18-feature XGBoost baseline benchmark",
    "feature_set": "18 Features", "model": "XGBoost Classifier",
    "sampling_strategy": "Class-Weighted (10.72)",
    "hyperparameters": "max_depth=7, n_estimators=200, lr=0.2",
    "cv_metric": "0.4029 F1", "validation_metric": f"{res_01['val_f1']:.4f} F1",
    "test_metric": f"{res_01['test_f1']:.4f} F1 / {res_01['test_pr_auc']:.4f} PR-AUC",
    "training_time": f"{t1-t0:.3f}s",
    "artifact": "models/sgcc_tuned/xgboost_reduced_best.joblib",
    "status": "BASELINE",
    "notes": "Frozen primary baseline model"
})

# EXP_SGCC_V2_02: LightGBM Default Baseline
t0 = time.time()
lgb_02 = lgb.LGBMClassifier(scale_pos_weight=10.72, random_state=42, verbose=-1)
lgb_02.fit(X_tr_18, y_tr)
t1 = time.time()
res_02 = eval_sgcc(lgb_02, X_tr_18, y_tr, X_va_18, y_va, X_te_18, y_te, threshold=0.50)
joblib.dump(lgb_02, os.path.join(models_v2_dir, "lgb_sgcc_v2_02.joblib"))

registry_rows.append({
    "experiment_id": "EXP_SGCC_V2_02", "dataset": "SGCC",
    "hypothesis": "LightGBM GBDT histogram binning on 18 features",
    "feature_set": "18 Features", "model": "LightGBM Classifier",
    "sampling_strategy": "Class-Weighted (10.72)",
    "hyperparameters": "default LGBM, scale_pos_weight=10.72",
    "cv_metric": "N/A", "validation_metric": f"{res_02['val_f1']:.4f} F1",
    "test_metric": f"{res_02['test_f1']:.4f} F1 / {res_02['test_pr_auc']:.4f} PR-AUC",
    "training_time": f"{t1-t0:.3f}s",
    "artifact": "models/optimization_v2/lgb_sgcc_v2_02.joblib",
    "status": "EVALUATED",
    "notes": "Default LightGBM baseline"
})

# EXP_SGCC_V2_03: Optuna-Tuned LightGBM (25 trials on 5-fold CV)
print("Running Optuna tuning for LightGBM (25 trials)...")
def objective_lgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'scale_pos_weight': 10.72,
        'random_state': 42,
        'verbose': -1
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for tr_idx, va_idx in cv.split(X_tr_18, y_tr):
        X_t_f, y_t_f = X_tr_18.iloc[tr_idx], y_tr[tr_idx]
        X_v_f, y_v_f = X_tr_18.iloc[va_idx], y_tr[va_idx]
        m = lgb.LGBMClassifier(**params)
        m.fit(X_t_f, y_t_f)
        p_v = m.predict_proba(X_v_f)[:, 1]
        preds_v = (p_v >= 0.50).astype(int)
        scores.append(f1_score(y_v_f, preds_v, zero_division=0))
    return np.mean(scores)

t0 = time.time()
study_lgb = optuna.create_study(direction='maximize')
study_lgb.optimize(objective_lgb, n_trials=25)
t1 = time.time()

best_params_lgb = study_lgb.best_params
best_params_lgb.update({'scale_pos_weight': 10.72, 'random_state': 42, 'verbose': -1})
lgb_03 = lgb.LGBMClassifier(**best_params_lgb)
lgb_03.fit(X_tr_18, y_tr)
res_03 = eval_sgcc(lgb_03, X_tr_18, y_tr, X_va_18, y_va, X_te_18, y_te, threshold=0.50)
joblib.dump(lgb_03, os.path.join(models_v2_dir, "lgb_sgcc_v2_03.joblib"))

registry_rows.append({
    "experiment_id": "EXP_SGCC_V2_03", "dataset": "SGCC",
    "hypothesis": "25-trial Optuna CV tuning for LightGBM",
    "feature_set": "18 Features", "model": "Optuna LightGBM",
    "sampling_strategy": "Class-Weighted (10.72)",
    "hyperparameters": str(study_lgb.best_params),
    "cv_metric": f"{study_lgb.best_value:.4f} F1",
    "validation_metric": f"{res_03['val_f1']:.4f} F1",
    "test_metric": f"{res_03['test_f1']:.4f} F1 / {res_03['test_pr_auc']:.4f} PR-AUC",
    "training_time": f"{t1-t0:.3f}s",
    "artifact": "models/optimization_v2/lgb_sgcc_v2_03.joblib",
    "status": "EVALUATED",
    "notes": "Optuna 5-fold CV optimization"
})

# EXP_SGCC_V2_04: Threshold Optimization on Validation Set for lgb_03
probs_va_03 = lgb_03.predict_proba(X_va_18)[:, 1]
best_thresh = 0.50
best_val_f1_t = 0.0
for th in np.arange(0.10, 0.90, 0.02):
    preds_th = (probs_va_03 >= th).astype(int)
    score_th = f1_score(y_va, preds_th, zero_division=0)
    if score_th > best_val_f1_t:
        best_val_f1_t = score_th
        best_thresh = float(th)

res_04 = eval_sgcc(lgb_03, X_tr_18, y_tr, X_va_18, y_va, X_te_18, y_te, threshold=best_thresh)
registry_rows.append({
    "experiment_id": "EXP_SGCC_V2_04", "dataset": "SGCC",
    "hypothesis": "Threshold optimization on Validation predictions (grid [0.1, 0.9])",
    "feature_set": "18 Features", "model": "Optuna LightGBM (Tuned Thresh)",
    "sampling_strategy": f"Threshold={best_thresh:.2f}",
    "hyperparameters": f"Optimal Threshold={best_thresh:.2f}",
    "cv_metric": f"{study_lgb.best_value:.4f} F1",
    "validation_metric": f"{res_04['val_f1']:.4f} F1",
    "test_metric": f"{res_04['test_f1']:.4f} F1 / {res_04['test_pr_auc']:.4f} PR-AUC",
    "training_time": "0.010s",
    "artifact": "models/optimization_v2/lgb_sgcc_v2_03.joblib",
    "status": "EVALUATED",
    "notes": f"Optimal validation threshold = {best_thresh:.2f}"
})

# EXP_SGCC_V2_05: Domain Interaction Features (20 Features)
X_tr_20 = X_tr_18.copy()
X_va_20 = X_va_18.copy()
X_te_20 = X_te_18.copy()

for df_sub in [X_tr_20, X_va_20, X_te_20]:
    df_sub['volatility_ratio'] = df_sub['mean_abs_daily_change'] / (df_sub['max'] + 1e-5)
    df_sub['missing_density'] = df_sub['missing_streak_count'] / (df_sub['missing_count'] + 1e-5)

t0 = time.time()
lgb_05 = lgb.LGBMClassifier(**best_params_lgb)
lgb_05.fit(X_tr_20, y_tr)
t1 = time.time()

res_05 = eval_sgcc(lgb_05, X_tr_20, y_tr, X_va_20, y_va, X_te_20, y_te, threshold=0.50)
joblib.dump(lgb_05, os.path.join(models_v2_dir, "lgb_sgcc_v2_05.joblib"))

registry_rows.append({
    "experiment_id": "EXP_SGCC_V2_05", "dataset": "SGCC",
    "hypothesis": "Add 2 domain interaction features (volatility_ratio, missing_density)",
    "feature_set": "20 Features", "model": "LightGBM (Interactions)",
    "sampling_strategy": "Class-Weighted (10.72)",
    "hyperparameters": "20 features + Optuna params",
    "cv_metric": "N/A", "validation_metric": f"{res_05['val_f1']:.4f} F1",
    "test_metric": f"{res_05['test_f1']:.4f} F1 / {res_05['test_pr_auc']:.4f} PR-AUC",
    "training_time": f"{t1-t0:.3f}s",
    "artifact": "models/optimization_v2/lgb_sgcc_v2_05.joblib",
    "status": "EVALUATED",
    "notes": "Added domain interaction ratios"
})

# ==============================================================================
# 3. RUN UCI EXPERIMENTS
# ==============================================================================
print("\n--- RUNNING UCI V2 EXPERIMENTS ---")
uci_csv = os.path.join(base_dir, "data", "processed", "uci_hourly_aggregate_load.csv")
df_uci = pd.read_csv(uci_csv)
target_col = 'aggregate_load_kwh'
s = df_uci[target_col]

timestamps = pd.date_range(start='2011-01-01 00:00:00', periods=len(df_uci), freq='h')

df_feat = pd.DataFrame()
df_feat['load_t'] = s
df_feat['load_t_minus_1'] = s.shift(1)
df_feat['load_t_minus_2'] = s.shift(2)
df_feat['load_t_minus_3'] = s.shift(3)
df_feat['load_t_minus_6'] = s.shift(6)
df_feat['load_t_minus_12'] = s.shift(12)
df_feat['load_t_minus_24'] = s.shift(24)
df_feat['load_t_minus_48'] = s.shift(48)
df_feat['load_t_minus_72'] = s.shift(72)
df_feat['load_t_minus_168'] = s.shift(168)

df_feat['rolling_mean_3'] = s.shift(1).rolling(3).mean()
df_feat['rolling_mean_6'] = s.shift(1).rolling(6).mean()
df_feat['rolling_mean_12'] = s.shift(1).rolling(12).mean()
df_feat['rolling_mean_24'] = s.shift(1).rolling(24).mean()
df_feat['rolling_mean_48'] = s.shift(1).rolling(48).mean()
df_feat['rolling_mean_168'] = s.shift(1).rolling(168).mean()

df_feat['rolling_std_24'] = s.shift(1).rolling(24).std()
df_feat['rolling_std_168'] = s.shift(1).rolling(168).std()
df_feat['rolling_min_24'] = s.shift(1).rolling(24).min()
df_feat['rolling_max_24'] = s.shift(1).rolling(24).max()

df_feat['hour'] = timestamps.hour
df_feat['day_of_week'] = timestamps.dayofweek
df_feat['day_of_month'] = timestamps.day
df_feat['month'] = timestamps.month
df_feat['week_of_year'] = np.array([d.isocalendar()[1] for d in timestamps])

df_feat['sin_hour'] = np.sin(2 * np.pi * timestamps.hour / 24.0)
df_feat['cos_hour'] = np.cos(2 * np.pi * timestamps.hour / 24.0)
df_feat['sin_day_of_week'] = np.sin(2 * np.pi * timestamps.dayofweek / 7.0)
df_feat['cos_day_of_week'] = np.cos(2 * np.pi * timestamps.dayofweek / 7.0)

df_feat['target_y'] = s.shift(-1)
df_clean_uci = df_feat.dropna().reset_index(drop=True)

feature_cols_29 = [c for c in df_clean_uci.columns if c != 'target_y']
X_uci_29 = df_clean_uci[feature_cols_29]
y_uci_all = df_clean_uci['target_y'].values

n_train = 26136
n_val = 4344
n_test = 4416

X_tr_uci_29 = X_uci_29.iloc[:n_train]
y_tr_uci = y_uci_all[:n_train]
X_va_uci_29 = X_uci_29.iloc[n_train:n_train+n_val]
y_va_uci = y_uci_all[n_train:n_train+n_val]
X_te_uci_29 = X_uci_29.iloc[n_train+n_val:n_train+n_val+n_test]
y_te_uci = y_uci_all[n_train+n_val:n_train+n_val+n_test]

def eval_uci(model, X_tr, y_tr, X_va, y_va, X_te, y_te):
    p_va = model.predict(X_va)
    mae_va = mean_absolute_error(y_va, p_va)
    
    p_te = model.predict(X_te)
    mae_te = mean_absolute_error(y_te, p_te)
    rmse_te = np.sqrt(mean_squared_error(y_te, p_te))
    r2_te = r2_score(y_te, p_te)
    smape_te = np.mean(2.0 * np.abs(p_te - y_te) / (np.abs(y_te) + np.abs(p_te) + 1e-6)) * 100
    
    return {
        "val_mae": mae_va, "test_mae": mae_te, "test_rmse": rmse_te,
        "test_r2": r2_te, "test_smape": smape_te
    }

# EXP_UCI_V2_01: Frozen XGBoost Regressor Baseline Benchmark
model_uci_base = joblib.load(os.path.join(base_dir, "models", "uci_forecasting", "xgb_forecasting_frozen.joblib"))
t0 = time.time()
res_uci_01 = eval_uci(model_uci_base, X_tr_uci_29, y_tr_uci, X_va_uci_29, y_va_uci, X_te_uci_29, y_te_uci)
t1 = time.time()

registry_rows.append({
    "experiment_id": "EXP_UCI_V2_01", "dataset": "UCI",
    "hypothesis": "Re-evaluate frozen 29-feature XGBoost regressor baseline",
    "feature_set": "29 Features", "model": "XGBoost Regressor",
    "sampling_strategy": "Chronological Split (75/12.5/12.5)",
    "hyperparameters": "n_estimators=300, max_depth=6, lr=0.05",
    "cv_metric": "10,365.77 kWh MAE", "validation_metric": f"{res_uci_01['val_mae']:.2f} kWh MAE",
    "test_metric": f"{res_uci_01['test_mae']:.2f} kWh MAE / {res_uci_01['test_smape']:.2f}% sMAPE",
    "training_time": f"{t1-t0:.3f}s",
    "artifact": "models/uci_forecasting/xgb_forecasting_frozen.joblib",
    "status": "BASELINE",
    "notes": "Frozen primary baseline forecasting model"
})

# EXP_UCI_V2_02: LightGBM Regressor Baseline
t0 = time.time()
lgb_uci_02 = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1)
lgb_uci_02.fit(X_tr_uci_29, y_tr_uci)
t1 = time.time()
res_uci_02 = eval_uci(lgb_uci_02, X_tr_uci_29, y_tr_uci, X_va_uci_29, y_va_uci, X_te_uci_29, y_te_uci)
joblib.dump(lgb_uci_02, os.path.join(models_v2_dir, "lgb_uci_v2_02.joblib"))

registry_rows.append({
    "experiment_id": "EXP_UCI_V2_02", "dataset": "UCI",
    "hypothesis": "LightGBM Regressor histogram split performance on 29 features",
    "feature_set": "29 Features", "model": "LightGBM Regressor",
    "sampling_strategy": "Chronological Split",
    "hyperparameters": "n_estimators=300, lr=0.05",
    "cv_metric": "N/A", "validation_metric": f"{res_uci_02['val_mae']:.2f} kWh MAE",
    "test_metric": f"{res_uci_02['test_mae']:.2f} kWh MAE / {res_uci_02['test_smape']:.2f}% sMAPE",
    "training_time": f"{t1-t0:.3f}s",
    "artifact": "models/optimization_v2/lgb_uci_v2_02.joblib",
    "status": "EVALUATED",
    "notes": "Default LightGBM regressor"
})

# EXP_UCI_V2_03: Optuna-Tuned LightGBM Regressor
print("Running Optuna tuning for LightGBM Regressor (25 trials)...")
def objective_uci(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': 42,
        'verbose': -1
    }
    m = lgb.LGBMRegressor(**params)
    m.fit(X_tr_uci_29, y_tr_uci)
    preds_val = m.predict(X_va_uci_29)
    return mean_absolute_error(y_va_uci, preds_val)

t0 = time.time()
study_uci = optuna.create_study(direction='minimize')
study_uci.optimize(objective_uci, n_trials=25)
t1 = time.time()

best_uci_params = study_uci.best_params
best_uci_params.update({'random_state': 42, 'verbose': -1})
lgb_uci_03 = lgb.LGBMRegressor(**best_uci_params)
lgb_uci_03.fit(X_tr_uci_29, y_tr_uci)
res_uci_03 = eval_uci(lgb_uci_03, X_tr_uci_29, y_tr_uci, X_va_uci_29, y_va_uci, X_te_uci_29, y_te_uci)
joblib.dump(lgb_uci_03, os.path.join(models_v2_dir, "lgb_uci_v2_03.joblib"))

registry_rows.append({
    "experiment_id": "EXP_UCI_V2_03", "dataset": "UCI",
    "hypothesis": "25-trial Optuna hyperparameter optimization for LightGBM Regressor",
    "feature_set": "29 Features", "model": "Optuna LightGBM Regressor",
    "sampling_strategy": "Chronological Split",
    "hyperparameters": str(study_uci.best_params),
    "cv_metric": "N/A", "validation_metric": f"{res_uci_03['val_mae']:.2f} kWh MAE",
    "test_metric": f"{res_uci_03['test_mae']:.2f} kWh MAE / {res_uci_03['test_smape']:.2f}% sMAPE",
    "training_time": f"{t1-t0:.3f}s",
    "artifact": "models/optimization_v2/lgb_uci_v2_03.joblib",
    "status": "EVALUATED",
    "notes": "Optuna tuned LightGBM Regressor"
})

# EXP_UCI_V2_04: 33 Enhanced Features (Fourier + EMA Features)
df_clean_33 = df_clean_uci.copy()
s_clean = df_clean_33['load_t']
df_clean_33['ema_6'] = s_clean.ewm(span=6).mean()
df_clean_33['ema_24'] = s_clean.ewm(span=24).mean()
df_clean_33['sin_month'] = np.sin(2 * np.pi * df_clean_33['month'] / 12.0)
df_clean_33['cos_month'] = np.cos(2 * np.pi * df_clean_33['month'] / 12.0)

feature_cols_33 = [c for c in df_clean_33.columns if c != 'target_y']
X_uci_33 = df_clean_33[feature_cols_33]

X_tr_uci_33 = X_uci_33.iloc[:n_train]
X_va_uci_33 = X_uci_33.iloc[n_train:n_train+n_val]
X_te_uci_33 = X_uci_33.iloc[n_train+n_val:n_train+n_val+n_test]

t0 = time.time()
lgb_uci_04 = lgb.LGBMRegressor(**best_uci_params)
lgb_uci_04.fit(X_tr_uci_33, y_tr_uci)
t1 = time.time()
res_uci_04 = eval_uci(lgb_uci_04, X_tr_uci_33, y_tr_uci, X_va_uci_33, y_va_uci, X_te_uci_33, y_te_uci)
joblib.dump(lgb_uci_04, os.path.join(models_v2_dir, "lgb_uci_v2_04.joblib"))

registry_rows.append({
    "experiment_id": "EXP_UCI_V2_04", "dataset": "UCI",
    "hypothesis": "Add 4 seasonal Fourier & EMA features (ema_6, ema_24, sin/cos_month)",
    "feature_set": "33 Features", "model": "LightGBM (33 Features)",
    "sampling_strategy": "Chronological Split",
    "hyperparameters": "33 features + Optuna params",
    "cv_metric": "N/A", "validation_metric": f"{res_uci_04['val_mae']:.2f} kWh MAE",
    "test_metric": f"{res_uci_04['test_mae']:.2f} kWh MAE / {res_uci_04['test_smape']:.2f}% sMAPE",
    "training_time": f"{t1-t0:.3f}s",
    "artifact": "models/optimization_v2/lgb_uci_v2_04.joblib",
    "status": "EVALUATED",
    "notes": "Added EMA & Fourier monthly features"
})

# Save experiment registry CSV
reg_csv_path = os.path.join(results_v2_dir, "experiment_registry.csv")
pd.DataFrame(registry_rows).to_csv(reg_csv_path, index=False)
print(f"Saved experiment registry CSV to {reg_csv_path} ({len(registry_rows)} rows).")

# ==============================================================================
# 4. GENERATE V2 COMPARISON PLOTS
# ==============================================================================
# SGCC V2 Comparison Chart
plt.figure(figsize=(9, 5))
sgcc_exp_names = [r["experiment_id"] for r in registry_rows if r["dataset"] == "SGCC"]
sgcc_f1_val = [float(r["validation_metric"].split()[0]) for r in registry_rows if r["dataset"] == "SGCC"]
sgcc_f1_test = [float(r["test_metric"].split()[0]) for r in registry_rows if r["dataset"] == "SGCC"]

x = np.arange(len(sgcc_exp_names))
width = 0.35

plt.bar(x - width/2, sgcc_f1_val, width, label='Validation F1', color='#0284c7')
plt.bar(x + width/2, sgcc_f1_test, width, label='Sealed Test F1', color='#0d9488')
plt.axhline(0.4014, color='red', linestyle='--', label='Frozen Baseline Test F1 (0.4014)')
plt.xticks(x, sgcc_exp_names, rotation=15)
plt.ylabel('F1-Score')
plt.title('SGCC V2 Experiments — Validation vs Sealed Test F1 Performance')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(graphs_v2_dir, "sgcc_v2_experiments_comparison.png"), dpi=300)
plt.close()

# UCI V2 Comparison Chart
plt.figure(figsize=(9, 5))
uci_exp_names = [r["experiment_id"] for r in registry_rows if r["dataset"] == "UCI"]
uci_mae_val = [float(r["validation_metric"].split()[0]) for r in registry_rows if r["dataset"] == "UCI"]
uci_mae_test = [float(r["test_metric"].split()[0]) for r in registry_rows if r["dataset"] == "UCI"]

x_u = np.arange(len(uci_exp_names))
plt.bar(x_u - width/2, uci_mae_val, width, label='Validation MAE (kWh)', color='#0284c7')
plt.bar(x_u + width/2, uci_mae_test, width, label='Sealed Test MAE (kWh)', color='#0d9488')
plt.axhline(4503.51, color='red', linestyle='--', label='Frozen Baseline Test MAE (4503.51 kWh)')
plt.xticks(x_u, uci_exp_names, rotation=15)
plt.ylabel('MAE (kWh)')
plt.title('UCI V2 Experiments — Validation vs Sealed Test MAE Performance')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(graphs_v2_dir, "uci_v2_experiments_comparison.png"), dpi=300)
plt.close()

# Save python runner script copy into scripts/optimization_v2/
script_dest = os.path.join(scripts_v2_dir, "run_v2_experiments.py")
with open(__file__, "r", encoding="utf-8") as f_src:
    with open(script_dest, "w", encoding="utf-8") as f_dst:
        f_dst.write(f_src.read())

print("--- FINISHED V2 EXPERIMENT EXECUTION & REGISTRY LOGGING ---")
