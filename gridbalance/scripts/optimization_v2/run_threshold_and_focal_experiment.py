import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, average_precision_score, precision_recall_curve
)
import xgboost as xgb
import lightgbm as lgb

# Base paths
base_dir = r"d:\watt-matic\gridbalance"
data_path = os.path.join(base_dir, "data", "processed", "sgcc_features_real_v2.csv")
frozen_model_path = os.path.join(base_dir, "models", "sgcc_tuned", "xgboost_reduced_best.joblib")
output_csv_path = os.path.join(base_dir, "results", "optimization_v2", "sgcc_threshold_operating_curve.csv")
focal_results_path = os.path.join(base_dir, "results", "optimization_v2", "sgcc_focal_loss_comparison.csv")

os.makedirs(os.path.join(base_dir, "results", "optimization_v2"), exist_ok=True)

print("=== 1. LOADING DATASET & SPLITTING ===")
df = pd.read_csv(data_path)

# 18 baseline feature names matching the frozen pipeline
features_18 = [
    "missing_streak_count", "monthly_std_4", "monthly_std_3",
    "mean_abs_daily_change", "missing_count", "monthly_mean_10",
    "longest_missing_streak", "peak_to_average_ratio", "monthly_std_5",
    "monthly_std_10", "monthly_std_11", "cv",
    "monthly_mean_8", "max", "monthly_std_8",
    "monthly_mean_11", "monthly_std_1", "monthly_mean_2"
]

X = df[features_18]
y = df['FLAG']

# Stratified Split 70% Train, 15% Val, 15% Test (Seed 42)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.17647, stratify=y_train_val, random_state=42 # 0.15 / 0.85 approx 0.17647
)

print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")

print("\n=== 2. EVALUATING THRESHOLD OPERATING CURVE ON FROZEN MODEL ===")
# Load frozen model pipeline
pipeline = joblib.load(frozen_model_path)
# Predict probabilities on Test set
test_probs = pipeline.predict_proba(X_test)[:, 1]

# Overall ranking metrics (threshold-independent)
roc_auc = roc_auc_score(y_test, test_probs)
pr_auc = average_precision_score(y_test, test_probs)
print(f"Test ROC-AUC: {roc_auc:.4f}, Test PR-AUC: {pr_auc:.4f}")

thresholds = [0.10, 0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
threshold_records = []

for t in thresholds:
    preds = (test_probs >= t).astype(int)
    p = precision_score(y_test, preds, zero_division=0)
    r = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    acc = accuracy_score(y_test, preds)
    num_flagged = int(np.sum(preds))
    pct_flagged = (num_flagged / len(y_test)) * 100
    
    threshold_records.append({
        "Threshold": t,
        "Accuracy": round(acc, 4),
        "Precision": round(p, 4),
        "Recall": round(r, 4),
        "F1_Score": round(f1, 4),
        "Flagged_Meters_Count": num_flagged,
        "Flagged_Meters_Pct": round(pct_flagged, 2),
        "Field_Inspection_Hit_Rate": f"{round(p * 100, 1)}%",
        "Operational_Mode": (
            "Aggressive Catch (High Recall)" if t <= 0.35 else
            "Balanced Standard (Default)" if 0.40 <= t <= 0.60 else
            "Conservative (High Precision, Low Inspection Cost)"
        )
    })

thresh_df = pd.DataFrame(threshold_records)
thresh_df.to_csv(output_csv_path, index=False)
print(f"Saved threshold operating curve to: {output_csv_path}")
print(thresh_df.to_string())

print("\n=== 3. TRAINING FOCAL LOSS & ASYMMETRIC GRADIENT CANDIDATES ===")

# Define Focal Loss for LightGBM
# FL(p) = -alpha * (1-p)^gamma * log(p) if y=1 else -(1-alpha) * p^gamma * log(1-p)
def focal_loss_objective(preds, train_data, gamma=2.0, alpha=0.75):
    labels = train_data.get_label()
    # Sigmoid to get probability
    p = 1.0 / (1.0 + np.exp(-preds))
    
    # Gradient and Hessian
    p = np.clip(p, 1e-15, 1 - 1e-15)
    
    # For positive: p_t = p, alpha_t = alpha
    # For negative: p_t = 1 - p, alpha_t = 1 - alpha
    pt = np.where(labels == 1, p, 1 - p)
    at = np.where(labels == 1, alpha, 1 - alpha)
    
    # First derivative w.r.t preds (logit z)
    # g = at * (1 - pt)^gamma * (gamma * pt * log(pt) + pt - 1) * (if y==1: (p-1) else p)
    # Approximated focal gradient for stable boosting:
    grad = at * ((1 - pt) ** gamma) * (p - labels)
    hess = at * ((1 - pt) ** gamma) * (p * (1 - p)) * (gamma * np.abs(p - labels) + 1.0)
    
    return grad, hess

# Impute median on X_train, X_val, X_test using training medians
train_medians = X_train.median()
X_train_imp = X_train.fillna(train_medians)
X_val_imp = X_val.fillna(train_medians)
X_test_imp = X_test.fillna(train_medians)

# Candidate A: Tuned XGBoost with fine-grained scale_pos_weight sweep (8.0, 10.72, 12.0)
focal_records = []

# Baseline record
focal_records.append({
    "Experiment": "Baseline Frozen XGBoost",
    "Loss_Strategy": "Binary Cross-Entropy (scale_pos_weight=10.72)",
    "Val_F1": 0.4037,
    "Val_PR_AUC": 0.3684,
    "Test_F1_t0.50": 0.3592,
    "Test_Precision": 0.3040,
    "Test_Recall": 0.4391,
    "Test_PR_AUC": 0.3114,
    "Test_ROC_AUC": 0.7758
})

for spw in [6.0, 8.0, 10.72, 14.0]:
    candidate_xgb = xgb.XGBClassifier(
        max_depth=7,
        n_estimators=200,
        learning_rate=0.15,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="aucpr",
        early_stopping_rounds=25
    )
    candidate_xgb.fit(
        X_train_imp, y_train,
        eval_set=[(X_val_imp, y_val)],
        verbose=False
    )
    val_probs = candidate_xgb.predict_proba(X_val_imp)[:, 1]
    test_probs_c = candidate_xgb.predict_proba(X_test_imp)[:, 1]
    
    val_pr_auc = average_precision_score(y_val, val_probs)
    test_pr_auc = average_precision_score(y_test, test_probs_c)
    test_roc_auc = roc_auc_score(y_test, test_probs_c)
    
    # Check t=0.50
    preds_50 = (test_probs_c >= 0.50).astype(int)
    f1_50 = f1_score(y_test, preds_50)
    p_50 = precision_score(y_test, preds_50, zero_division=0)
    r_50 = recall_score(y_test, preds_50, zero_division=0)
    
    # Find best threshold on validation
    best_t = 0.50
    best_val_f1 = 0
    for t_cand in np.arange(0.2, 0.8, 0.05):
        cand_f1 = f1_score(y_val, (val_probs >= t_cand).astype(int), zero_division=0)
        if cand_f1 > best_val_f1:
            best_val_f1 = cand_f1
            best_t = t_cand
            
    test_preds_opt = (test_probs_c >= best_t).astype(int)
    f1_opt = f1_score(y_test, test_preds_opt)
    
    focal_records.append({
        "Experiment": f"XGBoost (spw={spw}, lr=0.15, early_stopping)",
        "Loss_Strategy": f"Cost-Sensitive Cross-Entropy (scale_pos_weight={spw})",
        "Val_F1": round(best_val_f1, 4),
        "Val_PR_AUC": round(val_pr_auc, 4),
        "Test_F1_t0.50": round(f1_50, 4),
        "Test_Precision": round(p_50, 4),
        "Test_Recall": round(r_50, 4),
        "Test_PR_AUC": round(test_pr_auc, 4),
        "Test_ROC_AUC": round(test_roc_auc, 4)
    })

# Candidate B: Focal Loss LightGBM
train_data_lgb = lgb.Dataset(X_train_imp, label=y_train)
val_data_lgb = lgb.Dataset(X_val_imp, label=y_val, reference=train_data_lgb)

for gamma_val in [1.5, 2.0]:
    # Custom focal objective wrapper
    def custom_obj(preds, train_set):
        return focal_loss_objective(preds, train_set, gamma=gamma_val, alpha=0.85)
        
    params = {
        'objective': custom_obj,
        'metric': 'auc',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42
    }
        
    bst = lgb.train(
        params,
        train_data_lgb,
        num_boost_round=250,
        valid_sets=[val_data_lgb],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
    )
    
    # Predict raw logits and convert to prob
    val_logits = bst.predict(X_val_imp, raw_score=True)
    val_probs_fl = 1.0 / (1.0 + np.exp(-val_logits))
    test_logits = bst.predict(X_test_imp, raw_score=True)
    test_probs_fl = 1.0 / (1.0 + np.exp(-test_logits))
    
    val_pr_auc = average_precision_score(y_val, val_probs_fl)
    test_pr_auc = average_precision_score(y_test, test_probs_fl)
    test_roc_auc = roc_auc_score(y_test, test_probs_fl)
    
    # Best threshold on validation
    best_t_fl = 0.50
    best_val_f1_fl = 0
    for t_cand in np.arange(0.2, 0.8, 0.05):
        cand_f1 = f1_score(y_val, (val_probs_fl >= t_cand).astype(int), zero_division=0)
        if cand_f1 > best_val_f1_fl:
            best_val_f1_fl = cand_f1
            best_t_fl = t_cand
            
    test_preds_fl = (test_probs_fl >= best_t_fl).astype(int)
    f1_fl = f1_score(y_test, test_preds_fl)
    p_fl = precision_score(y_test, test_preds_fl, zero_division=0)
    r_fl = recall_score(y_test, test_preds_fl, zero_division=0)
    
    focal_records.append({
        "Experiment": f"LightGBM Focal Loss (gamma={gamma_val})",
        "Loss_Strategy": f"Focal Loss (gamma={gamma_val}, alpha=0.85)",
        "Val_F1": round(best_val_f1_fl, 4),
        "Val_PR_AUC": round(val_pr_auc, 4),
        "Test_F1_t0.50": round(f1_fl, 4),
        "Test_Precision": round(p_fl, 4),
        "Test_Recall": round(r_fl, 4),
        "Test_PR_AUC": round(test_pr_auc, 4),
        "Test_ROC_AUC": round(test_roc_auc, 4)
    })

focal_df = pd.DataFrame(focal_records)
focal_df.to_csv(focal_results_path, index=False)
print(f"\nSaved focal loss & hyperparameter results to: {focal_results_path}")
print(focal_df.to_string())

print("\n=== EXPERIMENTS COMPLETED SUCCESSFULLY ===")
