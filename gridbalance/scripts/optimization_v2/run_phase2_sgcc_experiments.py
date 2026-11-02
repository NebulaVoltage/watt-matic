import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, fbeta_score, roc_auc_score,
    average_precision_score, matthews_corrcoef, confusion_matrix
)
import lightgbm as lgb
import xgboost as xgb

base_dir = r"D:\watt-matic\gridbalance"
data_path = os.path.join(base_dir, "data", "processed", "sgcc_features_real_v2.csv")
focal_model_path = os.path.join(base_dir, "models", "optimization_v2", "lgbm_focal_loss_best.joblib")
frozen_xgb_path = os.path.join(base_dir, "models", "sgcc_tuned", "xgboost_reduced_best.joblib")
out_graphs = os.path.join(base_dir, "graphs", "optimization_v2", "phase2")
out_results = os.path.join(base_dir, "results", "optimization_v2")
os.makedirs(out_graphs, exist_ok=True)
os.makedirs(out_results, exist_ok=True)

print("=== 1. LOADING SGCC DATA AND SPLITTING ===")
df = pd.read_csv(data_path)
features_18 = [
    "missing_streak_count", "monthly_std_4", "monthly_std_3",
    "mean_abs_daily_change", "missing_count", "monthly_mean_10",
    "longest_missing_streak", "peak_to_average_ratio", "monthly_std_5",
    "monthly_std_10", "monthly_std_11", "cv",
    "monthly_mean_8", "max", "monthly_std_8",
    "monthly_mean_11", "monthly_std_1", "monthly_mean_2"
]
missing_features = ["missing_count", "missing_streak_count", "longest_missing_streak"]
consumption_features = [f for f in features_18 if f not in missing_features]
y = df["FLAG"].astype(int).values

X_indices = np.arange(len(y))
idx_train_val, idx_test, y_train_val, y_test = train_test_split(
    X_indices, y, test_size=0.15, stratify=y, random_state=42
)
idx_train, idx_val, y_train, y_val = train_test_split(
    idx_train_val, y_train_val, test_size=0.17647, stratify=y_train_val, random_state=42
)

# Impute with training medians strictly
train_medians = df.iloc[idx_train][features_18].median()
df_imp = df[features_18].fillna(train_medians)

X_train = df_imp.iloc[idx_train]
X_val = df_imp.iloc[idx_val]
X_test = df_imp.iloc[idx_test]

print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")

print("\n=== EXPERIMENT 1: FOCAL LOSS RECALCULATION AND COMPARISON ===")
focal_bundle = joblib.load(focal_model_path)
bst = focal_bundle["model"]
xgb_pipeline = joblib.load(frozen_xgb_path)

# Verify probability conversion on Test
focal_test_logits = bst.predict(X_test, raw_score=True)
focal_test_probs = 1.0 / (1.0 + np.exp(-focal_test_logits))
xgb_test_probs = xgb_pipeline.predict_proba(df.iloc[idx_test][features_18])[:, 1]

t = 0.50
focal_pred_50 = (focal_test_probs >= t).astype(int)
xgb_pred_50 = (xgb_test_probs >= t).astype(int)

exp1_results = {
    "Model": ["Frozen Baseline XGBoost", "LightGBM Focal Loss (gamma=2.0)"],
    "Sealed_Test_F1": [f1_score(y_test, xgb_pred_50), f1_score(y_test, focal_pred_50)],
    "Sealed_Test_PR_AUC": [average_precision_score(y_test, xgb_test_probs), average_precision_score(y_test, focal_test_probs)],
    "Sealed_Test_ROC_AUC": [roc_auc_score(y_test, xgb_test_probs), roc_auc_score(y_test, focal_test_probs)],
    "Sealed_Test_Precision": [precision_score(y_test, xgb_pred_50), precision_score(y_test, focal_pred_50)],
    "Sealed_Test_Recall": [recall_score(y_test, xgb_pred_50), recall_score(y_test, focal_pred_50)],
    "Sealed_Test_MCC": [matthews_corrcoef(y_test, xgb_pred_50), matthews_corrcoef(y_test, focal_pred_50)]
}
df_exp1 = pd.DataFrame(exp1_results)
print(df_exp1.to_string(index=False))

# 5-Fold Stratified CV on Training partition
print("\nRunning 5-Fold Stratified CV on Training Data...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def focal_loss_objective(preds, train_data, gamma=2.0, alpha=0.85):
    labels = train_data.get_label()
    p = 1.0 / (1.0 + np.exp(-preds))
    p = np.clip(p, 1e-15, 1 - 1e-15)
    pt = np.where(labels == 1, p, 1 - p)
    at = np.where(labels == 1, alpha, 1 - alpha)
    grad = at * ((1 - pt) ** gamma) * (p - labels)
    hess = at * ((1 - pt) ** gamma) * (p * (1 - p)) * (gamma * np.abs(p - labels) + 1.0)
    return grad, hess

focal_cv_f1, focal_cv_pr, focal_cv_roc = [], [], []
xgb_cv_f1, xgb_cv_pr, xgb_cv_roc = [], [], []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr_f, y_tr_f = X_train.iloc[tr_idx], y_train[tr_idx]
    X_va_f, y_va_f = X_train.iloc[val_idx], y_train[val_idx]
    
    # Train Focal Loss
    dtrain = lgb.Dataset(X_tr_f, label=y_tr_f)
    dval = lgb.Dataset(X_va_f, label=y_va_f, reference=dtrain)
    params = {
        "objective": focal_loss_objective,
        "metric": "auc",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "random_state": 42 + fold
    }
    m_focal = lgb.train(params, dtrain, num_boost_round=250, valid_sets=[dval], callbacks=[lgb.early_stopping(20, verbose=False)])
    val_logits = m_focal.predict(X_va_f, raw_score=True)
    val_probs = 1.0 / (1.0 + np.exp(-val_logits))
    focal_cv_f1.append(f1_score(y_va_f, (val_probs >= 0.50).astype(int)))
    focal_cv_pr.append(average_precision_score(y_va_f, val_probs))
    focal_cv_roc.append(roc_auc_score(y_va_f, val_probs))
    
    # Train XGBoost Baseline
    m_xgb = xgb.XGBClassifier(
        n_estimators=200, max_depth=7, learning_rate=0.10,
        scale_pos_weight=10.72, subsample=0.7, colsample_bytree=0.7,
        reg_lambda=100, reg_alpha=1, random_state=42 + fold, n_jobs=-1
    )
    m_xgb.fit(X_tr_f, y_tr_f)
    xgb_probs = m_xgb.predict_proba(X_va_f)[:, 1]
    xgb_cv_f1.append(f1_score(y_va_f, (xgb_probs >= 0.50).astype(int)))
    xgb_cv_pr.append(average_precision_score(y_va_f, xgb_probs))
    xgb_cv_roc.append(roc_auc_score(y_va_f, xgb_probs))

print(f"XGBoost CV F1: {np.mean(xgb_cv_f1):.4f} +/- {np.std(xgb_cv_f1):.4f} | PR-AUC: {np.mean(xgb_cv_pr):.4f} | ROC-AUC: {np.mean(xgb_cv_roc):.4f}")
print(f"Focal Loss CV F1: {np.mean(focal_cv_f1):.4f} +/- {np.std(focal_cv_f1):.4f} | PR-AUC: {np.mean(focal_cv_pr):.4f} | ROC-AUC: {np.mean(focal_cv_roc):.4f}")

# Save Exp 1 verification CSV
df_exp1_summary = pd.DataFrame([
    {"Model": "Frozen Baseline XGBoost", "CV_F1": f"{np.mean(xgb_cv_f1):.4f} +/- {np.std(xgb_cv_f1):.4f}", "CV_PR_AUC": np.mean(xgb_cv_pr), "CV_ROC_AUC": np.mean(xgb_cv_roc), "Test_F1": f1_score(y_test, xgb_pred_50), "Test_PR_AUC": average_precision_score(y_test, xgb_test_probs), "Test_ROC_AUC": roc_auc_score(y_test, xgb_test_probs), "Test_Precision": precision_score(y_test, xgb_pred_50), "Test_Recall": recall_score(y_test, xgb_pred_50)},
    {"Model": "LightGBM Focal Loss (gamma=2.0)", "CV_F1": f"{np.mean(focal_cv_f1):.4f} +/- {np.std(focal_cv_f1):.4f}", "CV_PR_AUC": np.mean(focal_cv_pr), "CV_ROC_AUC": np.mean(focal_cv_roc), "Test_F1": f1_score(y_test, focal_pred_50), "Test_PR_AUC": average_precision_score(y_test, focal_test_probs), "Test_ROC_AUC": roc_auc_score(y_test, focal_test_probs), "Test_Precision": precision_score(y_test, focal_pred_50), "Test_Recall": recall_score(y_test, focal_pred_50)}
])
df_exp1_summary.to_csv(os.path.join(out_results, "sgcc_exp1_focal_verification.csv"), index=False)

print("\n=== EXPERIMENT 2: MISSINGNESS DEPENDENCY ANALYSIS ===")
# Evaluate Partitions: Consumption-only (15), Missingness-only (3), Combined (18)
def evaluate_partition(feature_subset, name):
    m = xgb.XGBClassifier(
        n_estimators=200, max_depth=7, learning_rate=0.10,
        scale_pos_weight=10.72, subsample=0.7, colsample_bytree=0.7,
        reg_lambda=100, reg_alpha=1, random_state=42, n_jobs=-1
    )
    m.fit(X_train[feature_subset], y_train)
    val_probs = m.predict_proba(X_val[feature_subset])[:, 1]
    test_probs = m.predict_proba(X_test[feature_subset])[:, 1]
    
    val_pred = (val_probs >= 0.50).astype(int)
    test_pred = (test_probs >= 0.50).astype(int)
    
    return {
        "Partition": name,
        "Feature_Count": len(feature_subset),
        "Val_F1": f1_score(y_val, val_pred),
        "Val_PR_AUC": average_precision_score(y_val, val_probs),
        "Val_ROC_AUC": roc_auc_score(y_val, val_probs),
        "Val_Precision": precision_score(y_val, val_pred),
        "Val_Recall": recall_score(y_val, val_pred),
        "Val_MCC": matthews_corrcoef(y_val, val_pred),
        "Test_F1": f1_score(y_test, test_pred),
        "Test_PR_AUC": average_precision_score(y_test, test_probs),
        "Test_ROC_AUC": roc_auc_score(y_test, test_probs),
        "Test_Precision": precision_score(y_test, test_pred),
        "Test_Recall": recall_score(y_test, test_pred),
        "Test_MCC": matthews_corrcoef(y_test, test_pred)
    }

res_cons = evaluate_partition(consumption_features, "Consumption-Only (15 Features)")
res_miss = evaluate_partition(missing_features, "Missingness-Only (3 Features)")
res_comb = evaluate_partition(features_18, "Combined Baseline (18 Features)")

df_exp2 = pd.DataFrame([res_cons, res_miss, res_comb])
print(df_exp2.to_string(index=False))
df_exp2.to_csv(os.path.join(out_results, "sgcc_exp2_missingness_ablation.csv"), index=False)

# Coverage Groups Audit on Validation Set
val_raw = df.iloc[idx_val]
missing_pcts = val_raw["missing_count"] / 1034.0
cov_high = missing_pcts < 0.10
cov_med = (missing_pcts >= 0.10) & (missing_pcts < 0.30)
cov_low = missing_pcts >= 0.30

# Predict using combined model
m_comb = xgb.XGBClassifier(
    n_estimators=200, max_depth=7, learning_rate=0.10,
    scale_pos_weight=10.72, subsample=0.7, colsample_bytree=0.7,
    reg_lambda=100, reg_alpha=1, random_state=42, n_jobs=-1
)
m_comb.fit(X_train[features_18], y_train)
val_probs = m_comb.predict_proba(X_val[features_18])[:, 1]
val_pred = (val_probs >= 0.50).astype(int)

def group_metrics(mask, name):
    y_g = y_val[mask]
    p_g = val_pred[mask]
    prob_g = val_probs[mask]
    return {
        "Coverage_Tier": name,
        "Customer_Count": int(np.sum(mask)),
        "Theft_Count": int(np.sum(y_g == 1)),
        "Theft_Prevalence": np.mean(y_g),
        "F1_Score": f1_score(y_g, p_g),
        "PR_AUC": average_precision_score(y_g, prob_g),
        "Precision": precision_score(y_g, p_g, zero_division=0),
        "Recall": recall_score(y_g, p_g, zero_division=0)
    }

df_cov = pd.DataFrame([
    group_metrics(cov_high, "High Coverage (<10% Missing)"),
    group_metrics(cov_med, "Medium Coverage (10-30% Missing)"),
    group_metrics(cov_low, "Low Coverage (>=30% Missing)")
])
print("\n--- Coverage Group Performance on Validation Set ---")
print(df_cov.to_string(index=False))
df_cov.to_csv(os.path.join(out_results, "sgcc_exp2_coverage_groups.csv"), index=False)

# Plot Missingness Ablation
plt.figure(figsize=(10, 5), dpi=300)
metrics_to_plot = ["Val_F1", "Val_PR_AUC", "Val_ROC_AUC", "Val_Precision", "Val_Recall"]
df_plot = df_exp2.melt(id_vars=["Partition"], value_vars=metrics_to_plot, var_name="Metric", value_name="Score")
sns.barplot(data=df_plot, x="Metric", y="Score", hue="Partition", palette="Blues")
plt.title("SGCC Feature Partition Ablation (Validation Set)", fontsize=13, fontweight="bold")
plt.ylim(0, 1.0)
plt.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(out_graphs, "sgcc_missingness_ablation.png"))
plt.close()

print("\n=== EXPERIMENT 3: SGCC OPERATIONAL THRESHOLD ANALYSIS ===")
# On Validation set, evaluate thresholds 0.05 to 0.95
thresholds = np.arange(0.05, 0.96, 0.05)
thresh_records = []

# Using Focal Loss model on validation
val_logits_focal = bst.predict(X_val, raw_score=True)
val_probs_focal = 1.0 / (1.0 + np.exp(-val_logits_focal))

for t in thresholds:
    p_t = (val_probs_focal >= t).astype(int)
    p = precision_score(y_val, p_t, zero_division=0)
    r = recall_score(y_val, p_t, zero_division=0)
    f1 = f1_score(y_val, p_t, zero_division=0)
    f2 = fbeta_score(y_val, p_t, beta=2, zero_division=0)
    mcc = matthews_corrcoef(y_val, p_t)
    flagged = int(np.sum(p_t))
    
    thresh_records.append({
        "Threshold": round(t, 2),
        "Precision": round(p, 4),
        "Recall": round(r, 4),
        "F1": round(f1, 4),
        "F2": round(f2, 4),
        "MCC": round(mcc, 4),
        "Flagged_Count": flagged,
        "Flagged_Pct": round((flagged / len(y_val)) * 100, 2)
    })

df_thresh = pd.DataFrame(thresh_records)
df_thresh.to_csv(os.path.join(out_results, "sgcc_exp3_threshold_curves.csv"), index=False)

# Identify Operating Points
best_f1_row = df_thresh.loc[df_thresh["F1"].idxmax()]
# High-Recall: highest F1 subject to Recall >= 0.70
high_rec_candidates = df_thresh[df_thresh["Recall"] >= 0.70]
best_high_rec = high_rec_candidates.loc[high_rec_candidates["F1"].idxmax()]
# High-Precision: highest F1 subject to Precision >= 0.50
high_prec_candidates = df_thresh[df_thresh["Precision"] >= 0.50]
best_high_prec = high_prec_candidates.loc[high_prec_candidates["F1"].idxmax()]

operating_points = pd.DataFrame([
    {"Operating_Mode": "F1-Optimal Balanced", "Threshold": best_f1_row["Threshold"], "Precision": best_f1_row["Precision"], "Recall": best_f1_row["Recall"], "F1": best_f1_row["F1"], "F2": best_f1_row["F2"], "MCC": best_f1_row["MCC"], "Flagged_Pct": best_f1_row["Flagged_Pct"]},
    {"Operating_Mode": "High-Recall (Fraud Sweep)", "Threshold": best_high_rec["Threshold"], "Precision": best_high_rec["Precision"], "Recall": best_high_rec["Recall"], "F1": best_high_rec["F1"], "F2": best_high_rec["F2"], "MCC": best_high_rec["MCC"], "Flagged_Pct": best_high_rec["Flagged_Pct"]},
    {"Operating_Mode": "High-Precision (Targeted Audits)", "Threshold": best_high_prec["Threshold"], "Precision": best_high_prec["Precision"], "Recall": best_high_prec["Recall"], "F1": best_high_prec["F1"], "F2": best_high_prec["F2"], "MCC": best_high_prec["MCC"], "Flagged_Pct": best_high_prec["Flagged_Pct"]}
])
print("\n--- Operational Operating Points (Validation Set) ---")
print(operating_points.to_string(index=False))
operating_points.to_csv(os.path.join(out_results, "sgcc_exp3_operating_points.csv"), index=False)

# Plot Threshold curves
plt.figure(figsize=(9, 5), dpi=300)
plt.plot(df_thresh["Threshold"], df_thresh["Precision"], label="Precision", color="green", lw=2)
plt.plot(df_thresh["Threshold"], df_thresh["Recall"], label="Recall", color="blue", lw=2)
plt.plot(df_thresh["Threshold"], df_thresh["F1"], label="F1-Score", color="red", lw=2.5)
plt.plot(df_thresh["Threshold"], df_thresh["F2"], label="F2-Score (Recall-Weighted)", color="purple", lw=1.8, linestyle="--")
plt.axvline(best_f1_row["Threshold"], color="gray", linestyle=":", label=f"Optimal F1 (t={best_f1_row['Threshold']})")
plt.xlabel("Decision Threshold", fontsize=11, fontweight="semibold")
plt.ylabel("Metric Value", fontsize=11, fontweight="semibold")
plt.title("SGCC Operational Threshold Operating Curves (Validation Set)", fontsize=13, fontweight="bold")
plt.legend(loc="best")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_graphs, "sgcc_threshold_curves.png"))
plt.close()

# Cost Sensitivity Analysis
# Parametric Cost: Total Cost = FP * C_inspect + FN * C_theft
# Normalized by C_theft: Relative Cost = FP * (C_inspect / C_theft) + FN
cost_ratios = [0.05, 0.10, 0.20, 0.35, 0.50]
plt.figure(figsize=(9, 5), dpi=300)
for r in cost_ratios:
    costs = []
    for t in df_thresh["Threshold"]:
        p_t = (val_probs_focal >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, p_t).ravel()
        cost = fp * r + fn * 1.0
        costs.append(cost)
    plt.plot(df_thresh["Threshold"], costs, label=f"Cost Ratio r = {r}", lw=2)

plt.xlabel("Decision Threshold", fontsize=11, fontweight="semibold")
plt.ylabel("Normalized Loss (FP * r + FN)", fontsize=11, fontweight="semibold")
plt.title("Cost-Sensitivity Operating Curve across Inspection Cost Ratios", fontsize=13, fontweight="bold")
plt.legend(loc="upper center")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_graphs, "sgcc_cost_sensitivity.png"))
plt.close()

print("\n--- FINISHED ALL SGCC EXPERIMENTS SUCCESSFULLY ---")
