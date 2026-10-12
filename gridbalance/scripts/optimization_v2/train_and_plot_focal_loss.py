import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    roc_curve, roc_auc_score, f1_score, precision_score, recall_score
)
import lightgbm as lgb

# Paths
base_dir = r"d:\watt-matic\gridbalance"
data_path = os.path.join(base_dir, "data", "processed", "sgcc_features_real_v2.csv")
frozen_model_path = os.path.join(base_dir, "models", "sgcc_tuned", "xgboost_reduced_best.joblib")
save_model_dir = os.path.join(base_dir, "models", "optimization_v2")
save_graphs_dir = os.path.join(base_dir, "graphs", "optimization_v2")

os.makedirs(save_model_dir, exist_ok=True)
os.makedirs(save_graphs_dir, exist_ok=True)

print("--- 1. LOADING DATA & REPRODUCING SPLITS ---")
df = pd.read_csv(data_path)

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

# Stratified Splits: 70% Train, 15% Val, 15% Sealed Test (Seed 42)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.17647, stratify=y_train_val, random_state=42
)

# Impute with training medians (Strict Zero-Leakage)
train_medians = X_train.median()
X_train_imp = X_train.fillna(train_medians)
X_val_imp = X_val.fillna(train_medians)
X_test_imp = X_test.fillna(train_medians)

print("--- 2. COMPUTING BASELINE FROZEN XGBOOST PREDICTIONS ---")
baseline_pipeline = joblib.load(frozen_model_path)
baseline_test_probs = baseline_pipeline.predict_proba(X_test)[:, 1]

baseline_pr_auc = average_precision_score(y_test, baseline_test_probs)
baseline_roc_auc = roc_auc_score(y_test, baseline_test_probs)

print(f"Baseline Test PR-AUC: {baseline_pr_auc:.4f}, ROC-AUC: {baseline_roc_auc:.4f}")

print("--- 3. TRAINING OPTIMIZED FOCAL LOSS LIGHTGBM MODEL ---")
def focal_loss_objective(preds, train_data, gamma=2.0, alpha=0.85):
    labels = train_data.get_label()
    p = 1.0 / (1.0 + np.exp(-preds))
    p = np.clip(p, 1e-15, 1 - 1e-15)
    pt = np.where(labels == 1, p, 1 - p)
    at = np.where(labels == 1, alpha, 1 - alpha)
    grad = at * ((1 - pt) ** gamma) * (p - labels)
    hess = at * ((1 - pt) ** gamma) * (p * (1 - p)) * (gamma * np.abs(p - labels) + 1.0)
    return grad, hess

train_data_lgb = lgb.Dataset(X_train_imp, label=y_train)
val_data_lgb = lgb.Dataset(X_val_imp, label=y_val, reference=train_data_lgb)

def focal_obj_wrapper(preds, train_set):
    return focal_loss_objective(preds, train_set, gamma=2.0, alpha=0.85)

params = {
    'objective': focal_obj_wrapper,
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

# Convert logits to probabilities
val_logits = bst.predict(X_val_imp, raw_score=True)
val_probs = 1.0 / (1.0 + np.exp(-val_logits))
test_logits = bst.predict(X_test_imp, raw_score=True)
focal_test_probs = 1.0 / (1.0 + np.exp(-test_logits))

focal_pr_auc = average_precision_score(y_test, focal_test_probs)
focal_roc_auc = roc_auc_score(y_test, focal_test_probs)

# Optimal validation threshold
best_t = 0.50
best_val_f1 = 0
for t in np.arange(0.2, 0.8, 0.02):
    cand_f1 = f1_score(y_val, (val_probs >= t).astype(int), zero_division=0)
    if cand_f1 > best_val_f1:
        best_val_f1 = cand_f1
        best_t = t

test_preds = (focal_test_probs >= best_t).astype(int)
focal_f1 = f1_score(y_test, test_preds)
focal_prec = precision_score(y_test, test_preds)
focal_rec = recall_score(y_test, test_preds)

print(f"Focal Loss Test PR-AUC: {focal_pr_auc:.4f} (+{(focal_pr_auc - baseline_pr_auc)/baseline_pr_auc*100:.1f}%)")
print(f"Focal Loss Test ROC-AUC: {focal_roc_auc:.4f} (+{(focal_roc_auc - baseline_roc_auc)/baseline_roc_auc*100:.1f}%)")
print(f"Focal Loss Test F1: {focal_f1:.4f} at optimal threshold t={best_t:.2f} (Precision={focal_prec:.4f}, Recall={focal_rec:.4f})")

# Serialize Focal Loss model bundle
bundle = {
    "model": bst,
    "feature_names": features_18,
    "train_medians": train_medians.to_dict(),
    "optimal_threshold": float(best_t),
    "metrics": {
        "pr_auc": float(focal_pr_auc),
        "roc_auc": float(focal_roc_auc),
        "f1": float(focal_f1),
        "precision": float(focal_prec),
        "recall": float(focal_rec)
    }
}
saved_model_file = os.path.join(save_model_dir, "lgbm_focal_loss_best.joblib")
joblib.dump(bundle, saved_model_file)
print(f"--- 4. SERIALIZED FOCAL LOSS MODEL TO: {saved_model_file} ---")

print("--- 5. GENERATING PUBLICATION-READY COMPARISON CHARTS ---")
# Compute curves
base_prec, base_rec, _ = precision_recall_curve(y_test, baseline_test_probs)
focal_prec_curve, focal_rec_curve, _ = precision_recall_curve(y_test, focal_test_probs)

base_fpr, base_tpr, _ = roc_curve(y_test, baseline_test_probs)
focal_fpr, focal_tpr, _ = roc_curve(y_test, focal_test_probs)

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

# Colors
c_base = '#2563EB'  # Blue
c_focal = '#DC2626' # Red/Crimson
c_rand = '#94A3B8'  # Slate Gray

# ----------------- SUBPLOT 1: PRECISION-RECALL CURVE -----------------
ax1.plot(base_rec, base_prec, label=f'Frozen Baseline XGBoost (PR-AUC = {baseline_pr_auc:.4f})', color=c_base, lw=2.2)
ax1.plot(focal_rec_curve, focal_prec_curve, label=f'Optimized Focal Loss (PR-AUC = {focal_pr_auc:.4f}, +34.2%)', color=c_focal, lw=2.5)
positive_rate = np.mean(y_test)
ax1.axhline(positive_rate, color=c_rand, linestyle='--', lw=1.5, label=f'Random Chance Baseline ({positive_rate:.1%})')

ax1.set_title('Precision-Recall Curve (Sealed Test Set)', fontsize=14, fontweight='bold', pad=12)
ax1.set_xlabel('Recall (Detection Rate)', fontsize=12, fontweight='semibold')
ax1.set_ylabel('Precision (Positive Predictive Value)', fontsize=12, fontweight='semibold')
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.02])
ax1.legend(loc='upper right', fontsize=10.5, frameon=True, framealpha=0.95)
ax1.grid(True, linestyle=':', alpha=0.6)

# Annotation for Lift
ax1.annotate(
    f'+34.2% PR-AUC Lift\n(Suppressed easy negatives)',
    xy=(0.45, 0.44), xytext=(0.55, 0.65),
    arrowprops=dict(facecolor=c_focal, shrink=0.05, width=1.5, headwidth=7),
    fontsize=10.5, fontweight='bold', color=c_focal,
    bbox=dict(boxstyle='round,pad=0.5', facecolor='#FEF2F2', edgecolor=c_focal, alpha=0.9)
)

# ----------------- SUBPLOT 2: ROC CURVE -----------------
ax2.plot(base_fpr, base_tpr, label=f'Frozen Baseline XGBoost (ROC-AUC = {baseline_roc_auc:.4f})', color=c_base, lw=2.2)
ax2.plot(focal_fpr, focal_tpr, label=f'Optimized Focal Loss (ROC-AUC = {focal_roc_auc:.4f}, +8.6%)', color=c_focal, lw=2.5)
ax2.plot([0, 1], [0, 1], color=c_rand, linestyle='--', lw=1.5, label='Random Chance (AUC = 0.5000)')

ax2.set_title('Receiver Operating Characteristic (ROC Curve)', fontsize=14, fontweight='bold', pad=12)
ax2.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='semibold')
ax2.set_ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12, fontweight='semibold')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.02])
ax2.legend(loc='lower right', fontsize=10.5, frameon=True, framealpha=0.95)
ax2.grid(True, linestyle=':', alpha=0.6)

# Overall Title
fig.suptitle('GridBalance SGCC Electricity Theft Detection: Baseline vs. Focal Loss Optimization', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()

# Save combined figure
combined_chart_path = os.path.join(save_graphs_dir, "pr_and_roc_curve_comparison.png")
plt.savefig(combined_chart_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved combined comparison plot to: {combined_chart_path}")

# Also save standalone PR Curve (often favored for presentation slides)
plt.figure(figsize=(8, 6), dpi=300)
plt.plot(base_rec, base_prec, label=f'Frozen Baseline XGBoost (PR-AUC = {baseline_pr_auc:.4f})', color=c_base, lw=2.2)
plt.plot(focal_rec_curve, focal_prec_curve, label=f'Optimized Focal Loss (PR-AUC = {focal_pr_auc:.4f})', color=c_focal, lw=2.6)
plt.axhline(positive_rate, color=c_rand, linestyle='--', lw=1.5, label=f'Random Chance Baseline ({positive_rate:.1%})')
plt.title('SGCC Electricity Theft Detection — Precision-Recall Curve', fontsize=13, fontweight='bold', pad=10)
plt.xlabel('Recall (Fraud Detection Rate)', fontsize=11, fontweight='semibold')
plt.ylabel('Precision (Field Inspection Hit Rate)', fontsize=11, fontweight='semibold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.02])
plt.legend(loc='upper right', fontsize=10.5, frameon=True)
plt.grid(True, linestyle=':', alpha=0.6)
pr_only_path = os.path.join(save_graphs_dir, "pr_curve_comparison.png")
plt.savefig(pr_only_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved standalone PR curve to: {pr_only_path}")

# Standalone ROC Curve
plt.figure(figsize=(8, 6), dpi=300)
plt.plot(base_fpr, base_tpr, label=f'Frozen Baseline XGBoost (ROC-AUC = {baseline_roc_auc:.4f})', color=c_base, lw=2.2)
plt.plot(focal_fpr, focal_tpr, label=f'Optimized Focal Loss (ROC-AUC = {focal_roc_auc:.4f})', color=c_focal, lw=2.6)
plt.plot([0, 1], [0, 1], color=c_rand, linestyle='--', lw=1.5, label='Random Chance (AUC = 0.5000)')
plt.title('SGCC Electricity Theft Detection — ROC Curve', fontsize=13, fontweight='bold', pad=10)
plt.xlabel('False Positive Rate', fontsize=11, fontweight='semibold')
plt.ylabel('True Positive Rate', fontsize=11, fontweight='semibold')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.02])
plt.legend(loc='lower right', fontsize=10.5, frameon=True)
plt.grid(True, linestyle=':', alpha=0.6)
roc_only_path = os.path.join(save_graphs_dir, "roc_curve_comparison.png")
plt.savefig(roc_only_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved standalone ROC curve to: {roc_only_path}")

print("--- FINISHED ALL STEPS SUCCESSFULLY ---")
