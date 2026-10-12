import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, precision_recall_curve, matthews_corrcoef
)
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesClassifier

base_dir = r"d:\watt-matic\gridbalance"
data_path = os.path.join(base_dir, "data", "processed", "sgcc_features_real_v2.csv")
reports_dir = os.path.join(base_dir, "reports", "optimization_v2")
graphs_dir = os.path.join(base_dir, "graphs", "optimization_v2", "sgcc_dataset_diagnostics")

os.makedirs(reports_dir, exist_ok=True)
os.makedirs(graphs_dir, exist_ok=True)

print("="*70)
print("GRIDBALANCE SGCC DATASET LIMITATION & PERFORMANCE DIAGNOSTIC MASTER SUITE")
print("="*70)

# Load data
df = pd.read_csv(data_path)
print(f"Loaded processed dataset: {df.shape[0]} rows, {df.shape[1]} columns")

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
all_42_features = [c for c in df.columns if c not in ["CONS_NO", "FLAG"]]

y = df["FLAG"].astype(int).values

# 70% Train, 15% Val, 15% Sealed Test (Seed 42)
X_indices = np.arange(len(y))
idx_train_val, idx_test, y_train_val, y_test = train_test_split(
    X_indices, y, test_size=0.15, stratify=y, random_state=42
)
idx_train, idx_val, y_train, y_val = train_test_split(
    idx_train_val, y_train_val, test_size=0.17647, stratify=y_train_val, random_state=42
)

print(f"Dataset split sizes: Train={len(idx_train)}, Val={len(idx_val)}, Sealed Test={len(idx_test)}")
print(f"Positive prevalence: Train={np.mean(y_train):.4f}, Val={np.mean(y_val):.4f}, Sealed Test={np.mean(y_test):.4f}")

# Train imputation medians strictly on training fold
train_medians = df.iloc[idx_train][all_42_features].median()
df_imp = df.copy()
df_imp[all_42_features] = df_imp[all_42_features].fillna(train_medians)

# ------------------------------------------------------------------------------
# PHASE 1: DATASET AUDIT METRICS GATHERING
# ------------------------------------------------------------------------------
print("\n--- RUNNING PHASE 1: AUDIT OF DATASET ---")
total_rows = len(df)
pos_count = int(np.sum(y == 1))
neg_count = int(np.sum(y == 0))
pos_prev = pos_count / total_rows

# Check missingness in df before imputation
missing_cells = df[all_42_features].isna().sum().sum()
total_cells = len(df) * len(all_42_features)
pct_missing_cells = (missing_cells / total_cells) * 100

print(f"Total samples: {total_rows}")
print(f"Negatives (0): {neg_count} ({neg_count/total_rows*100:.2f}%), Positives (1): {pos_count} ({pos_prev*100:.2f}%)")
print(f"Missing values across 42 features: {missing_cells} / {total_cells} ({pct_missing_cells:.2f}%)")

# ------------------------------------------------------------------------------
# PHASE 2: CLASS SEPARABILITY AUDIT
# ------------------------------------------------------------------------------
print("\n--- RUNNING PHASE 2: CLASS SEPARABILITY AUDIT ---")
df_train = df_imp.iloc[idx_train]
y_tr = y_train

separability_rows = []

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(6, 3, figsize=(18, 24))
axes = axes.flatten()

for idx, feat in enumerate(features_18):
    vals_0 = df_train[df_train["FLAG"] == 0][feat].values
    vals_1 = df_train[df_train["FLAG"] == 1][feat].values
    
    # Missingness on un-imputed train
    raw_train_feat = df.iloc[idx_train][feat]
    miss_pct_0 = (df.iloc[idx_train][(df.iloc[idx_train]["FLAG"] == 0)][feat].isna().mean()) * 100
    miss_pct_1 = (df.iloc[idx_train][(df.iloc[idx_train]["FLAG"] == 1)][feat].isna().mean()) * 100
    miss_pct_all = (raw_train_feat.isna().mean()) * 100
    
    # Medians and IQRs
    med_0 = np.median(vals_0)
    q25_0, q75_0 = np.percentile(vals_0, 25), np.percentile(vals_0, 75)
    iqr_0 = q75_0 - q25_0
    
    med_1 = np.median(vals_1)
    q25_1, q75_1 = np.percentile(vals_1, 25), np.percentile(vals_1, 75)
    iqr_1 = q75_1 - q25_1
    
    # Effect sizes: Mann-Whitney U, Cliff's Delta, Cohen's d, Kolmogorov-Smirnov
    u_stat, p_val = stats.mannwhitneyu(vals_1, vals_0, alternative='two-sided')
    # Cliff's Delta = 2*U / (n1*n2) - 1
    cliffs_delta = (2.0 * u_stat) / (len(vals_1) * len(vals_0)) - 1.0
    
    # Cohen's d
    s_pooled = np.sqrt(((len(vals_0)-1)*np.var(vals_0, ddof=1) + (len(vals_1)-1)*np.var(vals_1, ddof=1)) / (len(vals_0)+len(vals_1)-2))
    cohens_d = (np.mean(vals_1) - np.mean(vals_0)) / s_pooled if s_pooled > 1e-12 else 0.0
    
    # KS test
    ks_stat, ks_pval = stats.ks_2samp(vals_1, vals_0)
    
    # Distribution Overlap: Overlapping coefficient of empirical histograms
    # Compute min bin overlap
    bins = np.histogram_bin_edges(np.concatenate([vals_0, vals_1]), bins=50)
    h0, _ = np.histogram(vals_0, bins=bins, density=True)
    h1, _ = np.histogram(vals_1, bins=bins, density=True)
    # Area overlap = sum(min(h0, h1) * bin_width)
    bin_widths = np.diff(bins)
    overlap_area = np.sum(np.minimum(h0, h1) * bin_widths)
    overlap_area = np.clip(overlap_area, 0.0, 1.0)
    
    separability_rows.append({
        "Feature": feat,
        "Category": "Missingness" if feat in missing_features else "Consumption",
        "Train_Missing_Pct_Benign": round(miss_pct_0, 3),
        "Train_Missing_Pct_Theft": round(miss_pct_1, 3),
        "Benign_Median": round(med_0, 4),
        "Benign_IQR": round(iqr_0, 4),
        "Theft_Median": round(med_1, 4),
        "Theft_IQR": round(iqr_1, 4),
        "Mann_Whitney_U": round(u_stat, 1),
        "Mann_Whitney_pval": f"{p_val:.2e}",
        "Cliffs_Delta": round(cliffs_delta, 4),
        "Cohens_d": round(cohens_d, 4),
        "KS_Statistic": round(ks_stat, 4),
        "Empirical_Distribution_Overlap": round(overlap_area, 4)
    })
    
    # Plotting on subplot
    ax = axes[idx]
    # Truncate at 99th percentile for clean visualization
    p99 = np.percentile(np.concatenate([vals_0, vals_1]), 99)
    p01 = np.percentile(np.concatenate([vals_0, vals_1]), 1)
    sub_0 = vals_0[(vals_0 >= p01) & (vals_0 <= p99)]
    sub_1 = vals_1[(vals_1 >= p01) & (vals_1 <= p99)]
    
    ax.hist(sub_0, bins=35, density=True, alpha=0.5, color='steelblue', label='Benign (0)')
    ax.hist(sub_1, bins=35, density=True, alpha=0.5, color='crimson', label='Theft (1)')
    ax.set_title(f"{feat} (Overlap={overlap_area:.2f}, KS={ks_stat:.2f})", fontsize=10, fontweight='bold')
    if idx == 0:
        ax.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plot_overlap_path = os.path.join(graphs_dir, "sgcc_feature_distribution_overlap.png")
fig.savefig(plot_overlap_path, dpi=300)
plt.close(fig)
print(f"Saved distribution overlap plot to {plot_overlap_path}")

df_sep = pd.DataFrame(separability_rows)
sep_csv_path = os.path.join(reports_dir, "SGCC_CLASS_SEPARABILITY.csv")
df_sep.to_csv(sep_csv_path, index=False)
print(f"Saved class separability CSV to {sep_csv_path}")

# ------------------------------------------------------------------------------
# PHASE 3: CONSUMPTION VS MISSINGNESS ABLATION
# ------------------------------------------------------------------------------
print("\n--- RUNNING PHASE 3: CONSUMPTION VS MISSINGNESS ABLATION ---")
feature_partitions = {
    "Consumption_Only": consumption_features,
    "Missingness_Only": missing_features,
    "Combined_18": features_18
}

ablation_rows = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for part_name, feats in feature_partitions.items():
    print(f"Evaluating Partition: {part_name} (Num Features: {len(feats)})")
    X_tr_sub = df_imp.iloc[idx_train][feats].values
    X_va_sub = df_imp.iloc[idx_val][feats].values
    
    # 5-fold CV on Training Set
    cv_f1s, cv_praucs, cv_rocaucs, cv_precs, cv_recs, cv_mccs = [], [], [], [], [], []
    for f_idx, (f_tr_idx, f_va_idx) in enumerate(skf.split(X_tr_sub, y_train)):
        f_Xtr, f_ytr = X_tr_sub[f_tr_idx], y_train[f_tr_idx]
        f_Xva, f_yva = X_tr_sub[f_va_idx], y_train[f_va_idx]
        
        clf = xgb.XGBClassifier(
            scale_pos_weight=10.72,
            max_depth=7,
            n_estimators=150,
            learning_rate=0.10,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )
        clf.fit(f_Xtr, f_ytr)
        
        probs = clf.predict_proba(f_Xva)[:, 1]
        preds = (probs >= 0.50).astype(int)
        
        cv_f1s.append(f1_score(f_yva, preds, zero_division=0))
        cv_praucs.append(average_precision_score(f_yva, probs))
        cv_rocaucs.append(roc_auc_score(f_yva, probs))
        cv_precs.append(precision_score(f_yva, preds, zero_division=0))
        cv_recs.append(recall_score(f_yva, preds, zero_division=0))
        cv_mccs.append(matthews_corrcoef(f_yva, preds))
        
    # Validation Evaluation (Trained on full X_train)
    full_clf = xgb.XGBClassifier(
        scale_pos_weight=10.72,
        max_depth=7,
        n_estimators=150,
        learning_rate=0.10,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    full_clf.fit(X_tr_sub, y_train)
    val_probs = full_clf.predict_proba(X_va_sub)[:, 1]
    val_preds = (val_probs >= 0.50).astype(int)
    
    ablation_rows.append({
        "Feature_Partition": part_name,
        "Feature_Count": len(feats),
        "CV_F1_Mean": round(np.mean(cv_f1s), 4),
        "CV_F1_Std": round(np.std(cv_f1s), 4),
        "CV_PR_AUC_Mean": round(np.mean(cv_praucs), 4),
        "CV_ROC_AUC_Mean": round(np.mean(cv_rocaucs), 4),
        "CV_Precision_Mean": round(np.mean(cv_precs), 4),
        "CV_Recall_Mean": round(np.mean(cv_recs), 4),
        "CV_MCC_Mean": round(np.mean(cv_mccs), 4),
        "Val_F1": round(f1_score(y_val, val_preds, zero_division=0), 4),
        "Val_PR_AUC": round(average_precision_score(y_val, val_probs), 4),
        "Val_ROC_AUC": round(roc_auc_score(y_val, val_probs), 4),
        "Val_Precision": round(precision_score(y_val, val_preds, zero_division=0), 4),
        "Val_Recall": round(recall_score(y_val, val_preds, zero_division=0), 4),
        "Val_MCC": round(matthews_corrcoef(y_val, val_preds), 4)
    })

df_ablation = pd.DataFrame(ablation_rows)
ablation_csv_path = os.path.join(reports_dir, "SGCC_MISSINGNESS_ABLATION.csv")
df_ablation.to_csv(ablation_csv_path, index=False)
print(f"Saved ablation CSV to {ablation_csv_path}")

# Plot Phase 3 Ablation Bar Chart
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(df_ablation))
width = 0.25

rects1 = ax.bar(x - width, df_ablation["Val_F1"], width, label='Val F1', color='#2b5c8f')
rects2 = ax.bar(x, df_ablation["Val_PR_AUC"], width, label='Val PR-AUC', color='#e06666')
rects3 = ax.bar(x + width, df_ablation["Val_ROC_AUC"], width, label='Val ROC-AUC', color='#6aa84f')

ax.set_ylabel('Metric Score', fontsize=12)
ax.set_title('Ablation: Consumption-Only vs. Missingness-Only vs. Combined 18 Features', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(df_ablation["Feature_Partition"], fontsize=11)
ax.set_ylim(0, 1.0)
ax.legend(fontsize=11)
for rect in rects1 + rects2 + rects3:
    h = rect.get_height()
    ax.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width()/2, h),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(graphs_dir, "sgcc_missingness_ablation_comparison.png"), dpi=300)
plt.close(fig)

# ------------------------------------------------------------------------------
# PHASE 4: DATA QUALITY AND COVERAGE GROUPS
# ------------------------------------------------------------------------------
print("\n--- RUNNING PHASE 4: DATA QUALITY AND COVERAGE GROUPS ---")
# Use missing_ratio from df to define coverage bins on training set:
# High Coverage: missing_ratio < 0.10 (low missingness)
# Medium Coverage: 0.10 <= missing_ratio < 0.30
# Low Coverage: missing_ratio >= 0.30 (high missingness)

df_val_raw = df.iloc[idx_val].copy()
val_missing_ratios = df_val_raw["missing_ratio"].values

val_groups = {}
val_groups["High_Coverage (<10% Missing)"] = np.where(val_missing_ratios < 0.10)[0]
val_groups["Medium_Coverage (10-30% Missing)"] = np.where((val_missing_ratios >= 0.10) & (val_missing_ratios < 0.30))[0]
val_groups["Low_Coverage (>=30% Missing)"] = np.where(val_missing_ratios >= 0.30)[0]

# Evaluate the Combined 18 model on validation subsets
model_base_18 = full_clf # Trained on Combined_18
val_probs_all = model_base_18.predict_proba(df_imp.iloc[idx_val][features_18].values)[:, 1]

coverage_rows = []
for grp_name, grp_indices in val_groups.items():
    sub_y = y_val[grp_indices]
    sub_probs = val_probs_all[grp_indices]
    sub_preds = (sub_probs >= 0.50).astype(int)
    
    n_cust = len(grp_indices)
    prev = np.mean(sub_y)
    f1 = f1_score(sub_y, sub_preds, zero_division=0)
    pr_auc = average_precision_score(sub_y, sub_probs) if len(np.unique(sub_y)) > 1 else np.nan
    rec = recall_score(sub_y, sub_preds, zero_division=0)
    prec = precision_score(sub_y, sub_preds, zero_division=0)
    
    coverage_rows.append({
        "Coverage_Group": grp_name,
        "Customer_Count": n_cust,
        "Positive_Count": int(np.sum(sub_y == 1)),
        "Positive_Prevalence": round(prev, 4),
        "Validation_F1": round(f1, 4),
        "Validation_PR_AUC": round(pr_auc, 4) if not np.isnan(pr_auc) else 0.0,
        "Validation_Precision": round(prec, 4),
        "Validation_Recall": round(rec, 4)
    })

df_coverage = pd.DataFrame(coverage_rows)
cov_csv_path = os.path.join(reports_dir, "SGCC_DATA_QUALITY_GROUPS.csv")
df_coverage.to_csv(cov_csv_path, index=False)
print(f"Saved data quality groups CSV to {cov_csv_path}")

# ------------------------------------------------------------------------------
# PHASE 5: LEARNING CURVES
# ------------------------------------------------------------------------------
print("\n--- RUNNING PHASE 5: LEARNING CURVES (20%, 40%, 60%, 80%, 100%) ---")
subsets = [0.20, 0.40, 0.60, 0.80, 1.00]
learning_curve_rows = []

X_train_full = df_imp.iloc[idx_train][features_18].values
y_train_full = y_train

for frac in subsets:
    print(f"Computing Learning Curve for Training Subset: {int(frac*100)}%")
    if frac < 1.0:
        _, sub_idx, _, sub_y = train_test_split(
            np.arange(len(y_train_full)), y_train_full,
            test_size=frac, stratify=y_train_full, random_state=42
        )
    else:
        sub_idx = np.arange(len(y_train_full))
        sub_y = y_train_full
        
    X_sub = X_train_full[sub_idx]
    y_sub = sub_y
    
    # 5-Fold Stratified CV
    tr_f1s, tr_praucs, tr_rocaucs = [], [], []
    cv_f1s, cv_praucs, cv_rocaucs = [], [], []
    
    for f_tr_idx, f_va_idx in skf.split(X_sub, y_sub):
        f_Xtr, f_ytr = X_sub[f_tr_idx], y_sub[f_tr_idx]
        f_Xva, f_yva = X_sub[f_va_idx], y_sub[f_va_idx]
        
        clf_lc = xgb.XGBClassifier(
            scale_pos_weight=10.72,
            max_depth=7,
            n_estimators=150,
            learning_rate=0.10,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )
        clf_lc.fit(f_Xtr, f_ytr)
        
        # Train fold metrics
        p_tr = clf_lc.predict_proba(f_Xtr)[:, 1]
        pred_tr = (p_tr >= 0.50).astype(int)
        tr_f1s.append(f1_score(f_ytr, pred_tr, zero_division=0))
        tr_praucs.append(average_precision_score(f_ytr, p_tr))
        tr_rocaucs.append(roc_auc_score(f_ytr, p_tr))
        
        # CV fold metrics
        p_cv = clf_lc.predict_proba(f_Xva)[:, 1]
        pred_cv = (p_cv >= 0.50).astype(int)
        cv_f1s.append(f1_score(f_yva, pred_cv, zero_division=0))
        cv_praucs.append(average_precision_score(f_yva, p_cv))
        cv_rocaucs.append(roc_auc_score(f_yva, p_cv))
        
    learning_curve_rows.append({
        "Training_Fraction": frac,
        "Training_Samples": len(sub_idx),
        "Train_F1_Mean": round(np.mean(tr_f1s), 4),
        "Train_F1_Std": round(np.std(tr_f1s), 4),
        "Train_PR_AUC_Mean": round(np.mean(tr_praucs), 4),
        "Train_ROC_AUC_Mean": round(np.mean(tr_rocaucs), 4),
        "CV_F1_Mean": round(np.mean(cv_f1s), 4),
        "CV_F1_Std": round(np.std(cv_f1s), 4),
        "CV_PR_AUC_Mean": round(np.mean(cv_praucs), 4),
        "CV_PR_AUC_Std": round(np.std(cv_praucs), 4),
        "CV_ROC_AUC_Mean": round(np.mean(cv_rocaucs), 4),
        "CV_ROC_AUC_Std": round(np.std(cv_rocaucs), 4)
    })

df_lc = pd.DataFrame(learning_curve_rows)
lc_csv_path = os.path.join(reports_dir, "SGCC_LEARNING_CURVES.csv")
df_lc.to_csv(lc_csv_path, index=False)
print(f"Saved learning curves CSV to {lc_csv_path}")

# Plot Learning Curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

pcts = [int(f*100) for f in df_lc["Training_Fraction"]]
ax1.plot(pcts, df_lc["Train_F1_Mean"], 'o--', color='crimson', label='Train F1')
ax1.plot(pcts, df_lc["CV_F1_Mean"], 'o-', color='navy', label='CV F1')
ax1.fill_between(pcts, df_lc["CV_F1_Mean"] - df_lc["CV_F1_Std"], df_lc["CV_F1_Mean"] + df_lc["CV_F1_Std"], alpha=0.15, color='navy')
ax1.set_title("F1-Score Learning Curve Across Training Sample Fractions", fontsize=12, fontweight='bold')
ax1.set_xlabel("% of Training Data", fontsize=11)
ax1.set_ylabel("F1 Score", fontsize=11)
ax1.set_ylim(0.2, 1.0)
ax1.legend(loc='center right', fontsize=10)

ax2.plot(pcts, df_lc["Train_PR_AUC_Mean"], 's--', color='darkorange', label='Train PR-AUC')
ax2.plot(pcts, df_lc["CV_PR_AUC_Mean"], 's-', color='teal', label='CV PR-AUC')
ax2.fill_between(pcts, df_lc["CV_PR_AUC_Mean"] - df_lc["CV_PR_AUC_Std"], df_lc["CV_PR_AUC_Mean"] + df_lc["CV_PR_AUC_Std"], alpha=0.15, color='teal')
ax2.set_title("PR-AUC Learning Curve Across Training Sample Fractions", fontsize=12, fontweight='bold')
ax2.set_xlabel("% of Training Data", fontsize=11)
ax2.set_ylabel("PR-AUC", fontsize=11)
ax2.set_ylim(0.2, 1.0)
ax2.legend(loc='center right', fontsize=10)

plt.tight_layout()
fig.savefig(os.path.join(graphs_dir, "sgcc_learning_curves.png"), dpi=300)
plt.close(fig)

# ------------------------------------------------------------------------------
# PHASE 6 & 7: FEATURE REPRESENTATION & MODEL DIAGNOSTICS BENCHMARK
# ------------------------------------------------------------------------------
print("\n--- RUNNING PHASE 6 & 7: REPRESENTATION & MODEL DIAGNOSTIC BENCHMARKS ---")
# Feature sets to evaluate:
# 1. Original 18-Feature Representation
# 2. Original 42-Feature Representation
# 3. Consumption-Only (15 features)
# 4. Missingness-Only (3 features)

# Model Architectures:
# A. XGBoost (Class-Weighted)
# B. LightGBM Focal Loss (gamma=2.0, alpha=0.85)
# C. ExtraTrees Classifier

def focal_loss_obj(preds, train_data, gamma=2.0, alpha=0.85):
    labels = train_data.get_label()
    p = 1.0 / (1.0 + np.exp(-preds))
    p = np.clip(p, 1e-15, 1 - 1e-15)
    pt = np.where(labels == 1, p, 1 - p)
    at = np.where(labels == 1, alpha, 1 - alpha)
    grad = at * ((1 - pt) ** gamma) * (p - labels)
    hess = at * ((1 - pt) ** gamma) * (p * (1 - p)) * (gamma * np.abs(p - labels) + 1.0)
    return grad, hess

model_diag_rows = []

representations = {
    "18_Features": features_18,
    "42_Features": all_42_features,
    "Consumption_Only_15": consumption_features,
    "Missingness_Only_3": missing_features
}

models_to_test = ["XGBoost", "LightGBM_FocalLoss", "ExtraTrees"]

for rep_name, feats in representations.items():
    X_tr_rep = df_imp.iloc[idx_train][feats].values
    X_va_rep = df_imp.iloc[idx_val][feats].values
    
    for m_name in models_to_test:
        print(f"Benchmarking Representation: {rep_name} | Model: {m_name}")
        
        # 5-fold CV
        cv_f1s, cv_praucs, cv_rocaucs = [], [], []
        
        for f_tr_idx, f_va_idx in skf.split(X_tr_rep, y_train):
            f_Xtr, f_ytr = X_tr_rep[f_tr_idx], y_train[f_tr_idx]
            f_Xva, f_yva = X_tr_rep[f_va_idx], y_train[f_va_idx]
            
            if m_name == "XGBoost":
                m = xgb.XGBClassifier(
                    scale_pos_weight=10.72, max_depth=7, n_estimators=150,
                    learning_rate=0.10, eval_metric="logloss", random_state=42, n_jobs=-1
                )
                m.fit(f_Xtr, f_ytr)
                p_cv = m.predict_proba(f_Xva)[:, 1]
            elif m_name == "LightGBM_FocalLoss":
                train_ds = lgb.Dataset(f_Xtr, label=f_ytr)
                def _fl_obj(preds, ds):
                    return focal_loss_obj(preds, ds, gamma=2.0, alpha=0.85)
                params = {
                    'objective': _fl_obj,
                    'num_leaves': 63, 'learning_rate': 0.05, 'feature_fraction': 0.8,
                    'bagging_fraction': 0.8, 'bagging_freq': 5, 'verbose': -1, 'random_state': 42
                }
                m = lgb.train(params, train_ds, num_boost_round=150)
                logits = m.predict(f_Xva, raw_score=True)
                p_cv = 1.0 / (1.0 + np.exp(-logits))
            elif m_name == "ExtraTrees":
                m = ExtraTreesClassifier(
                    n_estimators=150, max_depth=12, class_weight='balanced',
                    random_state=42, n_jobs=-1
                )
                m.fit(f_Xtr, f_ytr)
                p_cv = m.predict_proba(f_Xva)[:, 1]
                
            pred_cv = (p_cv >= 0.50).astype(int)
            cv_f1s.append(f1_score(f_yva, pred_cv, zero_division=0))
            cv_praucs.append(average_precision_score(f_yva, p_cv))
            cv_rocaucs.append(roc_auc_score(f_yva, p_cv))
            
        # Fit on full training set and evaluate on Validation
        if m_name == "XGBoost":
            full_m = xgb.XGBClassifier(
                scale_pos_weight=10.72, max_depth=7, n_estimators=150,
                learning_rate=0.10, eval_metric="logloss", random_state=42, n_jobs=-1
            )
            full_m.fit(X_tr_rep, y_train)
            p_tr_full = full_m.predict_proba(X_tr_rep)[:, 1]
            p_va_full = full_m.predict_proba(X_va_rep)[:, 1]
        elif m_name == "LightGBM_FocalLoss":
            train_ds_full = lgb.Dataset(X_tr_rep, label=y_train)
            def _fl_obj_full(preds, ds):
                return focal_loss_obj(preds, ds, gamma=2.0, alpha=0.85)
            params = {
                'objective': _fl_obj_full,
                'num_leaves': 63, 'learning_rate': 0.05, 'feature_fraction': 0.8,
                'bagging_fraction': 0.8, 'bagging_freq': 5, 'verbose': -1, 'random_state': 42
            }
            full_m = lgb.train(params, train_ds_full, num_boost_round=150)
            logits_tr = full_m.predict(X_tr_rep, raw_score=True)
            p_tr_full = 1.0 / (1.0 + np.exp(-logits_tr))
            logits_va = full_m.predict(X_va_rep, raw_score=True)
            p_va_full = 1.0 / (1.0 + np.exp(-logits_va))
        elif m_name == "ExtraTrees":
            full_m = ExtraTreesClassifier(
                n_estimators=150, max_depth=12, class_weight='balanced',
                random_state=42, n_jobs=-1
            )
            full_m.fit(X_tr_rep, y_train)
            p_tr_full = full_m.predict_proba(X_tr_rep)[:, 1]
            p_va_full = full_m.predict_proba(X_va_rep)[:, 1]
            
        pred_tr_full = (p_tr_full >= 0.50).astype(int)
        pred_va_full = (p_va_full >= 0.50).astype(int)
        
        tr_f1 = f1_score(y_train, pred_tr_full, zero_division=0)
        tr_prauc = average_precision_score(y_train, p_tr_full)
        val_f1 = f1_score(y_val, pred_va_full, zero_division=0)
        val_prauc = average_precision_score(y_val, p_va_full)
        val_rocauc = roc_auc_score(y_val, p_va_full)
        
        model_diag_rows.append({
            "Representation": rep_name,
            "Feature_Count": len(feats),
            "Model": m_name,
            "Train_F1": round(tr_f1, 4),
            "Train_PR_AUC": round(tr_prauc, 4),
            "CV_F1_Mean": round(np.mean(cv_f1s), 4),
            "CV_F1_Std": round(np.std(cv_f1s), 4),
            "CV_PR_AUC_Mean": round(np.mean(cv_praucs), 4),
            "CV_ROC_AUC_Mean": round(np.mean(cv_rocaucs), 4),
            "Val_F1": round(val_f1, 4),
            "Val_PR_AUC": round(val_prauc, 4),
            "Val_ROC_AUC": round(val_rocauc, 4),
            "Overfit_F1_Gap": round(tr_f1 - val_f1, 4)
        })

df_diag = pd.DataFrame(model_diag_rows)
diag_csv_path = os.path.join(reports_dir, "SGCC_DIAGNOSTIC_RESULTS.csv")
df_diag.to_csv(diag_csv_path, index=False)
print(f"Saved diagnostic results CSV to {diag_csv_path}")

print("\n--- ALL COMPUTATIONS COMPLETED SUCCESSFULLY ---")
