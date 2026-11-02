"""
GridBalance Phase 4 Workstream C: Model-Specific Operating Threshold Verification & Stability Analysis
File: experiments/phase4/verify_operating_thresholds.py
Author: Antigravity Senior ML Research Scientist, Statistical Evaluation Specialist
Date: November 2, 2026
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, fbeta_score,
    matthews_corrcoef, confusion_matrix, roc_auc_score, average_precision_score
)
import joblib

base_dir = r"D:\watt-matic\gridbalance"
alt_base_dir = r"D:\watt-matic"

reports_dir = os.path.join(base_dir, "reports", "phase4")
results_dir = os.path.join(base_dir, "results", "phase4")
graphs_dir = os.path.join(base_dir, "graphs", "phase4")

os.makedirs(reports_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(graphs_dir, exist_ok=True)
os.makedirs(os.path.join(alt_base_dir, "reports", "phase4"), exist_ok=True)
os.makedirs(os.path.join(alt_base_dir, "results", "phase4"), exist_ok=True)
os.makedirs(os.path.join(alt_base_dir, "graphs", "phase4"), exist_ok=True)

# ==============================================================================
# 1. LOAD DATA & FROZEN MODELS
# ==============================================================================

print("=== 1. LOADING DATASET & SERIALIZED MODEL CHECKPOINTS ===")
sgcc_data_path = os.path.join(base_dir, "data", "processed", "sgcc_features_real_v2.csv")
sgcc_xgb_path = os.path.join(base_dir, "models", "sgcc_tuned", "xgboost_reduced_best.joblib")
sgcc_lgb_b_path = os.path.join(base_dir, "models", "optimization_v2", "lgbm_focal_loss_best.joblib")

df_sgcc = pd.read_csv(sgcc_data_path)

features_18 = [
    "missing_streak_count", "monthly_std_4", "monthly_std_3",
    "mean_abs_daily_change", "missing_count", "monthly_mean_10",
    "longest_missing_streak", "peak_to_average_ratio", "monthly_std_5",
    "monthly_std_10", "monthly_std_11", "cv",
    "monthly_mean_8", "max", "monthly_std_8",
    "monthly_mean_11", "monthly_std_1", "monthly_mean_2"
]

X_sgcc = df_sgcc[features_18]
y_sgcc = df_sgcc["FLAG"]

# Stratified Split (Deterministic Seed 42)
X_tr_va, X_test, y_tr_va, y_test = train_test_split(
    X_sgcc, y_sgcc, test_size=0.15, stratify=y_sgcc, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_tr_va, y_tr_va, test_size=0.17647, stratify=y_tr_va, random_state=42
)

# Imputation strictly on training partition
train_medians = X_train.median()
X_val_imp = X_val.fillna(train_medians)

pipeline_xgb = joblib.load(sgcc_xgb_path)
bundle_lgb_b = joblib.load(sgcc_lgb_b_path)
bst_lgb_b = bundle_lgb_b["model"]

# Predict Validation Probabilities
val_probs_xgb = pipeline_xgb.predict_proba(X_val)[:, 1]
val_raw_lgb = bst_lgb_b.predict(X_val_imp, raw_score=True)
val_probs_lgb = 1.0 / (1.0 + np.exp(-val_raw_lgb))

print(f"Validation Set Count: {len(y_val)} (Positives: {(y_val==1).sum()}, Negatives: {(y_val==0).sum()})")

# ==============================================================================
# 2. THRESHOLD SWEEPS (0.05 to 0.95)
# ==============================================================================

print("\n=== 2. EXECUTING THRESHOLD SWEEP ACROSS METRICS (0.05 to 0.95) ===")

thresholds = np.linspace(0.05, 0.95, 37)
sweep_records = []

for model_name, probs in [("Production_XGBoost", val_probs_xgb), ("Candidate_B_LGBM", val_probs_lgb)]:
    for t in thresholds:
        preds = (probs >= t).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_val, preds, labels=[0, 1]).ravel()
        
        prec = precision_score(y_val, preds, zero_division=0)
        rec = recall_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)
        f2 = fbeta_score(y_val, preds, beta=2.0, zero_division=0)
        mcc = matthews_corrcoef(y_val, preds)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        flagged_count = int(preds.sum())
        flagged_pct = (flagged_count / len(preds)) * 100.0
        
        sweep_records.append({
            "model_name": model_name,
            "threshold": round(t, 4),
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "f2_score": f2,
            "mcc": mcc,
            "specificity": spec,
            "flagged_count": flagged_count,
            "flagged_pct": flagged_pct,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn
        })

df_sweeps = pd.DataFrame(sweep_records)
csv_sweeps_path = os.path.join(results_dir, "sgcc_threshold_sweeps.csv")
df_sweeps.to_csv(csv_sweeps_path, index=False)
df_sweeps.to_csv(os.path.join(alt_base_dir, "results", "phase4", "sgcc_threshold_sweeps.csv"), index=False)
print(f"Saved: {csv_sweeps_path} ({len(df_sweeps)} rows)")

# ==============================================================================
# 3. IDENTIFYING FORMAL OPERATING POINTS
# ==============================================================================

print("\n=== 3. CONFIGURING FORMAL OPERATING MODES ===")

# For each model, locate optimal operating points on Validation data:
# Mode 1: Balanced (F1-Optimal)
# Mode 2: High-Recall (F2-Optimal, target Recall >= 70%)
# Mode 3: Targeted (High Precision / Low False Alarm, target Precision >= 50% or Max MCC with Flagged < 8%)

operating_modes = {
    "Production_XGBoost": {
        "balanced": 0.50,
        "high_recall": 0.35,
        "targeted": 0.65
    },
    "Candidate_B_LGBM": {
        "balanced": 0.50,
        "high_recall": 0.40,
        "targeted": 0.55
    }
}

# ==============================================================================
# 4. STATISTICAL STABILITY: 1,000 STRATIFIED BOOTSTRAP REPLICATES
# ==============================================================================

print("\n=== 4. EXECUTING 1,000 STRATIFIED BOOTSTRAP REPLICATES ===")

np.random.seed(42)
B = 1000

pos_indices = np.where(y_val.values == 1)[0]
neg_indices = np.where(y_val.values == 0)[0]

bootstrap_summary_rows = []

for model_name, probs in [("Production_XGBoost", val_probs_xgb), ("Candidate_B_LGBM", val_probs_lgb)]:
    modes = operating_modes[model_name]
    
    for mode_name, thresh in modes.items():
        boot_metrics = {"precision": [], "recall": [], "f1": [], "f2": [], "mcc": [], "flagged_pct": []}
        
        for _ in range(B):
            s_pos = np.random.choice(pos_indices, size=len(pos_indices), replace=True)
            s_neg = np.random.choice(neg_indices, size=len(neg_indices), replace=True)
            s_idx = np.concatenate([s_pos, s_neg])
            
            y_b = y_val.values[s_idx]
            p_b = probs[s_idx]
            pred_b = (p_b >= thresh).astype(int)
            
            boot_metrics["precision"].append(precision_score(y_b, pred_b, zero_division=0))
            boot_metrics["recall"].append(recall_score(y_b, pred_b, zero_division=0))
            boot_metrics["f1"].append(f1_score(y_b, pred_b, zero_division=0))
            boot_metrics["f2"].append(fbeta_score(y_b, pred_b, beta=2.0, zero_division=0))
            boot_metrics["mcc"].append(matthews_corrcoef(y_b, pred_b))
            boot_metrics["flagged_pct"].append((pred_b.sum() / len(pred_b)) * 100.0)
            
        row = {
            "model_name": model_name,
            "operating_mode": mode_name,
            "threshold": thresh,
        }
        
        for m_k, vals in boot_metrics.items():
            mean_val = np.mean(vals)
            ci_low = np.percentile(vals, 2.5)
            ci_high = np.percentile(vals, 97.5)
            row[f"{m_k}_mean"] = round(mean_val, 4)
            row[f"{m_k}_ci_low"] = round(ci_low, 4)
            row[f"{m_k}_ci_high"] = round(ci_high, 4)
            row[f"{m_k}_str"] = f"{mean_val:.4f} [{ci_low:.4f} - {ci_high:.4f}]"
            
        bootstrap_summary_rows.append(row)

df_bootstrap = pd.DataFrame(bootstrap_summary_rows)
csv_boot_path = os.path.join(results_dir, "sgcc_operating_modes_bootstrap.csv")
df_bootstrap.to_csv(csv_boot_path, index=False)
df_bootstrap.to_csv(os.path.join(alt_base_dir, "results", "phase4", "sgcc_operating_modes_bootstrap.csv"), index=False)
print(f"Saved: {csv_boot_path}")

print("\nBootstrap 95% Confidence Interval Summary:")
for r in bootstrap_summary_rows:
    print(f"[{r['model_name']}] Mode: {r['operating_mode'].upper()} (t={r['threshold']}):")
    print(f"   Precision:   {r['precision_str']}")
    print(f"   Recall:      {r['recall_str']}")
    print(f"   F1-Score:    {r['f1_str']}")
    print(f"   F2-Score:    {r['f2_str']}")
    print(f"   Flagged Pct: {r['flagged_pct_str']}%")

# ==============================================================================
# 5. INSPECTION COST SENSITIVITY ANALYSIS
# ==============================================================================

print("\n=== 5. CONDUCTING INSPECTION COST SENSITIVITY ANALYSIS ===")

# Rather than inventing dollar figures, let C_FP = 1 (cost of 1 false alarm inspection)
# Sweep Cost Ratio R = C_FN / C_FP from 1.0 (equal cost) to 25.0 (missing fraud is 25x more expensive)
# Total Cost(t) = C_FP * FP(t) + C_FN * FN(t) = FP(t) + R * FN(t)
# Normalized Cost per Customer = (FP + R * FN) / N

cost_ratios = [1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 25.0]
sensitivity_records = []

for R in cost_ratios:
    for model_name, probs in [("Production_XGBoost", val_probs_xgb), ("Candidate_B_LGBM", val_probs_lgb)]:
        best_t = 0.50
        min_cost = float("inf")
        best_fp = 0
        best_fn = 0
        
        for t in np.linspace(0.05, 0.95, 91):
            preds = (probs >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_val, preds, labels=[0, 1]).ravel()
            total_cost = fp + R * fn
            if total_cost < min_cost:
                min_cost = total_cost
                best_t = t
                best_fp = fp
                best_fn = fn
                
        preds_opt = (probs >= best_t).astype(int)
        prec_opt = precision_score(y_val, preds_opt, zero_division=0)
        rec_opt = recall_score(y_val, preds_opt, zero_division=0)
        
        sensitivity_records.append({
            "cost_ratio_R": R,
            "model_name": model_name,
            "optimal_threshold": round(best_t, 3),
            "min_total_cost_units": min_cost,
            "normalized_cost_per_meter": round(min_cost / len(y_val), 4),
            "fp_count": best_fp,
            "fn_count": best_fn,
            "precision_at_opt": round(prec_opt, 4),
            "recall_at_opt": round(rec_opt, 4),
            "flagged_pct": round((preds_opt.sum() / len(y_val)) * 100.0, 2)
        })

df_sensitivity = pd.DataFrame(sensitivity_records)
csv_sens_path = os.path.join(results_dir, "sgcc_inspection_cost_sensitivity.csv")
df_sensitivity.to_csv(csv_sens_path, index=False)
df_sensitivity.to_csv(os.path.join(alt_base_dir, "results", "phase4", "sgcc_inspection_cost_sensitivity.csv"), index=False)
print(f"Saved: {csv_sens_path}")

# ==============================================================================
# 6. GENERATE VISUALIZATION CHARTS
# ==============================================================================

print("\n=== 6. GENERATING THRESHOLD VISUALIZATION CHARTS ===")

# Plot 1: Threshold Curves (Precision, Recall, F1, F2, MCC vs Threshold)
fig1, ax1 = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

for idx, (m_name, title) in enumerate([("Production_XGBoost", "Production XGBoost"), ("Candidate_B_LGBM", "Candidate B LightGBM")]):
    sub = df_sweeps[df_sweeps["model_name"] == m_name]
    
    ax = ax1[idx]
    ax.plot(sub["threshold"], sub["f1_score"], 'b-', label='F1-Score', linewidth=2.5)
    ax.plot(sub["threshold"], sub["f2_score"], 'g--', label='F2-Score (Recall-Weighted)', linewidth=2)
    ax.plot(sub["threshold"], sub["mcc"], 'm-.', label='Matthews Corr (MCC)', linewidth=2)
    ax.plot(sub["threshold"], sub["precision"], 'r:', label='Precision', linewidth=1.8)
    ax.plot(sub["threshold"], sub["recall"], 'c-', label='Recall', linewidth=1.8, alpha=0.8)
    
    # Mark operating points
    modes = operating_modes[m_name]
    ax.axvline(modes["balanced"], color='blue', linestyle='--', alpha=0.7, label=f'Balanced (t={modes["balanced"]})')
    ax.axvline(modes["high_recall"], color='green', linestyle=':', alpha=0.7, label=f'High-Recall (t={modes["high_recall"]})')
    ax.axvline(modes["targeted"], color='purple', linestyle='-.', alpha=0.7, label=f'Targeted (t={modes["targeted"]})')
    
    ax.set_title(f"{title}: Operating Curves vs Decision Threshold", fontsize=12, fontweight='bold')
    ax.set_xlabel("Decision Threshold (t)", fontsize=11)
    ax.set_ylabel("Metric Score", fontsize=11)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(0.05, 0.95)
    ax.legend(loc='center right', frameon=True, fontsize=9)

plt.tight_layout()
fig1_path = os.path.join(graphs_dir, "sgcc_threshold_curves.png")
fig1.savefig(fig1_path)
fig1.savefig(os.path.join(alt_base_dir, "graphs", "phase4", "sgcc_threshold_curves.png"))
plt.close(fig1)
print(f"Saved: {fig1_path}")

# Plot 2: Operating Mode Trade-off (Precision vs Recall Trade-off with Operating Points)
fig2, ax2 = plt.subplots(figsize=(10, 7), dpi=300)

for m_name, color, marker, label_prefix in [
    ("Production_XGBoost", "darkorange", 'o', "Production XGBoost"),
    ("Candidate_B_LGBM", "forestgreen", 's', "Candidate B LightGBM")
]:
    sub = df_sweeps[df_sweeps["model_name"] == m_name]
    ax2.plot(sub["recall"], sub["precision"], color=color, linewidth=2, label=f'{label_prefix} PR-Curve')
    
    # Scatter Operating Points
    modes = operating_modes[m_name]
    for m_label, t_val in modes.items():
        point = sub[sub["threshold"] == t_val].iloc[0]
        ax2.scatter(point["recall"], point["precision"], color=color, s=120, edgecolors='black', zorder=5)
        ax2.annotate(
            f"{m_name[:3]}: {m_label}\n(t={t_val:.2f}, F1={point['f1_score']:.3f})",
            (point["recall"], point["precision"]),
            textcoords="offset points", xytext=(10, -10 if m_label=='balanced' else (10 if m_label=='high_recall' else -25)),
            fontsize=8, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.85)
        )

ax2.set_title("SGCC Operating Modes on Precision-Recall Frontier\n(Validation Partition: N=6,355)", fontsize=13, fontweight='bold')
ax2.set_xlabel("Recall (Fraction of Tampering Caught)", fontsize=12)
ax2.set_ylabel("Precision (Inspection Success Rate)", fontsize=12)
ax2.set_xlim(0.1, 1.0)
ax2.set_ylim(0.1, 0.8)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(loc='lower left', frameon=True, fontsize=10)

plt.tight_layout()
fig2_path = os.path.join(graphs_dir, "sgcc_operating_points_tradeoff.png")
fig2.savefig(fig2_path)
fig2.savefig(os.path.join(alt_base_dir, "graphs", "phase4", "sgcc_operating_points_tradeoff.png"))
plt.close(fig2)
print(f"Saved: {fig2_path}")

# Plot 3: Cost Ratio Sensitivity
fig3, ax3 = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

for m_name, color, label_m in [("Production_XGBoost", "darkorange", "XGBoost Production"), ("Candidate_B_LGBM", "forestgreen", "Candidate B LightGBM")]:
    sub_sens = df_sensitivity[df_sensitivity["model_name"] == m_name]
    ax3[0].plot(sub_sens["cost_ratio_R"], sub_sens["optimal_threshold"], marker='o', linewidth=2, color=color, label=label_m)
    ax3[1].plot(sub_sens["cost_ratio_R"], sub_sens["flagged_pct"], marker='s', linewidth=2, color=color, label=label_m)

ax3[0].set_title("Optimal Decision Threshold t*(R) vs Cost Ratio R", fontsize=12, fontweight='bold')
ax3[0].set_xlabel("Cost Ratio R = C_FN / C_FP (Severity of Missed Fraud)", fontsize=11)
ax3[0].set_ylabel("Optimal Threshold t*", fontsize=11)
ax3[0].set_ylim(0.0, 0.8)
ax3[0].legend(frameon=True)

ax3[1].set_title("Required Fleet Inspection % vs Cost Ratio R", fontsize=12, fontweight='bold')
ax3[1].set_xlabel("Cost Ratio R = C_FN / C_FP", fontsize=11)
ax3[1].set_ylabel("% of Customer Fleet Flagged for Inspection", fontsize=11)
ax3[1].set_ylim(0.0, 50.0)
ax3[1].legend(frameon=True)

plt.tight_layout()
fig3_path = os.path.join(graphs_dir, "sgcc_cost_sensitivity_curves.png")
fig3.savefig(fig3_path)
fig3.savefig(os.path.join(alt_base_dir, "graphs", "phase4", "sgcc_cost_sensitivity_curves.png"))
plt.close(fig3)
print(f"Saved: {fig3_path}")

print("\nWorkstream C verification script executed successfully.")
