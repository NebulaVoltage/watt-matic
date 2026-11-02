"""
GridBalance Phase 4 Workstream A: Focal-Weighted Objective Mathematical Hardening & Independent Verification
File: experiments/phase4/focal_loss_formal_proof.py
Author: Antigravity Senior Mathematical ML Research Scientist
Date: October 31, 2026
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
import lightgbm as lgb
import joblib

# Ensure output directories exist
base_dir = r"D:\watt-matic\gridbalance"
alt_base_dir = r"D:\watt-matic"

reports_dir = os.path.join(base_dir, "reports", "phase4")
results_dir = os.path.join(base_dir, "results", "phase4")
graphs_dir = os.path.join(base_dir, "graphs", "phase4")
os.makedirs(reports_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(graphs_dir, exist_ok=True)

# Also ensure alt_base_dir directories exist for mirroring
os.makedirs(os.path.join(alt_base_dir, "reports", "phase4"), exist_ok=True)
os.makedirs(os.path.join(alt_base_dir, "results", "phase4"), exist_ok=True)
os.makedirs(os.path.join(alt_base_dir, "graphs", "phase4"), exist_ok=True)

# ==============================================================================
# 1. CORE MATHEMATICAL FORMULATIONS
# ==============================================================================

def sigmoid(z):
    """Numerically stable sigmoid."""
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))

def exact_focal_loss(z, y, alpha=0.85, gamma=2.0, eps=1e-15):
    """
    Conventional Focal Loss (Lin et al., 2017):
    L = - alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    p = np.clip(sigmoid(z), eps, 1.0 - eps)
    pt = np.where(y == 1, p, 1.0 - p)
    at = np.where(y == 1, alpha, 1.0 - alpha)
    return -at * ((1.0 - pt) ** gamma) * np.log(pt)

def exact_focal_gradient(z, y, alpha=0.85, gamma=2.0, eps=1e-15):
    """
    Exact analytical gradient dL/dz of conventional focal loss:
    For y=1: g1 = alpha * (1 - p)^gamma * [ gamma * p * log(p) + p - 1 ]
    For y=0: g0 = (1 - alpha) * p^gamma * [ p - gamma * (1 - p) * log(1 - p) ]
    """
    p = np.clip(sigmoid(z), eps, 1.0 - eps)
    if y == 1:
        return alpha * ((1.0 - p) ** gamma) * (gamma * p * np.log(p) + p - 1.0)
    else:
        return (1.0 - alpha) * (p ** gamma) * (p - gamma * (1.0 - p) * np.log(1.0 - p))

def exact_focal_hessian(z, y, alpha=0.85, gamma=2.0, eps=1e-15):
    """
    Exact analytical Hessian d^2L/dz^2 of conventional focal loss:
    For y=1: h1 = alpha * (1 - p)^gamma * p * [ gamma * (1 - (1 + gamma)*p) * log(p) + (1 - p)*(2*gamma + 1) ]
    For y=0: h0 = (1 - alpha) * p^gamma * (1 - p) * [ p*(2*gamma + 1) + gamma * (p - gamma*(1 - p)) * log(1 - p) ]
    """
    p = np.clip(sigmoid(z), eps, 1.0 - eps)
    if y == 1:
        return alpha * ((1.0 - p) ** gamma) * p * (
            gamma * (1.0 - (1.0 + gamma) * p) * np.log(p) + (1.0 - p) * (2.0 * gamma + 1.0)
        )
    else:
        return (1.0 - alpha) * (p ** gamma) * (1.0 - p) * (
            p * (2.0 * gamma + 1.0) + gamma * (p - gamma * (1.0 - p)) * np.log(1.0 - p)
        )

def implemented_pseudo_gradient(z, y, alpha=0.85, gamma=2.0, eps=1e-15):
    """
    Implemented focal-weighted pseudo-gradient in GridBalance:
    g = alpha_t * (1 - p_t)^gamma * (p - y)
    """
    p = np.clip(sigmoid(z), eps, 1.0 - eps)
    pt = np.where(y == 1, p, 1.0 - p)
    at = np.where(y == 1, alpha, 1.0 - alpha)
    return at * ((1.0 - pt) ** gamma) * (p - y)

def implemented_surrogate_hessian(z, y, alpha=0.85, gamma=2.0, eps=1e-15):
    """
    Implemented surrogate Hessian in GridBalance:
    h_surr = alpha_t * (1 - p_t)^gamma * p * (1 - p) * [ gamma * |p - y| + 1 ]
    """
    p = np.clip(sigmoid(z), eps, 1.0 - eps)
    pt = np.where(y == 1, p, 1.0 - p)
    at = np.where(y == 1, alpha, 1.0 - alpha)
    return at * ((1.0 - pt) ** gamma) * (p * (1.0 - p)) * (gamma * np.abs(p - y) + 1.0)

def true_derivative_of_pseudo_gradient(z, y, alpha=0.85, gamma=2.0, eps=1e-15):
    """
    The exact mathematical derivative dg_impl/dz of the implemented pseudo-gradient:
    dg_impl/dz = alpha_t * (gamma + 1) * p * (1 - p) * (1 - p_t)^gamma
    """
    p = np.clip(sigmoid(z), eps, 1.0 - eps)
    pt = np.where(y == 1, p, 1.0 - p)
    at = np.where(y == 1, alpha, 1.0 - alpha)
    return at * (gamma + 1.0) * p * (1.0 - p) * ((1.0 - pt) ** gamma)

# Numerical differentiation helpers
def finite_difference_gradient(f, z, y, alpha, gamma, delta=1e-6):
    return (f(z + delta, y, alpha, gamma) - f(z - delta, y, alpha, gamma)) / (2.0 * delta)

def finite_difference_hessian(f, z, y, alpha, gamma, delta=1e-5):
    return (f(z + delta, y, alpha, gamma) - 2.0 * f(z, y, alpha, gamma) + f(z - delta, y, alpha, gamma)) / (delta ** 2)

# ==============================================================================
# 2. NUMERICAL VERIFICATION & ERROR AUDIT
# ==============================================================================

print("=== 1. RUNNING HIGH-PRECISION NUMERICAL VERIFICATION ===")

z_grid = np.linspace(-10.0, 10.0, 401)
grad_records = []
hess_records = []

# Verify over comprehensive grid
configs = [
    (0.5, 0.0), (0.85, 0.0),
    (0.5, 1.0), (0.85, 1.0),
    (0.5, 2.0), (0.85, 2.0),
    (0.5, 3.0), (0.85, 3.0),
]

for alpha_val, gamma_val in configs:
    for y_val in [1, 0]:
        for z_val in z_grid:
            p_val = sigmoid(z_val)
            
            # Gradients
            g_exact = exact_focal_gradient(z_val, y_val, alpha_val, gamma_val)
            g_num = finite_difference_gradient(exact_focal_loss, z_val, y_val, alpha_val, gamma_val, delta=1e-6)
            g_impl = implemented_pseudo_gradient(z_val, y_val, alpha_val, gamma_val)
            
            grad_err = abs(g_exact - g_num)
            grad_diff = g_impl - g_exact
            
            grad_records.append({
                "alpha": alpha_val,
                "gamma": gamma_val,
                "y": y_val,
                "z": z_val,
                "p": p_val,
                "g_exact": g_exact,
                "g_num": g_num,
                "g_impl": g_impl,
                "exact_analyt_vs_num_err": grad_err,
                "impl_vs_exact_diff": grad_diff
            })
            
            # Hessians
            h_exact = exact_focal_hessian(z_val, y_val, alpha_val, gamma_val)
            h_num = finite_difference_hessian(exact_focal_loss, z_val, y_val, alpha_val, gamma_val, delta=1e-5)
            h_num_from_g_exact = (
                exact_focal_gradient(z_val + 1e-6, y_val, alpha_val, gamma_val) -
                exact_focal_gradient(z_val - 1e-6, y_val, alpha_val, gamma_val)
            ) / (2e-6)
            h_surr = implemented_surrogate_hessian(z_val, y_val, alpha_val, gamma_val)
            h_true_dg = true_derivative_of_pseudo_gradient(z_val, y_val, alpha_val, gamma_val)
            h_num_from_g_impl = (
                implemented_pseudo_gradient(z_val + 1e-6, y_val, alpha_val, gamma_val) -
                implemented_pseudo_gradient(z_val - 1e-6, y_val, alpha_val, gamma_val)
            ) / (2e-6)
            
            hess_records.append({
                "alpha": alpha_val,
                "gamma": gamma_val,
                "y": y_val,
                "z": z_val,
                "p": p_val,
                "h_exact": h_exact,
                "h_num": h_num,
                "h_num_from_g_exact": h_num_from_g_exact,
                "h_surr": h_surr,
                "h_true_dg": h_true_dg,
                "h_num_from_g_impl": h_num_from_g_impl,
                "exact_analyt_vs_num_err": abs(h_exact - h_num_from_g_exact),
                "true_dg_vs_num_err": abs(h_true_dg - h_num_from_g_impl),
                "surr_vs_exact_diff": h_surr - h_exact,
                "surr_vs_true_dg_diff": h_surr - h_true_dg
            })

df_grad = pd.DataFrame(grad_records)
df_hess = pd.DataFrame(hess_records)

csv_grad_path = os.path.join(results_dir, "focal_gradient_verification.csv")
csv_hess_path = os.path.join(results_dir, "focal_hessian_verification.csv")
df_grad.to_csv(csv_grad_path, index=False)
df_hess.to_csv(csv_hess_path, index=False)

# Mirror to alt_base_dir
df_grad.to_csv(os.path.join(alt_base_dir, "results", "phase4", "focal_gradient_verification.csv"), index=False)
df_hess.to_csv(os.path.join(alt_base_dir, "results", "phase4", "focal_hessian_verification.csv"), index=False)

print(f"Saved: {csv_grad_path} ({len(df_grad)} rows)")
print(f"Saved: {csv_hess_path} ({len(df_hess)} rows)")

# Gradient Verification Summary:
max_g_err = df_grad["exact_analyt_vs_num_err"].max()
mean_g_err = df_grad["exact_analyt_vs_num_err"].mean()
print(f"Exact Gradient vs Numerical: Max Abs Error = {max_g_err:.2e}, Mean Abs Error = {mean_g_err:.2e}")

# Hessian Verification Summary:
max_h_err = df_hess["exact_analyt_vs_num_err"].max()
mean_h_err = df_hess["exact_analyt_vs_num_err"].mean()
print(f"Exact Hessian vs Numerical: Max Abs Error = {max_h_err:.2e}, Mean Abs Error = {mean_h_err:.2e}")

max_true_dg_err = df_hess["true_dg_vs_num_err"].max()
print(f"True Pseudo-Grad Derivative vs Numerical: Max Abs Error = {max_true_dg_err:.2e}")

# Check minimum values of exact Hessian vs surrogate Hessian
min_exact_h = df_hess[df_hess["gamma"] == 2.0]["h_exact"].min()
min_surr_h = df_hess[df_hess["gamma"] == 2.0]["h_surr"].min()
print(f"Gamma=2.0: Minimum Exact Hessian = {min_exact_h:.5f} (STRICTLY NEGATIVE!)")
print(f"Gamma=2.0: Minimum Surrogate Hessian = {min_surr_h:.5f} (STRICTLY POSITIVE/NON-NEGATIVE)")

# ==============================================================================
# 3. SPECIAL INVESTIGATION: GAMMA = 0 ACROSS ALPHA VALUES
# ==============================================================================

print("\n=== 2. INVESTIGATING GAMMA = 0 SPECIAL CASE ACROSS ALPHAS ===")

alpha_test_list = [0.25, 0.50, 0.75, 0.85, 1.00]
gamma_zero_rows = []

for a_val in alpha_test_list:
    for y_val in [1, 0]:
        for z_val in [-2.0, 0.0, 2.0]:
            p_val = sigmoid(z_val)
            g_val = implemented_pseudo_gradient(z_val, y_val, alpha=a_val, gamma=0.0)
            
            # Theoretical equations:
            # y=1: a*(p - 1)
            # y=0: (1 - a)*p
            if y_val == 1:
                expected_g = a_val * (p_val - 1.0)
            else:
                expected_g = (1.0 - a_val) * p_val
                
            err = abs(g_val - expected_g)
            gamma_zero_rows.append({
                "alpha": a_val,
                "y": y_val,
                "z": z_val,
                "p": p_val,
                "g_computed": g_val,
                "g_expected": expected_g,
                "abs_error": err
            })

df_gamma_zero = pd.DataFrame(gamma_zero_rows)
print("Gamma=0 evaluation highlights:")
print(df_gamma_zero[df_gamma_zero["z"] == 0.0][["alpha", "y", "p", "g_computed", "g_expected"]])

# Document critical finding:
# For y=0, alpha=1.0: g_computed = 0.0!
g_neg_alpha_1 = df_gamma_zero[(df_gamma_zero["alpha"] == 1.0) & (df_gamma_zero["y"] == 0)]["g_computed"].values
print(f"Gradient on negative class (y=0) when alpha=1.0: {g_neg_alpha_1} -> EXACT ZERO!")

# ==============================================================================
# 4. PUBLICATION-QUALITY COMPARISON PLOTS
# ==============================================================================

print("\n=== 3. GENERATING COMPARISON GRAPHS ===")

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

z_plot = np.linspace(-5.0, 5.0, 250)

# 1. Gradient Comparison (alpha=0.85, gamma=2.0)
ax = axes[0, 0]
g1_exact = [exact_focal_gradient(z, 1, 0.85, 2.0) for z in z_plot]
g1_impl = [implemented_pseudo_gradient(z, 1, 0.85, 2.0) for z in z_plot]
g0_exact = [exact_focal_gradient(z, 0, 0.85, 2.0) for z in z_plot]
g0_impl = [implemented_pseudo_gradient(z, 0, 0.85, 2.0) for z in z_plot]

ax.plot(z_plot, g1_exact, 'b-', label='Exact Focal Grad (y=1)', linewidth=2)
ax.plot(z_plot, g1_impl, 'b--', label='Pseudo-Grad Impl (y=1)', linewidth=2)
ax.plot(z_plot, g0_exact, 'r-', label='Exact Focal Grad (y=0)', linewidth=2)
ax.plot(z_plot, g0_impl, 'r--', label='Pseudo-Grad Impl (y=0)', linewidth=2)
ax.set_title("Gradient Comparison: Exact Focal vs Pseudo-Gradient\n(alpha=0.85, gamma=2.0)", fontsize=12, fontweight='bold')
ax.set_xlabel("Logit z", fontsize=11)
ax.set_ylabel("Gradient g(z)", fontsize=11)
ax.axhline(0, color='gray', linestyle=':', alpha=0.7)
ax.legend(loc='best', frameon=True)

# 2. Gradient Discrepancy (impl - exact)
ax = axes[0, 1]
ax.plot(z_plot, np.array(g1_impl) - np.array(g1_exact), 'b-', label='y=1 Discrepancy: g_impl - g_exact', linewidth=2)
ax.plot(z_plot, np.array(g0_impl) - np.array(g0_exact), 'r-', label='y=0 Discrepancy: g_impl - g_exact', linewidth=2)
ax.set_title("Modulating Factor Derivative Impact\n(Omission of d(1-pt)^gamma/dz)", fontsize=12, fontweight='bold')
ax.set_xlabel("Logit z", fontsize=11)
ax.set_ylabel("Gradient Discrepancy", fontsize=11)
ax.axhline(0, color='gray', linestyle=':', alpha=0.7)
ax.legend(loc='best', frameon=True)

# 3. Hessian Comparison (alpha=0.85, gamma=2.0)
ax = axes[1, 0]
h1_exact = [exact_focal_hessian(z, 1, 0.85, 2.0) for z in z_plot]
h1_surr = [implemented_surrogate_hessian(z, 1, 0.85, 2.0) for z in z_plot]
h1_true_dg = [true_derivative_of_pseudo_gradient(z, 1, 0.85, 2.0) for z in z_plot]

ax.plot(z_plot, h1_exact, 'k-', label='Exact Focal Hessian (y=1)', linewidth=2)
ax.plot(z_plot, h1_surr, 'g--', label='Surrogate Hessian Impl (y=1)', linewidth=2)
ax.plot(z_plot, h1_true_dg, 'c-.', label='True d(g_impl)/dz (y=1)', linewidth=2)
ax.axhline(0, color='red', linestyle='--', label='Hessian = 0 Boundary', alpha=0.8)
ax.fill_between(z_plot, h1_exact, 0, where=(np.array(h1_exact) < 0), color='salmon', alpha=0.4, label='Non-Convex Region (h < 0)')
ax.set_title("Hessian Behavior & Non-Convexity (y=1)\n(alpha=0.85, gamma=2.0)", fontsize=12, fontweight='bold')
ax.set_xlabel("Logit z", fontsize=11)
ax.set_ylabel("Hessian h(z)", fontsize=11)
ax.legend(loc='best', frameon=True)

# 4. Hessian Comparison (y=0)
ax = axes[1, 1]
h0_exact = [exact_focal_hessian(z, 0, 0.85, 2.0) for z in z_plot]
h0_surr = [implemented_surrogate_hessian(z, 0, 0.85, 2.0) for z in z_plot]
h0_true_dg = [true_derivative_of_pseudo_gradient(z, 0, 0.85, 2.0) for z in z_plot]

ax.plot(z_plot, h0_exact, 'k-', label='Exact Focal Hessian (y=0)', linewidth=2)
ax.plot(z_plot, h0_surr, 'g--', label='Surrogate Hessian Impl (y=0)', linewidth=2)
ax.plot(z_plot, h0_true_dg, 'c-.', label='True d(g_impl)/dz (y=0)', linewidth=2)
ax.axhline(0, color='red', linestyle='--', label='Hessian = 0 Boundary', alpha=0.8)
ax.fill_between(z_plot, h0_exact, 0, where=(np.array(h0_exact) < 0), color='salmon', alpha=0.4, label='Non-Convex Region (h < 0)')
ax.set_title("Hessian Behavior & Non-Convexity (y=0)\n(alpha=0.85, gamma=2.0)", fontsize=12, fontweight='bold')
ax.set_xlabel("Logit z", fontsize=11)
ax.set_ylabel("Hessian h(z)", fontsize=11)
ax.legend(loc='best', frameon=True)

plt.tight_layout()
graph_grad_path = os.path.join(graphs_dir, "focal_gradient_comparison.png")
fig.savefig(graph_grad_path)
fig.savefig(os.path.join(alt_base_dir, "graphs", "phase4", "focal_gradient_comparison.png"))
plt.close(fig)
print(f"Saved: {graph_grad_path}")

# Second focused graph: Hessian zoomed with probability domain
fig2, ax2 = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
p_grid = np.linspace(1e-4, 1.0 - 1e-4, 500)
h1_exact_p = [exact_focal_hessian(np.log(p / (1.0 - p)), 1, 0.85, 2.0) for p in p_grid]
h1_surr_p = [implemented_surrogate_hessian(np.log(p / (1.0 - p)), 1, 0.85, 2.0) for p in p_grid]
h0_exact_p = [exact_focal_hessian(np.log(p / (1.0 - p)), 0, 0.85, 2.0) for p in p_grid]
h0_surr_p = [implemented_surrogate_hessian(np.log(p / (1.0 - p)), 0, 0.85, 2.0) for p in p_grid]

ax2[0].plot(p_grid, h1_exact_p, 'k-', label='Exact Focal Hessian (y=1)', linewidth=2)
ax2[0].plot(p_grid, h1_surr_p, 'g--', label='Surrogate Hessian (y=1)', linewidth=2)
ax2[0].axhline(0, color='red', linestyle=':', label='Zero Curvature')
ax2[0].fill_between(p_grid, h1_exact_p, 0, where=(np.array(h1_exact_p) < 0), color='salmon', alpha=0.4, label='h < 0 Region (p < 0.058)')
ax2[0].set_title("Positive Class Hessian vs Probability p", fontsize=12, fontweight='bold')
ax2[0].set_xlabel("Probability p = sigma(z)", fontsize=11)
ax2[0].set_ylabel("Hessian", fontsize=11)
ax2[0].legend(loc='best', frameon=True)

ax2[1].plot(p_grid, h0_exact_p, 'k-', label='Exact Focal Hessian (y=0)', linewidth=2)
ax2[1].plot(p_grid, h0_surr_p, 'g--', label='Surrogate Hessian (y=0)', linewidth=2)
ax2[1].axhline(0, color='red', linestyle=':', label='Zero Curvature')
ax2[1].fill_between(p_grid, h0_exact_p, 0, where=(np.array(h0_exact_p) < 0), color='salmon', alpha=0.4, label='h < 0 Region (p > 0.942)')
ax2[1].set_title("Negative Class Hessian vs Probability p", fontsize=12, fontweight='bold')
ax2[1].set_xlabel("Probability p = sigma(z)", fontsize=11)
ax2[1].set_ylabel("Hessian", fontsize=11)
ax2[1].legend(loc='best', frameon=True)

plt.tight_layout()
graph_hess_path = os.path.join(graphs_dir, "focal_hessian_comparison.png")
fig2.savefig(graph_hess_path)
fig2.savefig(os.path.join(alt_base_dir, "graphs", "phase4", "focal_hessian_comparison.png"))
plt.close(fig2)
print(f"Saved: {graph_hess_path}")

# ==============================================================================
# 5. CONTROLLED EMPIRICAL VERIFICATION ON SGCC DATASET
# ==============================================================================

print("\n=== 4. RUNNING CONTROLLED EMPIRICAL MODEL COMPARISON ===")

data_path = os.path.join(base_dir, "data", "processed", "sgcc_features_real_v2.csv")
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

# Stratified Split (Seed 42)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.17647, stratify=y_train_val, random_state=42
)

# Imputation strictly on training partition
train_medians = X_train.median()
X_train_imp = X_train.fillna(train_medians)
X_val_imp = X_val.fillna(train_medians)

# Define objective candidate functions for LightGBM
def obj_surrogate(preds, train_set):
    """Candidate B: Implemented Focal-Weighted Pseudo-Gradient with Surrogate Hessian."""
    labels = train_set.get_label()
    p = np.clip(sigmoid(preds), 1e-15, 1.0 - 1e-15)
    pt = np.where(labels == 1, p, 1.0 - p)
    at = np.where(labels == 1, 0.85, 0.15)
    grad = at * ((1.0 - pt) ** 2.0) * (p - labels)
    hess = at * ((1.0 - pt) ** 2.0) * (p * (1.0 - p)) * (2.0 * np.abs(p - labels) + 1.0)
    return grad, hess

def obj_exact_clipped(preds, train_set):
    """Candidate C: Exact Focal Gradient with Clipped Exact Hessian (max(h, 1e-6))."""
    labels = train_set.get_label()
    p = np.clip(sigmoid(preds), 1e-15, 1.0 - 1e-15)
    a, g = 0.85, 2.0
    g1 = a * ((1.0 - p) ** g) * (g * p * np.log(p) + p - 1.0)
    g0 = (1.0 - a) * (p ** g) * (p - g * (1.0 - p) * np.log(1.0 - p))
    grad = np.where(labels == 1, g1, g0)
    
    h1 = a * ((1.0 - p) ** g) * p * (g * (1.0 - (1.0 + g) * p) * np.log(p) + (1.0 - p) * (2.0 * g + 1.0))
    h0 = (1.0 - a) * (p ** g) * (1.0 - p) * (p * (2.0 * g + 1.0) + g * (p - g * (1.0 - p)) * np.log(1.0 - p))
    hess = np.where(labels == 1, h1, h0)
    hess = np.clip(hess, 1e-6, None) # Clip negative curvature
    return grad, hess

def obj_true_dg(preds, train_set):
    """Candidate D: Pseudo-Gradient with its True Mathematical Derivative (dg_impl/dz)."""
    labels = train_set.get_label()
    p = np.clip(sigmoid(preds), 1e-15, 1.0 - 1e-15)
    pt = np.where(labels == 1, p, 1.0 - p)
    at = np.where(labels == 1, 0.85, 0.15)
    grad = at * ((1.0 - pt) ** 2.0) * (p - labels)
    hess = at * 3.0 * p * (1.0 - p) * ((1.0 - pt) ** 2.0)
    return grad, hess

def eval_auc(preds, train_set):
    labels = train_set.get_label()
    probs = sigmoid(preds)
    return 'auc', roc_auc_score(labels, probs), True

candidates = {
    "Candidate B (Implemented Surrogate)": obj_surrogate,
    "Candidate C (Exact Grad + Clipped Hess)": obj_exact_clipped,
    "Candidate D (Pseudo-Grad + True dg/dz)": obj_true_dg
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
candidate_results = []

for name, fobj in candidates.items():
    print(f"\nEvaluating: {name}")
    
    # 5-Fold Stratified CV on Training Partition
    cv_f1s = []
    cv_pr_aucs = []
    cv_roc_aucs = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_imp, y_train)):
        X_tr, y_tr = X_train_imp.iloc[train_idx], y_train.iloc[train_idx]
        X_va, y_va = X_train_imp.iloc[val_idx], y_train.iloc[val_idx]
        
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)
        
        bst_cv = lgb.train(
            {
                'objective': fobj,
                'learning_rate': 0.05,
                'num_leaves': 63,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': 42 + fold
            },
            dtrain,
            num_boost_round=250,
            valid_sets=[dval],
            feval=eval_auc,
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )
        
        raw_cv = bst_cv.predict(X_va, raw_score=True)
        probs_cv = sigmoid(raw_cv)
        
        pr_auc = average_precision_score(y_va, probs_cv)
        roc_auc = roc_auc_score(y_va, probs_cv)
        
        # Best F1 on this fold
        best_fold_f1 = max(
            f1_score(y_va, (probs_cv >= t).astype(int), zero_division=0)
            for t in np.arange(0.2, 0.8, 0.02)
        )
        
        cv_f1s.append(best_fold_f1)
        cv_pr_aucs.append(pr_auc)
        cv_roc_aucs.append(roc_auc)
    
    # Train full model on X_train_imp, evaluate on X_val_imp
    dtrain_full = lgb.Dataset(X_train_imp, label=y_train)
    dval_full = lgb.Dataset(X_val_imp, label=y_val, reference=dtrain_full)
    
    bst_full = lgb.train(
        {
            'objective': fobj,
            'learning_rate': 0.05,
            'num_leaves': 63,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        },
        dtrain_full,
        num_boost_round=250,
        valid_sets=[dval_full],
        feval=eval_auc,
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
    )
    
    val_raw = bst_full.predict(X_val_imp, raw_score=True)
    val_probs = sigmoid(val_raw)
    
    val_pr_auc = average_precision_score(y_val, val_probs)
    val_roc_auc = roc_auc_score(y_val, val_probs)
    
    best_t = 0.50
    best_val_f1 = 0.0
    for t in np.arange(0.2, 0.8, 0.02):
        score = f1_score(y_val, (val_probs >= t).astype(int), zero_division=0)
        if score > best_val_f1:
            best_val_f1 = score
            best_t = t
            
    val_prec = precision_score(y_val, (val_probs >= best_t).astype(int), zero_division=0)
    val_rec = recall_score(y_val, (val_probs >= best_t).astype(int), zero_division=0)
    
    candidate_results.append({
        "candidate": name,
        "cv_f1_mean": np.mean(cv_f1s),
        "cv_f1_std": np.std(cv_f1s),
        "cv_pr_auc_mean": np.mean(cv_pr_aucs),
        "cv_roc_auc_mean": np.mean(cv_roc_aucs),
        "val_f1": best_val_f1,
        "val_pr_auc": val_pr_auc,
        "val_roc_auc": val_roc_auc,
        "optimal_val_threshold": best_t,
        "val_precision": val_prec,
        "val_recall": val_rec,
        "trees": bst_full.num_trees()
    })

df_candidates = pd.DataFrame(candidate_results)
csv_cand_path = os.path.join(results_dir, "sgcc_focal_candidates_comparison.csv")
df_candidates.to_csv(csv_cand_path, index=False)
df_candidates.to_csv(os.path.join(alt_base_dir, "results", "phase4", "sgcc_focal_candidates_comparison.csv"), index=False)

print("\n=== EMPIRICAL COMPARISON SUMMARY ===")
print(df_candidates[["candidate", "cv_f1_mean", "cv_f1_std", "val_f1", "val_pr_auc", "val_roc_auc", "trees"]])

print("\nWorkstream A mathematical verification script executed successfully.")
