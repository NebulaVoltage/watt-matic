import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Create directories
os.makedirs(r"d:\watt-matic\gridbalance\results\optimization_v2", exist_ok=True)
os.makedirs(r"d:\watt-matic\gridbalance\graphs\optimization_v2\phase3", exist_ok=True)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

# 1. Exact Conventional Focal Loss
def exact_focal_loss(z, y, alpha=0.85, gamma=2.0, eps=1e-15):
    p = sigmoid(z)
    p = np.clip(p, eps, 1.0 - eps)
    pt = np.where(y == 1, p, 1.0 - p)
    at = np.where(y == 1, alpha, 1.0 - alpha)
    return -at * ((1.0 - pt) ** gamma) * np.log(pt)

# 2. Exact Analytical Derivative of Conventional Focal Loss w.r.t logit z
def exact_focal_loss_grad_exact(z, y, alpha=0.85, gamma=2.0, eps=1e-15):
    p = sigmoid(z)
    p = np.clip(p, eps, 1.0 - eps)
    # d(FL)/dz = d(FL)/dp * dp/dz, where dp/dz = p * (1 - p)
    if y == 1:
        # FL = -alpha * (1 - p)^gamma * log(p)
        # dFL/dp = alpha * [gamma * (1-p)^(gamma-1) * log(p) - (1-p)^gamma / p]
        # dFL/dz = dFL/dp * p * (1-p) = alpha * (1-p)^gamma * [gamma * p * log(p) + p - 1]
        term1 = gamma * p * np.log(p)
        grad = alpha * ((1.0 - p) ** gamma) * (term1 + p - 1.0)
    else:
        # FL = -(1 - alpha) * p^gamma * log(1 - p)
        # dFL/dp = -(1 - alpha) * [gamma * p^(gamma-1) * log(1-p) - p^gamma / (1-p)]
        # dFL/dz = dFL/dp * p * (1-p) = (1 - alpha) * p^gamma * [-gamma * (1-p) * log(1-p) + p]
        term1 = -gamma * (1.0 - p) * np.log(1.0 - p)
        grad = (1.0 - alpha) * (p ** gamma) * (term1 + p)
    return grad

# Exact Analytical Hessian of Conventional Focal Loss w.r.t logit z
def exact_focal_loss_hess_exact(z, y, alpha=0.85, gamma=2.0, eps=1e-15):
    delta = 1e-5
    g_plus = exact_focal_loss_grad_exact(z + delta, y, alpha, gamma, eps)
    g_minus = exact_focal_loss_grad_exact(z - delta, y, alpha, gamma, eps)
    return (g_plus - g_minus) / (2.0 * delta)

# 3. Implemented LightGBM Custom Objective (Pseudo-Gradient / Modulated Gradient)
def implemented_focal_grad(z, y, alpha=0.85, gamma=2.0, eps=1e-15):
    p = sigmoid(z)
    p = np.clip(p, eps, 1.0 - eps)
    pt = np.where(y == 1, p, 1.0 - p)
    at = np.where(y == 1, alpha, 1.0 - alpha)
    grad = at * ((1.0 - pt) ** gamma) * (p - y)
    return grad

def implemented_focal_hess(z, y, alpha=0.85, gamma=2.0, eps=1e-15):
    p = sigmoid(z)
    p = np.clip(p, eps, 1.0 - eps)
    pt = np.where(y == 1, p, 1.0 - p)
    at = np.where(y == 1, alpha, 1.0 - alpha)
    hess = at * ((1.0 - pt) ** gamma) * (p * (1.0 - p)) * (gamma * np.abs(p - y) + 1.0)
    return hess

# 4. Numerical Finite Differences
def numerical_grad(z, y, alpha=0.85, gamma=2.0, delta=1e-6):
    f_plus = exact_focal_loss(z + delta, y, alpha, gamma)
    f_minus = exact_focal_loss(z - delta, y, alpha, gamma)
    return (f_plus - f_minus) / (2.0 * delta)

def numerical_hess(z, y, alpha=0.85, gamma=2.0, delta=1e-5):
    f_plus = exact_focal_loss(z + delta, y, alpha, gamma)
    f_0 = exact_focal_loss(z, y, alpha, gamma)
    f_minus = exact_focal_loss(z - delta, y, alpha, gamma)
    return (f_plus - 2.0 * f_0 + f_minus) / (delta ** 2)

# Run Grid Analysis
logits = np.linspace(-6, 6, 121)
results = []

for y_val in [1, 0]:
    for z_val in logits:
        p_val = sigmoid(z_val)
        
        # Exact FL numerical & analytical
        fl_val = exact_focal_loss(z_val, y_val, alpha=0.85, gamma=2.0)
        g_num = numerical_grad(z_val, y_val, alpha=0.85, gamma=2.0)
        g_exact = exact_focal_loss_grad_exact(z_val, y_val, alpha=0.85, gamma=2.0)
        h_num = numerical_hess(z_val, y_val, alpha=0.85, gamma=2.0)
        h_exact = exact_focal_loss_hess_exact(z_val, y_val, alpha=0.85, gamma=2.0)
        
        # Implemented formula
        g_impl = implemented_focal_grad(z_val, y_val, alpha=0.85, gamma=2.0)
        h_impl = implemented_focal_hess(z_val, y_val, alpha=0.85, gamma=2.0)
        
        # Discrepancy between exact analytical and numerical (Sanity check)
        exact_deriv_error = abs(g_exact - g_num)
        
        # Difference between implemented formula and exact derivative
        impl_vs_exact_grad = abs(g_impl - g_exact)
        
        impl_diff = g_impl - g_exact
        
        results.append({
            "y": y_val,
            "logit": z_val,
            "prob": p_val,
            "FL_loss": fl_val,
            "grad_numerical": g_num,
            "grad_exact_analytical": g_exact,
            "grad_implemented": g_impl,
            "exact_deriv_error": exact_deriv_error,
            "impl_vs_exact_diff": impl_diff,
            "hess_numerical": h_num,
            "hess_exact_analytical": h_exact,
            "hess_implemented": h_impl,
            "hess_impl_vs_exact_diff": (h_impl - h_exact)
        })

df_res = pd.DataFrame(results)
csv_out = r"d:\watt-matic\gridbalance\results\optimization_v2\focal_loss_math_verification.csv"
df_res.to_csv(csv_out, index=False)
print(f"Saved numerical verification to {csv_out}")

# Summary comparisons
print("\n--- Summary of Derivative Verification ---")
print(f"Max discrepancy between exact analytical derivative and finite differences: {df_res['exact_deriv_error'].max():.2e}")

# Check when gamma=0 (Should match standard cross-entropy exactly)
ce_check_diff = []
for z_val in [-4, -2, 0, 2, 4]:
    g_exact_0 = exact_focal_loss_grad_exact(z_val, 1, alpha=1.0, gamma=0.0)
    g_impl_0 = implemented_focal_grad(z_val, 1, alpha=1.0, gamma=0.0)
    ce_check_diff.append(abs(g_exact_0 - g_impl_0))
print(f"When gamma=0, alpha=1 (Standard Binary Cross Entropy), max difference: {max(ce_check_diff):.2e}")

# Check differences at gamma=2.0
df_y1 = df_res[df_res['y'] == 1]
df_y0 = df_res[df_res['y'] == 0]
print(f"For y=1 (gamma=2.0, alpha=0.85): Max |g_impl - g_exact|: {df_y1['impl_vs_exact_diff'].abs().max():.4f}")
print(f"For y=0 (gamma=2.0, alpha=0.85): Max |g_impl - g_exact|: {df_y0['impl_vs_exact_diff'].abs().max():.4f}")

# Check Hessian positivity
print(f"Implemented Hessian minimum value across grid: {df_res['hess_implemented'].min():.6f}")
print(f"Exact Hessian minimum value across grid: {df_res['hess_exact_analytical'].min():.6f}")

# Plot Comparison
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# Subplot 1: Gradient for y=1
axs[0, 0].plot(df_y1['logit'], df_y1['grad_exact_analytical'], label='Exact FL Derivative (Analytical)', color='#1f77b4', lw=2.5)
axs[0, 0].plot(df_y1['logit'], df_y1['grad_numerical'], '--', label='Numerical Finite Difference', color='#2ca02c', lw=1.5)
axs[0, 0].plot(df_y1['logit'], df_y1['grad_implemented'], ':', label='Implemented Objective (Pseudo-Grad)', color='#d62728', lw=2.5)
axs[0, 0].set_title('Positive Class (y=1) Gradient Comparison', fontsize=12, fontweight='bold')
axs[0, 0].set_xlabel('Logit (z)')
axs[0, 0].set_ylabel('Gradient')
axs[0, 0].legend()
axs[0, 0].grid(True, alpha=0.3)

# Subplot 2: Gradient for y=0
axs[0, 1].plot(df_y0['logit'], df_y0['grad_exact_analytical'], label='Exact FL Derivative (Analytical)', color='#1f77b4', lw=2.5)
axs[0, 1].plot(df_y0['logit'], df_y0['grad_numerical'], '--', label='Numerical Finite Difference', color='#2ca02c', lw=1.5)
axs[0, 1].plot(df_y0['logit'], df_y0['grad_implemented'], ':', label='Implemented Objective (Pseudo-Grad)', color='#d62728', lw=2.5)
axs[0, 1].set_title('Negative Class (y=0) Gradient Comparison', fontsize=12, fontweight='bold')
axs[0, 1].set_xlabel('Logit (z)')
axs[0, 1].set_ylabel('Gradient')
axs[0, 1].legend()
axs[0, 1].grid(True, alpha=0.3)

# Subplot 3: Hessian for y=1
axs[1, 0].plot(df_y1['logit'], df_y1['hess_exact_analytical'], label='Exact FL Hessian (Analytical)', color='#1f77b4', lw=2)
axs[1, 0].plot(df_y1['logit'], df_y1['hess_implemented'], label='Implemented Surrogate Hessian', color='#d62728', lw=2)
axs[1, 0].set_title('Positive Class (y=1) Hessian Comparison', fontsize=12, fontweight='bold')
axs[1, 0].set_xlabel('Logit (z)')
axs[1, 0].set_ylabel('Hessian')
axs[1, 0].legend()
axs[1, 0].grid(True, alpha=0.3)

# Subplot 4: Gradient Difference (Modulating Term Gap)
axs[1, 1].plot(df_y1['logit'], df_y1['impl_vs_exact_diff'], label='y=1 Discrepancy (Impl - Exact)', color='#9467bd', lw=2)
axs[1, 1].plot(df_y0['logit'], df_y0['impl_vs_exact_diff'], label='y=0 Discrepancy (Impl - Exact)', color='#8c564b', lw=2)
axs[1, 1].set_title('Gradient Difference: Implemented vs Exact Derivative', fontsize=12, fontweight='bold')
axs[1, 1].set_xlabel('Logit (z)')
axs[1, 1].set_ylabel('Discrepancy: g_impl - g_exact')
axs[1, 1].legend()
axs[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = r"d:\watt-matic\gridbalance\graphs\optimization_v2\phase3\focal_loss_mathematical_comparison.png"
plt.savefig(plot_path, dpi=300)
print(f"Saved plot to {plot_path}")
