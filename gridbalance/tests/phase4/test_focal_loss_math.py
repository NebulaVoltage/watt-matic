"""
GridBalance Phase 4 Workstream A: Automated Mathematical Verification Test Suite
Tests mathematical properties, gradients, Hessians, non-negativity, and edge-cases of Focal Loss.
"""

import pytest
import numpy as np


def sigmoid(z):
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def exact_focal_loss(z, y, alpha=0.85, gamma=2.0, eps=1e-15):
    p = np.clip(sigmoid(z), eps, 1.0 - eps)
    pt = np.where(y == 1, p, 1.0 - p)
    at = np.where(y == 1, alpha, 1.0 - alpha)
    return -at * ((1.0 - pt) ** gamma) * np.log(pt)


def exact_focal_gradient(z, y, alpha=0.85, gamma=2.0, eps=1e-15):
    p = np.clip(sigmoid(z), eps, 1.0 - eps)
    if y == 1:
        return alpha * ((1.0 - p) ** gamma) * (gamma * p * np.log(p) + p - 1.0)
    else:
        return (1.0 - alpha) * (p ** gamma) * (p - gamma * (1.0 - p) * np.log(1.0 - p))


def exact_focal_hessian(z, y, alpha=0.85, gamma=2.0, eps=1e-15):
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
    p = np.clip(sigmoid(z), eps, 1.0 - eps)
    pt = np.where(y == 1, p, 1.0 - p)
    at = np.where(y == 1, alpha, 1.0 - alpha)
    return at * ((1.0 - pt) ** gamma) * (p - y)


def implemented_surrogate_hessian(z, y, alpha=0.85, gamma=2.0, eps=1e-15):
    p = np.clip(sigmoid(z), eps, 1.0 - eps)
    pt = np.where(y == 1, p, 1.0 - p)
    at = np.where(y == 1, alpha, 1.0 - alpha)
    return at * ((1.0 - pt) ** gamma) * (p * (1.0 - p)) * (gamma * np.abs(p - y) + 1.0)


# ==============================================================================
# TEST CASES
# ==============================================================================

def test_exact_gradient_matches_numerical_finite_difference():
    """Verify exact analytical focal gradient matches central finite difference within 1e-5."""
    z_vals = np.linspace(-6, 6, 25)
    delta = 1e-6
    for y in [0, 1]:
        for z in z_vals:
            g_exact = exact_focal_gradient(z, y, alpha=0.85, gamma=2.0)
            g_num = (exact_focal_loss(z + delta, y, 0.85, 2.0) - exact_focal_loss(z - delta, y, 0.85, 2.0)) / (2 * delta)
            assert np.isclose(g_exact, g_num, atol=1e-5), f"Gradient mismatch at z={z}, y={y}"


def test_exact_hessian_matches_numerical_differentiation_of_gradient():
    """Verify exact analytical focal Hessian matches derivative of exact gradient within 1e-5."""
    z_vals = np.linspace(-6, 6, 25)
    delta = 1e-6
    for y in [0, 1]:
        for z in z_vals:
            h_exact = exact_focal_hessian(z, y, alpha=0.85, gamma=2.0)
            h_num = (exact_focal_gradient(z + delta, y, 0.85, 2.0) - exact_focal_gradient(z - delta, y, 0.85, 2.0)) / (2 * delta)
            assert np.isclose(h_exact, h_num, atol=1e-5), f"Hessian mismatch at z={z}, y={y}"


def test_surrogate_hessian_is_strictly_non_negative():
    """Verify implemented surrogate Hessian is >= 0 for all z in [-10, 10] and y in {0, 1}."""
    z_vals = np.linspace(-10, 10, 201)
    for y in [0, 1]:
        for z in z_vals:
            h_surr = implemented_surrogate_hessian(z, y, alpha=0.85, gamma=2.0)
            assert h_surr >= 0.0, f"Surrogate Hessian negative at z={z}, y={y}: {h_surr}"


def test_exact_focal_hessian_contains_negative_curvature_regions():
    """Verify exact focal loss is non-convex by demonstrating negative Hessian for hard misclassifications."""
    # For y=1 (positive), p close to 0 (hard false negative, z << 0)
    z_hard_fn = -4.0  # p ~ 0.018
    h1 = exact_focal_hessian(z_hard_fn, y=1, alpha=0.85, gamma=2.0)
    assert h1 < 0.0, f"Expected negative Hessian for y=1 at z={z_hard_fn}, got {h1}"
    
    # For y=0 (negative), p close to 1 (hard false positive, z >> 0)
    z_hard_fp = 4.0  # p ~ 0.982
    h0 = exact_focal_hessian(z_hard_fp, y=0, alpha=0.85, gamma=2.0)
    assert h0 < 0.0, f"Expected negative Hessian for y=0 at z={z_hard_fp}, got {h0}"


def test_gamma_zero_special_case_and_alpha_one_zeroing():
    """Verify gamma=0 properties: alpha=0.5 yields 0.5*(p-y); alpha=1.0 zeroes negative class gradient."""
    z_vals = np.linspace(-4, 4, 9)
    for z in z_vals:
        p = sigmoid(z)
        
        # At alpha=0.5, gamma=0: g = 0.5 * (p - y)
        g1_half = implemented_pseudo_gradient(z, y=1, alpha=0.5, gamma=0.0)
        g0_half = implemented_pseudo_gradient(z, y=0, alpha=0.5, gamma=0.0)
        assert np.isclose(g1_half, 0.5 * (p - 1.0), atol=1e-12)
        assert np.isclose(g0_half, 0.5 * (p - 0.0), atol=1e-12)
        
        # At alpha=1.0, gamma=0: negative class gradient is identically 0
        g0_one = implemented_pseudo_gradient(z, y=0, alpha=1.0, gamma=0.0)
        assert np.isclose(g0_one, 0.0, atol=1e-12), f"Negative gradient not zero at alpha=1.0: {g0_one}"
        
        # Positive class at alpha=1.0 is standard BCE: 1.0 * (p - 1)
        g1_one = implemented_pseudo_gradient(z, y=1, alpha=1.0, gamma=0.0)
        assert np.isclose(g1_one, p - 1.0, atol=1e-12)


def test_pseudo_gradient_differs_from_exact_focal_gradient():
    """Verify that implemented pseudo-gradient differs from exact focal gradient when gamma > 0."""
    z_vals = np.linspace(-3, 3, 10)
    discrepancies = []
    for y in [0, 1]:
        for z in z_vals:
            g_exact = exact_focal_gradient(z, y, alpha=0.85, gamma=2.0)
            g_impl = implemented_pseudo_gradient(z, y, alpha=0.85, gamma=2.0)
            discrepancies.append(abs(g_impl - g_exact))
            
    # Max discrepancy must be non-zero due to omission of modulating factor derivative
    assert max(discrepancies) > 0.05, f"Discrepancy too small: {max(discrepancies)}"


def test_pseudo_gradient_global_lipschitz_continuity():
    """Verify implemented pseudo-gradient satisfies Lipschitz continuity with K <= 0.27."""
    z_vals = np.linspace(-10, 10, 500)
    K_bound = 0.27
    for y in [0, 1]:
        g_vals = np.array([implemented_pseudo_gradient(z, y, alpha=0.85, gamma=2.0) for z in z_vals])
        dz = z_vals[1] - z_vals[0]
        empirical_dg_dz = np.abs(np.diff(g_vals) / dz)
        max_slope = np.max(empirical_dg_dz)
        assert max_slope <= K_bound, f"Empirical Lipschitz constant {max_slope} exceeds bound {K_bound}"
