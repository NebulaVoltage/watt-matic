# GridBalance — Phase 4 Workstream B: Candidate Objective Comparison
**Document ID:** `GB-P4-WB-OBJECTIVE-COMPARISON`  
**Research Stage:** Phase 4 — Research Hardening, Reproducibility & Scientific Verification  
**Author:** Antigravity Senior ML Research Scientist, Statistical Evaluation Specialist  
**Date:** October 31, 2026  
**Status:** Comparative Audit Complete — Strict Comparability Rules Enforced  

---

## 1. Principles of Fair Scientific Comparison

To avoid misleading scientific claims, all candidate comparisons in GridBalance are governed by the **Principle of Strict Invariance**:
> Two performance numbers may be directly compared to compute relative delta ($\Delta$) **only if** they share:
> 1. The exact same dataset partition and sample observations.
> 2. The exact same target label or regression ground truth.
> 3. The exact same evaluation metric implementation and averaging convention.
> 4. The exact same decision thresholding or post-processing protocol.
>
> **If any of these conditions are violated, the comparison is strictly labeled INCOMPARABLE.**

---

## 2. Comparison A: SGCC Production XGBoost vs Candidate B

* **Baseline:** Production XGBoost (`models/sgcc_tuned/xgboost_reduced_best.joblib`, 18 features).
* **Candidate B:** LightGBM Focal Loss (`models/optimization_v2/lgbm_focal_loss_best.joblib`, 18 features, $\alpha=0.85, \gamma=2.0$).

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                COMPARISON A: PRODUCTION XGBOOST vs CANDIDATE B                         │
├────────────────────┬─────────────────────────────┬───────────────────────────┬──────────────┬──────────┤
│ Metric Regime      │ Production XGBoost          │ Candidate B (Surrogate)   │ Delta (Δ)    │ Status   │
├────────────────────┼─────────────────────────────┼───────────────────────────┼──────────────┼──────────┤
│ 5-Fold CV F1       │ 0.4210 ± 0.0132             │ 0.4177 ± 0.0118           │ -0.0033      │ Valid    │
│ Validation F1      │ 0.4037                      │ 0.4469                    │ +0.0432      │ Valid    │
│ Validation PR-AUC  │ 0.4009                      │ 0.4252                    │ +0.0243      │ Valid    │
│ Validation ROC-AUC │ 0.8216                      │ 0.8445                    │ +0.0229      │ Valid    │
│ Held-Out Test F1   │ 0.4105 (Imputed) / 0.3592   │ 0.4479                    │ +0.0374      │ Valid    │
│ Held-Out Test PR   │ 0.3762                      │ 0.4178                    │ +0.0416      │ Valid    │
│ Raw Test vs Imputed│ 0.3592 (Raw) vs 0.4479      │ —                         │ INCOMPARABLE │ Mismatch │
└────────────────────┴─────────────────────────────┴───────────────────────────┴──────────────┴──────────┘
```

### Statistical Analysis:
* **CV Differences:** Across the 5 training folds, XGBoost and Candidate B yield statistically comparable performance ($p = 0.3981$ paired $t$-test; $p = 0.4375$ Wilcoxon signed-rank test).
* **Validation Separation:** Candidate B achieves a **$+10.7\%$ relative gain in Validation F1** ($0.4469$ vs $0.4037$) and **$+6.1\%$ in PR-AUC** ($0.4252$ vs $0.4009$). The focal pseudo-gradient dynamically downweights well-classified background samples, producing smoother decision surfaces on the imbalanced minority class.
* **Incomparable Trap:** Comparing raw XGBoost Test F1 ($0.3592$) to Candidate B Test F1 ($0.4479$) is labeled **INCOMPARABLE** because $0.3592$ used un-imputed NaNs, whereas Candidate B used median imputation. The valid test comparison is $0.4105$ vs $0.4479$ ($+9.1\%$).

---

## 3. Comparison B: Candidate B vs Candidate C (Exact Clipped)

* **Candidate B:** Pseudo-gradient + Surrogate Hessian ($h_{\mathrm{surr}} \ge 0$).
* **Candidate C:** Exact Focal Loss gradient + Clipped Exact Hessian ($h = \max(h_{\mathrm{exact}}, 10^{-6})$).

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                COMPARISON B: CANDIDATE B vs CANDIDATE C                                │
├────────────────────┬─────────────────────────────┬───────────────────────────┬──────────────┬──────────┤
│ Metric Regime      │ Candidate B (Surrogate)     │ Candidate C (Exact Clip)  │ Delta (Δ)    │ Status   │
├────────────────────┼─────────────────────────────┼───────────────────────────┼──────────────┼──────────┤
│ 5-Fold CV F1       │ 0.4177 ± 0.0118             │ 0.4275 ± 0.0116           │ +0.0098      │ Valid    │
│ Validation F1      │ 0.4469                      │ 0.4668                    │ +0.0199      │ Valid    │
│ Validation PR-AUC  │ 0.4252                      │ 0.4398                    │ +0.0146      │ Valid    │
│ Validation ROC-AUC │ 0.8445                      │ 0.8516                    │ +0.0071      │ Valid    │
│ Boosting Trees     │ 117                         │ 202                       │ +85 trees    │ Valid    │
└────────────────────┴─────────────────────────────┴───────────────────────────┴──────────────┴──────────┘
```

### Statistical Analysis:
* **CV Differences:** Candidate C achieves a higher CV F1 ($0.4275$ vs $0.4177$) that is **statistically significant** under a paired $t$-test ($p = 0.0247$).
* **Tree Depth & Regularization:** Candidate C requires 202 trees vs 117 trees for Candidate B. Clipping the exact negative Hessian to $10^{-6}$ artificially flattens the Newton step on hard false negatives, requiring more boosting rounds to converge.
* **Governance Status:** Candidate C is retained as an active research asset. Per Phase 4 preservation rules, it does **not** replace Candidate B in the experimental backend route.

---

## 4. Comparison C: Candidate B vs Candidate D (True Derivative $\frac{\partial g}{\partial z}$)

* **Candidate B:** Pseudo-gradient + Surrogate Hessian.
* **Candidate D:** Pseudo-gradient + True Second Derivative $\frac{\partial g_{\mathrm{impl}}}{\partial z} = \alpha_t (\gamma + 1) p (1 - p)(1 - p_t)^\gamma$.

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                COMPARISON C: CANDIDATE B vs CANDIDATE D                                │
├────────────────────┬─────────────────────────────┬───────────────────────────┬──────────────┬──────────┤
│ Metric Regime      │ Candidate B (Surrogate)     │ Candidate D (True Deriv)  │ Delta (Δ)    │ Status   │
├────────────────────┼─────────────────────────────┼───────────────────────────┼──────────────┼──────────┤
│ 5-Fold CV F1       │ 0.4177 ± 0.0118             │ 0.4342 ± 0.0116           │ +0.0165      │ Valid    │
│ Validation F1      │ 0.4469                      │ 0.4510                    │ +0.0041      │ Valid    │
│ Validation PR-AUC  │ 0.4252                      │ 0.4447                    │ +0.0195      │ Valid    │
│ Validation ROC-AUC │ 0.8445                      │ 0.8526                    │ +0.0081      │ Valid    │
│ Boosting Trees     │ 117                         │ 170                       │ +53 trees    │ Valid    │
└────────────────────┴─────────────────────────────┴───────────────────────────┴──────────────┴──────────┘
```

### Statistical Analysis:
* **CV Differences:** Candidate D yields the highest mean CV F1 ($0.4342 \pm 0.0116$) across all evaluated configurations ($p = 0.0467$ paired $t$-test vs Candidate B).
* **Mathematical Properties:** Because $\frac{\partial g_{\mathrm{impl}}}{\partial z} \ge 0$ everywhere, Candidate D preserves true curvature without artificial threshold clipping. It represents the most mathematically harmonious candidate for future retraining.

---

## 5. Comparison D: UCI Direct vs Difference Forecasting

* **Direct Model:** Predicts next-hour load level $\hat{y}(t+1)$ directly from 29 FIR lag/rolling features.
* **Difference Model:** Predicts next-hour load change $\hat{\Delta}(t+1) = \hat{y}(t+1) - y(t)$, reconstructing $\hat{y}(t+1) = y(t) + \hat{\Delta}(t+1)$.

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                             COMPARISON D: UCI DIRECT vs DIFFERENCE FORECASTING                         │
├────────────────────┬─────────────────────────────┬───────────────────────────┬──────────────┬──────────┤
│ Evaluation Window  │ XGBoost Direct (29 Feat)    │ XGBoost Difference (Delta)│ Delta (Δ)    │ Status   │
├────────────────────┼─────────────────────────────┼───────────────────────────┼──────────────┼──────────┤
│ H1 2014 Validation │ 3,914.41 kWh                │ 4,061.06 kWh              │ +146.65 kWh  │ Valid    │
│ H2 2014 Test MAE   │ 4,440.05 kWh                │ 4,350.13 kWh              │ -89.92 kWh   │ Valid    │
│ H2 2014 Test RMSE  │ 6,686.97 kWh                │ 6,512.27 kWh              │ -174.70 kWh  │ Valid    │
│ H2 2014 Test sMAPE │ 1.94%                       │ 1.92%                     │ -0.02%       │ Valid    │
└────────────────────┴─────────────────────────────┴───────────────────────────┴──────────────┴──────────┘
```

### Key Insight:
* The difference formulation delivers a **$2.0\%$ error reduction** ($4,350.13$ vs $4,440.05\text{ kWh}$) on the held-out test partition by anchoring forecasts directly to the current state $y(t)$, mitigating level drift during rapid ramp periods.
* However, on the Validation partition (H1 2014), the direct model outperformed difference forecasting ($3,914.41$ vs $4,061.06\text{ kWh}$). Therefore, the direct model is rightfully retained as production baseline.

---

## 6. Comparison E: UCI Direct vs EMA Forecasting

* **Direct Model:** Stateless 24-hour history requirement, 29 FIR features.
* **EMA Model:** Requires $\ge 336$-hour contiguous history, 33 features (including EMA-24).

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                COMPARISON E: UCI DIRECT vs EMA FORECASTING                             │
├────────────────────┬─────────────────────────────┬───────────────────────────┬──────────────┬──────────┤
│ Evaluation Window  │ XGBoost Direct (29 Feat)    │ LightGBM EMA (33 Feat)    │ Delta (Δ)    │ Status   │
├────────────────────┼─────────────────────────────┼───────────────────────────┼──────────────┼──────────┤
│ H2 2014 Test MAE   │ 4,440.05 kWh                │ 4,357.54 kWh              │ -82.51 kWh   │ Valid    │
│ H2 2014 Test RMSE  │ 6,686.97 kWh                │ 6,540.10 kWh              │ -146.87 kWh  │ Valid    │
│ History Contract   │ 24 Hours (Stateless)        │ 336 Hours (Contiguous)    │ +312 Hours   │ Valid    │
│ 24h History on EMA │ 4,440.05 kWh                │ 5,420.18 kWh (Corrupted!) │ INCOMPARABLE │ Failure  │
└────────────────────┴─────────────────────────────┴───────────────────────────┴──────────────┴──────────┘
```

> [!CAUTION]
> If LightGBM EMA is evaluated on only 24 hours of history, warmup bias corrupts the EMA features, causing Test MAE to explode to **$5,420.18\text{ kWh}$**. It is **strictly invalid** to serve the EMA model through the standard 24-hour API contract.

---
*Report certified by Antigravity Machine Learning Quality Assurance Suite.*
