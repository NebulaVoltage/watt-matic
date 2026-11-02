# GridBalance: Prioritized Research Improvement Roadmap & Execution Blueprint

> **Role & Authority**: Lead ML Research Scientist, Systems Architect  
> **Repository Root**: `d:\watt-matic\gridbalance`  
> **Status**: PROPOSED ACTIONABLE ROADMAP (Awaiting Faculty / User Approval)  
> **Reference Decision Matrix**: `results/GRIDBALANCE_IMPROVEMENT_PRIORITY.csv`  

---

## 1. ROADMAP PHILOSOPHY & PRIORITIZATION CRITERIA

Every experiment in this roadmap is strictly connected to an empirical weakness identified in the research audit. No experiment is proposed for theoretical novelty alone. Prioritization follows four strict tiers:
- **`P0 — Immediate Remediation`**: Must be fixed before running further modeling experiments. Focuses on metric consistency and benchmark integrity.
- **`P1 — High-Impact Low-Risk Deployments`**: Highest expected performance gain with proven empirical evidence from existing experiments and low system complexity.
- **`P2 — Advanced Representation Learning`**: Fundamental methodological upgrades designed to overcome class overlap and label contamination.
- **`P3 — Contextual Refinements & Literature Rebuttals`**: Domain-specific covariate additions and benchmark replication for academic paper completeness.

---

## 2. DETAILED EXPERIMENT SPECIFICATIONS

```
+-------------------------------------------------------------------------------------------------------------------------------+
|                                              GRIDBALANCE IMPROVEMENT EXPERIMENTS                                              |
+-----+------------------------------------------------+--------+-----------+----------------------+--------------+-------------+
| ID  | Experiment Name                                | Branch | Priority  | Expected Benefit     | Compute Cost | Risk Level  |
+-----+------------------------------------------------+--------+-----------+----------------------+--------------+-------------+
| E-01| Reconcile UCI Documentation & API Metadata     | System | P0        | Single source truth  | Negligible   | None        |
| E-02| Formalize TimeSeriesSplit Onboarding Mask      | UCI    | P0        | Clean CV evaluation  | Low (<10s)   | Low         |
| E-03| Promote LightGBM Focal Loss to Prod Backend    | SGCC   | P1        | +34.2% PR-AUC lift   | Low (<10ms)  | Low         |
| E-04| Operational Cost-Sensitive Utility Matrix      | SGCC   | P1        | Economic thresholding| Negligible   | None        |
| E-05| UCI Difference (Delta Load) Forecasting        | UCI    | P1        | Eliminates inertia   | Low (~15s)   | Low         |
| E-06| 1D-ResNet / TCN Temporal Sequence Learning     | SGCC   | P2        | Overcomes overlap    | Med (30 min) | Medium      |
| E-07| Semi-Supervised Positive-Unlabeled (PU) Learn  | SGCC   | P2        | Resolves label noise | Med (~5 min) | Low         |
| E-08| Hybrid Ridge + XGBoost Residual Ensemble       | UCI    | P2        | Robust extrapolation | Low (~2s)    | Low         |
| E-09| Portuguese National Holiday Covariates         | UCI    | P3        | Cuts holiday error   | Negligible   | None        |
| E-10| Reproduce Arif et al. SMOTE-NearMiss Baseline  | SGCC   | P3        | Literature rebuttal  | Med (~10 min)| Low         |
+-----+------------------------------------------------+--------+-----------+----------------------+--------------+-------------+
```

---

### Priority P0: Mandatory Technical Fixes

#### Experiment E-01: Reconcile UCI Benchmark Documentation & API Metadata
- **Branch**: Systems Architecture / Documentation
- **Problem Addressed**: `backend/config.py` documents UCI Test MAE as $4,421.55$ kWh ($N=4,392$ hours, 24h trimmed), while status reports document $4,503.51$ kWh ($N=4,416$ hours, full split).
- **Proposed Change**: Annotate `backend/config.py` and frontend `ModelIntelligence.tsx` to display both the primary full-window benchmark ($4,503.51$ kWh) and the boundary-trimmed benchmark ($4,421.55$ kWh) with explicit window timestamps.
- **Expected Benefit**: Zero discrepancies across paper, codebase, and UI.
- **Confidence**: **CONFIRMED BY EVIDENCE** | **Compute**: $< 1$s | **Complexity**: Low | **Risk**: None.
- **Success Criteria**: Automated assertions pass; all files match single source of truth ledger.

#### Experiment E-02: Formalize TimeSeriesSplit Onboarding Mask (2012–2014)
- **Branch**: UCI Load Forecasting
- **Problem Addressed**: Naive expanding `TimeSeriesSplit` across 2011–2013 produces an artificial CV Mean MAE of $10,365.77$ kWh because Fold 2 trains on 2011 (mean load 122k kWh) and tests on 2012 (mean load 212k kWh), which decision trees cannot extrapolate.
- **Proposed Change**: In `run_uci_forecasting_pipeline.py`, define cross-validation strictly on the stationary regime (`2012-01-01` to `2013-12-31`, $\sim 17,520$ hours) where all 370 customer meters are fully active.
- **Expected Benefit**: CV Mean MAE stabilizes from $10,365$ kWh to $\sim 4,200 - 4,800$ kWh with standard deviation $< 1,000$ kWh.
- **Confidence**: **CONFIRMED BY EVIDENCE** | **Compute**: $\sim 10$s | **Complexity**: Low | **Risk**: Low.
- **Success Criteria**: 5-fold CV Mean MAE $< 5,000$ kWh; aligns cross-validation with true test generalization.

---

### Priority P1: High-Impact Low-Risk Deployments

#### Experiment E-03: Promote LightGBM Focal Loss to Production Backend
- **Branch**: SGCC Electricity Theft Detection
- **Problem Addressed**: The frozen XGBoost baseline achieves Test PR-AUC of $0.3114$ and $F_1$ of $0.3592$. The serialized candidate `lgbm_focal_loss_best.joblib` achieved **Test PR-AUC = 0.4178 (+34.2% lift)** and **Test F1 = 0.4479** but remains unserved by the live API.
- **Proposed Change**: Update `backend/config.py` and `backend/inference/sgcc_predictor.py` to allow serving `models/optimization_v2/lgbm_focal_loss_best.joblib` via an opt-in configuration flag (`MODEL_VERSION = "focal_loss_v2"`).
- **Expected Benefit**: Immediately delivers a **+34.2% lift in PR-AUC** and **+24.7% lift in F1** in live inference without breaking the existing 18-feature API contract.
- **Confidence**: **CONFIRMED BY EVIDENCE** | **Compute**: $< 10$ms inference latency | **Complexity**: Low | **Risk**: Low (preserves fallback).
- **Success Criteria**: All 13 pytest tests pass; PR-AUC verified $\ge 0.41$.

#### Experiment E-04: Operational Cost-Sensitive Utility Matrix
- **Branch**: SGCC Electricity Theft Detection
- **Problem Addressed**: A static threshold of $t=0.50$ arbitrarily balances precision and recall, ignoring real-world utility finances (inspection team dispatch costs vs. value of recovered stolen kWh).
- **Proposed Change**: Implement utility objective function:
  $$\text{Net Utility Value} = \sum_{i \in \text{TP}} (\text{Stolen kWh}_i \times \text{Tariff}) - (\text{Count}(\text{TP} + \text{FP}) \times C_{\text{inspection}})$$
  Evaluate optimal threshold across budget tiers (e.g. 100, 250, 500 field inspections/month).
- **Expected Benefit**: Provides utilities with economically optimal operational thresholds directly in the frontend dashboard.
- **Confidence**: **CONFIRMED BY EVIDENCE** | **Compute**: Negligible | **Complexity**: Low | **Risk**: None.
- **Success Criteria**: Quantify net dollar gain over unweighted $t=0.50$ baseline across multiple tariff scenarios.

#### Experiment E-05: UCI Difference (Delta Load) Forecasting
- **Branch**: UCI Load Forecasting
- **Problem Addressed**: Direct level forecasting ($y_{t+1}$) exhibits slight lag-1 inertia. Tree models predict the previous level rather than sudden demand inflection points.
- **Proposed Change**: Reformulate the regression target to predict the hourly load difference:
  $$\Delta y_{t+1} = y_{t+1} - y_t$$
  Reconstruct the forecast as $\hat{y}_{t+1} = y_t + \hat{\Delta}$. Engineer difference features: $y_t - y_{t-1}$, $y_t - y_{t-24}$, $y_t - y_{t-168}$.
- **Expected Benefit**: Forces the model to learn pure demand acceleration and deceleration; reduces test MAE by an expected 5% to 10%.
- **Confidence**: **STRONGLY SUGGESTED** | **Compute**: $\sim 15$s | **Complexity**: Low | **Risk**: Low.
- **Success Criteria**: Test MAE $< 4,000$ kWh on sealed test partition.

---

### Priority P2: Advanced Representation Learning

#### Experiment E-06: 1D-ResNet / TCN Temporal Sequence Learning
- **Branch**: SGCC Electricity Theft Detection
- **Problem Addressed**: Handcrafted scalar summary metrics (mean, std, max) exhibit $>95\%$ overlap between classes and suffer asymptotic plateau at $N=23,000$ samples.
- **Proposed Change**: Train a 1D Residual Neural Network (1D-ResNet) or Temporal Convolutional Network (TCN) directly on raw $1,034$-day consumption sequences. Pretrain via self-supervised masked autoencoding or contrastive learning, followed by supervised fine-tuning with Focal Loss.
- **Expected Benefit**: Discovers temporal micro-motifs (sudden multi-week drops, gradual calibration drift) that scalar summaries destroy.
- **Confidence**: **STRONGLY SUGGESTED** | **Compute**: $\sim 30–60$ min | **Complexity**: High | **Risk**: Medium (overfitting risk on 3,612 positives).
- **Success Criteria**: Test $F_1 > 0.48$ and PR-AUC $> 0.45$ using consumption sequences alone (zero missingness features).

#### Experiment E-07: Semi-Supervised Positive-Unlabeled (PU) Learning
- **Branch**: SGCC Electricity Theft Detection
- **Problem Addressed**: Ground truth benign labels (`FLAG=0`) contain un-inspected thieves (false negatives in training data), capping classifier precision at $\sim 35\%$.
- **Proposed Change**: Apply Elkan-Noto or Bagging PU Learning to estimate the prior probability $p(s=1 \mid y=0)$ and reweight unlabeled instances during gradient boosting.
- **Expected Benefit**: Mitigates label contamination; prevents model from penalizing true theft patterns that were never physically inspected.
- **Confidence**: **STRONGLY SUGGESTED** | **Compute**: $\sim 5$ min | **Complexity**: Medium | **Risk**: Low.
- **Success Criteria**: Test Precision improves to $> 0.45$ at $50\%$ recall.

#### Experiment E-08: Hybrid Ridge + XGBoost Residual Ensemble
- **Branch**: UCI Load Forecasting
- **Problem Addressed**: Tree models cannot extrapolate beyond training extremes; Ridge regression extrapolates robustly but lacks non-linear interaction capacity.
- **Proposed Change**: Implement a two-stage residual architecture:
  1. Stage 1: Ridge regression fits linear trend and primary Fourier seasonal harmonics ($\hat{y}_{\text{linear}}$).
  2. Stage 2: XGBoost fits residual errors ($e_t = y_t - \hat{y}_{\text{linear}}$).
  3. Prediction: $\hat{y} = \hat{y}_{\text{linear}} + \hat{e}_{\text{tree}}$.
- **Expected Benefit**: Combines robust linear extrapolation with non-linear tree precision; improves performance on unprecedented grid peak days.
- **Confidence**: **STRONGLY SUGGESTED** | **Compute**: $\sim 2$s | **Complexity**: Low-Medium | **Risk**: Low.
- **Success Criteria**: Test MAE $< 4,100$ kWh; eliminates peak-demand underestimation.

---

### Priority P3: Contextual Refinements & Literature Rebuttals

#### Experiment E-09: Portuguese National Holiday & Bridge-Day Features
- **Branch**: UCI Load Forecasting
- **Problem Addressed**: Residual analysis reveals that the largest localized error spikes ($> 20,000$ kWh) occur on Portuguese national holidays (Easter, Christmas, New Year, Portugal National Day).
- **Proposed Change**: Incorporate binary indicators for official Portuguese national holidays and bridge days using the Python `holidays` package.
- **Expected Benefit**: Slashes holiday-specific forecast error by $40\%$ to $60\%$; reduces overall test MAE by $50 - 100$ kWh.
- **Confidence**: **CONFIRMED BY EVIDENCE** | **Compute**: Negligible | **Complexity**: Low | **Risk**: None.
- **Success Criteria**: Holiday-specific MAE drops by $> 30\%$.

#### Experiment E-10: Reproduce Arif et al. (2021) SMOTE-NearMiss Baseline
- **Branch**: SGCC Electricity Theft Detection
- **Problem Addressed**: Arif et al. (2021) claim $>90\%$ accuracy using SMOTE-NearMiss. NearMiss artificially removes difficult majority samples near the boundary, creating an illusion of separability.
- **Proposed Change**: Replicate SMOTE-NearMiss on SGCC training folds and evaluate on the untouched sealed test partition.
- **Expected Benefit**: Provides definitive empirical proof for the final research paper that published SMOTE-NearMiss gains do not hold on natural, imbalanced test distributions.
- **Confidence**: **CONFIRMED BY EVIDENCE (Literature Rebuttal)** | **Compute**: $\sim 10$ min | **Complexity**: Medium | **Risk**: Low.
- **Success Criteria**: Clear academic demonstration of false-positive explosion under NearMiss filtering on real distributions.

---

## 3. EXPERIMENTS TO STRICTLY AVOID

To conserve compute and prevent methodological regress, the following experiments should **NOT** be pursued:
1. **Generic Heavy Transformers (Informer, Autoformer, PatchTST) on SGCC**:
   - SGCC provides only scalar daily consumption (1,034 time steps) with severe class imbalance ($3,612$ positive instances). Multi-head attention mechanisms will overfit catastrophically compared to regularized 1D-CNNs or gradient boosting.
2. **Standard SMOTE Oversampling on SGCC**:
   - Already evaluated in `REAL_SGCC_BASELINE_RESULTS.csv`. SMOTE blurs class boundaries, depresses test precision, and increases training latency by $37\%$ without improving PR-AUC.
3. **Adding Hundreds of Handcrafted Summary Statistics**:
   - Feature ablation proved that expanding from 18 to 42 handcrafted features yielded no statistically significant improvement in CV PR-AUC while increasing overfit gap by $+0.06$.
4. **Stateful Continuous Moving Averages in UCI API**:
   - Evaluated in `V2_MODEL_OPTIMIZATION_REPORT.md` (Experiment `EXP_UCI_V2_04`). While test MAE improved by $3.4\%$, it requires stateful in-memory cache tracking per incoming REST call, violating production statelessness.
