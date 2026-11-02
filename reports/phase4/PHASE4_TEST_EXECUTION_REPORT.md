# GRIDBALANCE — RESEARCH HARDENING & SCIENTIFIC VERIFICATION
## WORKSTREAM G: AUTOMATED REGRESSION & INTEGRATION TEST EXECUTION REPORT

**Author:** GridBalance Senior Software Quality Architect & ML Systems Engineer  
**Date:** November 2, 2026  
**Document Version:** 1.0.0-FINAL  
**Repository Branch:** `main` (Audited Baseline)  
**Execution Environment:** Python 3.11.9, Pytest 9.1.1, FastAPI, Starlette TestClient, LightGBM 4.7.0, XGBoost 2.1.3  

---

### EXECUTIVE SUMMARY

As part of Phase 4 Workstream G, the GridBalance automated regression test suite was comprehensively expanded to cover all mathematical invariants, numerical stability bounds, boundary truncation rules, input defense sanitization, operational thresholding behaviors, and non-theft terminology requirements established across Phase 4.

Across the unified test suite comprising 35 comprehensive test cases, **35 out of 35 tests passed cleanly (100.0% Pass Rate)** with an execution duration of $3.58\text{ seconds}$. Zero regressions were introduced to existing production baselines or frozen contracts.

---

### 1. TEST SUITE ARCHITECTURE & COVERAGE TAXONOMY

The test suite spans 7 specialized modules, testing the system at the mathematical, algorithmic, service, and API integration layers:

**Table 1: Test Suite Coverage Matrix**

| Test Module | Domain / Focus Area | Tests Executed | Passed | Failed | Duration |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `tests/phase4/test_focal_loss_math.py` | Focal Objective Formal Mathematics & Hessians | 7 | 7 | 0 | 0.09s |
| `tests/phase4/test_phase4_hardening.py` | Operational Thresholds, Input Defense, Parity, Purity | 8 | 8 | 0 | 0.85s |
| `tests/test_api_endpoints.py` | Production HTTP API Endpoints & Status Codes | 6 | 6 | 0 | 0.35s |
| `tests/test_model_loading.py` | Serialized Joblib Artifact Integrity & Signatures | 2 | 2 | 0 | 0.20s |
| `tests/test_phase3_controlled_integration.py` | Dual-Model Dynamic Selection & Mode Overrides | 7 | 7 | 0 | 0.75s |
| `tests/test_sgcc_inference.py` | SGCC Feature Extraction & Pattern Classification | 3 | 3 | 0 | 0.45s |
| `tests/test_uci_forecasting.py` | UCI FIR Baseline Feature Extraction & Forecast | 2 | 2 | 0 | 0.89s |
| **TOTAL UNIFIED TEST SUITE** | **Complete GridBalance System Integration** | **35** | **35** | **0** | **3.58s** |

---

### 2. DETAILED TEST EXECUTION BREAKDOWN

#### 2.1 Focal Loss Mathematical Invariants (`test_focal_loss_math.py`)
1. `test_exact_gradient_matches_numerical_finite_difference`: Verifies analytical derivatives against central difference numerical differentiation across $z \in [-10, 10]$ ($\text{err} < 10^{-6}$). **PASSED**.
2. `test_exact_hessian_matches_numerical_differentiation_of_gradient`: Verifies exact Hessian equations match numerical differentiation of gradients. **PASSED**.
3. `test_surrogate_hessian_is_strictly_non_negative`: Verifies that Candidate B surrogate Hessian $h_{\mathrm{surr}} \ge 0$ everywhere, preventing boosting tree split aborts. **PASSED**.
4. `test_exact_focal_hessian_contains_negative_curvature_regions`: Proves conventional focal loss exact Hessian is negative for hard false negatives ($h_{\min} = -0.04011$). **PASSED**.
5. `test_gamma_zero_special_case_and_alpha_one_zeroing`: Validates limit behavior $\gamma = 0$, $\alpha = 1.0$. **PASSED**.
6. `test_pseudo_gradient_differs_from_exact_focal_gradient`: Confirms omission of modulating factor derivative. **PASSED**.
7. `test_pseudo_gradient_global_lipschitz_continuity`: Confirms global Lipschitz constant $K \le 0.26895$. **PASSED**.

#### 2.2 Phase 4 System Hardening Invariants (`test_phase4_hardening.py`)
1. `test_sgcc_operating_modes_threshold_boundaries`: Validates decision logic across Balanced ($t=0.50$), High-Recall ($t=0.40$), and Targeted ($t=0.55$) modes. Verifies boundary behavior for intermediate probabilities ($p=0.45, 0.52$). **PASSED**.
2. `test_sgcc_api_non_theft_terminology_invariant`: Enforces that the classification label strictly outputs `potential_tampering` or `normal`, and under no circumstance returns `confirmed_theft`. **PASSED**.
3. `test_uci_ema_input_defense_truncated_history`: Asserts that history lengths below 336 hours ($w=335$) are blocked with `HTTP 400` or `HTTP 422`. **PASSED**.
4. `test_uci_ema_input_defense_empty_and_nan_injection`: Asserts empty arrays and NaN injections are rejected at the network boundary. **PASSED**.
5. `test_uci_ema_input_defense_malformed_timestamp`: Asserts corrupted datetime strings trigger explicit validation errors. **PASSED**.
6. `test_uci_ema_input_defense_out_of_order_robustness`: Asserts shuffled telemetry is sorted chronologically and produces identical forecasts to in-order telemetry ($\Delta \hat{y} < 10^{-4}$). **PASSED**.
7. `test_data_leakage_imputer_purity`: Verifies the production pipeline imputer was fitted strictly on $X_{\text{train}}$ with exactly $0.0$ discrepancy. **PASSED**.
8. `test_canonical_benchmark_ledger_integrity`: Asserts `CANONICAL_BENCHMARK_LEDGER.csv` contains all 35 authoritative rows with zero missing values. **PASSED**.

#### 2.3 Production API & Model Serving Regression Tests
1. Health and metadata inspection: `/api/v1/health` and `/api/v1/models` return 200 OK with correct schemas. **PASSED**.
2. SGCC dynamic model selection: Requests specifying `model_type='xgboost'` or `model_type='lightgbm_focal'` execute their respective underlying models correctly. **PASSED**.
3. Operating mode overrides: API properly overrides operating threshold when requested. **PASSED**.
4. UCI FIR baseline vs EMA experimental routes: `/api/v1/forecast/next-hour` (29 features) and `/api/v1/forecast/next-hour-experimental-ema` (33 features) both execute and output standardized payloads. **PASSED**.

---

### 3. DEFECT REMEDIATION LOG

During the authoring of Workstream G tests, one critical algorithmic edge case in the experimental EMA serving layer was discovered, isolated, and remediated:

- **Defect Description**: When out-of-order timestamps were provided to `/api/v1/forecast/next-hour-experimental-ema`, the FIR feature extractor (`preprocess_and_extract_features`) properly sorted the series, but the downstream EMA computation extracted `loads` directly from the raw unsorted payload list, resulting in slightly divergent EMA calculations for shuffled input.
- **Remediation**: Updated `backend/inference/uci_forecaster.py` to sort and deduplicate timestamps prior to computing `s.ewm(span=6)` and `s.ewm(span=24)`.
- **Verification**: `test_uci_ema_input_defense_out_of_order_robustness` confirmed that shuffled payloads produce forecasts matching in-order payloads within machine precision ($\Delta = 0.00\text{ kWh}$).

---

### 4. CONCLUSION

All 35 unit, integration, boundary, and mathematical tests pass with 100% reliability in $3.58\text{ seconds}$. The software systems and machine learning inference pipelines are fully verified, hardened, and regression-tested. Workstream G is approved and complete.
