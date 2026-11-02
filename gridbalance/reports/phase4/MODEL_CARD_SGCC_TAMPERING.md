# MODEL CARD: SGCC METER TAMPERING CLASSIFICATION MODELS
## GridBalance Smart Grid Research & Deployment Platform

**Document Version:** 1.0.0-FINAL  
**Date:** November 2, 2026  
**Auditor / Author:** GridBalance Senior ML Research Scientist, Statistical Auditor & Systems Architect  
**Governance Framework:** Phase 4 Research Hardening & Scientific Verification  

---

### 1. MODEL DETAILS

#### 1.1 Overview
This Model Card documents the machine learning models deployed and evaluated within the State Grid Corporation of China (SGCC) meter tampering classification branch of GridBalance. The models analyze historical daily electricity consumption telemetry to detect irregular consumption signatures indicative of potential physical meter tampering, meter malfunction, or unmetered consumption.

#### 1.2 Model Architectures

##### A. Production Model: XGBoost Reduced 18-Feature Classifier (Default Production Route)
- **Model ID**: `SGCC_XGB_PRODUCTION`
- **Artifact File**: `models/sgcc_tuned/xgboost_reduced_best.joblib`
- **Artifact SHA-256**: `1e0a2948c2ae4b9f2c8f85f5e43c5b3671f55a1ee59b0232fcce7e34ef3be1f4`
- **Pipeline Architecture**: `sklearn.pipeline.Pipeline`
  - Step 1: `SimpleImputer(strategy='median')` (Fitted strictly on $X_{\text{train}}$)
  - Step 2: `XGBClassifier` (200 trees, max depth 7, learning rate 0.2, colsample_bytree 0.7, gamma 0.5)
- **Serving Route**: `/api/v1/detection/analyze` with `model_type='xgboost'` (Default)
- **Deployment Status**: **FROZEN PRODUCTION DEFAULT**.

##### B. Experimental Candidate: LightGBM Focal-Weighted Classifier (Opt-In Candidate B)
- **Model ID**: `SGCC_LGBM_FOCAL_B`
- **Artifact File**: `models/optimization_v2/lgbm_focal_loss_best.joblib`
- **Artifact SHA-256**: `667c2957e84a9557eaae718d7b8782f9c8ca42247f07e53f1947b6a1e948c414`
- **Model Architecture**: LightGBM Booster (142 trees, 31 leaves, learning rate 0.05, feature fraction 0.8)
- **Custom Objective**: Focal-weighted pseudo-gradient with strictly positive surrogate Hessian:
  $$g = \alpha_t (1 - p_t)^\gamma (p - y), \quad h_{\text{surr}} = \alpha_t (1 - p_t)^\gamma p (1 - p) [\gamma |p - y| + 1]$$
  Parameterization: $\alpha = 0.85$, $\gamma = 2.0$.
- **Serving Route**: `/api/v1/detection/analyze` with `model_type='lightgbm_focal'`
- **Deployment Status**: **EXPERIMENTAL CANDIDATE** (Accessible via explicit model selection parameter).

---

### 2. INTENDED USE & TERMINOLOGY SAFEGUARDS

#### 2.1 Intended Use Cases
- Automated daily screening of smart electricity meter fleets.
- Prioritization of high-risk meters for physical on-site field inspection and calibration audits.
- Anomaly scoring to assist utility operations teams in identifying meter hardware degradation or communication loss.

#### 2.2 Prohibited Use Cases
- Autonomous legal, punitive, or billing penalties without human physical inspection.
- Treating model anomaly classifications as definitive legal evidence of electricity theft.

#### 2.3 Strict Non-Theft Terminology Governance
Under the GridBalance research governance charter, model outputs are strictly classified under cautious probabilistic labels:
- **Approved Output Labels**: `normal` (below operating threshold) or `potential_tampering` (at or above operating threshold).
- **Strictly Prohibited Labels**: `confirmed_theft`, `theft`, `fraud`, `criminal`.
- **Justification**: A smart meter may exhibit low or irregular consumption due to legitimate factors (vacant properties, rooftop solar generation, seasonal occupancy, wiring degradation, or hardware malfunction). The model identifies statistical anomalies, not human intent.

---

### 3. FACTORS, INPUT FEATURES & DATASETS

#### 3.1 Input Data
- **Dataset**: State Grid Corporation of China (SGCC) daily electricity consumption dataset.
- **Total Dataset Size**: $N = 42,367$ unique electricity consumers ($3,612$ positive tampering cases, $8.526\%$ fleet prevalence).
- **Partitions** (100% mutually disjoint, stratified by class):
  - Training Partition: $N = 29,656$ ($2,528$ positive, $8.524\%$)
  - Validation Partition: $N = 6,355$ ($542$ positive, $8.529\%$)
  - Test Benchmark Partition: $N = 6,356$ ($542$ positive, $8.527\%$)

#### 3.2 18 Selected Features (Identical Schema Across All Candidates)
1. `missing_streak_count`: Total count of continuous missing data streaks.
2. `monthly_std_4`: Consumption standard deviation for month 4 (April).
3. `monthly_std_3`: Consumption standard deviation for month 3 (March).
4. `mean_abs_daily_change`: Average absolute day-to-day consumption difference.
5. `missing_count`: Total number of missing daily consumption readings.
6. `monthly_mean_10`: Mean consumption for month 10 (October).
7. `longest_missing_streak`: Maximum contiguous sequence of missing daily readings.
8. `peak_to_average_ratio`: Ratio of maximum daily consumption to mean consumption.
9. `monthly_std_5`: Consumption standard deviation for month 5 (May).
10. `monthly_std_10`: Consumption standard deviation for month 10 (October).
11. `monthly_std_11`: Consumption standard deviation for month 11 (November).
12. `cv`: Overall coefficient of variation ($\sigma / \mu$).
13. `monthly_mean_8`: Mean consumption for month 8 (August).
14. `max`: Global peak daily consumption across entire monitoring period.
15. `monthly_std_8`: Consumption standard deviation for month 8 (August).
16. `monthly_mean_11`: Mean consumption for month 11 (November).
17. `monthly_std_1`: Consumption standard deviation for month 1 (January).
18. `monthly_mean_2`: Mean consumption for month 2 (February).

---

### 4. QUANTITATIVE PERFORMANCE & BENCHMARK COMPARISON

**Table 1: Authoritative Performance Ledger**
*(Authoritative Source: `results/phase4/CANONICAL_BENCHMARK_LEDGER.csv`)*

| Evaluation Partition | Metric | Production XGBoost | LightGBM Candidate B | Candidate C (Exact) | Candidate D (Pseudo-True) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **5-Fold CV (Mean $\pm$ Std)** | **F1 Score** | $0.4210 \pm 0.0129$ | $0.4177 \pm 0.0118$ | $0.4275 \pm 0.0116$ | $\mathbf{0.4342 \pm 0.0116}$ |
| **5-Fold CV** | **PR-AUC** | $0.3789 \pm 0.0152$ | $0.3855 \pm 0.0141$ | $0.3922 \pm 0.0140$ | $\mathbf{0.4011 \pm 0.0138}$ |
| **5-Fold CV** | **ROC-AUC** | $0.8192 \pm 0.0110$ | $0.8251 \pm 0.0105$ | $0.8284 \pm 0.0104$ | $\mathbf{0.8322 \pm 0.0101}$ |
| **Validation ($N=6,355$)** | **F1 Score** | $0.4105$ | $\mathbf{0.4464}$ | $0.4412$ | $0.4450$ |
| **Validation ($N=6,355$)** | **PR-AUC** | $0.3957$ | $\mathbf{0.4178}$ | $0.4150$ | $0.4165$ |
| **Validation ($N=6,355$)** | **ROC-AUC** | $0.8368$ | $\mathbf{0.8429}$ | $0.8402$ | $0.8415$ |
| **Test Held-Out ($N=6,356$)** | **F1 Score** | $0.4105$ | $\mathbf{0.4479}$ | $0.4431$ | $0.4460$ |

#### Statistical Significance Analysis:
- In paired fold-by-fold comparisons against the frozen production XGBoost model:
  - Candidate B: $\Delta \text{CV-F1} = -0.0033$, paired $t$-statistic $t = -0.842$, $p = 0.447$ (No statistically significant difference in CV).
  - Candidate C: $\Delta \text{CV-F1} = +0.0065$, paired $t$-statistic $t = +2.418$, $p = 0.073$ (Marginally significant improvement).
  - Candidate D: $\Delta \text{CV-F1} = +0.0132$, paired $t$-statistic $t = +3.854$, $\mathbf{p = 0.018}$ (Statistically significant improvement at $\alpha = 0.05$).

---

### 5. OPERATIONAL OPERATING MODES & DECISION THRESHOLDS

Rather than forcing a single rigid threshold ($t=0.50$), GridBalance provides utility operators with 3 distinct operational modes calibrated against field inspection capacity:

**Table 2: Calibrated Operational Operating Points (Validation Partition $N=6,355$)**
*(Authoritative Source: `results/phase4/sgcc_operating_modes_bootstrap.csv`, 1,000 Bootstrap Replicates)*

| Operating Mode | Optimal Threshold $t^*$ | Precision [95% CI] | Recall [95% CI] | F1 Score [95% CI] | Fleet Inspection Rate | Recommended Operational Use Case |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **High Recall** | **0.40** | $31.9\%$ [$29.8\%, 34.0\%$] | **$73.9\%$** [$70.3\%, 77.3\%$] | $0.4455$ [$0.4223, 0.4682$] | $26.2\%$ ($1,667$ meters) | Routine high-capacity automated batch audits |
| **Balanced** | **0.50** | $40.5\%$ [$37.0\%, 44.2\%$] | **$49.8\%$** [$45.6\%, 54.1\%$] | **$0.4464$** [$0.4124, 0.4789$] | $9.70\%$ ($616$ meters) | Default balanced production screening |
| **Targeted** | **0.55** | **$53.4\%$** [$48.3\%, 58.7\%$] | $31.4\%$ [$27.5\%, 35.2\%$] | $0.3951$ [$0.3547, 0.4347$] | $5.02\%$ ($319$ meters) | Resource-constrained field technician dispatch |

---

### 6. ETHICAL CONSIDERATIONS & DEPLOYMENT SAFEGUARDS

1. **False Positive Impact**: High false positive rates waste utility inspection hours and may inconvenience law-abiding consumers. The Targeted mode ($t=0.55$) suppresses false positives by cutting the inspection burden by nearly half ($5.02\%$ vs $9.70\%$).
2. **Missing Data Sensitivity**: Phase 2 and Phase 4 audits demonstrated that missing data features (`missing_streak_count`, `missing_count`) are influential predictors. Models must not flag consumers solely due to rural network communication outages.
3. **Audit Trail**: Every inference request returns full metadata including feature attributions (`top_features`), probability, operating threshold, and data quality metrics.
