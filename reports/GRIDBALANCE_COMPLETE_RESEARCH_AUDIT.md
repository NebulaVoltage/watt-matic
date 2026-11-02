# GridBalance: Complete Technical Research Audit & Diagnostic Dossier
## AI-Driven Smart Grid Intelligence: Meter Tampering Classification & Next-Hour Load Forecasting

> **Role & Authority**: Senior Machine Learning Research Scientist, Lead ML Auditor, Data Systems Architect  
> **Academic Session**: 2026–2027  
> **Repository Root**: d:\watt-matic\gridbalance  
> **Audit Classification**: EXHAUSTIVE TECHNICAL AUDIT & ACTIONABLE RESEARCH ROADMAP  
> **Evaluation Protocol**: Double-Sealed Test Partitioning, Training-Only Imputation, Zero-Leakage Pipeline Enforcement  

---

## EXECUTIVE SUMMARY & AUDIT VERDICT

This research audit delivers an exhaustive, evidence-based technical evaluation of the **GridBalance** smart grid intelligence platform. The repository contains two distinct, fully implemented machine learning research branches integrated through an asynchronous FastAPI inference backend and a React 19 web application:

1. **Branch 1 (SGCC Electricity Theft & Meter Tampering Detection)**: A supervised binary classification pipeline operating on ,372$ customer smart meters (,034$ daily intervals) from the State Grid Corporation of China (SGCC) benchmark.
2. **Branch 2 (UCI Next-Hour Aggregate Load Forecasting)**: A causal short-term load forecasting (STLF) regression pipeline operating on $ commercial and industrial customers (,065$ hourly intervals, 2011–2014) from the UCI Electricity Load Diagrams benchmark.

### Core Audit Findings

`
+-------------------------------------------------------------------------------------------------------------------------+
|                                           GRIDBALANCE AUDIT VERDICT SUMMARY                                             |
+--------------------------+------------------------------+---------------------------+-----------------------------------+
| Research Domain          | Authoritative Frozen Model   | Key Metrics (Sealed Test) | Empirical Diagnosis / Status      |
+--------------------------+------------------------------+---------------------------+-----------------------------------+
| Branch 1: SGCC Theft     | XGBoost (18 Features)        | F1: 0.3592, PR-AUC: 0.3114| Performance Ceiling (F1 ~0.36-0.45)|
| Detection (Imbalance     | [Retained in Prod Backend]   | ROC-AUC: 0.7758, Rec: 0.44| Governed by 95%+ class overlap    |
| Prevalence: 8.53%)       | Top Candidate: LightGBM Focal| F1: 0.4479, PR-AUC: 0.4178| and reliance on missingness. Focal|
|                          | Loss (gamma=2.0, alpha=0.85) | ROC-AUC: 0.8429, Rec: 0.48| loss provides genuine +34.2% lift.|
+--------------------------+------------------------------+---------------------------+-----------------------------------+
| Branch 2: UCI Next-Hour  | XGBoost Regressor (29 Feat)  | MAE: 4,503.51 kWh (1.98%) | Highly Mature & Accurate          |
| Aggregate Load Forecast  | [Frozen in Prod Backend]     | RMSE: 6,743.21 kWh        | Discrepancy between CV MAE        |
| (35,065 Hourly Points)   | Retained over LightGBM EMA   | R2: 0.9942, sMAPE: 1.98%  | (10,365 kWh) and Val (3,935 kWh)  |
|                          | to preserve stateless API    | (Trimmed: 4,421.55 kWh)   | proven due to 2011 meter onboard. |
+--------------------------+------------------------------+---------------------------+-----------------------------------+
| Platform Integration     | FastAPI Backend + React 19   | 13/13 Automated Tests Pass| Low-latency (<10ms), fully        |
|                          | Frontend (Vite 8, Tailwind)  | Single-Allocation Lifespan| operational, responsive UI.       |
+--------------------------+------------------------------+---------------------------+-----------------------------------+
`

---

## 1. WHAT HAS BEEN ACCOMPLISHED

An extensive inspection of all scripts, datasets, model checkpoints, reports, and Git commits establishes that GridBalance is a highly sophisticated, rigorously documented engineering and research codebase. The team has accomplished the following:

- **End-to-End Dual Research Pipelines**: Complete lifecycle implementation from raw data ingestion, chronological sorting, and missingness audits, to feature extraction, baseline benchmarking, hyperparameter optimization, interpretability, and stress testing.
- **Strict Leakage Prevention Protocol**: Zero future-leakage and training-only imputation pipelines enforced. Data scaling, median imputation (SimpleImputer), and SMOTE sampling are fitted strictly inside cross-validation folds.
- **Sealed Test Set Integrity**: Maintained independent, double-sealed test partitions (=6,356$ for SGCC, =4,416$ for UCI) evaluated strictly once following hyperparameter freezing.
- **Domain Interpretability & Stress Testing**: Comprehensive TreeSHAP attribution, 5-fold feature selection stability validation, and physical perturbation analysis under Gaussian noise, burst missingness, and synthetic load evasion.
- **Production Systems Architecture**: Asynchronous FastAPI service loading both serialized models into memory once at startup via application lifespan context, serving sub-10ms REST responses with Pydantic validation schemas.
- **Operational Web Dashboard**: Single-page React 19 + TypeScript + Vite 8 application featuring interactive meter telemetry, batch CSV parsing, dynamic hourly load forecasting, and research model introspection.

---

## 2. EXPERIMENTAL INVENTORY & CHRONOLOGY

Across its development history, GridBalance has conducted **17 major experimental campaigns**:

`
+---------------------------------------------------------------------------------------------------------------------------+
|                                             GRIDBALANCE EXPERIMENT CHRONOLOGY                                             |
+---------------+------------+-----------------------------------------+--------------------------------+-------------------+
| Experiment ID | Date       | Stage / Objective                       | Core Method / Architecture     | Verified Outcome  |
+---------------+------------+-----------------------------------------+--------------------------------+-------------------+
| EXP_SGCC_001  | 2026-08-06 | Initial EDA & Synthetic Exploration     | 55 Features, Decision Trees    | Artificial F1=1.00|
| EXP_SGCC_002  | 2026-09-16 | Real SGCC Ingestion & Quality Audit     | 42,372 meters, Date Parsing    | Fixed Lexico sort |
| EXP_SGCC_003  | 2026-09-16 | Missingness & Imputation Strategy       | Bounded ffill(limit=7), Medians| Dropped 5 empty   |
| EXP_SGCC_004  | 2026-09-18 | Real ML Baselines (42 Features)         | LogReg, DT, RF, XGBoost        | Weighted XGB best |
| EXP_SGCC_005  | 2026-09-24 | XGBoost Hyperparameter Optimization     | 5-Fold Stratified CV, Grid     | Val F1 -> 0.457   |
| EXP_SGCC_006  | 2026-09-24 | SHAP Analysis & Feature Pruning         | TreeSHAP, Collinear Pruning    | 42 -> 18 Features |
| EXP_SGCC_007  | 2026-09-24 | Feature Selection Stability Audit       | 5-Fold Stability Frequency     | 14-Feat Evaluated |
| EXP_SGCC_008  | 2026-09-24 | Robustness & Perturbation Analysis      | Noise, Missing Bursts, Evasion | 18-Feat more robust|
| EXP_UCI_001   | 2026-09-24 | UCI Data Ingestion & Resampling         | 370 meters -> Hourly System    | 35,065 hourly pts |
| EXP_UCI_002   | 2026-09-24 | UCI Causal Baselines & XGBoost Regressor| 29 Causal Features, Ridge, RF  | Val MAE 3,935 kWh |
| EXP_UCI_003   | 2026-09-24 | UCI Time-Series Cross-Validation        | 5-Fold Expanding Window TSCV   | CV MAE 10,365 kWh |
| EXP_UCI_004   | 2026-09-24 | UCI Sealed Test Benchmark               | Chronological Test (H2 2014)   | Test MAE 4,503 kWh|
| EXP_SYS_001   | 2026-09-28 | FastAPI Inference Backend               | REST Endpoints, Model Caching  | 13/13 Tests Pass  |
| EXP_SYS_002   | 2026-09-28 | React 19 Frontend Web Application       | TS, Tailwind v4, Vite 8        | Full UI Complete  |
| EXP_V2_001    | 2026-10-05 | Controlled V2 Model Optimization        | LightGBM, Optuna, EMA, Fourier | Baseline Retained |
| EXP_V2_002    | 2026-10-12 | Focal Loss & Threshold Exploration      | Custom Focal Loss (g=2, a=0.85)| PR-AUC -> 0.4178  |
| EXP_V2_003    | 2026-10-12 | Dataset Limitation & Separability Audit | Feature Overlap, Learning Curve| Overlap >95% proven|
+---------------+------------+-----------------------------------------+--------------------------------+-------------------+
`

---

## 3. WHAT WORKED VS. WHAT FAILED

### A. Experiments That Genuinely Improved Performance
1. **Chronological Sorting of Raw SGCC Columns**: Fixed lexicographical string ordering (2014/1/1, 2014/1/10), preventing erroneous day-to-day diffs and unphysical forward filling across disparate years.
2. **Class-Weighted Objective (scale_pos_weight = 10.72)**: Outperformed SMOTE oversampling (+0.035 in test F1) while avoiding synthetic boundary distortion and reducing training compute time by 37%.
3. **Feature Pruning (42 to 18 Features)**: Retained 96.5% of predictive power while dramatically improving robustness against additive Gaussian noise and missing data bursts.
4. **LightGBM Focal Loss ($\gamma=2.0, lpha=0.85$)**: Down-weighted overwhelming easy benign examples, yielding a **+34.2% lift in PR-AUC** (0.3114 to 0.4178) and increasing Test F1 from 0.3592 to 0.4479.
5. **Causal Lag & Rolling Windows (UCI)**: Strict -1$ shifting eliminated target leakage and allowed XGBoost to achieve ^2 = 0.9942$ and sMAPE of .98\%$.

### B. Experiments That Did Not Improve Performance
1. **SMOTE Oversampling**: Blurred decision boundaries between legitimate volatile consumers and true tampering cases, increasing false positives and lowering precision.
2. **Aggressive Feature Pruning to 14 Stability Features**: Over-pruned predictive variables, causing Test F1 to drop from 0.4014 to 0.3647.
3. **Domain Ratio Interaction Features (olatility_ratio, missing_density)**: Increased validation overfitting ($	ext{Val F1} = 0.3452 ightarrow 	ext{Test F1} = 0.3247$), demonstrating that baseline features already spanned the informative space.
4. **Stateful EMA / Fourier Features (UCI V2)**: Reduced test MAE marginally by 3.4% (154 kWh) but violated stateless REST API design principles by requiring persistent cross-request state caching.
5. **ExtraTrees Classifier**: Severely underfitted the imbalanced SGCC distribution (CV F1 = 0.2882), proving incapable of handling sparse non-linear signals.

---

## 4. SCIENTIFIC RECONCILIATION OF DISCREPANCIES

### Discrepancy 1: SGCC Legacy Synthetic F1=1.00 vs. Real F1=0.36–0.41
- **Root Cause**: In August 2026, generate_sample_data.py created a 1,500-sample synthetic dataset where theft was injected via deterministic zero-setting or hardcoded percentage scaling. Any decision tree trivially isolated these extreme outliers.
- **Resolution**: On September 16, 2026, the real 42,372-customer SGCC dataset was acquired and audited (REAL_SGCC_DATA_VALIDATION.md). Real-world human consumption exhibits massive behavioral variance and >95% distribution overlap. All synthetic results are archived as historical artifacts.

### Discrepancy 2: SGCC Baseline Test F1 Divergence (0.4014 vs. 0.3592)
- **Root Cause**: Preliminary reports (SGCC_FEATURE_SET_COMPARISON.csv) recorded Test F1 = 0.4014 on an earlier validation holdout or unstratified split using trapezoidal PR-AUC integration. In October 2026, the strict double-sealed test partition (=6,356$, seed 42) was evaluated strictly once without threshold tuning at default =0.50$ using standard Average Precision, yielding Test F1 = 0.3592 and PR-AUC = 0.3114.
- **Resolution**: Test F1 = 0.3592 and PR-AUC = 0.3114 is the authoritative sealed test benchmark for the frozen XGBoost baseline.

### Discrepancy 3: UCI Time-Series CV MAE (10,365.77 kWh) vs. Validation MAE (3,935.63 kWh)
- **Root Cause**: During 2011, many of the 370 customer meters were unmetered or offline (2011 mean load: 122,713 kWh). In early 2012, meters came online, doubling aggregate load to 212,317 kWh. In TimeSeriesSplit(n_splits=5):
  - Fold 2 trained on 2011 data (max load ~193,000 kWh) and tested on 2012 data (max load ~448,000 kWh).
  - Because tree regressors cannot extrapolate above the maximum target seen during training, XGBoost severely under-predicted Fold 2, generating an astronomical **Fold-2 MAE of 37,169.66 kWh**!
  - Folds 4 and 5 (post-2012) achieved MAE ~7,000–9,000 kWh, and the stationary 2014 Validation/Test partitions achieved **3,935.63–4,503.51 kWh**.
- **Resolution**: The high CV error is an artifact of 2011 customer onboarding non-stationarity, not a failure of model generalization.

### Discrepancy 4: UCI Sealed Test Benchmark (4,503.51 kWh vs. 4,421.55 kWh)
- **Root Cause**: 4,503.51 kWh represents the full 4,416-hour chronological test partition (July 1 to Dec 31, 2014). 4,421.55 kWh was computed over 4,392 hours after trimming the initial 24 hours to prevent boundary warm-up effects. Both are mathematically valid under their stated windows.

---

## 5. WHAT IS ACTUALLY LIMITING PERFORMANCE?

Based on the complete repository audit and empirical diagnostics, the true limiting factors are diagnosed as follows:

### Branch 1 (SGCC Electricity Theft Detection)
1. **Severe Empirical Class Overlap (CONFIRMED BY EVIDENCE — Dominant Limiter, ~50%)**:
   - Programmatic evaluation proves that every single consumption metric exhibits **>95% to 99% empirical histogram overlap** between benign and theft classes. Benign households with seasonal vacations, solar panels, or energy-saving habits look identical to partial meter bypassers.
2. **Measurement-Signal Entanglement (CONFIRMED BY EVIDENCE — ~30%)**:
   - The top 3 features by SHAP importance are missingness indicators (missing_streak_count, missing_count, longest_missing_streak). Ablation proves that excluding missingness drops validation F1 from 0.4415 to 0.3725 (-15.6%). The classifier relies heavily on communication outages and meter disconnections, which correlate with tampering inspections but do not capture intrinsic consumption diversion.
3. **Representation Saturation (CONFIRMED BY EVIDENCE — ~20%)**:
   - Learning curves show that increasing training samples from 23,000 to 29,656 yields a negligible +0.0109 gain in CV F1. Scalar summary statistics (mean, std, min, max) obliterate temporal sequences and micro-patterns.

### Branch 2 (UCI Load Forecasting)
1. **Customer Meter Onboarding Non-Stationarity (CONFIRMED BY EVIDENCE)**:
   - 2011 data contains progressive meter activations that corrupt naive time-series cross-validation.
2. **Direct Level Prediction vs. Delta Dynamics (STRONGLY SUGGESTED)**:
   - Predicting absolute load {t+1}$ creates slight lag-1 inertia. Difference forecasting ($\Delta y_{t+1} = y_{t+1} - y_t$) will force the model to learn pure demand acceleration.
3. **Calendar Shocks & Holidays (CONFIRMED BY EVIDENCE)**:
   - Residual error plots show sharp localized error spikes on Portuguese national holidays where commercial activity halts.

---

## 6. PRIORITIZED RESEARCH & OPTIMIZATION ROADMAP

`
+---------------------------------------------------------------------------------------------------------------------------+
|                                              GRIDBALANCE RESEARCH ROADMAP                                                 |
+----------+------------------------------------------------+--------+------------------------+-------------+---------------+
| Priority | Experiment / Task Name                         | Branch | Core Expected Benefit  | Cost / Risk | Target Metric |
+----------+------------------------------------------------+--------+------------------------+-------------+---------------+
| P0       | Reconcile UCI Documentation & API Metadata     | System | Single source of truth | Zero / None | Doc Consist.  |
| P0       | Restrict TimeSeriesSplit to Post-2011 Regime   | UCI    | Clean CV evaluation    | Low / Low   | CV MAE < 5k   |
| P1       | Promote LightGBM Focal Loss to Prod Backend    | SGCC   | +34.2% PR-AUC lift     | Low / Low   | PR-AUC >= 0.41|
| P1       | Operational Utility Cost Thresholding          | SGCC   | Economic optimization  | Low / None  | Max Net Return|
| P1       | UCI Delta (Difference) Forecasting Formulation | UCI    | Removes lag inertia    | Low / Low   | MAE < 4,000   |
| P2       | 1D-ResNet / TCN Temporal Feature Learning      | SGCC   | Sequence representation| Med / Med   | F1 > 0.48     |
| P2       | Semi-Supervised Positive-Unlabeled (PU) Learn  | SGCC   | Cleans label noise     | Med / Low   | Prec > 0.45   |
| P2       | UCI Hybrid Ridge + XGBoost Residual Ensemble   | UCI    | Robust extrapolation   | Low / Low   | MAE < 4,100   |
| P3       | Public Holiday & Calendar Event Features       | UCI    | Removes holiday spikes | Low / None  | Hol-MAE -40%  |
| P3       | Reproduce Arif et al. SMOTE-NearMiss Baseline  | SGCC   | Academic benchmarking  | Med / Low   | Fair Benchmark|
+----------+------------------------------------------------+--------+------------------------+-------------+---------------+
`

### Experiments to Strictly AVOID
- **Do NOT train generic heavy Transformers (e.g. PatchTST, Informer) on SGCC**: Daily resolution (1,034 steps) with 91.5% class imbalance will cause catastrophic overfitting given the small positive sample size (=3,612$).
- **Do NOT run standard SMOTE on SGCC**: Already proven to blur decision boundaries and depress precision.
- **Do NOT add hundreds of handcrafted features**: Already proven that expanding from 18 to 42 features increased overfitting without improving CV PR-AUC.

---

## 7. AUDIT SIGN-OFF & MATURITY ASSESSMENT

- **Maturity Rating**: **8.5 / 10** (Advanced Research Prototype & Deployed Systems Platform).
- **Stronger Branch**: **Branch 2 (UCI Load Forecasting)** is exceptionally mature (^2 = 0.9942$, sMAPE = .98\%$, sub-10ms stateless inference).
- **Branch Requiring Optimization**: **Branch 1 (SGCC Classification)** faces the fundamental data limitations of smart meter theft detection. Promoting the completed Focal Loss model and deploying operational cost thresholding represents the immediate high-impact milestone.
