# GridBalance: Complete Chronological Experiment History & Development Log
## Comprehensive Progression from Concept to Controlled Optimization and Deployment

> **Role & Authority**: Lead ML Auditor, Research Historian  
> **Repository Root**: `d:\watt-matic\gridbalance`  
> **Temporal Coverage**: August 2026 – October 2026  
> **Experiment Ledger Reference**: `results/GRIDBALANCE_EXPERIMENT_INVENTORY.csv`  

---

## 1. RECONSTRUCTED DEVELOPMENT TIMELINE

The GridBalance platform evolved across six distinct developmental epochs:

```
                                  GRIDBALANCE DEVELOPMENT TIMELINE
                                  
  AUG 2026      SEPT 16-18, 2026       SEPT 24, 2026       SEPT 28, 2026       OCT 05, 2026        OCT 12, 2026
+------------+ +--------------------+ +-----------------+ +-----------------+ +-----------------+ +-----------------+
| Phase 1:   | | Phase 2:           | | Phase 3:        | | Phase 4:        | | Phase 5:        | | Phase 6:        |
| Concept &  | | Real SGCC Ingest,  | | Pruning (18 F), | | FastAPI Backend | | V2 Controlled   | | Focal Loss LBM  |
| Synthetic  | | Date Sorting Bug,  | | UCI Forecasting | | & React 19 Web  | | Optimization &  | | (PR-AUC 0.4178),|
| EDA Suite  | | Baseline Models    | | Pipeline & Test | | Application Int | | Optuna Tuning   | | Separability Aud|
+------------+ +--------------------+ +-----------------+ +-----------------+ +-----------------+ +-----------------+
```

---

## 2. DETAILED PHASE-BY-PHASE AUDIT LOG

### Phase 1: Initial Concept & Synthetic EDA (August 2026)
- **Git Commit**: `f39101e` (*EDA Report*)
- **Objective**: Establish project architecture and exploratory data analysis suite for smart grid meter tampering.
- **Implementation**: Created `main.py`, `data_loader.py`, `generate_sample_data.py`, `overview_analyzer.py`, `eda_visualizer.py`, and `pdf_report_generator.py`.
- **Results**: Produced 22 publication plots and executive PDF report (`EDA_Report.pdf`). However, models trained on the data achieved artificial $F_1 = 1.0000$ and $\text{Accuracy} = 1.0000$.
- **Outcome / Diagnostic Finding**: The pipeline had automatically triggered the fallback synthetic generator `generate_sample_data.py` (1,500 samples), which injected deterministic zeros and multipliers for theft. Real SGCC data was not yet integrated.
- **Classification**: **COMPLETED** (Historical Synthetic Milestone).

---

### Phase 2: Real SGCC Ingestion, Lexicographical Sorting Bug & Baselines (Sept 16–18, 2026)
- **Git Commit**: `b147f99` (*feat: complete GridBalance SGCC research pipeline*)
- **Objective**: Ingest genuine SGCC dataset ($42,372$ meters), audit data quality, formulate missingness strategy, and establish real ML baselines.
- **Implementation**:
  - `audit_real_sgcc.py` and `REAL_SGCC_DATA_VALIDATION.md`: Discovered that raw column headers were ordered lexicographically as strings (`2014/1/1`, `2014/1/10`... `2014/1/2`).
  - `audit_and_fix_missingness.py` and `SGCC_Missingness_Strategy_Report.md`: Identified 5 fully-missing meters (dropped); implemented bounded `ffill(limit=7)`.
  - `run_fe_real_pipeline.py`: Implemented chronological parsing, sorted date vectors, and extracted 42 domain features into `data/processed/sgcc_features_real_v2.csv`.
  - `run_ml_baseline_real.py`: Benchmarked Logistic Regression, Decision Tree, Random Forest, and XGBoost under Unweighted, Class-Weighted (`scale_pos_weight = 10.72`), and SMOTE strategies.
- **Results**:
  - Weighted XGBoost achieved top Validation $F_1 = 0.4127$ and Test $F_1 = 0.3802$ (PR-AUC: 0.3438, ROC-AUC: 0.8030).
  - SMOTE yielded $F_1 = 0.3767$ with 37% higher compute cost and increased false positives.
- **Outcome**: Established the first genuine empirical baseline on real-world SGCC data.
- **Classification**: **COMPLETED**.

---

### Phase 3: XGBoost Tuning, TreeSHAP Pruning & UCI Forecasting (Sept 24, 2026)
- **Git Commits**: `b147f99`, `081abc3` (*feat: complete UCI electricity load forecasting research pipeline*)
- **Objective**: Optimize SGCC XGBoost hyperparameters, conduct TreeSHAP interpretability, prune features to 18, perform robustness stress tests, and develop Branch 2 (UCI forecasting).
- **Implementation**:
  - `run_xgb_tuning.py` / `SGCC_XGBOOST_TUNING_REPORT.md`: 5-fold Stratified CV hyperparameter tuning and threshold analysis ($t=0.50$ optimal).
  - `run_shap_ablation.py`: Computed TreeSHAP attributions; discovered missingness features (`missing_streak_count`, `missing_count`, `longest_missing_streak`) rank at the top; pruned 42 features to 18.
  - `run_feature_stability.py`: 5-fold feature stability frequency analysis (14-feature model evaluated).
  - `SGCC_ROBUSTNESS_ANALYSIS.md`: Perturbation testing under Gaussian noise (5%, 20%), data loss (isolated, 3-7d, 14-30d), and synthetic evasion ($\alpha \in [0.9, 0.5, 0.1]$). Proved 18-feature model is more robust than 42-feature model.
  - `process_uci_and_audit.py` & `run_uci_forecasting_pipeline.py`: Ingested 370 customer records ($140,256$ intervals), resampled to $35,065$ hourly intervals, extracted 29 causal lag/rolling/calendar features, benchmarked 10 models, froze XGBoost regressor, and evaluated on sealed H2 2014 test set.
- **Results**:
  - SGCC: 18-feature model achieved Test $F_1 = 0.4014$ (preliminary) / $0.3592$ (strict double-sealed).
  - UCI: Frozen XGBoost achieved Validation MAE = $3,935.63\text{ kWh}$, Sealed Test MAE = $4,503.51\text{ kWh}$ ($R^2 = 0.9942$, sMAPE = $1.98\%$).
- **Outcome**: Both research branches fully validated and frozen into serialized checkpoints.
- **Classification**: **COMPLETED**.

---

### Phase 4: Production FastAPI Backend & React 19 Frontend (Sept 28, 2026)
- **Git Commits**: `a72ce4e`, `08b87aa`, `1aa7e78`, `f2d980f`
- **Objective**: Construct low-latency REST inference backend and web dashboard for operational grid telemetry.
- **Implementation**:
  - `backend/`: FastAPI application loading `xgboost_reduced_best.joblib` and `xgb_forecasting_frozen.joblib` once during lifespan startup. Endpoints: `/health`, `/models`, `/detection/analyze`, `/detection/batch`, `/forecast/next-hour`.
  - `frontend/`: React 19, TypeScript, Vite 8, Tailwind CSS v4. Interactive meter inspection, CSV batch upload, real-time load forecasting, model intelligence drawer.
  - `tests/`: 13 automated integration tests in `test_api_endpoints.py`, `test_model_loading.py`, `test_sgcc_inference.py`, `test_uci_forecasting.py`.
- **Results**: 13/13 tests passed in 6.69s. Sub-10ms inference latency achieved.
- **Classification**: **COMPLETED**.

---

### Phase 5: Controlled V2 Optimization Research (October 5, 2026)
- **Git Commit**: `9c5486a` (*feat(v2): implement controlled model improvement framework*)
- **Objective**: Test advanced LightGBM, Optuna tuning, domain interaction ratios, and multi-scale EMA/Fourier series in an isolated workspace under a double-sealed test protocol.
- **Implementation**: `scripts/optimization_v2/run_v2_experiments.py`, `configs/optimization_v2/v2_experiment_config.json`.
- **Results**:
  - SGCC: Frozen XGBoost baseline ($F_1 = 0.3592$) outperformed all LightGBM candidates (top LightGBM $F_1 = 0.3366$).
  - UCI: 33-feature LightGBM achieved a 3.4% MAE reduction ($4,348.93$ vs $4,503.51$ kWh) but required stateful continuous EMA updating across REST API calls.
- **Outcome**: Baseline models retained in production to uphold stateless architecture and proven performance.
- **Classification**: **COMPLETED**.

---

### Phase 6: Focal Loss Optimization & Dataset Limitation Diagnostic (October 12, 2026)
- **Git Commits**: `3f37e60`, `4d50ca1`, `a08de15`, `1fbdafe`
- **Objective**: Implement custom Focal Loss objective, evaluate threshold operating curves, and conduct exhaustive programmatic audit of SGCC dataset limitations and class separability.
- **Implementation**:
  - `scripts/optimization_v2/train_and_plot_focal_loss.py`: Custom LightGBM Focal Loss ($\gamma=2.0, \alpha=0.85$).
  - `scripts/optimization_v2/run_threshold_and_focal_experiment.py`: Threshold operating curve across $t \in [0.10, 0.90]$.
  - `scripts/optimization_v2/run_dataset_limitation_diagnostics.py`: Overlap calculation across all features, missingness ablation, coverage group stratification, sample efficiency learning curves.
- **Results**:
  - LightGBM Focal Loss achieved **Test PR-AUC = 0.4178 (+34.2% lift)**, **Test ROC-AUC = 0.8429**, and **Test F1 = 0.4479** (serialized to `models/optimization_v2/lgbm_focal_loss_best.joblib`).
  - Proved $>95\%$ distribution overlap in consumption features and learning curve asymptotic saturation at $N=23,000$.
- **Outcome**: Successfully broken the baseline PR-AUC ceiling and definitively diagnosed root causes of performance limits.
- **Classification**: **COMPLETED**.

---

## 3. MASTER ACTIVITY CLASSIFICATION TABLE

Below is the verified classification of all 24 project activities:

```
+-------------------------------------------------------------------------------------------------------------------------+
|                                           GRIDBALANCE ACTIVITY CLASSIFICATION                                           |
+----+-----------------------------------------------------+-----------------------+--------------------------------------+
| #  | Activity Description                                | Audit Classification  | Empirical Repository Evidence        |
+----+-----------------------------------------------------+-----------------------+--------------------------------------+
| 1  | SGCC Raw Data Ingestion & Uniqueness Validation     | COMPLETED             | audit_real_sgcc.py (42,372 meters)   |
| 2  | SGCC Chronological Date Column Sorting              | COMPLETED             | run_fe_real_pipeline.py              |
| 3  | SGCC Missingness & Bounded Imputation Policy        | COMPLETED             | SGCC_Missingness_Strategy_Report.md  |
| 4  | SGCC 42-Feature Domain Engineering Pipeline         | COMPLETED             | sgcc_features_real_v2.csv            |
| 5  | SGCC ML Baselines (LogReg, DT, RF, XGBoost)         | COMPLETED             | REAL_SGCC_BASELINE_RESULTS.csv       |
| 6  | SGCC Class Imbalance Benchmarking (Weighted, SMOTE) | COMPLETED             | REAL_SGCC_BASELINE_MODEL_REPORT.md   |
| 7  | SGCC XGBoost Hyperparameter Search (Stratified CV)  | COMPLETED             | SGCC_XGBOOST_TUNING_REPORT.md        |
| 8  | SGCC TreeSHAP Attribution & Feature Pruning (18 F)  | COMPLETED             | SGCC_SHAP_FEATURE_IMPORTANCE.csv     |
| 9  | SGCC 5-Fold Feature Selection Stability Audit (14 F)| COMPLETED             | SGCC_FEATURE_SELECTION_STABILITY.md  |
| 10 | SGCC Perturbation Stress Testing (Noise, Loss, Evas)| COMPLETED             | SGCC_ROBUSTNESS_ANALYSIS.md          |
| 11 | SGCC Double-Sealed Test Partition Evaluation        | COMPLETED             | GRIDBALANCE_TRAIN_VAL_TEST_RESULTS   |
| 12 | SGCC LightGBM Focal Loss Experiment (gamma=2.0)     | COMPLETED             | lgbm_focal_loss_best.joblib          |
| 13 | SGCC Class Overlap & Empirical Separability Audit   | COMPLETED             | SGCC_DATASET_LIMITATION_AUDIT.md     |
| 14 | UCI 15-Minute Raw Ingestion & Hourly Aggregation    | COMPLETED             | process_uci_and_audit.py (35k hours) |
| 15 | UCI 29-Feature Causal Engineering Pipeline          | COMPLETED             | run_uci_forecasting_pipeline.py      |
| 16 | UCI Forecasting Baselines (Persistence, Linear, RF) | COMPLETED             | UCI_MODEL_COMPARISON.csv             |
| 17 | UCI Time-Series Expanding Window Cross-Validation   | COMPLETED             | UCI_TIME_SERIES_CV.csv               |
| 18 | UCI Double-Sealed Test Evaluation (H2 2014)         | COMPLETED             | GRIDBALANCE_TRAIN_VAL_TEST_RESULTS   |
| 19 | UCI V2 Optimization Experiments (EMA & Fourier)     | COMPLETED             | V2_MODEL_OPTIMIZATION_REPORT.md      |
| 20 | FastAPI Asynchronous REST Inference Backend         | COMPLETED             | backend/main.py (13/13 tests pass)   |
| 21 | React 19 + TypeScript + Vite 8 Web Application      | COMPLETED             | frontend/src/App.tsx                 |
| 22 | Operational Utility Cost Thresholding Matrix        | PLANNED (P1)          | Proposed in Improvement Roadmap      |
| 23 | UCI Difference Forecasting (Delta Load Formulation) | PLANNED (P1)          | Proposed in Improvement Roadmap      |
| 24 | 1D-ResNet / TCN Self-Supervised Sequence Learning   | PLANNED (P2)          | Proposed in Improvement Roadmap      |
+----+-----------------------------------------------------+-----------------------+--------------------------------------+
```
