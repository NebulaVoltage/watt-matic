# GridBalance: Step-by-Step Implementation Protocol for Next Research Phase

> **Role & Authority**: Lead ML Research Scientist, Systems Architect  
> **Repository Root**: `d:\watt-matic\gridbalance`  
> **Protocol Target**: Actionable Implementation Specifications for Approved Roadmap Experiments  
> **Precondition**: Strict Zero-Leakage, Frozen Baseline Preservation, Double-Sealed Test Integrity  

---

## 1. PHASE A — IMMEDIATE SYSTEM DEPLOYMENTS & CALIBRATION (WEEK 1)

```
+---------------------------------------------------------------------------------------------------------+
|                                    PHASE A IMPLEMENTATION SEQUENCE                                      |
+-----+----------------------------------------+------------------------------------+---------------------+
| Step| Task Name                              | Files Modified                     | Target Metric       |
+-----+----------------------------------------+------------------------------------+---------------------+
| A.1 | Reconcile UCI Documentation & API      | backend/config.py, frontend/src/   | Zero Discrepancy    |
| A.2 | Restrict TimeSeriesSplit to 2012–2013  | run_uci_forecasting_pipeline.py    | CV MAE < 5,000 kWh  |
| A.3 | Integrate LightGBM Focal Loss in Prod  | backend/inference/sgcc_predictor.py| PR-AUC >= 0.4178    |
| A.4 | Deploy Operational Utility Matrix      | backend/schemas/sgcc.py, frontend  | Max Economic Return |
+-----+----------------------------------------+------------------------------------+---------------------+
```

---

### Step A.1: Reconcile UCI Documentation & API Metadata
- **Why Do It**: Eliminate the minor 24-hour test window discrepancy between backend configuration ($4,421.55$ kWh) and final status reports ($4,503.51$ kWh).
- **Files Modified**:
  - `backend/config.py`:
    ```python
    UCI_METRICS = {
        "model_name": "XGBoost Regressor (29-Feature)",
        "feature_count": 29,
        "forecast_horizon": "next_hour",
        "test_mae_full_split_kwh": 4503.51,      # Full chronological H2 2014 test split (4,416 hours)
        "test_mae_trimmed_kwh": 4421.55,         # Lag warm-up boundary trimmed split (4,392 hours)
        "test_rmse_kwh": 6743.21,
        "test_r2": 0.9942,
        "test_smape_pct": 1.98,
        "sealed_test_size": 4416,
        "disclaimer": "Evaluated on sealed chronological test split (July 1 - Dec 31, 2014). Model frozen."
    }
    ```
  - `frontend/src/pages/ModelIntelligence.tsx`: Update displayed benchmark cards to present both windows.
- **Verification Command**: `pytest tests/test_api_endpoints.py`
- **Success Criteria**: All endpoint tests pass; `/api/v1/models` returns unified benchmark metadata.

---

### Step A.2: Restrict TimeSeriesSplit to the 2012–2013 Regime
- **Why Do It**: Fix the misleading $10,365.77$ kWh CV MAE caused by Fold-2 training on 2011 (partial network onboarding) and testing on 2012 (demand doubled).
- **Files Modified**: `run_uci_forecasting_pipeline.py`.
- **Implementation Logic**:
  ```python
  # Restrict cross-validation training window to stationary full-meter operation
  cv_mask = (features.index >= '2012-01-01 00:00:00') & (features.index <= '2013-12-31 23:00:00')
  X_cv, y_cv = features.loc[cv_mask], target.loc[cv_mask]
  tscv = TimeSeriesSplit(n_splits=5)
  ```
- **What Results to Measure**: Mean MAE, Std MAE, Mean RMSE, Mean $R^2$ across the 5 folds.
- **Success Criteria**: 5-fold CV Mean MAE drops below **$5,000\text{ kWh}$** with standard deviation $< 1,000\text{ kWh}$.
- **Fallback Plan**: If early 2012 still shows residual onboarding volatility, advance CV start date to `2012-03-01`.

---

### Step A.3: Integrate LightGBM Focal Loss into Production Backend
- **Why Do It**: Provide immediate access to the **+34.2% PR-AUC lift** (0.3114 to 0.4178) and higher detection recall achieved by `models/optimization_v2/lgbm_focal_loss_best.joblib`.
- **Files Modified**:
  - `backend/config.py`: Add `SGCC_FOCAL_MODEL_PATH = PROJECT_ROOT / "models" / "optimization_v2" / "lgbm_focal_loss_best.joblib"`.
  - `backend/inference/sgcc_predictor.py`: Add optional loader for focal loss bundle (`bst`, `optimal_threshold = 0.50`).
- **Inference Verification**:
  Ensure that incoming requests pass identical 18-feature extraction, compute raw logits, and convert via sigmoid:
  $$p = \frac{1}{1 + e^{-z}}$$
- **Success Criteria**: Live API returns `prediction: "potential_tampering"` matching the calibrated Focal Loss predictions with inference latency $< 10$ms.

---

### Step A.4: Deploy Operational Cost-Sensitive Utility Matrix
- **Why Do It**: Replace arbitrary $t=0.50$ threshold with quantified utility cost optimization (inspection costs vs. recovered stolen revenue).
- **Files Modified**:
  - `backend/schemas/sgcc.py`: Add `economic_impact` schema to `MeterDetectionResponse`.
  - `backend/services/sgcc_service.py`: Calculate estimated stolen kWh recovery:
    $$\text{Expected Recovery} = \hat{p} \times (\text{Peak} - \text{Mean}) \times 30\text{ days} \times \text{Tariff}$$
- **Success Criteria**: Operators receive estimated field inspection ROI on flagged meters.

---

## 2. PHASE B — CORE MODELING INNOVATIONS (WEEK 2–3)

### Step B.1: Formulate UCI Difference (Delta Load) Forecasting
- **Why Do It**: Eliminate residual lag-1 persistence and force the regressor to learn true load acceleration.
- **Files Modified**: `scripts/research/run_uci_delta_forecasting.py`.
- **Implementation Formulation**:
  1. Define target: $\Delta y_{t+1} = y_{t+1} - y_t$.
  2. Features: Include lag differences $\Delta y_t = y_t - y_{t-1}$, $\Delta_{24} y_t = y_t - y_{t-24}$, and cyclic hour/day features.
  3. Predict: $\widehat{\Delta y}_{t+1}$.
  4. Reconstruct: $\hat{y}_{t+1} = y_t + \widehat{\Delta y}_{t+1}$.
- **What Results to Measure**: Sealed Test MAE, RMSE, $R^2$, sMAPE on H2 2014 test set.
- **Success Criteria**: Sealed Test MAE drops below **$4,000\text{ kWh}$** (vs. $4,503.51$ kWh baseline).
- **Fallback Plan**: If difference formulation increases variance during night trough hours, blend via simple 50/50 weighted average: $\hat{y}_{\text{final}} = 0.5 \hat{y}_{\text{level}} + 0.5 \hat{y}_{\text{delta}}$.

---

### Step B.2: Implement Semi-Supervised Positive-Unlabeled (PU) Learning on SGCC
- **Why Do It**: Overcome label noise where un-inspected thieves are labeled as benign (`FLAG=0`), capping precision.
- **Files Modified**: `scripts/research/run_sgcc_pu_learning.py`.
- **Algorithm**: Two-step Elkan-Noto PU Learning:
  1. Train an initial probabilistic classifier on observed flags $s \in \{0, 1\}$.
  2. Estimate the labeling frequency constant $c = P(s=1 \mid y=1)$ from a high-confidence validation slice.
  3. Reweight unlabeled instances with importance weights $w_i = \frac{1 - c}{c} \frac{P(s=1 \mid x_i)}{1 - P(s=1 \mid x_i)}$.
  4. Retrain LightGBM Focal Loss using instance reweighting.
- **Success Criteria**: Test Precision improves to $> 0.45$ while maintaining Recall $\ge 50\%$.

---

## 3. PHASE C — DEEP SEQUENCE REPRESENTATION & REBUTTALS (WEEK 4)

### Step C.1: 1D-ResNet / Temporal Convolutional Network (TCN)
- **Why Do It**: Break the $F_1 \approx 0.45$ ceiling caused by handcrafted summary statistics obliterating temporal sequence shapes.
- **Files Modified**: `scripts/research/train_resnet_sgcc.py`.
- **Architecture**:
  - Input: Raw daily consumption sequence ($1 \times 1,034$ tensor).
  - Layers: 4 residual convolutional blocks with kernel size 7, dilated convolutions (dilation = 1, 2, 4, 8), batch normalization, and dropout ($p=0.3$).
  - Global average pooling followed by linear classification head with Focal Loss ($\gamma=2.0$).
- **Success Criteria**: Test $F_1 > 0.48$ using consumption series alone without missingness features.

---

### Step C.2: Reproduce Arif et al. (2021) SMOTE-NearMiss Literature Rebuttal
- **Why Do It**: Definitively rebut claims in published literature that SMOTE-NearMiss achieves $>90\%$ accuracy on SGCC.
- **Files Modified**: `scripts/research/benchmark_arif_comparison.py`.
- **Experiment Setup**:
  - Apply SMOTE-NearMiss to training folds (generating a 1:1 balanced training set).
  - Train Random Forest and Decision Tree classifiers following Arif et al.
  - Evaluate on both:
    1. A rebalanced test set (reproducing Arif's inflated metrics).
    2. The untouched, naturally imbalanced sealed test set ($8.53\%$ positive prevalence).
- **Success Criteria**: Demonstrate mathematically and empirically that while the model scores $>90\%$ accuracy on the artificial balanced test set, its precision drops to $<15\%$ on the real imbalanced test set due to NearMiss eliminating realistic boundary negatives. Include this finding as a key contribution in the final academic paper.
