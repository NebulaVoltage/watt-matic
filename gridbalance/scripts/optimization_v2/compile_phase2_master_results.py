import pandas as pd
import numpy as np
import os

# Create master summary dataframe
records = [
    # Exp 1: SGCC Focal Loss Verification
    {
        "Branch": "SGCC",
        "Experiment_ID": "EXP_01",
        "Experiment_Name": "Focal Loss Verification",
        "Model": "LightGBM Focal Loss (alpha=0.85, gamma=2.0)",
        "Split": "5-Fold CV Train",
        "Primary_Metric_Name": "F1-Score",
        "Primary_Metric_Value": 0.4166,
        "Secondary_Metric_1_Name": "F1_Std",
        "Secondary_Metric_1_Value": 0.0121,
        "Secondary_Metric_2_Name": "Baseline_CV_F1",
        "Secondary_Metric_2_Value": 0.3874,
        "Status": "Verified Superior (CV + Val)",
        "Key_Finding": "Outperforms frozen XGBoost (+2.92% CV F1). Stable custom loss, valid sigmoid logits, no label leakage."
    },
    {
        "Branch": "SGCC",
        "Experiment_ID": "EXP_01",
        "Experiment_Name": "Focal Loss Verification",
        "Model": "LightGBM Focal Loss (alpha=0.85, gamma=2.0)",
        "Split": "Validation",
        "Primary_Metric_Name": "F1-Score",
        "Primary_Metric_Value": 0.4469,
        "Secondary_Metric_1_Name": "PR-AUC",
        "Secondary_Metric_1_Value": 0.4215,
        "Secondary_Metric_2_Name": "ROC-AUC",
        "Secondary_Metric_2_Value": 0.8419,
        "Status": "Verified Superior (CV + Val)",
        "Key_Finding": "Validation F1 improved from 0.4037 to 0.4469 (+10.70%). PR-AUC improved to 0.4215."
    },
    {
        "Branch": "SGCC",
        "Experiment_ID": "EXP_01",
        "Experiment_Name": "Focal Loss Verification",
        "Model": "LightGBM Focal Loss (alpha=0.85, gamma=2.0)",
        "Split": "Sealed Test",
        "Primary_Metric_Name": "F1-Score",
        "Primary_Metric_Value": 0.4479,
        "Secondary_Metric_1_Name": "PR-AUC",
        "Secondary_Metric_1_Value": 0.4178,
        "Secondary_Metric_2_Name": "ROC-AUC",
        "Secondary_Metric_2_Value": 0.8429,
        "Status": "Verified Candidate",
        "Key_Finding": "Test F1 matches reported 0.4479 exactly; PR-AUC matches 0.4178 (+34.2% over XGBoost test 0.3113)."
    },

    # Exp 2: SGCC Missingness Dependency
    {
        "Branch": "SGCC",
        "Experiment_ID": "EXP_02",
        "Experiment_Name": "Missingness Dependency Ablation",
        "Model": "LightGBM (Consumption Features Only, 15 Feats)",
        "Split": "Validation",
        "Primary_Metric_Name": "F1-Score",
        "Primary_Metric_Value": 0.3653,
        "Secondary_Metric_1_Name": "PR-AUC",
        "Secondary_Metric_1_Value": 0.3588,
        "Secondary_Metric_2_Name": "MCC",
        "Secondary_Metric_2_Value": 0.3082,
        "Status": "Performance Degradation",
        "Key_Finding": "Removing missingness drops Val F1 by -8.2% and Val PR-AUC by -11.6%."
    },
    {
        "Branch": "SGCC",
        "Experiment_ID": "EXP_02",
        "Experiment_Name": "Missingness Dependency Ablation",
        "Model": "LightGBM (Missingness Features Only, 3 Feats)",
        "Split": "Validation",
        "Primary_Metric_Name": "F1-Score",
        "Primary_Metric_Value": 0.2964,
        "Secondary_Metric_1_Name": "PR-AUC",
        "Secondary_Metric_1_Value": 0.3189,
        "Secondary_Metric_2_Name": "MCC",
        "Secondary_Metric_2_Value": 0.2447,
        "Status": "Incomplete Signal",
        "Key_Finding": "Missingness alone yields modest F1 (0.2964); telemetry gaps capture tampering and comm drops."
    },
    {
        "Branch": "SGCC",
        "Experiment_ID": "EXP_02",
        "Experiment_Name": "Missingness Dependency Ablation",
        "Model": "LightGBM (Combined Features, 18 Feats)",
        "Split": "Validation",
        "Primary_Metric_Name": "F1-Score",
        "Primary_Metric_Value": 0.3981,
        "Secondary_Metric_1_Name": "PR-AUC",
        "Secondary_Metric_1_Value": 0.4057,
        "Secondary_Metric_2_Name": "MCC",
        "Secondary_Metric_2_Value": 0.3475,
        "Status": "Optimal Synergy",
        "Key_Finding": "Combined model yields highest PR-AUC and F1; missingness features are legitimate telemetry indicators."
    },

    # Exp 3: SGCC Operational Threshold Analysis
    {
        "Branch": "SGCC",
        "Experiment_ID": "EXP_03",
        "Experiment_Name": "Operational Threshold Analysis",
        "Model": "LightGBM Focal Loss (Threshold = 0.50, F1-Optimal)",
        "Split": "Validation",
        "Primary_Metric_Name": "F1-Score",
        "Primary_Metric_Value": 0.4469,
        "Secondary_Metric_1_Name": "Precision",
        "Secondary_Metric_1_Value": 0.4198,
        "Secondary_Metric_2_Name": "Recall",
        "Secondary_Metric_2_Value": 0.4779,
        "Status": "Balanced Operating Point",
        "Key_Finding": "Flag rate 9.71%. Optimal balance of false discoveries and missed fraud for standard audits."
    },
    {
        "Branch": "SGCC",
        "Experiment_ID": "EXP_03",
        "Experiment_Name": "Operational Threshold Analysis",
        "Model": "LightGBM Focal Loss (Threshold = 0.40, High-Recall)",
        "Split": "Validation",
        "Primary_Metric_Name": "Recall",
        "Primary_Metric_Value": 0.7399,
        "Secondary_Metric_1_Name": "Precision",
        "Secondary_Metric_1_Value": 0.2406,
        "Secondary_Metric_2_Name": "F2-Score",
        "Secondary_Metric_2_Value": 0.5228,
        "Status": "Fraud Sweep Operating Point",
        "Key_Finding": "Catches 74.0% of all tampering at the cost of inspecting 26.23% of grid meters."
    },
    {
        "Branch": "SGCC",
        "Experiment_ID": "EXP_03",
        "Experiment_Name": "Operational Threshold Analysis",
        "Model": "LightGBM Focal Loss (Threshold = 0.55, High-Precision)",
        "Split": "Validation",
        "Primary_Metric_Name": "Precision",
        "Primary_Metric_Value": 0.5346,
        "Secondary_Metric_1_Name": "Recall",
        "Secondary_Metric_1_Value": 0.3137,
        "Secondary_Metric_2_Name": "Flag_Rate",
        "Secondary_Metric_2_Value": 0.0500,
        "Status": "Resource-Constrained Point",
        "Key_Finding": "53.5% hit rate by dispatching crews to only top 5.0% of suspicious meters."
    },

    # Exp 4: UCI CV Regime Audit
    {
        "Branch": "UCI",
        "Experiment_ID": "EXP_04",
        "Experiment_Name": "CV Regime Audit",
        "Model": "XGBoost 29-Feat (Protocol A: 2012-2013)",
        "Split": "TSCV 5-Fold Mean",
        "Primary_Metric_Name": "MAE_kWh",
        "Primary_Metric_Value": 15578.43,
        "Secondary_Metric_1_Name": "RMSE_kWh",
        "Secondary_Metric_1_Value": 21212.69,
        "Secondary_Metric_2_Name": "sMAPE_pct",
        "Secondary_Metric_2_Value": 6.59,
        "Status": "Regime-Aware Baseline",
        "Key_Finding": "Reflects fully deployed 370-meter network. Folds 2-5 MAE avg 5,380 kWh."
    },
    {
        "Branch": "UCI",
        "Experiment_ID": "EXP_04",
        "Experiment_Name": "CV Regime Audit",
        "Model": "XGBoost 29-Feat (Protocol B: 2011-2013)",
        "Split": "TSCV 5-Fold Mean",
        "Primary_Metric_Name": "MAE_kWh",
        "Primary_Metric_Value": 10265.63,
        "Secondary_Metric_1_Name": "RMSE_kWh",
        "Secondary_Metric_1_Value": 16229.22,
        "Secondary_Metric_2_Name": "sMAPE_pct",
        "Secondary_Metric_2_Value": 5.01,
        "Status": "Full-History Stress Benchmark",
        "Key_Finding": "Fold 2 tests 2011->2012 onboarding shift (MAE jumps to 30,622 kWh due to tree extrapolation failure)."
    },

    # Exp 5: UCI Difference Forecasting
    {
        "Branch": "UCI",
        "Experiment_ID": "EXP_05",
        "Experiment_Name": "Difference Forecasting Benchmark",
        "Model": "XGBoost Direct (29 Feat)",
        "Split": "Sealed Test (H2 2014)",
        "Primary_Metric_Name": "MAE_kWh",
        "Primary_Metric_Value": 4440.05,
        "Secondary_Metric_1_Name": "RMSE_kWh",
        "Secondary_Metric_1_Value": 6686.97,
        "Secondary_Metric_2_Name": "sMAPE_pct",
        "Secondary_Metric_2_Value": 1.94,
        "Status": "Frozen Baseline",
        "Key_Finding": "Direct level forecasting maintains strong R2=0.9943; Val MAE = 3914.41 kWh."
    },
    {
        "Branch": "UCI",
        "Experiment_ID": "EXP_05",
        "Experiment_Name": "Difference Forecasting Benchmark",
        "Model": "XGBoost Difference (Delta)",
        "Split": "Sealed Test (H2 2014)",
        "Primary_Metric_Name": "MAE_kWh",
        "Primary_Metric_Value": 4350.13,
        "Secondary_Metric_1_Name": "RMSE_kWh",
        "Secondary_Metric_1_Value": 6512.27,
        "Secondary_Metric_2_Name": "sMAPE_pct",
        "Secondary_Metric_2_Value": 1.92,
        "Status": "Improved Test Accuracy",
        "Key_Finding": "Predicts delta y(t+1) - y(t). Marginally improves test MAE (-89.92 kWh) and lowers rising load error."
    },
    {
        "Branch": "UCI",
        "Experiment_ID": "EXP_05",
        "Experiment_Name": "Difference Forecasting Benchmark",
        "Model": "LightGBM Direct (29 Feat)",
        "Split": "Sealed Test (H2 2014)",
        "Primary_Metric_Name": "MAE_kWh",
        "Primary_Metric_Value": 4394.26,
        "Secondary_Metric_1_Name": "RMSE_kWh",
        "Secondary_Metric_1_Value": 6700.19,
        "Secondary_Metric_2_Name": "sMAPE_pct",
        "Secondary_Metric_2_Value": 1.92,
        "Status": "Validation-Optimal",
        "Key_Finding": "Best validation MAE (3869.53 kWh). Fast inference (~2.8 ms), excellent test R2=0.9943."
    },
    {
        "Branch": "UCI",
        "Experiment_ID": "EXP_05",
        "Experiment_Name": "Difference Forecasting Benchmark",
        "Model": "LightGBM Difference (Delta)",
        "Split": "Sealed Test (H2 2014)",
        "Primary_Metric_Name": "MAE_kWh",
        "Primary_Metric_Value": 4373.05,
        "Secondary_Metric_1_Name": "RMSE_kWh",
        "Secondary_Metric_1_Value": 6550.11,
        "Secondary_Metric_2_Name": "sMAPE_pct",
        "Secondary_Metric_2_Value": 1.94,
        "Status": "Alternative Architecture",
        "Key_Finding": "Test MAE 4373.05 kWh; slight reduction in peak error but slightly higher Val MAE (3961.96 kWh)."
    },

    # Exp 6: UCI EMA Inference Audit
    {
        "Branch": "UCI",
        "Experiment_ID": "EXP_06",
        "Experiment_Name": "EMA Stateless Inference Audit",
        "Model": "LightGBM 33-Feat (24h Finite Window)",
        "Split": "Inference Simulation",
        "Primary_Metric_Name": "EMA_24_Mean_Bias_kWh",
        "Primary_Metric_Value": 9802.85,
        "Secondary_Metric_1_Name": "EMA_24_Max_Bias_kWh",
        "Secondary_Metric_1_Value": 22286.33,
        "Secondary_Metric_2_Name": "Deployment_Decision",
        "Secondary_Metric_2_Value": "REJECT_STATELESS_24H",
        "Status": "Critical Contract Breach",
        "Key_Finding": "Dynamic EMA calculation on 24h history produces ~10% artificial bias. Requires 168h (7 days) minimum."
    },
    {
        "Branch": "UCI",
        "Experiment_ID": "EXP_06",
        "Experiment_Name": "EMA Stateless Inference Audit",
        "Model": "LightGBM 33-Feat (168h Finite Window)",
        "Split": "Inference Simulation",
        "Primary_Metric_Name": "EMA_24_Mean_Bias_kWh",
        "Primary_Metric_Value": 0.0611,
        "Secondary_Metric_1_Name": "EMA_24_Max_Bias_kWh",
        "Secondary_Metric_1_Value": 0.1464,
        "Secondary_Metric_2_Name": "Deployment_Decision",
        "Secondary_Metric_2_Value": "CONDITIONALLY_ACCEPTABLE",
        "Status": "Stateless Warmup Saturated",
        "Key_Finding": "Warmup bias reduces to <0.1 kWh if API contract expands payload from 24h to 168h."
    },
    {
        "Branch": "UCI",
        "Experiment_ID": "EXP_06",
        "Experiment_Name": "EMA Stateless Inference Audit",
        "Model": "XGBoost / LightGBM Direct (29 Feat)",
        "Split": "Inference Simulation",
        "Primary_Metric_Name": "Warmup_Bias_kWh",
        "Primary_Metric_Value": 0.00,
        "Secondary_Metric_1_Name": "Payload_Requirement_Hours",
        "Secondary_Metric_1_Value": 24.0,
        "Secondary_Metric_2_Name": "Deployment_Decision",
        "Secondary_Metric_2_Value": "PRODUCTION_APPROVED",
        "Status": "Strictly Stateless & Safe",
        "Key_Finding": "29-feature contract requires zero infinite-impulse recursive state; 100% causal and stateless."
    },

    # Exp 7: Final Candidate Decisions
    {
        "Branch": "SGCC",
        "Experiment_ID": "EXP_07",
        "Experiment_Name": "Final Candidate Decision",
        "Model": "LightGBM Focal Loss (alpha=0.85, gamma=2.0)",
        "Split": "Candidate Selection Audit",
        "Primary_Metric_Name": "CV_F1",
        "Primary_Metric_Value": 0.4166,
        "Secondary_Metric_1_Name": "Val_F1",
        "Secondary_Metric_1_Value": 0.4469,
        "Secondary_Metric_2_Name": "Promotion_Decision",
        "Secondary_Metric_2_Value": "RECOMMENDED_FOR_PROMOTION",
        "Status": "Approved by CV & Validation",
        "Key_Finding": "Selection validated on CV and Validation sets. Outperforms baseline XGBoost across all metrics."
    },
    {
        "Branch": "UCI",
        "Experiment_ID": "EXP_07",
        "Experiment_Name": "Final Candidate Decision",
        "Model": "XGBoost Direct 29-Feat (Baseline) / LightGBM Direct 29-Feat",
        "Split": "Candidate Selection Audit",
        "Primary_Metric_Name": "Val_MAE_kWh",
        "Primary_Metric_Value": 3869.53,
        "Secondary_Metric_1_Name": "Test_MAE_kWh",
        "Secondary_Metric_1_Value": 4394.26,
        "Secondary_Metric_2_Name": "Promotion_Decision",
        "Secondary_Metric_2_Value": "RETAIN_FROZEN_BASELINE_OR_DROPIN_LGBM29",
        "Status": "Baseline Retained (Contract Safe)",
        "Key_Finding": "33-feat rejected for stateless 24h API. Difference modeling yields marginal test gain. Baseline retained."
    }
]

df = pd.DataFrame(records)

# Save to both paths
out_p1 = r"D:\watt-matic\gridbalance\results\optimization_v2\PHASE2_MASTER_RESULTS.csv"
out_p2 = r"D:\watt-matic\results\optimization_v2\PHASE2_MASTER_RESULTS.csv"

os.makedirs(os.path.dirname(out_p1), exist_ok=True)
os.makedirs(os.path.dirname(out_p2), exist_ok=True)

df.to_csv(out_p1, index=False)
df.to_csv(out_p2, index=False)
print("Saved Master CSV to:")
print(f" - {out_p1}")
print(f" - {out_p2}")
print(f"Total rows: {len(df)}")
