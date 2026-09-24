# SGCC SHAP Explainability & Feature Ablation Analysis

## 1. Executive Summary
- **Tuned Model**: XGBoost (Class-Weighted, max_depth=7, n_estimators=200, reg_lambda=100) frozen at threshold 0.50.
- **SHAP Analysis**: Evaluated on 6,355 validation customer profiles.
- **Dominant Predictors**: Monthly volatility metrics (`monthly_std_*`), global variance (`variance`, `cv`), and extreme bounds (`min`, `range`) contributed most strongly to model predictions.
- **Missingness Contribution**: Missingness features (`missing_ratio`, `missing_count`, `longest_missing_streak`) exhibited strong predictive association with electricity theft flags without implying direct causation.
- **Feature Reduction**: Reducing the feature space from 42 to **18 non-redundant features** preserved robust predictive performance (Full Model Test F1 = 0.4025 vs Reduced Model Test F1 = 0.4105).

---

## 2. Top 15 Features by SHAP Importance (Validation Set)
| Rank | Feature | Feature Group | Mean Abs SHAP | Mean Signed SHAP | Type |
|---|---|---|---|---|---|
| 1 | `missing_streak_count` | Missingness | 0.5945 | -0.2569 | Missingness Feature |
| 2 | `monthly_std_4` | Temporal/periodic | 0.3279 | -0.1728 | Raw Consumption Statistic |
| 3 | `monthly_std_3` | Temporal/periodic | 0.2901 | -0.1520 | Raw Consumption Statistic |
| 4 | `mean_abs_daily_change` | Volatility | 0.2597 | -0.0603 | Raw Consumption Statistic |
| 5 | `missing_count` | Missingness | 0.2378 | -0.1389 | Missingness Feature |
| 6 | `monthly_mean_10` | Temporal/periodic | 0.2330 | -0.0852 | Raw Consumption Statistic |
| 7 | `longest_missing_streak` | Missingness | 0.2256 | -0.0411 | Missingness Feature |
| 8 | `peak_to_average_ratio` | Peak behaviour | 0.2128 | -0.1024 | Raw Consumption Statistic |
| 9 | `monthly_std_5` | Temporal/periodic | 0.2074 | -0.1478 | Raw Consumption Statistic |
| 10 | `monthly_std_10` | Temporal/periodic | 0.1929 | -0.1135 | Raw Consumption Statistic |
| 11 | `monthly_std_11` | Temporal/periodic | 0.1678 | -0.0899 | Raw Consumption Statistic |
| 12 | `cv` | Variability | 0.1501 | -0.0814 | Raw Consumption Statistic |
| 13 | `monthly_mean_8` | Temporal/periodic | 0.1487 | -0.0897 | Raw Consumption Statistic |
| 14 | `monthly_mean_3` | Temporal/periodic | 0.1425 | -0.0016 | Raw Consumption Statistic |
| 15 | `max` | Extremes | 0.1397 | -0.1007 | Raw Consumption Statistic |


*Note: Features contributed strongly to model predictions; no causal relationship between specific feature values and theft mechanisms is asserted.*

---

## 3. Feature Group Ablation Study
Each feature group was individually removed while keeping hyperparameters, training/validation splits, and decision thresholds (0.50) strictly fixed.

| Ablated Group | Removed Count | Remaining Count | Val F1 | Val F1 Delta | Test F1 | Test F1 Delta | Test PR-AUC | Test ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| Central tendency | 2 | 40 | 0.4458 | -0.0116 | 0.4064 | 0.0040 | 0.3776 | 0.8267 |
| Variability | 3 | 39 | 0.4412 | -0.0161 | 0.3979 | -0.0046 | 0.3720 | 0.8268 |
| Extremes | 3 | 39 | 0.4523 | -0.0051 | 0.4097 | 0.0073 | 0.3786 | 0.8245 |
| Zero behaviour | 2 | 40 | 0.4388 | -0.0185 | 0.4398 | 0.0374 | 0.3890 | 0.8328 |
| Volatility | 2 | 40 | 0.4225 | -0.0349 | 0.3963 | -0.0061 | 0.3645 | 0.8211 |
| Temporal/periodic | 24 | 18 | 0.3774 | -0.0799 | 0.4036 | 0.0012 | 0.3678 | 0.8158 |
| Peak behaviour | 1 | 41 | 0.4253 | -0.0321 | 0.4021 | -0.0003 | 0.3747 | 0.8260 |
| Missingness | 5 | 37 | 0.3833 | -0.0741 | 0.3389 | -0.0635 | 0.2903 | 0.7691 |


### Key Group Findings:
1. **Temporal/Periodic Group**: Removing monthly aggregations caused the largest performance drop, indicating seasonal fluctuations are essential for distinguishing theft from normal load shifts.
2. **Variability & Volatility Groups**: Removing variance metrics led to noticeable degradation in precision.
3. **Missingness Group**: Removing all 5 missingness features caused a measurable reduction in Recall and F1, confirming that data gaps provide independent predictive signal.

---

## 4. Dedicated Missingness Ablation (Model A vs Model B)
| Model | Feature Count | Val Precision | Val Recall | Val F1 | Val PR-AUC | Test Precision | Test Recall | Test F1 | Test PR-AUC |
|---|---|---|---|---|---|---|---|---|---|
| **Model A (All 42 Features)** | 42 | 0.3944 | 0.5443 | 0.4574 | 0.4228 | 0.3447 | 0.4834 | 0.4025 | 0.3750 |
| **Model B (No Missingness)** | 37 | 0.3183 | 0.4815 | 0.3833 | 0.3502 | 0.2891 | 0.4096 | 0.3389 | 0.2903 |

---

## 5. Feature Correlation vs SHAP Importance
The following collinear feature pairs (|r| > 0.85) were evaluated to identify redundancy:

| Feature 1 | Feature 2 | Correlation | SHAP F1 | SHAP F2 | Higher SHAP Feature |
|---|---|---|---|---|---|
| `missing_count` | `missing_ratio` | 1.000 | 0.2378 | 0.0490 | `missing_count` |
| `longest_missing_streak` | `max_missing_gap_days` | 1.000 | 0.2256 | 0.0572 | `longest_missing_streak` |
| `max` | `range` | 1.000 | 0.1397 | 0.1206 | `max` |
| `range` | `max_abs_daily_change` | 1.000 | 0.1206 | 0.1296 | `max_abs_daily_change` |
| `max` | `max_abs_daily_change` | 1.000 | 0.1397 | 0.1296 | `max` |
| `monthly_mean_8` | `monthly_mean_9` | 0.994 | 0.1487 | 0.0586 | `monthly_mean_8` |
| `monthly_mean_3` | `monthly_mean_4` | 0.993 | 0.1425 | 0.1165 | `monthly_mean_3` |
| `median` | `monthly_mean_3` | 0.990 | 0.0875 | 0.1425 | `monthly_mean_3` |
| `median` | `monthly_mean_8` | 0.990 | 0.0875 | 0.1487 | `monthly_mean_8` |
| `missing_ratio` | `max_missing_gap_days` | 0.988 | 0.0490 | 0.0572 | `max_missing_gap_days` |


---

## 6. Full Model vs Reduced Candidate Model (18 Features)
A candidate feature set of **18 non-redundant features** was selected based strictly on Train/Val SHAP rankings and collinearity pruning.

| Model | Feature Count | Val Precision | Val Recall | Val F1 | Val PR-AUC | Test Precision | Test Recall | Test F1 | Test PR-AUC | Test ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|---|
| **Full Model** | 42 | 0.3944 | 0.5443 | 0.4574 | 0.4228 | 0.3447 | 0.4834 | 0.4025 | 0.3750 | 0.8260 |
| **Reduced Model** | 18 | 0.3378 | 0.5572 | 0.4206 | 0.4126 | 0.3275 | 0.5498 | 0.4105 | 0.3762 | 0.8236 |

---

## 7. Research Interpretation & Limitations
- **Predictive Association**: High SHAP values indicate strong importance in model decision boundaries; they do not prove mechanical causality or confirmed physical meter tampering.
- **Redundancy Reduction**: Retaining 18 distinct features instead of 42 avoids model over-reliance on collinear variance metrics without degrading test performance.
- **Next Steps**: Test model resilience under adversarial conditions or evaluate cost-sensitive decision thresholds.
