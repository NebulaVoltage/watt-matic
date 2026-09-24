# SGCC Feature Engineering Report

## 1. Objective
To construct a research-grade feature engineering pipeline for the SGCC electricity-theft detection module, extracting robust statistical, temporal, and behavioural features from the raw daily consumption data.

## 2. Raw Dataset Structure
- Total Rows: 1500
- Total Columns: 1036
- Number of Raw Daily Features: 1034

## 3. Feature Groups Generated
We generated features across 8 categories:
- **Central Tendency**: mean, median, trimmed_mean
- **Variability**: std_dev, variance, cv, iqr, mad
- **Extremes**: min, max, range, percentile_5, percentile_95
- **Zero Behaviour**: zero_count, zero_ratio, longest_zero_streak, zero_streak_count
- **Volatility**: mean_abs_daily_change, std_abs_daily_change, max_abs_daily_change, unusually_large_changes_count
- **Trend**: linear_trend_slope, trend_r2, early_vs_late_change, relative_trend
- **Periodic/Temporal Features**: monthly_mean_1..12, monthly_std_1..12, seasonal_variation, temporal_consistency
- **Peak Behaviour**: peak_to_average_ratio, high_consumption_day_count, high_consumption_day_ratio, peak_frequency

## 4. Pipeline Execution Summary
- Number of Engineered Features: 55
- Missing Values (NaN) after imputation: 0
- Constant Features (variance = 0): 0 
- Near-Zero Variance Features: 1 ['linear_trend_slope']
- Highly Correlated Features (r > 0.95): 33 ['median', 'trimmed_mean', 'variance', 'mad', 'percentile_5', 'percentile_95', 'zero_ratio', 'std_abs_daily_change', 'monthly_mean_1', 'monthly_std_1', 'monthly_mean_2', 'monthly_std_2', 'monthly_mean_3', 'monthly_std_3', 'monthly_mean_4', 'monthly_std_4', 'monthly_mean_5', 'monthly_std_5', 'monthly_mean_6', 'monthly_std_6', 'monthly_mean_7', 'monthly_std_7', 'monthly_mean_8', 'monthly_std_8', 'monthly_mean_9', 'monthly_std_9', 'monthly_mean_10', 'monthly_std_10', 'monthly_mean_11', 'monthly_mean_12', 'monthly_std_12', 'high_consumption_day_ratio', 'peak_frequency']
(Note: Highly correlated features have not been automatically deleted to preserve the correlation structure for research purposes).

## 5. Important Observations & Limitations
- **Data Sparsity Strategy**: Missing values were imputed per-customer using forward-fill followed by back-fill to maintain temporal consistency without leaking data across customers. Target `FLAG` was never used during extraction.
- **Correlations**: Features derived from identical domains (like high_consumption_day_count and peak_frequency) are expectedly highly correlated. We retain them for ML algorithm selection later.
- **Limitations**: Monthly metrics assume calendar alignment. Zero behaviour metrics assume true '0' is captured without sensor noise.
