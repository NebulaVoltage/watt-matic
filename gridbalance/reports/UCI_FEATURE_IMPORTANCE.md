# UCI Forecasting Feature Importance & Interpretability Report

## 1. Feature Importance Summary
The top predictive features driving next-hour load forecasts:

1. `load_t` (Importance: 0.7578)
2. `cos_hour` (Importance: 0.0770)
3. `rolling_max_24` (Importance: 0.0432)
4. `sin_hour` (Importance: 0.0401)
5. `rolling_mean_24` (Importance: 0.0328)
6. `hour` (Importance: 0.0183)
7. `rolling_std_24` (Importance: 0.0073)
8. `rolling_min_24` (Importance: 0.0052)
9. `rolling_std_168` (Importance: 0.0030)
10. `load_t_minus_12` (Importance: 0.0022)

## 2. Key Domain Interpretations
- **Lag 0 (`load_t`) and Lag 1 (`load_t_minus_1`)**: The most immediate prior load values dominate predictions, reflecting strong autoregressive continuity.
- **Daily Seasonality (`load_t_minus_24`)**: 24-hour lag features strongly anchor forecasts to same-time-yesterday usage.
- **Rolling Averages (`rolling_mean_24`, `rolling_mean_168`)**: Provide smooth baseline trend indicators for multi-day shifts.
