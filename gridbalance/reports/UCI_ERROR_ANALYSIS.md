# UCI Load Forecasting Error Analysis Report

## 1. Overall Test Performance (Sealed Evaluation)
- **Model**: XGBoost Regressor
- **MAE**: 4421.55 kWh
- **RMSE**: 6654.94 kWh
- **R²**: 0.9944
- **sMAPE**: 1.93%
- **MAE Improvement vs Daily Seasonal Naive**: 78.75%

## 2. Failure Mode Analysis
### A. Performance by Demand Quantile
- **Low Demand (Q1)**: MAE = 2529.40 kWh
- **Peak Demand (Q5)**: MAE = 5327.04 kWh
*Observation*: Forecast errors scale proportionally with aggregate load magnitude. Absolute errors peak during maximum load hours (e.g. 18:00 - 20:00).

### B. Hourly Error Distribution
- Lowest Error Hour: Hour 2 (MAE = 1987.11 kWh)
- Highest Error Hour: Hour 22 (MAE = 8891.01 kWh)
