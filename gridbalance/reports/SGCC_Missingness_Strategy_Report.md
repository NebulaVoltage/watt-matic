# SGCC Missingness Strategy & Audit Report

## 1. Missingness Audit
- **100% Missing Customers**: 5 (FLAG distribution: {1: 3, 0: 2})
- **99-100% Missing**: 8
- **Partial Missingness (>0, <100%)**: 42367

## 2. Missing Gap Distribution
Across the dataset, missing values occur in contiguous blocks (gaps):
- **Total gaps**: 234308
- **Median gap length**: 2.0 days
- **95th Percentile gap length**: 400.0 days
- **Maximum gap length**: 1034 days
- **Gaps > 7 days**: 37656
- **Gaps > 30 days**: 26307
- **Gaps > 90 days**: 22767

## 3. Imputation Strategy Justification
Given 26307 communication outages exceeding 30 days, unrestricted `ffill()` artificially carries consumption values across months, destroying seasonality and volatility.
**Strategy**: `ffill(limit=7)`. We will only forward-fill missing values up to 7 days to smooth brief meter disconnections/errors. Gaps longer than 7 days will remain `NaN`. Statistical features will use `np.nanmean`, `np.nanstd` directly on observed data, ensuring missingness does not distort mathematical realities.

## 4. Exclusion Criteria
The 5 completely empty customers will be explicitly dropped from the feature set. Because they lack any consumption history, they provide zero signal.

## 5. Next ML Steps
Remaining NaNs in the engineered feature dataset (caused by customers having mostly NaNs and unable to form trend/volatility metrics) should be handled via a Scikit-Learn `SimpleImputer` (e.g., median) inside a pipeline fitted STRICTLY on the training set to prevent leakage.
