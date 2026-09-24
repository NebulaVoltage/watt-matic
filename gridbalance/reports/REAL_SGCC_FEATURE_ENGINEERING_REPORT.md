# REAL SGCC Feature Engineering Report

## 1. Input Dataset Details
- **Source**: `D:\wattmaticdataset\data.csv`
- **Row Count**: 42372
- **Original Column Count**: 1036
- **Consumption Columns**: 1034
- **Class Distribution**: FLAG=0: 38757, FLAG=1: 3615

## 2. Chronological Parsing & Sorting
- **Date Parsing Method**: `pandas.to_datetime` automatically parsing strings (e.g., '2014/1/1'). All successfully parsed.
- **Sorting Method**: Dates were sorted using pandas `sort_values` on the datetime representations. The sorted column name list was then passed to `FeatureExtractor`.
- **Temporal Sanity Check**:
  - Raw (First 5): ['2014/1/1', '2014/1/10', '2014/1/11', '2014/1/12', '2014/1/13'] (Lexicographical)
  - Sorted (First 5): ['2014/1/1', '2014/1/2', '2014/1/3', '2014/1/4', '2014/1/5'] (Chronological)
  - Sorting guarantees that `np.diff()` computes day-to-day changes correctly, and that temporal sliding/aggregations respect true calendar progression.

## 3. Imputation Review
- **Method**: `ffill(axis=1).bfill(axis=1)` operating on the chronologically sorted row vector per customer.
- **Imputation Metric**: Approximately 25.64% of missing values were imputed this way.
- **Scientific Impact**: While this enables numerical pipelines to run without NaNs, carrying-forward (`ffill`) over multi-month gaps is problematic for highly seasonal load curves, as it "freezes" summer peaks into winter. Missingness patterns (zero-ratio, missing gaps) should ideally be decoupled from raw fill values. However, for this baseline comparison, the method was preserved strictly for compatibility.

## 4. Feature Extraction Validation
- **Output Shape**: (42372, 57) (Includes CONS_NO and FLAG)
- **Feature Count**: 55
- **NaN / Inf Count**: 235 NaNs, 0 Infs
- **Constant Features**: 0 
- **Near-Zero Variance**: 0 

## 5. Preprocessing Concerns Before ML Baseline
- **Correlation**: High collinearity remains among variance/MAD metrics.
- **Imputation Limitations**: Ffill/Bfill creates artificial flatlines during communication outages, potentially depressing volatility metrics.
- **Target Leakage**: The extreme deterministic separability observed in the synthetic data has been eliminated. The current feature distributions (see `graphs/real_sgcc_feature_engineering/`) show highly overlapping metrics, which is expected for realistic human behavior data.

## 6. Output Files
- **Dataset**: `data/processed/sgcc_features_real.csv`
- **Plots**: `graphs/real_sgcc_feature_engineering/feature_distributions_real.png`
