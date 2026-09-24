import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
warnings.filterwarnings('ignore')

def create_dirs():
    os.makedirs('reports', exist_ok=True)
    os.makedirs('graphs/real_sgcc_missingness', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)

def analyze_gaps(bool_mat):
    # bool_mat is boolean array: True where missing, False where valid
    # We find lengths of contiguous True streaks
    padded = np.pad(bool_mat, ((0,0), (1,1)), mode='constant', constant_values=False)
    diffs = np.diff(padded.astype(int), axis=1)
    
    all_gaps = []
    max_gaps = np.zeros(bool_mat.shape[0])
    streak_counts = np.zeros(bool_mat.shape[0])
    
    for i in range(bool_mat.shape[0]):
        starts = np.where(diffs[i] == 1)[0]
        ends = np.where(diffs[i] == -1)[0]
        gaps = ends - starts
        if len(gaps) > 0:
            all_gaps.extend(gaps)
            max_gaps[i] = np.max(gaps)
            streak_counts[i] = len(gaps)
            
    return np.array(all_gaps), max_gaps, streak_counts

def main():
    create_dirs()
    input_file = r"D:\wattmaticdataset\data.csv"
    print(f"[INFO] Loading REAL SGCC data from {input_file}...")
    df = pd.read_csv(input_file)
    
    id_col = 'CONS_NO'
    target_col = 'FLAG'
    consump_cols_raw = [c for c in df.columns if c not in [id_col, target_col]]
    
    parsed_dates = pd.to_datetime(consump_cols_raw)
    df_dates = pd.DataFrame({'col_name': consump_cols_raw, 'date': parsed_dates})
    sorted_date_cols = df_dates.sort_values('date')['col_name'].tolist()
    
    # 1. Audit Missingness on Raw Data
    print("[INFO] Auditing missingness...")
    X_raw_df = df[sorted_date_cols]
    X_raw = X_raw_df.values
    missing_mask = np.isnan(X_raw)
    
    missing_count = missing_mask.sum(axis=1)
    missing_ratio = missing_count / X_raw.shape[1]
    
    # Identify empty customers
    is_empty = (missing_ratio == 1.0)
    num_empty = is_empty.sum()
    empty_flags = df.loc[is_empty, target_col].value_counts().to_dict()
    
    # Gap analysis
    all_gaps, max_gaps, streak_counts = analyze_gaps(missing_mask)
    
    if len(all_gaps) > 0:
        gap_median = np.median(all_gaps)
        gap_95 = np.percentile(all_gaps, 95)
        gap_max = np.max(all_gaps)
        gaps_gt_7 = np.sum(all_gaps > 7)
        gaps_gt_30 = np.sum(all_gaps > 30)
        gaps_gt_90 = np.sum(all_gaps > 90)
    else:
        gap_median = gap_95 = gap_max = gaps_gt_7 = gaps_gt_30 = gaps_gt_90 = 0

    # Save Missingness Audit
    audit_report = f"""# SGCC Missingness Strategy & Audit Report

## 1. Missingness Audit
- **100% Missing Customers**: {num_empty} (FLAG distribution: {empty_flags})
- **99-100% Missing**: {np.sum(missing_ratio >= 0.99)}
- **Partial Missingness (>0, <100%)**: {np.sum((missing_ratio > 0) & (missing_ratio < 1))}

## 2. Missing Gap Distribution
Across the dataset, missing values occur in contiguous blocks (gaps):
- **Total gaps**: {len(all_gaps)}
- **Median gap length**: {gap_median} days
- **95th Percentile gap length**: {gap_95} days
- **Maximum gap length**: {gap_max} days
- **Gaps > 7 days**: {gaps_gt_7}
- **Gaps > 30 days**: {gaps_gt_30}
- **Gaps > 90 days**: {gaps_gt_90}

## 3. Imputation Strategy Justification
Given {gaps_gt_30} communication outages exceeding 30 days, unrestricted `ffill()` artificially carries consumption values across months, destroying seasonality and volatility.
**Strategy**: `ffill(limit=7)`. We will only forward-fill missing values up to 7 days to smooth brief meter disconnections/errors. Gaps longer than 7 days will remain `NaN`. Statistical features will use `np.nanmean`, `np.nanstd` directly on observed data, ensuring missingness does not distort mathematical realities.

## 4. Exclusion Criteria
The {num_empty} completely empty customers will be explicitly dropped from the feature set. Because they lack any consumption history, they provide zero signal.

## 5. Next ML Steps
Remaining NaNs in the engineered feature dataset (caused by customers having mostly NaNs and unable to form trend/volatility metrics) should be handled via a Scikit-Learn `SimpleImputer` (e.g., median) inside a pipeline fitted STRICTLY on the training set to prevent leakage.
"""
    with open('reports/SGCC_Missingness_Strategy_Report.md', 'w') as f:
        f.write(audit_report)
    
    # 2. Extract Missingness Features BEFORE imputation
    features = pd.DataFrame(index=df.index)
    features['missing_count'] = missing_count
    features['missing_ratio'] = missing_ratio
    features['longest_missing_streak'] = max_gaps
    features['missing_streak_count'] = streak_counts
    features['max_missing_gap_days'] = max_gaps # same as longest streak for missingness
    
    # 3. Bounded Imputation
    limit = 7
    print(f"[INFO] Performing bounded imputation (ffill limit={limit})...")
    X_filled_df = X_raw_df.ffill(axis=1, limit=limit) # don't chain bfill to avoid reverse leakage
    X = X_filled_df.values
    
    # 4. Extract standard features using nan-aware functions
    print("[INFO] Extracting domain features...")
    # Central tendency
    features['mean'] = np.nanmean(X, axis=1)
    features['median'] = np.nanmedian(X, axis=1)
    
    # Variability
    features['std_dev'] = np.nanstd(X, axis=1)
    features['variance'] = np.nanvar(X, axis=1)
    features['cv'] = features['std_dev'] / (features['mean'] + 1e-6)
    
    # Extremes
    features['min'] = np.nanmin(X, axis=1)
    features['max'] = np.nanmax(X, axis=1)
    features['range'] = features['max'] - features['min']
    
    # Zero Behaviour
    is_zero = (X == 0)
    features['zero_count'] = np.sum(is_zero, axis=1)
    # Using observed days for ratio
    observed_days = (~np.isnan(X)).sum(axis=1)
    features['zero_ratio'] = np.where(observed_days > 0, features['zero_count'] / observed_days, np.nan)
    
    # Volatility
    diffs = np.diff(X, axis=1)
    abs_diffs = np.abs(diffs)
    features['mean_abs_daily_change'] = np.nanmean(abs_diffs, axis=1)
    features['max_abs_daily_change'] = np.nanmax(abs_diffs, axis=1)
    
    # Monthly Periodic
    monthly_means = X_filled_df.T.groupby(X_filled_df.columns.month).mean().T
    monthly_stds = X_filled_df.T.groupby(X_filled_df.columns.month).std().T
    for m in range(1, 13):
        features[f'monthly_mean_{m}'] = monthly_means[m]
        features[f'monthly_std_{m}'] = monthly_stds[m]
        
    features['peak_to_average_ratio'] = features['max'] / (features['mean'] + 1e-6)

    # 5. Exclude Empty Customers
    print("[INFO] Excluding empty customers...")
    features_clean = features[~is_empty].copy()
    valid_df = df[~is_empty].copy()
    
    # Assemble
    out_df = features_clean.copy()
    out_df.insert(0, id_col, valid_df[id_col])
    out_df.insert(1, target_col, valid_df[target_col])
    
    # Final checks
    nans = out_df.isna().sum().sum()
    print(f"Remaining NaNs in feature table: {nans}")
    
    out_path = 'data/processed/sgcc_features_real_v2.csv'
    out_df.to_csv(out_path, index=False)
    print(f"[SUCCESS] Saved to {out_path}")
    
if __name__ == '__main__':
    main()

