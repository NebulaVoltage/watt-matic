import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import SGCCDataLoader
from run_fe_extractor import FeatureExtractor

def generate_visualizations(features_df, df, target_col):
    out_dir = "graphs/feature_engineering/"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Feature distributions (select 4 interesting ones)
    cols_to_plot = ['mean', 'zero_ratio', 'linear_trend_slope', 'cv']
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for i, col in enumerate(cols_to_plot):
        ax = axes[i//2, i%2]
        sns.histplot(features_df[col], kde=True, ax=ax)
        ax.set_title(f'Distribution of {col}')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "feature_distributions.png"))
    plt.close()
    
    # 2. Correlation Heatmap
    corr = features_df.corr()
    plt.figure(figsize=(16, 12))
    sns.heatmap(corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "correlation_heatmap.png"))
    plt.close()
    
    # 3. Class-wise feature distributions
    features_with_target = features_df.copy()
    features_with_target['FLAG'] = df[target_col].values
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for i, col in enumerate(cols_to_plot):
        ax = axes[i//2, i%2]
        sns.boxplot(x='FLAG', y=col, data=features_with_target, ax=ax)
        ax.set_title(f'{col} by Target Class')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "class_wise_distributions.png"))
    plt.close()
    
def generate_report(features_df, nans_after_imputation, corr_matrix, df, date_cols):
    # Detect constant/near-zero variance features
    variances = features_df.var()
    constant_features = variances[variances == 0].index.tolist()
    near_zero_features = variances[(variances > 0) & (variances < 1e-4)].index.tolist()
    
    # Detect highly correlated pairs
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    highly_correlated = [column for column in upper.columns if any(upper[column].abs() > 0.95)]
    
    report_content = f"""# SGCC Feature Engineering Report

## 1. Objective
To construct a research-grade feature engineering pipeline for the SGCC electricity-theft detection module, extracting robust statistical, temporal, and behavioural features from the raw daily consumption data.

## 2. Raw Dataset Structure
- Total Rows: {df.shape[0]}
- Total Columns: {df.shape[1]}
- Number of Raw Daily Features: {len(date_cols)}

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
- Number of Engineered Features: {features_df.shape[1]}
- Missing Values (NaN) after imputation: {nans_after_imputation}
- Constant Features (variance = 0): {len(constant_features)} {constant_features if constant_features else ''}
- Near-Zero Variance Features: {len(near_zero_features)} {near_zero_features if near_zero_features else ''}
- Highly Correlated Features (r > 0.95): {len(highly_correlated)} {highly_correlated if highly_correlated else ''}
(Note: Highly correlated features have not been automatically deleted to preserve the correlation structure for research purposes).

## 5. Important Observations & Limitations
- **Data Sparsity Strategy**: Missing values were imputed per-customer using forward-fill followed by back-fill to maintain temporal consistency without leaking data across customers. Target `FLAG` was never used during extraction.
- **Correlations**: Features derived from identical domains (like high_consumption_day_count and peak_frequency) are expectedly highly correlated. We retain them for ML algorithm selection later.
- **Limitations**: Monthly metrics assume calendar alignment. Zero behaviour metrics assume true '0' is captured without sensor noise.
"""
    os.makedirs("reports", exist_ok=True)
    with open("reports/SGCC_Feature_Engineering_Report.md", "w") as f:
        f.write(report_content)
    
def main():
    loader = SGCCDataLoader()
    df = loader.df
    
    extractor = FeatureExtractor(df, loader.date_cols, loader.id_col, loader.target_col)
    features_df = extractor.extract_all()
    
    print("[INFO] Generating Visualizations...")
    generate_visualizations(features_df, df, loader.target_col)
    
    print("[INFO] Generating Report...")
    corr_matrix = features_df.corr()
    generate_report(features_df, extractor.nans_after_imputation, corr_matrix, df, loader.date_cols)
    
    print("[INFO] Saving ML-Ready Feature Matrix...")
    out_df = features_df.copy()
    out_df.insert(0, loader.id_col, df[loader.id_col])
    out_df.insert(1, loader.target_col, df[loader.target_col])
    
    out_path = "data/processed/sgcc_features.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"[SUCCESS] ML-Ready features saved to {out_path}")

if __name__ == "__main__":
    main()

