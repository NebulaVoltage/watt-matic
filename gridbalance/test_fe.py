import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def extract_features(df, date_cols):
    X_raw = df[date_cols].values
    
    # 1. Handle NaNs for feature extraction
    # Using pandas for ffill and bfill along axis=1
    X_df = df[date_cols].ffill(axis=1).bfill(axis=1)
    X = X_df.values
    
    # Track NaNs after imputation
    nans_after = np.isnan(X).sum()
    print(f"NaNs after imputation: {nans_after}")
    
    features = pd.DataFrame(index=df.index)
    
    # A. CENTRAL TENDENCY
    features['mean'] = np.mean(X, axis=1)
    features['median'] = np.median(X, axis=1)
    features['trimmed_mean'] = stats.trim_mean(X, proportiontocut=0.1, axis=1)
    
    # B. VARIABILITY
    features['std_dev'] = np.std(X, axis=1)
    features['variance'] = np.var(X, axis=1)
    features['cv'] = features['std_dev'] / (features['mean'] + 1e-6)
    q75, q25 = np.percentile(X, [75 ,25], axis=1)
    features['iqr'] = q75 - q25
    features['mad'] = np.mean(np.abs(X - features['mean'].values[:, None]), axis=1)
    
    # C. EXTREMES
    features['min'] = np.min(X, axis=1)
    features['max'] = np.max(X, axis=1)
    features['range'] = features['max'] - features['min']
    features['percentile_5'] = np.percentile(X, 5, axis=1)
    features['percentile_95'] = np.percentile(X, 95, axis=1)
    
    # D. ZERO BEHAVIOUR
    is_zero = (X == 0)
    features['zero_count'] = np.sum(is_zero, axis=1)
    features['zero_ratio'] = features['zero_count'] / X.shape[1]
    
    # Longest zero streak & zero streak count
    def get_zero_streaks(row):
        # padd with False at both ends to detect edges
        padded = np.pad(row, (1, 1), mode='constant', constant_values=False)
        edges = np.diff(padded.astype(int))
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0]
        streaks = ends - starts
        if len(streaks) == 0:
            return 0, 0
        return np.max(streaks), len(streaks)
    
    streak_info = np.array([get_zero_streaks(r) for r in is_zero])
    features['longest_zero_streak'] = streak_info[:, 0]
    features['zero_streak_count'] = streak_info[:, 1]
    
    # E. VOLATILITY
    diffs = np.diff(X, axis=1)
    abs_diffs = np.abs(diffs)
    features['mean_abs_daily_change'] = np.mean(abs_diffs, axis=1)
    features['std_abs_daily_change'] = np.std(diffs, axis=1) # std of daily changes
    features['max_abs_daily_change'] = np.max(abs_diffs, axis=1)
    
    # Unusually large changes (e.g. > mean + 3*std of absolute changes globally or per customer)
    # Let's do per customer:
    threshold = features['mean_abs_daily_change'].values[:, None] + 3 * np.std(abs_diffs, axis=1)[:, None]
    features['unusually_large_changes_count'] = np.sum(abs_diffs > threshold, axis=1)
    
    # F. TREND
    t = np.arange(X.shape[1])
    # polyfit vectorization using np.polyfit or lstsq. np.polyfit is fast enough for 1500 rows if we transpose
    # fit line y = mx + c
    # x: t, y: X.T
    p = np.polyfit(t, X.T, 1)
    features['linear_trend_slope'] = p[0]
    
    # R2
    y_pred = p[0][:, None] * t + p[1][:, None]
    ss_res = np.sum((X - y_pred)**2, axis=1)
    ss_tot = np.sum((X - features['mean'].values[:, None])**2, axis=1)
    features['trend_r2'] = np.where(ss_tot == 0, 0, 1 - ss_res / (ss_tot + 1e-6))
    
    # early vs late
    first_30 = np.mean(X[:, :30], axis=1)
    last_30 = np.mean(X[:, -30:], axis=1)
    features['early_vs_late_change'] = last_30 - first_30
    features['relative_trend'] = (last_30 - first_30) / (first_30 + 1e-6)
    
    # G. PERIODIC/TEMPORAL FEATURES
    dates = pd.to_datetime(date_cols)
    df_X = pd.DataFrame(X, columns=dates)
    
    # Monthly means
    monthly_means = df_X.T.groupby(df_X.columns.month).mean().T
    monthly_stds = df_X.T.groupby(df_X.columns.month).std().T
    for m in range(1, 13):
        features[f'monthly_mean_{m}'] = monthly_means[m]
        features[f'monthly_std_{m}'] = monthly_stds[m]
    
    # Seasons: Winter(12,1,2), Spring(3,4,5), Summer(6,7,8), Fall(9,10,11)
    def get_season(month):
        if month in [12, 1, 2]: return 'Winter'
        if month in [3, 4, 5]: return 'Spring'
        if month in [6, 7, 8]: return 'Summer'
        return 'Fall'
    
    seasons = df_X.columns.month.map(get_season)
    seasonal_means = df_X.T.groupby(seasons).mean().T
    
    # seasonal variation (max season mean - min season mean)
    features['seasonal_variation'] = seasonal_means.max(axis=1) - seasonal_means.min(axis=1)
    features['temporal_consistency'] = monthly_means.std(axis=1)
    
    # H. PEAK BEHAVIOUR
    features['peak_to_average_ratio'] = features['max'] / (features['mean'] + 1e-6)
    high_threshold = features['mean'] + 1.5 * features['std_dev']
    features['high_consumption_day_count'] = np.sum(X > high_threshold.values[:, None], axis=1)
    features['high_consumption_day_ratio'] = features['high_consumption_day_count'] / X.shape[1]
    features['peak_frequency'] = features['high_consumption_day_count'] # same as above, or similar
    
    return features

if __name__ == '__main__':
    print("Loading data...")
    df = pd.read_csv('data.csv', nrows=50) # test on 50 rows
    date_cols = df.columns[2:]
    print("Extracting features...")
    features = extract_features(df, date_cols)
    print(features.head())
    print(features.shape)

