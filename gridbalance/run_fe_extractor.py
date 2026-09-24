import os
import pandas as pd
import numpy as np
from scipy import stats
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

class FeatureExtractor:
    def __init__(self, df, date_cols, id_col, target_col):
        self.df = df
        self.date_cols = date_cols
        self.id_col = id_col
        self.target_col = target_col
        self.features_df = None
        
    def extract_all(self):
        print("[INFO] Starting Feature Extraction...")
        X_df = self.df[self.date_cols].ffill(axis=1).bfill(axis=1)
        X = X_df.values
        
        self.nans_after_imputation = np.isnan(X).sum()
        
        features = pd.DataFrame(index=self.df.index)
        
        self._extract_central_tendency(features, X)
        self._extract_variability(features, X)
        self._extract_extremes(features, X)
        self._extract_zero_behaviour(features, X)
        self._extract_volatility(features, X)
        self._extract_trend(features, X)
        self._extract_periodic(features, X, X_df.columns)
        self._extract_peak_behaviour(features, X)
        
        self.features_df = features
        print(f"[SUCCESS] Extracted {self.features_df.shape[1]} features for {self.features_df.shape[0]} customers.")
        return self.features_df
    
    def _extract_central_tendency(self, features, X):
        features['mean'] = np.mean(X, axis=1)
        features['median'] = np.median(X, axis=1)
        features['trimmed_mean'] = stats.trim_mean(X, proportiontocut=0.1, axis=1)

    def _extract_variability(self, features, X):
        features['std_dev'] = np.std(X, axis=1)
        features['variance'] = np.var(X, axis=1)
        features['cv'] = features['std_dev'] / (features['mean'] + 1e-6)
        q75, q25 = np.percentile(X, [75 ,25], axis=1)
        features['iqr'] = q75 - q25
        features['mad'] = np.mean(np.abs(X - features['mean'].values[:, None]), axis=1)

    def _extract_extremes(self, features, X):
        features['min'] = np.min(X, axis=1)
        features['max'] = np.max(X, axis=1)
        features['range'] = features['max'] - features['min']
        features['percentile_5'] = np.percentile(X, 5, axis=1)
        features['percentile_95'] = np.percentile(X, 95, axis=1)

    def _extract_zero_behaviour(self, features, X):
        is_zero = (X == 0)
        features['zero_count'] = np.sum(is_zero, axis=1)
        features['zero_ratio'] = features['zero_count'] / X.shape[1]
        
        def get_zero_streaks(row):
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

    def _extract_volatility(self, features, X):
        diffs = np.diff(X, axis=1)
        abs_diffs = np.abs(diffs)
        features['mean_abs_daily_change'] = np.mean(abs_diffs, axis=1)
        features['std_abs_daily_change'] = np.std(diffs, axis=1)
        features['max_abs_daily_change'] = np.max(abs_diffs, axis=1)
        
        threshold = features['mean_abs_daily_change'].values[:, None] + 3 * np.std(abs_diffs, axis=1)[:, None]
        features['unusually_large_changes_count'] = np.sum(abs_diffs > threshold, axis=1)

    def _extract_trend(self, features, X):
        t = np.arange(X.shape[1])
        p = np.polyfit(t, X.T, 1)
        features['linear_trend_slope'] = p[0]
        
        y_pred = p[0][:, None] * t + p[1][:, None]
        ss_res = np.sum((X - y_pred)**2, axis=1)
        ss_tot = np.sum((X - features['mean'].values[:, None])**2, axis=1)
        features['trend_r2'] = np.where(ss_tot == 0, 0, 1 - ss_res / (ss_tot + 1e-6))
        
        first_30 = np.mean(X[:, :30], axis=1)
        last_30 = np.mean(X[:, -30:], axis=1)
        features['early_vs_late_change'] = last_30 - first_30
        features['relative_trend'] = (last_30 - first_30) / (first_30 + 1e-6)

    def _extract_periodic(self, features, X, cols):
        dates = pd.to_datetime(cols)
        df_X = pd.DataFrame(X, columns=dates)
        
        monthly_means = df_X.T.groupby(df_X.columns.month).mean().T
        monthly_stds = df_X.T.groupby(df_X.columns.month).std().T
        for m in range(1, 13):
            features[f'monthly_mean_{m}'] = monthly_means[m]
            features[f'monthly_std_{m}'] = monthly_stds[m]
            
        def get_season(month):
            if month in [12, 1, 2]: return 'Winter'
            if month in [3, 4, 5]: return 'Spring'
            if month in [6, 7, 8]: return 'Summer'
            return 'Fall'
            
        seasons = df_X.columns.month.map(get_season)
        seasonal_means = df_X.T.groupby(seasons).mean().T
        
        features['seasonal_variation'] = seasonal_means.max(axis=1) - seasonal_means.min(axis=1)
        features['temporal_consistency'] = monthly_means.std(axis=1)

    def _extract_peak_behaviour(self, features, X):
        features['peak_to_average_ratio'] = features['max'] / (features['mean'] + 1e-6)
        high_threshold = features['mean'] + 1.5 * features['std_dev']
        features['high_consumption_day_count'] = np.sum(X > high_threshold.values[:, None], axis=1)
        features['high_consumption_day_ratio'] = features['high_consumption_day_count'] / X.shape[1]
        features['peak_frequency'] = features['high_consumption_day_count']

