"""
SGCC Dataset Statistics Calculator (Part 4)
Calculates Min, Max, Mean, Median, Variance, Std, Skewness, Kurtosis, IQR,
and Percentiles (25, 50, 75, 90, 95, 99) for dataset and exports Statistics.csv.
"""

import os
import pandas as pd
import numpy as np
from scipy import stats

class StatisticsCalculator:
    def __init__(self, loader):
        self.loader = loader
        self.consumption_df = loader.get_consumption_matrix()
        self.stats_df = None

    def compute_statistics(self):
        """Computes feature-level and global statistics across daily consumption features."""
        print("[INFO] Computing high-dimensional statistical metrics across all daily features...")
        
        # Column-wise statistical metrics
        mins = self.consumption_df.min(axis=0)
        maxs = self.consumption_df.max(axis=0)
        means = self.consumption_df.mean(axis=0)
        medians = self.consumption_df.median(axis=0)
        variances = self.consumption_df.var(axis=0)
        stds = self.consumption_df.std(axis=0)
        skewness = self.consumption_df.skew(axis=0)
        kurtosis = self.consumption_df.kurtosis(axis=0)
        
        p25 = self.consumption_df.quantile(0.25, axis=0)
        p50 = self.consumption_df.quantile(0.50, axis=0)
        p75 = self.consumption_df.quantile(0.75, axis=0)
        p90 = self.consumption_df.quantile(0.90, axis=0)
        p95 = self.consumption_df.quantile(0.95, axis=0)
        p99 = self.consumption_df.quantile(0.99, axis=0)
        iqr = p75 - p25

        self.stats_df = pd.DataFrame({
            "Feature_Date": self.loader.date_cols,
            "Minimum": mins.values,
            "Maximum": maxs.values,
            "Mean": means.values,
            "Median": medians.values,
            "Variance": variances.values,
            "Standard_Deviation": stds.values,
            "Skewness": skewness.values,
            "Kurtosis": kurtosis.values,
            "IQR": iqr.values,
            "Percentile_25": p25.values,
            "Percentile_50": p50.values,
            "Percentile_75": p75.values,
            "Percentile_90": p90.values,
            "Percentile_95": p95.values,
            "Percentile_99": p99.values,
        })
        return self.stats_df

    def get_global_summary_table(self):
        """Calculates global dataset-level aggregated statistics."""
        if self.stats_df is None:
            self.compute_statistics()
            
        vals = self.consumption_df.values.flatten()
        vals_clean = vals[~np.isnan(vals)]
        
        g_min = float(np.min(vals_clean))
        g_max = float(np.max(vals_clean))
        g_mean = float(np.mean(vals_clean))
        g_median = float(np.median(vals_clean))
        g_var = float(np.var(vals_clean))
        g_std = float(np.std(vals_clean))
        g_skew = float(stats.skew(vals_clean))
        g_kurt = float(stats.kurtosis(vals_clean))
        
        g_p25, g_p50, g_p75, g_p90, g_p95, g_p99 = np.percentile(vals_clean, [25, 50, 75, 90, 95, 99])
        g_iqr = g_p75 - g_p25

        global_metrics = {
            "Minimum": g_min,
            "Maximum": g_max,
            "Mean": g_mean,
            "Median": g_median,
            "Variance": g_var,
            "Standard Deviation": g_std,
            "Skewness": g_skew,
            "Kurtosis": g_kurt,
            "Interquartile Range (IQR)": g_iqr,
            "Percentile (25th)": g_p25,
            "Percentile (50th / Median)": g_p50,
            "Percentile (75th)": g_p75,
            "Percentile (90th)": g_p90,
            "Percentile (95th)": g_p95,
            "Percentile (99th)": g_p99,
        }
        return pd.DataFrame(list(global_metrics.items()), columns=["Metric", "Global_Dataset_Value"])

    def print_table(self):
        """Prints a clean tabular summary of global statistics."""
        summary = self.get_global_summary_table()
        print("\n" + "="*70)
        print("                 PART 4 — DATASET GLOBAL STATISTICS                  ")
        print("="*70)
        for _, row in summary.iterrows():
            print(f"  • {row['Metric']:<30}: {row['Global_Dataset_Value']:12.4f}")
        print("="*70 + "\n")

    def export_csv(self, output_path="Statistics.csv"):
        """Exports full feature-level Statistics.csv."""
        if self.stats_df is None:
            self.compute_statistics()
        self.stats_df.to_csv(output_path, index=False)
        print(f"[SUCCESS] High-dimensional statistics exported to '{output_path}'.")
        return self.stats_df

if __name__ == "__main__":
    from data_loader import SGCCDataLoader
    loader = SGCCDataLoader()
    calc = StatisticsCalculator(loader)
    calc.print_table()
    calc.export_csv()
