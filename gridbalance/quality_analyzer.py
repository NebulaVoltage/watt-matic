"""
SGCC Data Quality Analyzer (Part 2)
Analyzes missingness, zero consumption, negative/invalid values, outliers per feature & consumer.
Generates MissingValues.csv.
"""

import os
import pandas as pd
import numpy as np

class DataQualityAnalyzer:
    def __init__(self, loader):
        self.loader = loader
        self.df = loader.df
        self.consumption_df = loader.get_consumption_matrix()
        self.quality_summary = {}

    def analyze_quality(self):
        """Performs deep data quality checks across rows and columns."""
        # 1. Column-wise missing values
        col_missing = self.consumption_df.isna().sum()
        col_missing_pct = (col_missing / len(self.df)) * 100

        # 2. Row-wise missing values per customer
        row_missing = self.consumption_df.isna().sum(axis=1)
        row_missing_pct = (row_missing / len(self.loader.date_cols)) * 100

        # 3. Zero consumption analysis
        zero_counts = (self.consumption_df == 0).sum(axis=1)
        zero_pct = (zero_counts / len(self.loader.date_cols)) * 100

        # 4. Negative values (invalid readings)
        neg_counts = (self.consumption_df < 0).sum().sum()

        # 5. Outlier detection using IQR on overall non-null consumption
        vals = self.consumption_df.values.flatten()
        vals_clean = vals[~np.isnan(vals)]
        q25, q75 = np.percentile(vals_clean, [25, 75])
        iqr = q75 - q25
        upper_bound = q75 + 1.5 * iqr
        outlier_count = int((vals_clean > upper_bound).sum())
        outlier_pct = float((outlier_count / len(vals_clean)) * 100)

        # Build feature-wise missing values DataFrame
        missing_df = pd.DataFrame({
            "Feature_Date": self.loader.date_cols,
            "Missing_Count": col_missing.values,
            "Missing_Percentage": col_missing_pct.values
        })

        self.quality_summary = {
            "Total_Feature_Columns": len(self.loader.date_cols),
            "Max_Column_Missing_Pct": float(col_missing_pct.max()),
            "Mean_Column_Missing_Pct": float(col_missing_pct.mean()),
            "Customers_With_Zero_Missing": int((row_missing == 0).sum()),
            "Customers_With_High_Missing_GT_50pct": int((row_missing_pct > 50).sum()),
            "Negative_Value_Readings": int(neg_counts),
            "Total_Zero_Readings": int((self.consumption_df == 0).sum().sum()),
            "Zero_Reading_Percentage": float(((self.consumption_df == 0).sum().sum() / self.consumption_df.size) * 100),
            "Statistical_Outliers_Count": outlier_count,
            "Outlier_Percentage": outlier_pct,
            "IQR_Upper_Threshold": float(upper_bound)
        }

        return self.quality_summary, missing_df

    def print_quality_report(self):
        """Prints formatted data quality report."""
        if not self.quality_summary:
            self.analyze_quality()

        print("\n" + "="*70)
        print("                  PART 2 — DATA QUALITY REPORT                       ")
        print("="*70)
        for key, val in self.quality_summary.items():
            if isinstance(val, float):
                print(f"  • {key:<35}: {val:.2f}")
            elif isinstance(val, int):
                print(f"  • {key:<35}: {val:,}")
            else:
                print(f"  • {key:<35}: {val}")
        print("="*70 + "\n")

    def export_csv(self, missing_csv_path="MissingValues.csv"):
        """Exports MissingValues.csv."""
        _, missing_df = self.analyze_quality()
        missing_df.to_csv(missing_csv_path, index=False)
        print(f"[SUCCESS] Missing Values detailed report exported to '{missing_csv_path}'.")
        return missing_df

if __name__ == "__main__":
    from data_loader import SGCCDataLoader
    loader = SGCCDataLoader()
    quality = DataQualityAnalyzer(loader)
    quality.print_quality_report()
    quality.export_csv()
