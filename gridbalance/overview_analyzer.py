"""
SGCC Dataset Overview Analyzer (Part 1)
Generates dataset shape, memory, metadata, missing counts, duplicate analysis,
prints clean report, and exports Dataset_Summary.csv.
"""

import os
import pandas as pd
import numpy as np

class OverviewAnalyzer:
    def __init__(self, loader):
        self.loader = loader
        self.df = loader.df
        self.summary_dict = {}

    def analyze(self):
        """Computes comprehensive dataset overview metrics."""
        shape = self.df.shape
        size = self.df.size
        num_features = shape[1]
        num_numerical = self.df.select_dtypes(include=[np.number]).shape[1]
        num_categorical = num_features - num_numerical
        
        total_missing = int(self.df.isna().sum().sum())
        missing_pct = (total_missing / size) * 100
        
        duplicate_rows = int(self.df.duplicated().sum())
        duplicate_ids = int(self.df[self.loader.id_col].duplicated().sum())
        
        mem_bytes = self.df.memory_usage(deep=True).sum()
        mem_mb = mem_bytes / (1024 ** 2)
        mem_gb = mem_bytes / (1024 ** 3)
        
        dtypes_summary = self.df.dtypes.value_counts().to_dict()
        dtypes_str = ", ".join([f"{k}: {v}" for k, v in dtypes_summary.items()])

        self.summary_dict = {
            "Dataset Shape (Rows, Cols)": f"{shape[0]:,} rows x {shape[1]:,} cols",
            "Total Cells (Dataset Size)": f"{size:,}",
            "Total Features": shape[1],
            "Numerical Features": num_numerical,
            "Categorical/ID Features": num_categorical,
            "Total Missing Values": f"{total_missing:,}",
            "Overall Missing Percentage": f"{missing_pct:.2f}%",
            "Duplicate Rows": duplicate_rows,
            "Duplicate Customer IDs": duplicate_ids,
            "Memory Usage (MB)": f"{mem_mb:.2f} MB",
            "Memory Usage (GB)": f"{mem_gb:.4f} GB",
            "Data Types Breakdown": dtypes_str
        }
        return self.summary_dict

    def print_report(self):
        """Prints a beautifully formatted ASCII text report."""
        if not self.summary_dict:
            self.analyze()
            
        print("\n" + "="*70)
        print("                   PART 1 — DATASET OVERVIEW REPORT                   ")
        print("="*70)
        for key, val in self.summary_dict.items():
            print(f"  • {key:<35}: {val}")
        print("="*70 + "\n")

    def export_csv(self, output_path="Dataset_Summary.csv"):
        """Exports dataset overview to CSV."""
        if not self.summary_dict:
            self.analyze()
        
        summary_df = pd.DataFrame(list(self.summary_dict.items()), columns=["Metric", "Value"])
        summary_df.to_csv(output_path, index=False)
        print(f"[SUCCESS] Dataset Summary exported to '{output_path}'.")
        return summary_df

if __name__ == "__main__":
    from data_loader import SGCCDataLoader
    loader = SGCCDataLoader()
    analyzer = OverviewAnalyzer(loader)
    analyzer.print_report()
    analyzer.export_csv()
