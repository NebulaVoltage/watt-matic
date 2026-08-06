"""
SGCC Machine Learning Insights Engine (Part 5)
Automated analysis of class imbalance, sparsity, redundancy, missing values,
and comprehensive recommendations for Classification, Regression, Clustering, & Anomaly Detection.
"""

import os
import pandas as pd
import numpy as np

class MLInsightsEngine:
    def __init__(self, loader):
        self.loader = loader
        self.df = loader.df
        self.consumption_df = loader.get_consumption_matrix()
        self.insights = {}

    def analyze_insights(self):
        """Generates automated machine learning diagnostic insights."""
        n_total = len(self.df)
        n_theft = int((self.df[self.loader.target_col] == 1).sum())
        n_normal = n_total - n_theft
        imbalance_ratio = float(n_normal / max(1, n_theft))
        theft_pct = (n_theft / n_total) * 100

        # Sparsity (% zero readings)
        total_cells = self.consumption_df.size
        zero_cells = int((self.consumption_df == 0).sum().sum())
        sparsity_pct = (zero_cells / total_cells) * 100

        # Missing values
        missing_cells = int(self.consumption_df.isna().sum().sum())
        missing_pct = (missing_cells / total_cells) * 100

        # Redundancy & Correlation sample check
        # Take 20 random dates to assess inter-day correlation
        cols_sample = self.loader.date_cols[::max(1, len(self.loader.date_cols) // 20)]
        corr_matrix = self.consumption_df[cols_sample].corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_count = int((upper_tri > 0.85).sum().sum())

        self.insights = {
            "Class Imbalance": f"Severe Class Imbalance (~1:{imbalance_ratio:.1f}). Theft instances represent only {theft_pct:.2f}% of customers.",
            "Data Sparsity": f"{sparsity_pct:.2f}% of all reading matrix entries are exact 0 kWh values (meter zeroing/disconnections).",
            "Missing Value Challenges": f"{missing_pct:.2f}% missing entries overall. Requires imputation (forward-fill, linear interpolation, or XGBoost native missing handling).",
            "Feature Redundancy": f"High daily feature redundancy ({high_corr_count} date pairs with r > 0.85). Raw 1,034-day columns have temporal autocorrelation.",
            "Feature Engineering Opportunities": "Time-series aggregation (rolling means, volatility, trend slopes, zero-streak counts, weekend ratios) can compress 1,034 columns into ~30 high-impact domain features.",
        }

        return self.insights

    def get_algorithm_recommendations(self):
        """Returns detailed recommendations for ML tasks with justification."""
        return {
            "Classification": [
                {
                    "Algorithm": "XGBoost / LightGBM / CatBoost",
                    "Type": "Gradient Boosted Decision Trees (GBDT)",
                    "Why": "State-of-the-art performance on tabular & engineered features; handles missing values natively; robust against non-linear interactions; supports scale_pos_weight for class imbalance."
                },
                {
                    "Algorithm": "1D-CNN + LSTM / GRU",
                    "Type": "Deep Temporal Neural Networks",
                    "Why": "Extracts spatial feature motifs via 1D convolutions and captures long-term sequential memory across 1,034 days of raw daily consumption sequences."
                },
                {
                    "Algorithm": "Random Forest Classifier",
                    "Type": "Ensemble Trees",
                    "Why": "Provides strong benchmark accuracy, feature importance rankings, and resistance to overfitting."
                }
            ],
            "Regression": [
                {
                    "Algorithm": "LightGBM Regressor / Prophet / SARIMAX",
                    "Type": "Time-Series Forecasting Models",
                    "Why": "Used for expected baseline load estimation. Deviations between predicted baseline and actual metered load indicate potential theft or meter tampering."
                }
            ],
            "Clustering": [
                {
                    "Algorithm": "K-Means / DBSCAN / Hierarchical Clustering",
                    "Type": "Unsupervised Profile Segmentation",
                    "Why": "Groups customers by consumption behavior (residential, commercial, seasonal). Enables segment-specific anomaly thresholds and cluster profile validation."
                }
            ],
            "Anomaly Detection": [
                {
                    "Algorithm": "Isolation Forest / One-Class SVM / Autoencoders",
                    "Type": "Unsupervised Anomaly Detection",
                    "Why": "Effective when theft labels are sparse or unverified. Autoencoders reconstruct normal load; high reconstruction error signals unmetered energy loss."
                }
            ]
        }

    def print_insights_report(self):
        """Prints formatted ML insights report."""
        if not self.insights:
            self.analyze_insights()

        print("\n" + "="*70)
        print("                PART 5 — MACHINE LEARNING INSIGHTS                   ")
        print("="*70)
        for key, val in self.insights.items():
            print(f"  • {key:<35}: {val}")
        print("="*70 + "\n")

if __name__ == "__main__":
    from data_loader import SGCCDataLoader
    loader = SGCCDataLoader()
    engine = MLInsightsEngine(loader)
    engine.print_insights_report()
