"""
SGCC Feature Engineering Suite (Part 6)
Suggests, computes, and documents 12 categories of domain-specific engineered features.
Generates FeatureImportanceIdeas.csv.
"""

import os
import pandas as pd
import numpy as np

class FeatureEngineeringSuite:
    def __init__(self, loader):
        self.loader = loader
        self.df = loader.df
        self.date_cols = loader.date_cols

    def get_feature_ideas_dataframe(self):
        """Returns structured DataFrame explaining each engineered feature category, formula, and domain utility."""
        feature_ideas = [
            {
                "Feature_Category": "Rolling Mean (7-day & 30-day)",
                "Engineered_Features": "rolling_mean_7d, rolling_mean_30d",
                "Mathematical_Formula": "Mean of consumption over moving window of 7 and 30 days",
                "Why_It_Helps": "Smooths daily high-frequency noise to establish short-term and medium-term baseline consumption levels."
            },
            {
                "Feature_Category": "Rolling Std Dev (7-day & 30-day)",
                "Engineered_Features": "rolling_std_7d, rolling_std_30d",
                "Mathematical_Formula": "Standard deviation over moving window of 7 and 30 days",
                "Why_It_Helps": "Measures short-term load volatility. Theft customers often exhibit artificially suppressed or zero volatility during tampering periods."
            },
            {
                "Feature_Category": "Daily Difference / Change Rate",
                "Engineered_Features": "daily_diff_mean, daily_diff_max",
                "Mathematical_Formula": "C(t) - C(t-1) for daily time series",
                "Why_It_Helps": "Detects abrupt sudden drops or spikes caused by physical meter bypass installation or removal."
            },
            {
                "Feature_Category": "Consumption Variance & Volatility",
                "Engineered_Features": "overall_variance, overall_std",
                "Mathematical_Formula": "Variance of all non-missing daily readings per customer",
                "Why_It_Helps": "Captures overall customer load predictability; normal consumers exhibit natural human routine variances whereas bypass theft lowers variance."
            },
            {
                "Feature_Category": "Weekend vs Weekday Ratio",
                "Engineered_Features": "weekend_mean, weekday_mean, weekend_weekday_ratio",
                "Mathematical_Formula": "Mean(Weekend kWh) / Mean(Weekday kWh)",
                "Why_It_Helps": "Captures lifestyle/occupancy behavior. Discrepancies between weekend vs weekday ratios highlight abnormal non-residential or tampered usage."
            },
            {
                "Feature_Category": "Monthly Average & Seasonality",
                "Engineered_Features": "monthly_mean_Jan..Dec, summer_winter_ratio",
                "Mathematical_Formula": "Mean(Summer kWh) / Mean(Winter kWh)",
                "Why_It_Helps": "Electric heating/HVAC drives natural seasonal surges. Theft customers fail to show expected summer/winter peak variations."
            },
            {
                "Feature_Category": "Consumption Growth Rate",
                "Engineered_Features": "yearly_growth_rate, H2_H1_ratio",
                "Mathematical_Formula": "(Mean_Year3 - Mean_Year1) / Mean_Year1",
                "Why_It_Helps": "Identifies progressive long-term reduction in billed consumption as theft activity intensifies."
            },
            {
                "Feature_Category": "Peak Consumption & Peak-to-Average",
                "Engineered_Features": "peak_consumption, peak_to_mean_ratio",
                "Mathematical_Formula": "Max(kWh) / Mean(kWh)",
                "Why_It_Helps": "Identifies intermittent partial theft where peaks occur briefly but average consumption is severely depressed."
            },
            {
                "Feature_Category": "Consumption Ratios & Zero Count",
                "Engineered_Features": "zero_count, zero_ratio, min_to_mean_ratio",
                "Mathematical_Formula": "Count(C(t) == 0) / Total_Days",
                "Why_It_Helps": "Directly quantifies complete meter zeroing or shunting. High zero ratios are strong indicators of electricity theft."
            },
            {
                "Feature_Category": "Trend Features (Linear Slope)",
                "Engineered_Features": "trend_slope, trend_intercept",
                "Mathematical_Formula": "Linear regression slope of C(t) against time index t",
                "Why_It_Helps": "Quantifies downward structural trend trajectory in customer usage over time."
            },
            {
                "Feature_Category": "Customer Statistics Summary",
                "Engineered_Features": "mean_kWh, median_kWh, iqr_kWh, skewness, kurtosis",
                "Mathematical_Formula": "Standard distributional moments across temporal sequence",
                "Why_It_Helps": "Summarizes the overall shape of the customer load distribution into fixed-size ML features."
            },
            {
                "Feature_Category": "Data Availability / Missing Ratio",
                "Engineered_Features": "missing_count, missing_ratio",
                "Mathematical_Formula": "Count(NaN) / Total_Days",
                "Why_It_Helps": "Meter communication tampering or intentional physical disconnects often manifest as high missing reading rates."
            }
        ]
        return pd.DataFrame(feature_ideas)

    def export_csv(self, output_path="FeatureImportanceIdeas.csv"):
        """Exports FeatureImportanceIdeas.csv."""
        df_ideas = self.get_feature_ideas_dataframe()
        df_ideas.to_csv(output_path, index=False)
        print(f"[SUCCESS] Feature Engineering ideas & domain logic exported to '{output_path}'.")
        return df_ideas

if __name__ == "__main__":
    from data_loader import SGCCDataLoader
    loader = SGCCDataLoader()
    fe = FeatureEngineeringSuite(loader)
    fe.export_csv()
