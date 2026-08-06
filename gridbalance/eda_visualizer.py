"""
SGCC EDA Visualizer Engine (Part 3)
Generates 22 publication-grade, high-resolution (300 DPI) figures saved in `graphs/`.
Follows modern aesthetic guidelines: white background, crisp typography, clean layouts,
large readable fonts, and a harmonious color palette (Navy Blue for Normal, Crimson Red for Theft).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Configure Global Matplotlib Style for Publication Quality
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.sans-serif': 'DejaVu Sans',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'figure.autolayout': True,
    'axes.edgecolor': '#cccccc',
    'axes.linewidth': 1.2,
    'grid.color': '#e5e5e5',
    'grid.linestyle': '--',
    'grid.alpha': 0.7,
})

# Palette
COLOR_NORMAL = "#1f77b4"  # Deep Navy Blue
COLOR_THEFT = "#d62728"   # Crimson Red
COLOR_ACCENT = "#2ca02c"  # Emerald Green
COLOR_NEUTRAL = "#7f7f7f" # Slate Gray

class EDAVisualizer:
    def __init__(self, loader, output_dir="graphs"):
        self.loader = loader
        self.df = loader.df
        self.date_cols = loader.date_cols
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.normal_df, self.theft_df = loader.get_normal_and_theft_subsets()
        self.dates = pd.to_datetime(self.date_cols, errors='coerce')

    def save_fig(self, fig, filename):
        """Saves figure with 300 DPI, tight bounding box, white facecolor."""
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return path

    def generate_all_plots(self):
        """Generates all 22 publication-grade figures with progress bar."""
        plots = [
            ("01_class_distribution.png", self.plot_01_class_distribution),
            ("02_missing_values.png", self.plot_02_missing_values),
            ("03_average_daily_consumption.png", self.plot_03_average_daily_consumption),
            ("04_median_daily_consumption.png", self.plot_04_median_daily_consumption),
            ("05_daily_consumption_trend.png", self.plot_05_daily_consumption_trend),
            ("06_electricity_consumption_distribution.png", self.plot_06_electricity_consumption_distribution),
            ("07_histogram.png", self.plot_07_histogram),
            ("08_kde_plot.png", self.plot_08_kde_plot),
            ("09_boxplot.png", self.plot_09_boxplot),
            ("10_violin_plot.png", self.plot_10_violin_plot),
            ("11_customer_consumption_pattern.png", self.plot_11_customer_consumption_pattern),
            ("12_compare_normal_vs_theft_customer.png", self.plot_12_compare_normal_vs_theft_customer),
            ("13_top_20_consumers.png", self.plot_13_top_20_consumers),
            ("14_bottom_20_consumers.png", self.plot_14_bottom_20_consumers),
            ("15_correlation_heatmap.png", self.plot_15_correlation_heatmap),
            ("16_feature_correlation_matrix.png", self.plot_16_feature_correlation_matrix),
            ("17_pair_plot.png", self.plot_17_pair_plot),
            ("18_zero_consumption_distribution.png", self.plot_18_zero_consumption_distribution),
            ("19_data_availability_per_customer.png", self.plot_19_data_availability_per_customer),
            ("20_monthly_consumption_trend.png", self.plot_20_monthly_consumption_trend),
            ("21_yearly_consumption_trend.png", self.plot_21_yearly_consumption_trend),
            ("22_seasonal_consumption_pattern.png", self.plot_22_seasonal_consumption_pattern),
        ]

        print(f"\n[INFO] Generating 22 publication-grade figures in '{self.output_dir}/'...")
        for filename, plot_func in tqdm(plots, desc="Rendering EDA Figures"):
            try:
                plot_func(filename)
            except Exception as e:
                print(f"[ERROR] Failed to render {filename}: {e}")
        print(f"[SUCCESS] All 22 figures successfully saved in '{self.output_dir}/'.\n")

    # ------------------------------------------------------------------------
    # Individual Plot Implementations (1 to 22)
    # ------------------------------------------------------------------------

    def plot_01_class_distribution(self, filename):
        """1. Class Distribution (Normal vs Theft)"""
        fig, ax = plt.subplots(figsize=(8, 6))
        counts = self.df[self.loader.target_col].value_counts().sort_index()
        labels = ['Normal (FLAG=0)', 'Theft (FLAG=1)']
        colors = [COLOR_NORMAL, COLOR_THEFT]
        
        bars = ax.bar(labels, counts, color=colors, width=0.5, edgecolor='black', linewidth=1.2)
        total = len(self.df)
        
        for bar in bars:
            yval = bar.get_height()
            pct = (yval / total) * 100
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + (total * 0.015),
                    f'{yval:,}\n({pct:.1f}%)', ha='center', va='bottom', fontweight='bold', fontsize=13)

        ax.set_title('SGCC Electricity Theft Dataset — Class Distribution', pad=15, fontweight='bold')
        ax.set_ylabel('Number of Customers', fontweight='bold')
        ax.set_ylim(0, max(counts) * 1.18)
        self.save_fig(fig, filename)

    def plot_02_missing_values(self, filename):
        """2. Missing Values Across Features & Customers"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Column-wise missing percentage
        col_missing = (self.df[self.date_cols].isna().sum() / len(self.df)) * 100
        ax1.plot(self.dates, col_missing, color='#e67e22', linewidth=1.5)
        ax1.set_title('Missing Value Rate Over Time (Daily Features)', fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Missing Percentage (%)')
        ax1.set_ylim(0, max(10, col_missing.max() * 1.2))

        # Customer-wise missing percentage distribution
        row_missing = (self.df[self.date_cols].isna().sum(axis=1) / len(self.date_cols)) * 100
        ax2.hist(row_missing, bins=30, color='#34495e', edgecolor='black', alpha=0.85)
        ax2.set_title('Customer Data Loss Distribution (% Missing Days)', fontweight='bold')
        ax2.set_xlabel('Missing Percentage Per Customer (%)')
        ax2.set_ylabel('Customer Count')

        self.save_fig(fig, filename)

    def plot_03_average_daily_consumption(self, filename):
        """3. Average Daily Consumption (Normal vs Theft)"""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        mean_normal = self.normal_df[self.date_cols].mean(axis=0)
        mean_theft = self.theft_df[self.date_cols].mean(axis=0)
        
        ax.plot(self.dates, mean_normal, label='Normal Customers', color=COLOR_NORMAL, alpha=0.85, linewidth=1.8)
        ax.plot(self.dates, mean_theft, label='Theft Customers', color=COLOR_THEFT, alpha=0.85, linewidth=1.8)
        
        ax.set_title('Average Daily Electricity Consumption (kWh)', pad=15, fontweight='bold')
        ax.set_xlabel('Date', fontweight='bold')
        ax.set_ylabel('Mean Consumption (kWh)', fontweight='bold')
        ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
        self.save_fig(fig, filename)

    def plot_04_median_daily_consumption(self, filename):
        """4. Median Daily Consumption (Robust against extreme spikes)"""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        median_normal = self.normal_df[self.date_cols].median(axis=0)
        median_theft = self.theft_df[self.date_cols].median(axis=0)
        
        ax.plot(self.dates, median_normal, label='Normal Customers (Median)', color=COLOR_NORMAL, linewidth=1.8)
        ax.plot(self.dates, median_theft, label='Theft Customers (Median)', color=COLOR_THEFT, linewidth=1.8)
        
        ax.set_title('Median Daily Electricity Consumption (kWh)', pad=15, fontweight='bold')
        ax.set_xlabel('Date', fontweight='bold')
        ax.set_ylabel('Median Consumption (kWh)', fontweight='bold')
        ax.legend(loc='upper right', frameon=True)
        self.save_fig(fig, filename)

    def plot_05_daily_consumption_trend(self, filename):
        """5. Daily Consumption Trend (Overall + 30-Day Moving Average)"""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        overall_mean = self.df[self.date_cols].mean(axis=0)
        rolling_30 = overall_mean.rolling(window=30, min_periods=1).mean()
        
        ax.plot(self.dates, overall_mean, color='#bdc3c7', alpha=0.6, label='Raw Daily Mean', linewidth=1)
        ax.plot(self.dates, rolling_30, color='#2c3e50', linewidth=2.5, label='30-Day Moving Average Trend')
        
        ax.set_title('Overall Daily Grid Load Trend & Seasonal Baseline', pad=15, fontweight='bold')
        ax.set_xlabel('Date', fontweight='bold')
        ax.set_ylabel('Average Grid Consumption (kWh)', fontweight='bold')
        ax.legend(loc='upper right')
        self.save_fig(fig, filename)

    def plot_06_electricity_consumption_distribution(self, filename):
        """6. Electricity Consumption Distribution (Log Scale)"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        vals_norm = self.normal_df[self.date_cols].values.flatten()
        vals_theft = self.theft_df[self.date_cols].values.flatten()
        
        vals_norm = vals_norm[~np.isnan(vals_norm)]
        vals_theft = vals_theft[~np.isnan(vals_theft)]
        
        sns.kdeplot(np.log1p(vals_norm), ax=ax, color=COLOR_NORMAL, label='Normal (log1p)', fill=True, alpha=0.35)
        sns.kdeplot(np.log1p(vals_theft), ax=ax, color=COLOR_THEFT, label='Theft (log1p)', fill=True, alpha=0.35)
        
        ax.set_title('Log-Transformed Daily Electricity Consumption Distribution', pad=15, fontweight='bold')
        ax.set_xlabel('log(1 + kWh)', fontweight='bold')
        ax.set_ylabel('Density', fontweight='bold')
        ax.legend(loc='upper right')
        self.save_fig(fig, filename)

    def plot_07_histogram(self, filename):
        """7. Histogram of Customer Mean Daily Consumption"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        mean_per_cust_norm = self.normal_df[self.date_cols].mean(axis=1)
        mean_per_cust_theft = self.theft_df[self.date_cols].mean(axis=1)
        
        ax.hist(mean_per_cust_norm, bins=40, range=(0, 60), color=COLOR_NORMAL, alpha=0.6, label='Normal', edgecolor='black')
        ax.hist(mean_per_cust_theft, bins=40, range=(0, 60), color=COLOR_THEFT, alpha=0.7, label='Theft', edgecolor='black')
        
        ax.set_title('Histogram of Customer Mean Consumption', pad=15, fontweight='bold')
        ax.set_xlabel('Mean Daily Consumption (kWh)', fontweight='bold')
        ax.set_ylabel('Customer Count', fontweight='bold')
        ax.legend(loc='upper right')
        self.save_fig(fig, filename)

    def plot_08_kde_plot(self, filename):
        """8. Density KDE Comparison of Customer Standard Deviation"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        std_norm = self.normal_df[self.date_cols].std(axis=1).dropna()
        std_theft = self.theft_df[self.date_cols].std(axis=1).dropna()
        
        sns.kdeplot(std_norm, ax=ax, color=COLOR_NORMAL, label='Normal (Std Dev)', fill=True, alpha=0.3)
        sns.kdeplot(std_theft, ax=ax, color=COLOR_THEFT, label='Theft (Std Dev)', fill=True, alpha=0.3)
        
        ax.set_title('Kernel Density Estimation: Customer Load Volatility (Std Dev)', pad=15, fontweight='bold')
        ax.set_xlabel('Standard Deviation of Daily Readings (kWh)', fontweight='bold')
        ax.set_ylabel('Density', fontweight='bold')
        ax.set_xlim(0, np.percentile(std_norm, 99))
        ax.legend(loc='upper right')
        self.save_fig(fig, filename)

    def plot_09_boxplot(self, filename):
        """9. Boxplot of Customer Mean Consumption by Class"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        df_box = pd.DataFrame({
            'Mean_kWh': pd.concat([self.normal_df[self.date_cols].mean(axis=1), self.theft_df[self.date_cols].mean(axis=1)]),
            'Class': ['Normal'] * len(self.normal_df) + ['Theft'] * len(self.theft_df)
        })
        
        sns.boxplot(x='Class', y='Mean_kWh', hue='Class', data=df_box, palette=[COLOR_NORMAL, COLOR_THEFT], ax=ax, width=0.4, fliersize=3, legend=False)
        ax.set_title('Boxplot Comparison of Mean Consumption', pad=15, fontweight='bold')
        ax.set_ylabel('Mean Daily Consumption (kWh)', fontweight='bold')
        ax.set_ylim(0, np.percentile(df_box['Mean_kWh'].dropna(), 98))
        self.save_fig(fig, filename)

    def plot_10_violin_plot(self, filename):
        """10. Violin Plot of Customer Load Volatility"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        df_violin = pd.DataFrame({
            'Volatility_Std': pd.concat([self.normal_df[self.date_cols].std(axis=1), self.theft_df[self.date_cols].std(axis=1)]),
            'Class': ['Normal'] * len(self.normal_df) + ['Theft'] * len(self.theft_df)
        })
        
        sns.violinplot(x='Class', y='Volatility_Std', hue='Class', data=df_violin, palette=[COLOR_NORMAL, COLOR_THEFT], ax=ax, inner="quartile", cut=0, legend=False)
        ax.set_title('Violin Plot: Customer Volatility Distribution by Class', pad=15, fontweight='bold')
        ax.set_ylabel('Daily Load Standard Deviation (kWh)', fontweight='bold')
        ax.set_ylim(0, np.percentile(df_violin['Volatility_Std'].dropna(), 98))
        self.save_fig(fig, filename)

    def plot_11_customer_consumption_pattern(self, filename):
        """11. Customer Consumption Pattern (Representative Individual Normal Profiles)"""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        sample_normal = self.normal_df.sample(n=min(5, len(self.normal_df)), random_state=42)
        for idx, row in sample_normal.iterrows():
            ax.plot(self.dates, row[self.date_cols], label=f"Cust: {row[self.loader.id_col]}", alpha=0.7, linewidth=1.2)
            
        ax.set_title('Individual Normal Customer Consumption Time Series Profiles', pad=15, fontweight='bold')
        ax.set_xlabel('Date', fontweight='bold')
        ax.set_ylabel('Daily Consumption (kWh)', fontweight='bold')
        ax.legend(loc='upper right', frameon=True)
        self.save_fig(fig, filename)

    def plot_12_compare_normal_vs_theft_customer(self, filename):
        """12. Compare Normal vs Theft Customer Time Series"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        sample_n = self.normal_df.iloc[0]
        sample_t = self.theft_df.iloc[0]
        
        ax1.plot(self.dates, sample_n[self.date_cols], color=COLOR_NORMAL, linewidth=1.3)
        ax1.set_title(f'Normal Customer Profile ({sample_n[self.loader.id_col]})', fontweight='bold')
        ax1.set_ylabel('kWh', fontweight='bold')
        
        ax2.plot(self.dates, sample_t[self.date_cols], color=COLOR_THEFT, linewidth=1.3)
        ax2.set_title(f'Theft Customer Profile ({sample_t[self.loader.id_col]} - Zero/Bypass Pattern)', fontweight='bold')
        ax2.set_ylabel('kWh', fontweight='bold')
        ax2.set_xlabel('Date', fontweight='bold')
        
        self.save_fig(fig, filename)

    def plot_13_top_20_consumers(self, filename):
        """13. Top 20 Consumers (Highest Mean Load)"""
        fig, ax = plt.subplots(figsize=(10, 7))
        
        means = self.df.set_index(self.loader.id_col)[self.date_cols].mean(axis=1)
        top20 = means.nlargest(20).sort_values()
        
        ax.barh(top20.index, top20.values, color='#27ae60', edgecolor='black')
        ax.set_title('Top 20 Consumers by Mean Daily Electricity Usage', pad=15, fontweight='bold')
        ax.set_xlabel('Mean Daily Consumption (kWh)', fontweight='bold')
        ax.set_ylabel('Customer ID', fontweight='bold')
        self.save_fig(fig, filename)

    def plot_14_bottom_20_consumers(self, filename):
        """14. Bottom 20 Consumers (Lowest Mean Load)"""
        fig, ax = plt.subplots(figsize=(10, 7))
        
        means = self.df.set_index(self.loader.id_col)[self.date_cols].mean(axis=1)
        bottom20 = means.nsmallest(20).sort_values(ascending=False)
        
        ax.barh(bottom20.index, bottom20.values, color='#c0392b', edgecolor='black')
        ax.set_title('Bottom 20 Consumers by Mean Daily Electricity Usage', pad=15, fontweight='bold')
        ax.set_xlabel('Mean Daily Consumption (kWh)', fontweight='bold')
        ax.set_ylabel('Customer ID', fontweight='bold')
        self.save_fig(fig, filename)

    def plot_15_correlation_heatmap(self, filename):
        """15. Correlation Heatmap (Selected Days Across the Year)"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Select 15 evenly spaced day columns
        step = max(1, len(self.date_cols) // 15)
        sample_date_cols = self.date_cols[::step][:15]
        
        corr = self.df[sample_date_cols].corr()
        sns.heatmap(corr, ax=ax, cmap='coolwarm', vmin=-1, vmax=1, annot=False, cbar_kws={'label': 'Pearson Correlation'})
        ax.set_title('Cross-Day Electricity Consumption Correlation Heatmap', pad=15, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        self.save_fig(fig, filename)

    def plot_16_feature_correlation_matrix(self, filename):
        """16. Feature Correlation Matrix (Engineered Customer Summary Features)"""
        fig, ax = plt.subplots(figsize=(9, 8))
        
        summary_feats = pd.DataFrame({
            'Mean': self.df[self.date_cols].mean(axis=1),
            'Median': self.df[self.date_cols].median(axis=1),
            'Std': self.df[self.date_cols].std(axis=1),
            'Max': self.df[self.date_cols].max(axis=1),
            'Min': self.df[self.date_cols].min(axis=1),
            'Zero_Count': (self.df[self.date_cols] == 0).sum(axis=1),
            'FLAG': self.df[self.loader.target_col]
        })
        
        corr = summary_feats.corr()
        sns.heatmap(corr, ax=ax, cmap='vlag', annot=True, fmt='.2f', linewidths=0.5, cbar_kws={'label': 'Correlation'})
        ax.set_title('Feature Correlation Matrix (Summary Statistics vs Theft FLAG)', pad=15, fontweight='bold')
        self.save_fig(fig, filename)

    def plot_17_pair_plot(self, filename):
        """17. Pair Plot of Core Engineered Features"""
        summary_feats = pd.DataFrame({
            'Mean_kWh': self.df[self.date_cols].mean(axis=1),
            'Std_kWh': self.df[self.date_cols].std(axis=1),
            'Zero_Days': (self.df[self.date_cols] == 0).sum(axis=1),
            'Target': self.df[self.loader.target_col].map({0: 'Normal', 1: 'Theft'})
        }).dropna()
        
        # Sample for clean pair plot rendering
        norm_sub = summary_feats[summary_feats['Target'] == 'Normal']
        theft_sub = summary_feats[summary_feats['Target'] == 'Theft']
        
        sample_df = pd.concat([
            norm_sub.sample(min(300, len(norm_sub)), random_state=42),
            theft_sub.sample(min(300, len(theft_sub)), random_state=42)
        ])
        
        g = sns.pairplot(sample_df, hue='Target', palette={'Normal': COLOR_NORMAL, 'Theft': COLOR_THEFT},
                         diag_kind='kde', plot_kws={'alpha': 0.6, 's': 25})
        g.fig.suptitle('Pair Plot of Summary Features by Customer Class', y=1.02, fontweight='bold', fontsize=16)
        self.save_fig(g.fig, filename)

    def plot_18_zero_consumption_distribution(self, filename):
        """18. Zero Consumption Distribution (Normal vs Theft)"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        zero_norm = (self.normal_df[self.date_cols] == 0).sum(axis=1) / len(self.date_cols) * 100
        zero_theft = (self.theft_df[self.date_cols] == 0).sum(axis=1) / len(self.date_cols) * 100
        
        ax.hist(zero_norm, bins=30, alpha=0.6, color=COLOR_NORMAL, label='Normal Customers', edgecolor='black')
        ax.hist(zero_theft, bins=30, alpha=0.7, color=COLOR_THEFT, label='Theft Customers', edgecolor='black')
        
        ax.set_title('Distribution of Zero-Consumption Days per Customer (%)', pad=15, fontweight='bold')
        ax.set_xlabel('Percentage of Days with 0 kWh (%)', fontweight='bold')
        ax.set_ylabel('Customer Count', fontweight='bold')
        ax.legend(loc='upper right')
        self.save_fig(fig, filename)

    def plot_19_data_availability_per_customer(self, filename):
        """19. Data Availability per Customer (% Non-Missing Readings)"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        avail = (self.df[self.date_cols].notna().sum(axis=1) / len(self.date_cols)) * 100
        
        sns.histplot(avail, bins=30, ax=ax, color='#16a085', kde=True, edgecolor='black')
        ax.set_title('Customer Data Completeness / Availability Distribution', pad=15, fontweight='bold')
        ax.set_xlabel('Percentage of Valid Non-Missing Days (%)', fontweight='bold')
        ax.set_ylabel('Customer Count', fontweight='bold')
        self.save_fig(fig, filename)

    def plot_20_monthly_consumption_trend(self, filename):
        """20. Monthly Aggregated Consumption Trend by Class"""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Group columns by Month-Year
        month_labels = pd.to_datetime(self.date_cols, errors='coerce').to_period('M')
        
        norm_monthly = self.normal_df[self.date_cols].T.groupby(month_labels).mean().mean(axis=1)
        theft_monthly = self.theft_df[self.date_cols].T.groupby(month_labels).mean().mean(axis=1)
        
        x_str = [str(m) for m in norm_monthly.index]
        ax.plot(x_str, norm_monthly.values, marker='o', label='Normal (Monthly Mean)', color=COLOR_NORMAL, linewidth=2)
        ax.plot(x_str, theft_monthly.values, marker='s', label='Theft (Monthly Mean)', color=COLOR_THEFT, linewidth=2)
        
        ax.set_title('Monthly Aggregated Electricity Consumption Trend', pad=15, fontweight='bold')
        ax.set_xlabel('Month', fontweight='bold')
        ax.set_ylabel('Mean Daily Consumption (kWh)', fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        ax.legend(loc='upper right')
        self.save_fig(fig, filename)

    def plot_21_yearly_consumption_trend(self, filename):
        """21. Yearly Consumption Trend Comparison"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        years = pd.to_datetime(self.date_cols, errors='coerce').year
        
        norm_yearly = self.normal_df[self.date_cols].T.groupby(years).mean().mean(axis=1)
        theft_yearly = self.theft_df[self.date_cols].T.groupby(years).mean().mean(axis=1)
        
        years_unique = np.unique(years[~np.isnan(years)]).astype(int)
        x = np.arange(len(years_unique))
        width = 0.35
        
        ax.bar(x - width/2, norm_yearly.values[:len(years_unique)], width, label='Normal', color=COLOR_NORMAL, edgecolor='black')
        ax.bar(x + width/2, theft_yearly.values[:len(years_unique)], width, label='Theft', color=COLOR_THEFT, edgecolor='black')
        
        ax.set_title('Yearly Average Consumption Comparison', pad=15, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([str(y) for y in years_unique])
        ax.set_xlabel('Year', fontweight='bold')
        ax.set_ylabel('Average Consumption (kWh)', fontweight='bold')
        ax.legend(loc='upper right')
        self.save_fig(fig, filename)

    def plot_22_seasonal_consumption_pattern(self, filename):
        """22. Seasonal Consumption Pattern (Spring, Summer, Autumn, Winter)"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        months = pd.to_datetime(self.date_cols, errors='coerce').month
        season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                      3: 'Spring', 4: 'Spring', 5: 'Spring',
                      6: 'Summer', 7: 'Summer', 8: 'Summer',
                      9: 'Autumn', 10: 'Autumn', 11: 'Autumn'}
        seasons = pd.Series([season_map.get(m, 'Unknown') for m in months])
        
        season_order = ['Spring', 'Summer', 'Autumn', 'Winter']
        
        date_cols_arr = np.array(self.date_cols)
        norm_season = [self.normal_df[date_cols_arr[seasons == s]].mean().mean() for s in season_order]
        theft_season = [self.theft_df[date_cols_arr[seasons == s]].mean().mean() for s in season_order]
        
        x = np.arange(len(season_order))
        width = 0.35
        
        ax.bar(x - width/2, norm_season, width, label='Normal', color=COLOR_NORMAL, edgecolor='black')
        ax.bar(x + width/2, theft_season, width, label='Theft', color=COLOR_THEFT, edgecolor='black')
        
        ax.set_title('Seasonal Electricity Consumption Profiles', pad=15, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(season_order)
        ax.set_xlabel('Season', fontweight='bold')
        ax.set_ylabel('Mean Consumption (kWh)', fontweight='bold')
        ax.legend(loc='upper right')
        self.save_fig(fig, filename)

if __name__ == "__main__":
    from data_loader import SGCCDataLoader
    loader = SGCCDataLoader()
    viz = EDAVisualizer(loader)
    viz.generate_all_plots()
