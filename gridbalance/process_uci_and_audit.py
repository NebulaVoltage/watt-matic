import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def create_directories():
    os.makedirs('reports', exist_ok=True)
    os.makedirs('docs', exist_ok=True)
    os.makedirs('configs', exist_ok=True)
    os.makedirs('graphs/uci/eda', exist_ok=True)
    os.makedirs('graphs/uci/forecasting', exist_ok=True)
    os.makedirs('graphs/uci/interpretability', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('tests', exist_ok=True)

def audit_dataset():
    create_directories()
    raw_path = 'data/uci/LD2011_2014.txt'
    print(f"[INFO] Reading UCI Electricity dataset from {raw_path}...")
    
    # Read text file. Column 0 is timestamp, rest are MT_001 to MT_370
    # Decimals are represented by commas in the file
    df_raw = pd.read_csv(raw_path, sep=';', decimal=',', parse_dates=[0], index_col=0)
    
    num_timestamps, num_meters = df_raw.shape
    start_date = df_raw.index.min()
    end_date = df_raw.index.max()
    freq = pd.infer_freq(df_raw.index[:100])
    
    # Meter stats
    missing_cells = df_raw.isna().sum().sum()
    zero_cells = (df_raw == 0).sum().sum()
    neg_cells = (df_raw < 0).sum().sum()
    dup_timestamps = df_raw.index.duplicated().sum()
    dup_rows = df_raw.duplicated().sum()
    
    # Values are in kW per 15-min interval.
    # To aggregate to system hourly load in kWh (or average kW):
    # Sum across all 370 meters for each 15-min timestamp, then resample to 1-hour
    print("[INFO] Resampling 15-min system load to 1-hour aggregate load...")
    df_raw['system_load_kw'] = df_raw.sum(axis=1)
    
    # Resample to 1-hour resolution: sum of 15-min kW values / 4 = hourly energy in kWh
    hourly_df = df_raw['system_load_kw'].resample('1h').sum() / 4.0
    hourly_df = hourly_df.to_frame(name='aggregate_load_kwh')
    
    # Save processed hourly system load
    hourly_df.to_csv('data/processed/uci_hourly_aggregate_load.csv')
    print(f"[SUCCESS] Saved hourly aggregate load: {hourly_df.shape[0]} hours.")
    
    # Generate Audit CSV
    stats_df = pd.DataFrame({
        'Metric': [
            'Total Meters', 'Total 15-min Timestamps', 'Hourly Timestamps',
            'Start Date', 'End Date', 'Sampling Frequency',
            'Missing Values', 'Zero Values (15-min)', 'Negative Values',
            'Duplicate Timestamps', 'Duplicate Rows', 'Total Raw Cells'
        ],
        'Value': [
            num_meters, num_timestamps, len(hourly_df),
            str(start_date), str(end_date), '15-minute / 1-hour resampled',
            missing_cells, zero_cells, neg_cells,
            dup_timestamps, dup_rows, num_timestamps * num_meters
        ]
    })
    stats_df.to_csv('reports/UCI_DATASET_STATISTICS.csv', index=False)
    
    # Generate EDA Plots
    print("[INFO] Generating EDA plots under graphs/uci/eda/...")
    
    # 1. Aggregate load over time
    plt.figure(figsize=(14, 5))
    plt.plot(hourly_df.index, hourly_df['aggregate_load_kwh'], color='navy', alpha=0.8, linewidth=0.7)
    plt.title('Aggregate System Electricity Load Over Time (2011 - 2014)')
    plt.xlabel('Date')
    plt.ylabel('Hourly Load (kWh)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('graphs/uci/eda/01_aggregate_load_over_time.png')
    plt.close()
    
    # 2. Daily load profile (Hourly average)
    hourly_df['hour'] = hourly_df.index.hour
    hourly_df['day_of_week'] = hourly_df.index.dayofweek
    
    plt.figure(figsize=(10, 5))
    sns.lineplot(x='hour', y='aggregate_load_kwh', data=hourly_df, color='darkorange', ci=95)
    plt.title('Average Daily Load Profile (Hour-of-Day)')
    plt.xlabel('Hour of Day (0 - 23)')
    plt.ylabel('Load (kWh)')
    plt.xticks(range(0, 24))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('graphs/uci/eda/02_daily_load_profile.png')
    plt.close()
    
    # 3. Weekly load profile
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    plt.figure(figsize=(10, 5))
    sns.boxplot(x='day_of_week', y='aggregate_load_kwh', data=hourly_df, palette='Set2')
    plt.xticks(range(7), days)
    plt.title('Weekly Load Profile by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Load (kWh)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('graphs/uci/eda/03_weekly_load_profile.png')
    plt.close()
    
    # 4. Load distribution
    plt.figure(figsize=(9, 5))
    sns.histplot(hourly_df['aggregate_load_kwh'], kde=True, color='teal', bins=50)
    plt.title('Distribution of Aggregate Hourly Electricity Load')
    plt.xlabel('Hourly Load (kWh)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('graphs/uci/eda/04_load_distribution.png')
    plt.close()
    
    # 5. Missingness over time (Zero load meters count over time)
    zero_meters_per_ts = (df_raw.iloc[:, :370] == 0).sum(axis=1)
    plt.figure(figsize=(14, 4))
    plt.plot(df_raw.index, zero_meters_per_ts, color='crimson', linewidth=0.5)
    plt.title('Number of Uninitialized / Inactive Meters (Zero Load) Over Time')
    plt.xlabel('Date')
    plt.ylabel('Count of Inactive Meters')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('graphs/uci/eda/05_missingness_over_time.png')
    plt.close()
    
    # 6. Customer-level load distributions (sample 10 random meters)
    sample_meters = df_raw.columns[:10]
    plt.figure(figsize=(12, 6))
    for m in sample_meters:
        sns.kdeplot(df_raw[m][df_raw[m]>0], label=m, alpha=0.5)
    plt.title('Customer-Level Active Load Distributions (Sample of 10 Meters)')
    plt.xlabel('15-min Consumption (kW)')
    plt.ylabel('Density')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('graphs/uci/eda/06_customer_load_distributions.png')
    plt.close()
    
    # 7. Seasonal/periodic behavior
    hourly_df['month'] = hourly_df.index.month
    plt.figure(figsize=(10, 5))
    sns.barplot(x='month', y='aggregate_load_kwh', data=hourly_df, palette='viridis')
    plt.title('Monthly Average Load Profile (Seasonal Variation)')
    plt.xlabel('Month')
    plt.ylabel('Average Load (kWh)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('graphs/uci/eda/07_seasonal_behavior.png')
    plt.close()
    
    # Create Audit MD
    audit_md = f"""# UCI Electricity Load Diagrams Dataset Audit Report

## 1. Dataset Dimensions & Metadata
- **Source File**: `data/uci/LD2011_2014.txt`
- **Total Customer Meters**: {num_meters} (`MT_001` to `MT_370`)
- **Raw Timestamps Count**: {num_timestamps:,} (15-minute resolution)
- **Hourly Resampled Timestamps**: {len(hourly_df):,} hours
- **Date Range**: `{start_date}` to `{end_date}`
- **Measurement Units**: Raw values are recorded in kW per 15-min interval. Values were converted to hourly aggregate energy consumption in **kWh** via $\\sum_{{i=1}}^{{370}} \\text{{meter}}_i / 4$.

## 2. Quality & Integrity Assessment
- **Missing Cells (`NaN`)**: {missing_cells}
- **Zero Consumption Cells**: {zero_cells:,} ({zero_cells/(num_timestamps*num_meters)*100:.2f}%)
  - *Observation*: Early 2011 records contain high zero counts because several meters were brought online progressively during 2011 and 2012.
- **Negative Values**: {neg_cells}
- **Duplicate Timestamps**: {dup_timestamps}
- **Duplicate Rows**: {dup_rows}

## 3. Aggregate System vs Customer-Level Tradeoffs
The full 370-customer 15-minute matrix contains 51,894,720 values. For next-hour system load forecasting, aggregating the meters into a single hourly aggregate system load vector ($\text{{load}}_t = \\sum_{{i=1}}^{{370}} \\text{{load}}_{{i,t}}$) enables high-precision grid-level demand forecasting while maintaining computational tractability.

## 4. Output Artifacts
- **Statistics CSV**: `reports/UCI_DATASET_STATISTICS.csv`
- **Processed Hourly Load**: `data/processed/uci_hourly_aggregate_load.csv`
- **EDA Plots**: `graphs/uci/eda/`
"""
    with open('reports/UCI_DATASET_AUDIT.md', 'w') as f:
        f.write(audit_md)
        
    # Create Problem Definition Markdown
    problem_md = """# UCI Electricity Next-Hour Forecasting Problem Definition

## 1. Mathematical Formulation
Given a time series of hourly system load observations $y_1, y_2, \\dots, y_t$, the objective is to predict the next-hour load:

$$\\hat{y}_{t+1} = f(\\mathbf{X}_{\\le t})$$

where $\\mathbf{X}_{\\le t}$ is a feature vector derived strictly from historical observations $y_{\\tau}$ where $\\tau \\le t$, alongside deterministic temporal variables (e.g., hour, day of week) known at time $t$.

## 2. Temporal Correctness & Leakage Prevention Rules
1. **Causal Features Only**: All lag features ($y_t, y_{t-1}, \\dots$) and rolling window statistics (e.g., `rolling_mean_24`) are strictly computed using past values via `shift(1)`.
2. **No Future Lookahead**: No centered rolling windows, future interpolation, or global target normalization are permitted.
3. **Chronological Splitting**: Train, Validation, and Test splits are partitioned chronologically. Shuffling is strictly prohibited.
"""
    with open('docs/UCI_FORECASTING_PROBLEM_DEFINITION.md', 'w') as f:
        f.write(problem_md)
        
    print("[SUCCESS] Phase 1 & Phase 2 completed.")

if __name__ == '__main__':
    audit_dataset()
