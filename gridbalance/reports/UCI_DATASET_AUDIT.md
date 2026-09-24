# UCI Electricity Load Diagrams Dataset Audit Report

## 1. Dataset Dimensions & Metadata
- **Source File**: `data/uci/LD2011_2014.txt`
- **Total Customer Meters**: 370 (`MT_001` to `MT_370`)
- **Raw Timestamps Count**: 140,256 (15-minute resolution)
- **Hourly Resampled Timestamps**: 35,065 hours
- **Date Range**: `2011-01-01 00:15:00` to `2015-01-01 00:00:00`
- **Measurement Units**: Raw values are recorded in kW per 15-min interval. Values were converted to hourly aggregate energy consumption in **kWh** via $\sum_{i=1}^{370} \text{meter}_i / 4$.

## 2. Quality & Integrity Assessment
- **Missing Cells (`NaN`)**: 0
- **Zero Consumption Cells**: 10,457,342 (20.15%)
  - *Observation*: Early 2011 records contain high zero counts because several meters were brought online progressively during 2011 and 2012.
- **Negative Values**: 0
- **Duplicate Timestamps**: 0
- **Duplicate Rows**: 0

## 3. Aggregate System vs Customer-Level Tradeoffs
The full 370-customer 15-minute matrix contains 51,894,720 values. For next-hour system load forecasting, aggregating the meters into a single hourly aggregate system load vector ($	ext{load}_t = \sum_{i=1}^{370} \text{load}_{i,t}$) enables high-precision grid-level demand forecasting while maintaining computational tractability.

## 4. Output Artifacts
- **Statistics CSV**: `reports/UCI_DATASET_STATISTICS.csv`
- **Processed Hourly Load**: `data/processed/uci_hourly_aggregate_load.csv`
- **EDA Plots**: `graphs/uci/eda/`
