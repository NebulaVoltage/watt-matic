# Real SGCC Data Validation Report

## 1. Dataset Identification
- **File**: `D:\wattmaticdataset\data.csv`
- **Dimensions**: 42372 rows, 1036 columns
- **CONS_NO Uniqueness**: 42372 unique IDs (Duplicates: 0)
- **FLAG Distribution**: {0: 38757, 1: 3615}
- **Number of Consumption Columns**: 1034
- **Date Parsing Success**: True
- **Earliest Date**: 2014-01-01
- **Latest Date**: 2016-10-31

## 2. Chronological Order
- **Is Chronologically Sorted?**: False
- **Is Lexicographically Sorted?**: True
- **Current First 20 Columns**: 
  `['2014/1/1', '2014/1/10', '2014/1/11', '2014/1/12', '2014/1/13', '2014/1/14', '2014/1/15', '2014/1/16', '2014/1/17', '2014/1/18', '2014/1/19', '2014/1/2', '2014/1/20', '2014/1/21', '2014/1/22', '2014/1/23', '2014/1/24', '2014/1/25', '2014/1/26', '2014/1/27']`
- **Chronological First 20 Columns**: 
  `['2014/1/1', '2014/1/2', '2014/1/3', '2014/1/4', '2014/1/5', '2014/1/6', '2014/1/7', '2014/1/8', '2014/1/9', '2014/1/10', '2014/1/11', '2014/1/12', '2014/1/13', '2014/1/14', '2014/1/15', '2014/1/16', '2014/1/17', '2014/1/18', '2014/1/19', '2014/1/20']`

*Note: The dataset columns are ordered lexicographically (alphabetically by string) rather than chronologically by actual date (e.g., Jan 1st of different years appear consecutively).*

## 3. Data Quality
- **Total Missing Cells**: 11,233,528 (25.64%)
  - FLAG=0 Missingness: 25.10%
  - FLAG=1 Missingness: 31.47%
- **Total Zeros**: 5,788,603 (13.21%)
  - FLAG=0 Zeros: 13.67%
  - FLAG=1 Zeros: 8.31%
- **Negative Values**: 0
- **Infinite Values**: 0
- **Duplicate Rows**: 0

## 4. Raw Distribution (Non-NaN values)
| Statistic | Overall | FLAG=0 (Normal) | FLAG=1 (Theft) |
|-----------|---------|-----------------|----------------|
| **Min** | 0.00 | 0.00 | 0.00 |
| **Max** | 800003.32 | 800003.32 | 514991.78 |
| **Mean** | 9.26 | 7.70 | 27.53 |
| **Median**| 4.59 | 4.43 | 7.01 |
| **Std Dev**| 273.22 | 245.39 | 493.37 |
| **5th Pct**| 0.00 | 0.00 | 0.00 |
| **95th Pct**| 25.24 | 23.05 | 73.73 |

## 5. Target Leakage Audit
- Unlike the synthetic dataset, the minimum consumption values for Normal (0.00) and Theft (0.00) are statistically identical. 
- Zeros (13.67% vs 8.31%) and Missing values (25.10% vs 31.47%) exhibit slightly different rates between classes, representing behavioral signals rather than deterministic data-generator artifacts.
- No obvious deterministic proxy for FLAG is present in the raw data matrix.

## 6. Comparison Against Synthetic Data
- **Synthetic Data**: 1,500 rows. Zeroes and missing values were hardcoded deterministic patterns. Perfect separation was trivially achieved.
- **Real SGCC Data**: 42,372 rows. Contains genuine noise, non-chronological columns, and real-world missingness (e.g., communication failures rather than synthetic drops).

## 7. Feature Pipeline Compatibility (Required Changes)
The existing 55-feature pipeline (`run_fe_extractor.py`) must be modified before processing this data:
1. **Chronological Sorting**: The pipeline currently relies on `np.diff(X)` for volatility features (daily changes) and `polyfit` for linear trend. Since columns in this real dataset are ordered lexicographically (e.g., `2014/1/1`, `2014/1/10`), applying `diff()` right now will compute nonsense transitions between arbitrary dates. The dataset columns must be sorted chronologically first.
2. **Missing Value Handling**: The current pipeline uses `ffill(axis=1).bfill(axis=1)`. However, if columns are not chronological, `ffill` will propagate data across non-adjacent temporal blocks.
3. **Monthly Aggregations**: Handled safely because dates are parsed independently by pandas.
