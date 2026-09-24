# ⚡ SGCC Electricity Theft Detection — Complete EDA & Machine Learning Analytics Suite

This repository provides an industry-grade, publication-ready Exploratory Data Analysis (EDA) suite for the **State Grid Corporation of China (SGCC) Electricity Theft Detection Dataset**.

## 📌 Deliverables

1. **Jupyter Notebook**: [`SGCC_Electricity_Theft_EDA.ipynb`](file:///d:/gridbalance/SGCC_Electricity_Theft_EDA.ipynb) — Interactive self-contained notebook covering all 8 EDA parts.
2. **Publication Plots**: [`graphs/`](file:///d:/gridbalance/graphs) — 22 high-resolution (300 DPI) figures saved individually.
3. **Executive PDF Report**: [`EDA_Report.pdf`](file:///d:/gridbalance/EDA_Report.pdf) — Presentation-ready PDF report compiling all visual analytics, tables, ML insights, and presentation summaries.
4. **Summary & Data CSV Reports**:
   - `Dataset_Summary.csv`
   - `Statistics.csv`
   - `MissingValues.csv`
   - `FeatureImportanceIdeas.csv`
15. **Feature Engineering Pipeline Outputs**:
   - `data/processed/sgcc_features.csv` (ML-ready feature matrix)
   - `reports/SGCC_Feature_Engineering_Report.md` (Detailed feature engineering report)
   - `graphs/feature_engineering/` (Distributions, heatmaps)

## 📁 Repository Structure

```
d:\gridbalance\
├── SGCC_Electricity_Theft_EDA.ipynb  # Interactive Jupyter Notebook
├── main.py                          # Master execution pipeline script
├── data_loader.py                   # Data loader & SGCC schema validator
├── overview_analyzer.py             # Part 1: Dataset Overview module
├── quality_analyzer.py              # Part 2: Data Quality & Missing Value module
├── eda_visualizer.py                # Part 3: 22 High-DPI Plot Visualizer Engine
├── stats_calculator.py              # Part 4: High-Dimensional Statistics Calculator
├── ml_insights.py                   # Part 5: Machine Learning Insights & Model Recommender
├── feature_engineering.py           # Part 6: Domain Feature Engineering Ideas
├── run_fe_extractor.py              # Part 6b: Modular Feature Extractor
├── run_fe_pipeline.py               # Part 6c: Feature Engineering Pipeline execution
├── pdf_report_generator.py          # Part 7 & 8: Executive PDF Report Generator
├── generate_sample_data.py          # Synthetic fallback data generator
├── requirements.txt                 # Project dependencies
├── graphs/                          # Directory for 22 generated PNG figures
├── data/processed/                  # ML-ready processed datasets
├── reports/                         # Markdown reports for pipelines
└── data.csv                         # SGCC Dataset file (auto-generated if missing)
```

## 🚀 Execution Instructions

### Run EDA Pipeline
Simply run the master EDA pipeline:

```bash
python main.py
```
All 22 plots, CSV files, and `EDA_Report.pdf` will be updated automatically!

### Run Feature Engineering Pipeline
To process the raw daily consumption dataset into an ML-ready structured format with 55 statistical/temporal features:

```bash
python run_fe_pipeline.py
```
