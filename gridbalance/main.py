"""
SGCC Electricity Theft EDA Pipeline — Master Entry Point
Executes the full end-to-end Exploratory Data Analysis workflow,
generates all 22 publication-quality plots in `graphs/`,
computes high-dimensional statistics, exports all required CSV reports,
and compiles the presentation-ready `EDA_Report.pdf`.
"""

import sys
import time
from tqdm import tqdm

from data_loader import SGCCDataLoader
from overview_analyzer import OverviewAnalyzer
from quality_analyzer import DataQualityAnalyzer
from eda_visualizer import EDAVisualizer
from stats_calculator import StatisticsCalculator
from ml_insights import MLInsightsEngine
from feature_engineering import FeatureEngineeringSuite
from pdf_report_generator import PDFReportGenerator

def run_pipeline():
    start_time = time.time()
    print("="*75)
    print("      SGCC ELECTRICITY THEFT DETECTION — COMPLETE EDA PIPELINE      ")
    print("="*75)

    pipeline_steps = [
        ("Step 1/7: Data Loading & Schema Parsing", "loader"),
        ("Step 2/7: Part 1 — Dataset Overview Analysis", "overview"),
        ("Step 3/7: Part 2 — Data Quality & Missing Value Analysis", "quality"),
        ("Step 4/7: Part 3 — Generating 22 Publication Figures", "visualizer"),
        ("Step 5/7: Part 4 — Dataset Statistics Computation", "stats"),
        ("Step 6/7: Part 5 & 6 — ML Insights & Feature Engineering", "ml_fe"),
        ("Step 7/7: Part 7 & 8 — PDF Report Compilation & CSV Exports", "pdf_export"),
    ]

    # Initialize loader
    print("\n[STEP 1/7] Initializing SGCC Data Loader...")
    loader = SGCCDataLoader(filepath="data.csv")

    # Step 2: Overview
    print("\n[STEP 2/7] Running Dataset Overview Analysis...")
    overview = OverviewAnalyzer(loader)
    overview.print_report()
    overview.export_csv("Dataset_Summary.csv")

    # Step 3: Data Quality
    print("\n[STEP 3/7] Running Data Quality Analysis...")
    quality = DataQualityAnalyzer(loader)
    quality.print_quality_report()
    quality.export_csv("MissingValues.csv")

    # Step 4: Visualizations
    print("\n[STEP 4/7] Generating 22 Publication-Grade Plots...")
    viz = EDAVisualizer(loader, output_dir="graphs")
    viz.generate_all_plots()

    # Step 5: Statistics
    print("\n[STEP 5/7] Computing High-Dimensional Dataset Statistics...")
    stats_calc = StatisticsCalculator(loader)
    stats_calc.print_table()
    stats_calc.export_csv("Statistics.csv")

    # Step 6: ML Insights & Feature Engineering
    print("\n[STEP 6/7] Computing Machine Learning Insights & Feature Engineering Ideas...")
    ml_engine = MLInsightsEngine(loader)
    ml_engine.print_insights_report()
    
    fe_suite = FeatureEngineeringSuite(loader)
    fe_suite.export_csv("FeatureImportanceIdeas.csv")

    # Step 7: PDF Compilation
    print("\n[STEP 7/7] Compiling Presentation-Ready PDF Report...")
    pdf_gen = PDFReportGenerator(loader, graphs_dir="graphs", output_pdf="EDA_Report.pdf")
    pdf_gen.generate_pdf()

    elapsed = time.time() - start_time
    print("\n" + "="*75)
    print(f"[SUCCESS] SGCC EDA Pipeline Execution Completed in {elapsed:.2f} seconds.")
    print("Deliverables Generated:")
    print("  • graphs/ (22 high-resolution 300 DPI PNG figures)")
    print("  • EDA_Report.pdf (Executive PDF summary report)")
    print("  • Dataset_Summary.csv")
    print("  • Statistics.csv")
    print("  • MissingValues.csv")
    print("  • FeatureImportanceIdeas.csv")
    print("="*75 + "\n")

if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Pipeline execution failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
