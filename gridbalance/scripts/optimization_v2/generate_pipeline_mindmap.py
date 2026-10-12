import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# High-resolution canvas setup
fig = plt.figure(figsize=(24, 32), dpi=300)
ax = fig.add_subplot(111)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Color palette (Modern Dark-Mode / Technical Architecture Aesthetic)
BG_COLOR = "#0F172A"       # Slate 900
PANEL_BG = "#1E293B"       # Slate 800
PANEL_BORDER = "#334155"   # Slate 700
TEXT_MAIN = "#F8FAFC"      # Slate 50
TEXT_MUTED = "#94A3B8"     # Slate 400
TEXT_ACCENT = "#38BDF8"    # Sky 400

# Branch Colors
COLOR_DATA = "#3B82F6"     # Blue (Data)
COLOR_SPLIT = "#8B5CF6"    # Purple (Splits)
COLOR_FEAT = "#EC4899"     # Pink (Feature Engineering)
COLOR_MODEL = "#F59E0B"    # Amber (Model & Loss)
COLOR_EVAL = "#10B981"     # Emerald (Evaluation)
COLOR_PROD = "#06B6D4"     # Cyan (Production / Integrity)

fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

# Header Title Block
title_box = patches.FancyBboxPatch(
    (5, 93), 90, 5.5,
    boxstyle="round,pad=0.5,rounding_size=0.8",
    facecolor="#1E1B4B", edgecolor="#6366F1", linewidth=2.5
)
ax.add_patch(title_box)
ax.text(50, 96.5, "GRIDBALANCE — END-TO-END PIPELINE & DIAGNOSTIC ARCHITECTURE MINDMAP",
        fontsize=22, fontweight='bold', color="#FFFFFF", ha='center', va='center', fontfamily='sans-serif')
ax.text(50, 94.2, "Complete Step-by-Step Flow: Raw Ingestion → Leak-Free Splitting → Feature Engineering → Focal GBDT → Integrity Verification",
        fontsize=13, color="#A5B4FC", ha='center', va='center', fontfamily='sans-serif')

def draw_node(x, y, w, h, title, items, color, badge=""):
    # Main container
    box = patches.FancyBboxPatch(
        (x, y - h), w, h,
        boxstyle="round,pad=0.4,rounding_size=0.6",
        facecolor=PANEL_BG, edgecolor=color, linewidth=2.0
    )
    ax.add_patch(box)
    
    # Header bar
    header_box = patches.FancyBboxPatch(
        (x, y - 2.8), w, 2.8,
        boxstyle="round,pad=0.2,rounding_size=0.4",
        facecolor=color, edgecolor="none", alpha=0.25
    )
    ax.add_patch(header_box)
    
    # Title & Badge
    ax.text(x + 1.2, y - 1.4, title, fontsize=12.5, fontweight='bold', color=TEXT_MAIN, va='center')
    if badge:
        ax.text(x + w - 1.2, y - 1.4, badge, fontsize=9.5, fontweight='bold', color=color,
                ha='right', va='center', bbox=dict(boxstyle='round,pad=0.2', facecolor=PANEL_BG, edgecolor=color, lw=1))
    
    # Bullet items
    curr_y = y - 3.8
    for item in items:
        # Check if item is a sub-header or key-value
        if item.startswith("• "):
            ax.text(x + 1.2, curr_y, item, fontsize=9.8, color=TEXT_MAIN, va='top')
        elif item.startswith("  - "):
            ax.text(x + 2.5, curr_y, item, fontsize=9.0, color=TEXT_MUTED, va='top')
        elif item.startswith("  * "):
            ax.text(x + 2.5, curr_y, item, fontsize=9.0, color="#FCA5A5", va='top') # alert color
        elif item.startswith("  ✔ "):
            ax.text(x + 2.5, curr_y, item, fontsize=9.0, color="#86EFAC", va='top') # success color
        else:
            ax.text(x + 1.2, curr_y, item, fontsize=9.5, color=TEXT_MUTED, va='top')
        curr_y -= 1.45

def draw_arrow(x1, y1, x2, y2, color, label=""):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2, mutation_scale=16,
                        connectionstyle="arc3,rad=0.0")
    )
    if label:
        mx, my = (x1 + x2)/2, (y1 + y2)/2
        ax.text(mx, my + 0.6, label, fontsize=8.5, fontweight='bold', color=color,
                ha='center', va='center', bbox=dict(boxstyle='round,pad=0.15', facecolor=BG_COLOR, edgecolor='none'))

def draw_elbow_arrow(x1, y1, xmid, y2, x2, color):
    # Two-segment orthogonal arrow
    ax.plot([x1, xmid, xmid, x2], [y1, y1, y2, y2], color=color, lw=2.0)
    ax.annotate(
        "", xy=(x2, y2), xytext=(xmid + (0.5 if x2 > xmid else -0.5), y2),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0, mutation_scale=14)
    )

# ==============================================================================
# COLUMN 1: STAGE 1 (INGESTION) & STAGE 2 (LEAK-FREE SPLITTING) (x: 4 to 32)
# ==============================================================================

# Node 1.1: Raw Dual Data Ingestion
draw_node(
    4, 90, 27, 13,
    "STAGE 1: DUAL-DATASET INGESTION",
    [
        "• SGCC State Grid Corporation of China:",
        "  - Raw: 42,372 consumers × 1,036 dates (2014-2016)",
        "  - 43.8M cells | 25.64% missing (11.23M NaNs)",
        "  - Target FLAG: 91.47% Normal (0) vs 8.53% Theft (1)",
        "  - 5 all-NaN meters excluded → 42,367 clean rows",
        "• UCI Electricity Load Forecasting:",
        "  - 370 smart meters (15-min kW readings 2011-2014)",
        "  - Aggregated to 34,896 Hourly System Load (kWh)",
        "  - Pure numeric timeseries (zero missing values)"
    ],
    COLOR_DATA, "INGESTION"
)

# Node 1.2: Strict Double-Sealed Splitting
draw_node(
    4, 74, 27, 13,
    "STAGE 2: DOUBLE-SEALED SPLITTING",
    [
        "• SGCC Partitioning (Seed=42, Stratified):",
        "  - Train Set (70%): N = 29,656 (2,528 Theft)",
        "  - Validation Set (15%): N = 6,355 (542 Theft)",
        "  - Sealed Test Set (15%): N = 6,356 (542 Theft)",
        "• UCI Partitioning (Chronological Walk-Forward):",
        "  - Train (75%): 26,136 hours (~3 years)",
        "  - Validation (12.5%): 4,344 hours (~6 months)",
        "  - Sealed Test (12.5%): 4,416 hours (~6 months)",
        "• ZERO LEAKAGE LAW:",
        "  ✔ Sealed test isolated under double-seal protocol",
        "  ✔ Never inspected during tuning or feature selection"
    ],
    COLOR_SPLIT, "SPLITTING"
)

# Connect Stage 1 -> Stage 2
draw_arrow(17.5, 77, 17.5, 74, COLOR_SPLIT, "Partitioning")

# ==============================================================================
# COLUMN 2: STAGE 3 (FEATURE ENG) & STAGE 4 (DIAGNOSTICS) (x: 35 to 64)
# ==============================================================================

# Node 2.1: Feature Engineering Pipeline
draw_node(
    36, 90, 28, 14,
    "STAGE 3: ZERO-LEAKAGE FEATURE PIPELINE",
    [
        "• Training-Median Imputation:",
        "  - Imputer fitted strictly on Train (N=29,656)",
        "  - Transformed onto Val & Test without lookahead",
        "• 42-Feature Comprehensive Engineering:",
        "  - Missingness Dynamics: missing_count, streak_count,",
        "    longest_streak, max_missing_gap_days (5 feats)",
        "  - Consumption Statistics: mean, median, std, cv, iqr,",
        "    min, max, zero_count, zero_ratio (9 feats)",
        "  - Volatility: mean_abs_daily_change, max_daily_change",
        "  - Monthly Rolling Windows: 12 means + 12 stds",
        "• Frozen 18-Feature Dimensionality Reduction:",
        "  - Preserves 97% predictive power with minimal variance"
    ],
    COLOR_FEAT, "FEATURES"
)

# Node 2.2: Empirical Diagnostics & Ablation
draw_node(
    36, 73, 28, 14,
    "STAGE 4: EMPIRICAL DIAGNOSTICS & ABLATION",
    [
        "• Phase 2: Class Separability Audit:",
        "  - 95% - 99.9% histogram overlap on consumption",
        "  - Missing streak features show lowest overlap (81.8%)",
        "• Phase 3: Missingness vs Consumption Ablation:",
        "  - Consumption Only (15 feats): Val F1=0.372, PR=0.352",
        "  - Missingness Only (3 feats): Val F1=0.313, PR=0.333",
        "  - Combined 18 (All): Val F1=0.441, PR=0.422",
        "• Phase 4: Coverage Quality Groups:",
        "  - High coverage (<10% miss): Val F1=0.396 (low theft)",
        "  - Low coverage (≥30% miss): Val F1=0.462 (high theft)",
        "• Phase 5: Learning Curves (20% - 100%):",
        "  - CV F1 plateaus at 0.403 (sample saturation reached)"
    ],
    COLOR_MODEL, "DIAGNOSTICS"
)

# Connect Stage 2 -> Stage 3
draw_elbow_arrow(31, 67.5, 33.5, 83, 36, COLOR_FEAT)
# Connect Stage 3 -> Stage 4
draw_arrow(50, 76, 50, 73, COLOR_MODEL, "Feature Space")

# ==============================================================================
# COLUMN 3: STAGE 5 (MODEL TRAINING) & STAGE 6 (EVALUATION) (x: 68 to 96)
# ==============================================================================

# Node 3.1: Model Optimization & Loss Engineering
draw_node(
    68, 90, 28, 14,
    "STAGE 5: MODEL ARCHITECTURES & LOSS",
    [
        "• Frozen Baseline: XGBoost (18 Features):",
        "  - max_depth=7, n_est=200, scale_pos_weight=10.72",
        "  - Baseline Test: F1=0.410, PR-AUC=0.376, ROC=0.824",
        "• Candidate: LightGBM with Custom Focal Loss:",
        "  - Alpha=0.85 (Minority weight), Gamma=2.0 (Focusing)",
        "  - Loss = -alpha * (1 - pt)^gamma * log(pt)",
        "  - num_leaves=63, lr=0.05, feature_fraction=0.8",
        "  - Directly targets hard-to-classify borderline theft",
        "• Benchmark Comparison (5-Fold CV):",
        "  - ExtraTrees: Failed (CV F1=0.288, PR-AUC=0.300)",
        "  - XGBoost: Solid (CV F1=0.404, PR-AUC=0.393)",
        "  - LightGBM Focal: WINNER (CV F1=0.426, PR=0.413)"
    ],
    COLOR_MODEL, "MODELING"
)

# Node 3.2: Operational Decision Threshold & Sealed Test
draw_node(
    68, 73, 28, 14,
    "STAGE 6: THRESHOLDING & SEALED TEST",
    [
        "• Validation Threshold Optimization (Zero Test Leak):",
        "  - Scan threshold t ∈ [0.20, 0.80] strictly on Val",
        "  - Peak Val F1 achieved at t = 0.50 (balanced trade-off)",
        "• Final Sealed Test Verification (Evaluated ONCE):",
        "  - Frozen XGBoost Baseline:",
        "    Precision=0.3275 | Recall=0.5498 | F1=0.4105",
        "    PR-AUC=0.3762 | ROC-AUC=0.8236",
        "  - Optimized LightGBM Focal Loss Candidate:",
        "    Precision=0.4200 (+28.2%) | Recall=0.4797",
        "    F1=0.4479 (+9.1%) | PR-AUC=0.4178 (+11.1%)",
        "    ROC-AUC=0.8429 (+2.3%)",
        "  ✔ Statistically validated sealed-test improvement"
    ],
    COLOR_EVAL, "EVALUATION"
)

# Connect Stage 4 -> Stage 5
draw_elbow_arrow(64, 66, 66, 83, 68, COLOR_MODEL)
# Connect Stage 5 -> Stage 6
draw_arrow(82, 76, 82, 73, COLOR_EVAL, "Optimal Weights")

# ==============================================================================
# ROW 3: DETAILED DEEP-DIVE PANELS (y: 28 to 55)
# ==============================================================================

# Deep Dive Panel A: Full Step-by-Step Execution Sequence
draw_node(
    4, 56, 44, 26,
    "DETAILED STEP-BY-STEP EXECUTION SEQUENCE (ORDERWISE & LINEWISE)",
    [
        "• STEP 1: DATA INGESTION & CLEANING",
        "  - 1.1 Ingest data/sgcc_data.csv (42,372 rows) and data/uci/LD2011_2014.txt",
        "  - 1.2 Remove 5 records with 100% missing historical readings (42,367 clean rows)",
        "  - 1.3 Verify target distribution: 38,755 class 0 (91.47%), 3,612 class 1 (8.53%)",
        "• STEP 2: LEAKAGE-FREE DOUBLE-SEALED SPLITTING",
        "  - 2.1 Set global random seed = 42 for complete experimental reproducibility",
        "  - 2.2 StratifiedKFold / train_test_split: 70% Train (29,656), 15% Val (6,355), 15% Test (6,356)",
        "  - 2.3 Seal Test Set (N=6,356) in vault; freeze labels until final candidate gate",
        "• STEP 3: STRICT TRAIN-ONLY FEATURE IMPUTATION",
        "  - 3.1 Compute medians for all numeric columns strictly on Train (N=29,656)",
        "  - 3.2 Impute NaNs in Validation and Test sets using training medians exclusively",
        "• STEP 4: 42-FEATURE EXTRACTION & 18-FEATURE SELECTION",
        "  - 4.1 Missingness streaks: missing_count, longest_missing_streak, streak_count",
        "  - 4.2 Dispersion: cv, iqr, mean_abs_daily_change, max, peak_to_average_ratio",
        "  - 4.3 Seasonality: 12 monthly means and 12 monthly standard deviations",
        "  - 4.4 Prune to 18 most stable features via stability selection and SHAP importance",
        "• STEP 5: 5-FOLD STRATIFIED CROSS-VALIDATION TRAINING",
        "  - 5.1 StratifiedKFold(n_splits=5, shuffle=True, random_state=42) on Train",
        "  - 5.2 Train models with class weighting (scale_pos_weight=10.72) and Focal Loss",
        "  - 5.3 Record out-of-fold metrics: F1, PR-AUC (Average Precision), ROC-AUC, MCC",
        "• STEP 6: VALIDATION-BASED THRESHOLD OPTIMIZATION",
        "  - 6.1 Evaluate probability calibration on Validation set (N=6,355)",
        "  - 6.2 Optimize decision threshold t on Validation F1 curve (t=0.50 optimal)",
        "• STEP 7: SINGLE-EVALUATION SEALED TEST VERIFICATION",
        "  - 7.1 Freeze candidate weights; load sealed test set (N=6,356)",
        "  - 7.2 Run forward inference once; compute final test confusion matrix & metrics",
        "  - 7.3 Log results to reports/optimization_v2/ and graphs/optimization_v2/"
    ],
    COLOR_SPLIT, "EXECUTION ORDER"
)

# Deep Dive Panel B: Root Cause Diagnostics & Limitations
draw_node(
    52, 56, 44, 26,
    "WHY MODEL METRICS PLATEAU: EMPIRICAL ROOT CAUSE DIAGNOSIS",
    [
        "• FACTOR 1: WEAK & OVERLAPPING CONSUMPTION SIGNALS (~50% IMPACT)",
        "  - Empirical distribution overlap between normal and theft exceeds 95% - 99.9%",
        "  - Mann-Whitney effect sizes: Cohen's d < 0.40, KS stat < 0.32 across all 18 features",
        "  - Legitimate residential consumers exhibit high seasonal volatility and vacation zero-draws",
        "  - High overlap makes linear or shallow boundary separation impossible",
        "• FACTOR 2: MEASUREMENT-SIGNAL ENTANGLEMENT (~30% IMPACT)",
        "  - Feature Importance & SHAP shows missingness features drive huge predictive weight",
        "  - Ablation proves removing missingness drops Val F1 from 0.441 to 0.372 (-15.6%)",
        "  - Physical Dilemma: Is missingness actual line tampering or cellular telecom dropout?",
        "  - If telecom infrastructure improves, current models will experience severe performance drift",
        "• FACTOR 3: CLASS IMBALANCE MULTIPLIER (8.53% THEFT PREVALENCE)",
        "  - 10.7 benign consumers for every 1 theft consumer",
        "  - In 10,000 customers: 853 thieves vs 9,147 normal consumers",
        "  - A tiny 5% False Positive Rate produces 457 false alarms, capping Precision at ~42%",
        "• FACTOR 4: SAMPLE EFFICIENCY PLATEAU (LEARNING CURVE EVIDENCE)",
        "  - Training sample size scaling from 23,725 (80%) to 29,656 (100%) yielded only +0.010 F1 gain",
        "  - The model has exhausted tabular summary statistics; more rows will not help",
        "• FACTOR 5: GROUND TRUTH LABEL NOISE (UN-INSPECTED CONSUMERS)",
        "  - Class 0 includes normal consumers AND un-detected thieves (False Negative noise)",
        "  - Explains why ExtraTrees and high-capacity models easily overfit (Train F1=0.85, CV F1=0.41)"
    ],
    COLOR_FEAT, "ROOT CAUSE AUDIT"
)

# Connect Execution Panel -> Root Cause Panel
draw_arrow(48, 43, 52, 43, COLOR_FEAT, "Diagnostic Audit")

# ==============================================================================
# ROW 4: PRODUCTION ROADMAP & NEXT EXPERIMENTS (y: 2 to 26)
# ==============================================================================

# Bottom Panel 1: UCI Forecasting Pipeline Parallel Architecture
draw_node(
    4, 26, 44, 24,
    "PARALLEL WORKSTREAM: UCI LOAD FORECASTING PIPELINE",
    [
        "• Dataset: 34,896 Hourly System Load Readings (2011-2014)",
        "• Chronological Splits: Train (26,136h) → Val (4,344h) → Sealed Test (4,416h)",
        "• 29 Lagged & Rolling Temporal Features:",
        "  - Autoregressive Lags: lag_1h, lag_2h, lag_3h, lag_24h, lag_48h, lag_168h",
        "  - Rolling Windows: rolling_mean_24h, rolling_std_24h, rolling_max_168h",
        "  - Calendar Encodings: hour_of_day, day_of_week, month, is_weekend, cyclical sin/cos",
        "• Model Progression & Benchmarks:",
        "  - Frozen XGBoost Baseline: MAE = 210.89 kW, RMSE = 345.12 kW, R² = 0.954",
        "  - LightGBM Regressor (Tuned): MAE = 194.20 kW (-7.9%), RMSE = 318.45 kW (-7.7%)",
        "  - Multi-Horizon Forecasting: 1-hour ahead, 6-hour ahead, 24-hour ahead",
        "• Production FastAPI Serving:",
        "  - Endpoint: POST /api/v1/forecast/load",
        "  - Latency: <15ms inference, zero feature leakage"
    ],
    COLOR_PROD, "FORECASTING"
)

# Bottom Panel 2: The 3 Recommended Next Scientific Breakthroughs
draw_node(
    52, 26, 44, 24,
    "THE 3 RECOMMENDED NEXT EXPERIMENTS (BREAKING THE 0.45 F1 CEILING)",
    [
        "• EXPERIMENT 1: CONTRASTIVE TIMESERIES PRETRAINING (TS2Vec / TCN)",
        "  - Problem: Tabular summary statistics (mean, std, max) lose temporal sequence shape",
        "  - Solution: Train a 1D-CNN / Temporal Convolutional Network autoencoder",
        "  - Contrastive Loss learns sudden drop-offs, recovery, and periodic phase shifts",
        "  - Eliminates reliance on communication packet loss / missingness counts",
        "• EXPERIMENT 2: POSITIVE-UNLABELED (PU) LEARNING",
        "  - Problem: Label 0 contains undetected thieves who were never inspected by utility teams",
        "  - Solution: Treat Class 0 as Unlabeled (U) rather than Negative (N)",
        "  - Apply Elkan-Noto or Bagging-PU to reweight samples and avoid penalizing genuine theft",
        "• EXPERIMENT 3: OPERATIONAL COST-SENSITIVE UTILITY MATRIX",
        "  - Problem: Symmetric F1 treats false alarms and missed theft identically",
        "  - Solution: Optimize threshold based on monetary return:",
        "    Utility = (Stolen kWh Recovered * Tariff) - (Inspection Dispatch Cost * FP Count)",
        "  - Yields optimal business threshold rather than arbitrary statistical threshold"
    ],
    COLOR_EVAL, "RECOMMENDED NEXT STEPS"
)

# Save the master diagram
output_path = os.path.join(r"d:\watt-matic\gridbalance\graphs\optimization_v2", "sgcc_end_to_end_pipeline_mindmap.png")
plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
plt.close(fig)

print(f"High-resolution Mindmap successfully saved to: {output_path}")
