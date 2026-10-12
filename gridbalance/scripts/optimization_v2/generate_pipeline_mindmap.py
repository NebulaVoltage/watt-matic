import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Premium Ultrawide Layout: 38 inches wide x 44 inches tall, 300 DPI
# Canvas dimensions: 100 x 100 coordinate space
fig = plt.figure(figsize=(38, 44), dpi=300)
ax = fig.add_subplot(111)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Color Theme: Deep Modern Engineering Navy & Sleek Card Accents
BG_COLOR = "#0A0F1D"       # Deep Obsidian Slate
CARD_BG = "#131C31"        # Dark Navy Panel
BORDER_SUBTLE = "#1E293B"  # Card Boundary
TEXT_PRIMARY = "#FFFFFF"   # Pure White
TEXT_BODY = "#E2E8F0"      # Crisp Slate White
TEXT_MUTED = "#94A3B8"     # Soft Muted Gray
TEXT_CODE = "#38BDF8"      # Terminal Sky Blue
ACCENT_GREEN = "#10B981"   # Mint Emerald
ACCENT_AMBER = "#F59E0B"   # Warm Gold
ACCENT_ROSE = "#F43F5E"    # Alert Rose
ACCENT_PURPLE = "#A855F7"  # Electric Purple
ACCENT_BLUE = "#3B82F6"    # Cobalt Blue
ACCENT_CYAN = "#06B6D4"    # Precision Cyan

fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

# ==============================================================================
# 1. HEADER BANNER
# ==============================================================================
header_box = patches.FancyBboxPatch(
    (3, 93.5), 94, 4.8,
    boxstyle="round,pad=0.4,rounding_size=0.6",
    facecolor="#161F38", edgecolor="#6366F1", linewidth=2.5
)
ax.add_patch(header_box)

ax.text(50, 96.5, "GRIDBALANCE — END-TO-END PIPELINE & PERFORMANCE DIAGNOSTIC MINDMAP",
        fontsize=26, fontweight='heavy', color="#FFFFFF", ha='center', va='center')
ax.text(50, 94.7, "Complete Architecture: Ingestion → Zero-Leakage Splitting → Feature Engineering → Focal Loss GBDT → Sealed Test Audit",
        fontsize=15, fontweight='medium', color="#A5B4FC", ha='center', va='center')

# ==============================================================================
# CARD DRAWING HELPER WITH AUTOMATIC LINE-SPACING
# ==============================================================================
def draw_card(x, y, w, h, title, subtitle, items, border_color, badge="", badge_color="#38BDF8"):
    # Main Box Shadow & Body
    shadow = patches.FancyBboxPatch(
        (x + 0.2, y - h - 0.2), w, h,
        boxstyle="round,pad=0.3,rounding_size=0.5",
        facecolor="#050811", edgecolor="none", alpha=0.6
    )
    ax.add_patch(shadow)

    card = patches.FancyBboxPatch(
        (x, y - h), w, h,
        boxstyle="round,pad=0.3,rounding_size=0.5",
        facecolor=CARD_BG, edgecolor=border_color, linewidth=2.2
    )
    ax.add_patch(card)

    # Header Bar
    header_bar = patches.FancyBboxPatch(
        (x, y - 2.8), w, 2.8,
        boxstyle="round,pad=0.1,rounding_size=0.4",
        facecolor=border_color, edgecolor="none", alpha=0.20
    )
    ax.add_patch(header_bar)

    # Header Text
    ax.text(x + 1.2, y - 1.4, title, fontsize=15.0, fontweight='heavy', color=TEXT_PRIMARY, va='center')
    if badge:
        ax.text(x + w - 1.2, y - 1.4, badge, fontsize=11.0, fontweight='bold', color=badge_color,
                ha='right', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor="#0E1726", edgecolor=badge_color, lw=1.2))

    # Subtitle / Metadata row
    if subtitle:
        ax.text(x + 1.2, y - 3.7, subtitle, fontsize=11.5, fontweight='bold', color=TEXT_CODE, va='top')
        curr_y = y - 5.3
    else:
        curr_y = y - 3.8

    # Items rendering with dynamic line heights
    for item in items:
        if isinstance(item, tuple):
            prefix, text, style = item
            if style == "header":
                curr_y -= 0.3
                ax.text(x + 1.2, curr_y, text, fontsize=12.0, fontweight='bold', color="#F8FAFC", va='top')
                curr_y -= 1.6
            elif style == "success":
                ax.text(x + 1.6, curr_y, prefix, fontsize=11.0, fontweight='heavy', color=ACCENT_GREEN, va='top')
                ax.text(x + 3.4, curr_y, text, fontsize=11.0, color=TEXT_BODY, va='top')
                curr_y -= 1.45
            elif style == "alert":
                ax.text(x + 1.6, curr_y, prefix, fontsize=11.0, fontweight='heavy', color=ACCENT_ROSE, va='top')
                ax.text(x + 3.4, curr_y, text, fontsize=11.0, color=TEXT_BODY, va='top')
                curr_y -= 1.45
            elif style == "code":
                ax.text(x + 1.6, curr_y, prefix, fontsize=11.0, fontweight='bold', color=TEXT_CODE, va='top')
                ax.text(x + 3.4, curr_y, text, fontsize=11.0, color=TEXT_BODY, va='top')
                curr_y -= 1.45
            else:
                ax.text(x + 1.6, curr_y, prefix, fontsize=11.0, color=TEXT_MUTED, va='top')
                ax.text(x + 3.4, curr_y, text, fontsize=11.0, color=TEXT_BODY, va='top')
                curr_y -= 1.45
        else:
            ax.text(x + 1.2, curr_y, item, fontsize=11.0, color=TEXT_BODY, va='top')
            curr_y -= 1.45

def draw_flow_arrow(x1, y1, x2, y2, color, label=""):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=3.0, mutation_scale=22,
                        connectionstyle="arc3,rad=0.0")
    )
    if label:
        mx, my = (x1 + x2)/2, (y1 + y2)/2
        ax.text(mx, my, label, fontsize=11.0, fontweight='heavy', color=color,
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.25', facecolor=BG_COLOR, edgecolor=color, lw=1.2))

def draw_corner_connector(x1, y1, xmid, y2, x2, color, label=""):
    ax.plot([x1, xmid, xmid, x2], [y1, y1, y2, y2], color=color, lw=2.6)
    ax.annotate(
        "", xy=(x2, y2), xytext=(xmid + (0.6 if x2 > xmid else -0.6), y2),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=2.6, mutation_scale=20)
    )
    if label:
        ax.text(xmid, (y1 + y2)/2, label, fontsize=10.5, fontweight='heavy', color=color,
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor=BG_COLOR, edgecolor=color, lw=1.0))

# ==============================================================================
# ROW 1: 3 HIGH-LEVEL FLOW STAGES (y: 91 down to 73, height = 18.0)
# ==============================================================================

# CARD 1: Dual Ingestion & Preprocessing
draw_card(
    3.0, 91.0, 29.0, 18.0,
    "STAGE 1: DUAL DATASET INGESTION",
    "Source Telemetry & Initial Data Cleaning",
    [
        ("", "SGCC Electricity Theft Telemetry:", "header"),
        ("•", "Raw Volume: 42,372 meters × 1,036 dates (2014-2016)", "normal"),
        ("•", "Total Cells: 43,812,648 daily electricity measurements", "normal"),
        ("•", "Missingness: 25.64% overall (11,233,528 NaN values)", "alert"),
        ("•", "Target Ratio: 38,757 Normal (91.47%) vs 3,615 Theft (8.53%)", "normal"),
        ("✔", "Dropped 5 records with 100% missing data → 42,367 clean rows", "success"),
        ("", "UCI Electricity Hourly Load Demand:", "header"),
        ("•", "Raw Volume: 370 consumers aggregated to grid level", "normal"),
        ("•", "Clean Series: 34,896 hourly steps across 2011 to 2014", "normal"),
        ("✔", "Zero missing values | Clean physical continuous load (kWh)", "success")
    ],
    ACCENT_BLUE, "PHASE 1", ACCENT_BLUE
)

# CARD 2: Zero-Leakage Double-Sealed Splitting
draw_card(
    35.5, 91.0, 29.0, 18.0,
    "STAGE 2: DOUBLE-SEALED SPLITTING",
    "Strict Leak-Free Partitioning Architecture",
    [
        ("", "SGCC Stratified Partitioning (Seed = 42):", "header"),
        ("•", "Train Set (70%): N = 29,656 (27,128 Normal, 2,528 Theft)", "normal"),
        ("•", "Validation Set (15%): N = 6,355 (5,813 Normal, 542 Theft)", "normal"),
        ("•", "Sealed Test Set (15%): N = 6,356 (5,814 Normal, 542 Theft)", "normal"),
        ("!", "SEALED TEST VAULT: Kept completely isolated and untouched", "alert"),
        ("!", "Never inspected for feature selection, tuning, or thresholding", "alert"),
        ("", "UCI Temporal Walk-Forward Partitioning:", "header"),
        ("•", "Train (75%): 26,136 hourly observations (~3.0 years)", "normal"),
        ("•", "Validation (12.5%): 4,344 hourly observations (~6.0 months)", "normal"),
        ("•", "Sealed Test (12.5%): 4,416 hourly observations (~6.0 months)", "normal")
    ],
    ACCENT_PURPLE, "PHASE 2", ACCENT_PURPLE
)

# CARD 3: Feature Engineering & Imputation
draw_card(
    68.0, 91.0, 29.0, 18.0,
    "STAGE 3: ZERO-LEAKAGE FEATURE PIPELINE",
    "Training-Only Imputation & Signal Extraction",
    [
        ("", "Strict Train-Only Imputation Rule:", "header"),
        ("✔", "SimpleImputer medians computed strictly on Train (N=29,656)", "success"),
        ("✔", "Val and Test imputed with Train medians (Zero Data Leakage)", "success"),
        ("", "42-Feature Dimensionality Overview:", "header"),
        ("•", "Missingness (5 feats): missing_count, streak_count, longest_streak", "code"),
        ("•", "Dispersion (9 feats): mean, median, std_dev, cv, iqr, min, max", "code"),
        ("•", "Volatility (2 feats): mean_abs_daily_change, max_abs_daily_change", "code"),
        ("•", "Rolling Dynamics (24 feats): 12 monthly means + 12 monthly stds", "code"),
        ("•", "Ratios (2 feats): zero_ratio, peak_to_average_ratio", "code"),
        ("✔", "Reduced 18-Feature Subset: Captures 97% predictive power", "success")
    ],
    ACCENT_CYAN, "PHASE 3", ACCENT_CYAN
)

# Row 1 Connectors
draw_flow_arrow(32.0, 82.0, 35.5, 82.0, ACCENT_BLUE, "Split")
draw_flow_arrow(64.5, 82.0, 68.0, 82.0, ACCENT_PURPLE, "Extract")

# ==============================================================================
# ROW 2: 3 MODELING & EVALUATION STAGES (y: 70.0 down to 52.0, height = 18.0)
# ==============================================================================

# CARD 4: Diagnostics & Ablation
draw_card(
    3.0, 70.0, 29.0, 18.0,
    "STAGE 4: EMPIRICAL DIAGNOSTICS AUDIT",
    "Class Separability, Ablation & Learning Curves",
    [
        ("", "Phase 2 Class Separability Findings:", "header"),
        ("•", "Severe Class Overlap: 95% to 99.9% histogram overlap on consumption", "alert"),
        ("•", "No Linear Separator: Cohen's d < 0.40, KS-stat < 0.32 across all feats", "normal"),
        ("", "Phase 3 Feature Ablation Results (Val Split):", "header"),
        ("•", "Consumption Only (15 feats): F1 = 0.3725 | PR-AUC = 0.3520", "normal"),
        ("•", "Missingness Only (3 feats): F1 = 0.3128 | PR-AUC = 0.3332", "normal"),
        ("•", "Combined Baseline (18 feats): F1 = 0.4415 | PR-AUC = 0.4221", "success"),
        ("", "Phase 5 Learning Curve Plateau:", "header"),
        ("•", "20% Train: CV F1 = 0.328 | 80% Train: CV F1 = 0.393", "normal"),
        ("•", "100% Train: CV F1 = 0.404 (Sample size saturation confirmed)", "alert")
    ],
    ACCENT_AMBER, "PHASE 4", ACCENT_AMBER
)

# CARD 5: Model Training & Focal Loss
draw_card(
    35.5, 70.0, 29.0, 18.0,
    "STAGE 5: MODEL TRAINING & FOCAL LOSS",
    "Class-Weighted GBDTs & Custom Objectives",
    [
        ("", "Frozen Baseline Model (18 Features):", "header"),
        ("•", "Algorithm: XGBoost Classifier (scale_pos_weight = 10.72)", "normal"),
        ("•", "Parameters: max_depth = 7, learning_rate = 0.10, n_est = 150", "code"),
        ("•", "Validation Score: Val F1 = 0.4415 | Val PR-AUC = 0.4221", "normal"),
        ("", "Optimized Candidate: LightGBM with Focal Loss:", "header"),
        ("•", "Loss Function: FL(p_t) = -α * (1 - p_t)^γ * log(p_t)", "code"),
        ("•", "Hyperparameters: α = 0.85 (theft weight), γ = 2.0 (focusing)", "code"),
        ("•", "Gradient Tuning: Targets hard-to-classify borderline consumers", "success"),
        ("", "5-Fold CV Multi-Model Comparison:", "header"),
        ("•", "ExtraTrees: CV F1 = 0.2882 | XGBoost: CV F1 = 0.4036", "normal"),
        ("✔", "LightGBM Focal Loss: CV F1 = 0.4264 | CV PR-AUC = 0.4133", "success")
    ],
    ACCENT_ROSE, "PHASE 5", ACCENT_ROSE
)

# CARD 6: Threshold Selection & Sealed Test
draw_card(
    68.0, 70.0, 29.0, 18.0,
    "STAGE 6: THRESHOLDING & SEALED TEST GATE",
    "Operational Threshold Tuning & Vault Evaluation",
    [
        ("", "Validation Decision Threshold Optimization:", "header"),
        ("•", "Threshold Sweep: t ∈ [0.20, 0.80] scanned strictly on Val set", "code"),
        ("✔", "Optimal Threshold: t = 0.50 yields maximum Val F1 = 0.4438", "success"),
        ("", "Final Sealed Test Evaluation (Evaluated Exactly ONCE):", "header"),
        ("•", "Frozen XGBoost Baseline (18 feats, t = 0.50):", "normal"),
        ("  ", "Precision: 0.3275 | Recall: 0.5498 | F1: 0.4105 | PR-AUC: 0.3762", "normal"),
        ("•", "Candidate LightGBM Focal Loss (18 feats, t = 0.50):", "header"),
        ("  ", "Precision: 0.4200 (+28.2%) | Recall: 0.4797 (-12.7%)", "success"),
        ("  ", "F1 Score:  0.4479 (+9.1%)  | PR-AUC: 0.4178 (+11.1%)", "success"),
        ("  ", "ROC-AUC:   0.8429 (+2.3%)  | Brier Score: 0.0541 (-18.2%)", "success"),
        ("✔", "Decision Gate: Candidate survives sealed test verification", "success")
    ],
    ACCENT_GREEN, "PHASE 6", ACCENT_GREEN
)

# Row 2 Connectors & Inter-Row Connectors
draw_corner_connector(82.5, 73.0, 82.5, 71.5, 32.0, ACCENT_AMBER, "Audit")
draw_flow_arrow(32.0, 61.0, 35.5, 61.0, ACCENT_AMBER, "Train")
draw_flow_arrow(64.5, 61.0, 68.0, 61.0, ACCENT_ROSE, "Tune")

# ==============================================================================
# ROW 3: DETAILED DEEP-DIVE PANELS (y: 49.0 down to 26.0, height = 23.0)
# ==============================================================================

# PANEL 7: Orderwise & Linewise Execution Blueprint
draw_card(
    3.0, 49.0, 46.5, 23.0,
    "FULL ORDERWISE & LINEWISE EXECUTION BLUEPRINT",
    "Chronological Pipeline Implementation Details (Code-Level Protocol)",
    [
        ("", "PHASE 1: INGESTION & DATA INTEGRITY VERIFICATION", "header"),
        ("1.1", "Load raw sgcc_data.csv (42,372 rows) and uci_hourly_aggregate_load.csv (34,896 rows)", "code"),
        ("1.2", "Verify customer ID uniqueness (0 duplicate CONS_NO); verify non-negative readings", "code"),
        ("1.3", "Filter 5 records with 100% NaN missing records; preserve 42,367 clean customer rows", "code"),
        ("", "PHASE 2: ZERO-LEAKAGE STRATIFIED PARTITIONING", "header"),
        ("2.1", "Enforce global random_state = 42 across all random number generators", "code"),
        ("2.2", "Execute train_test_split: 70% Train (29,656), 15% Val (6,355), 15% Test (6,356)", "code"),
        ("2.3", "Lock sealed test indices in vault; block test access from all training scripts", "code"),
        ("", "PHASE 3: FEATURE IMPUTATION & REDUCTION", "header"),
        ("3.1", "Fit SimpleImputer(strategy='median') strictly on X_train (N = 29,656)", "code"),
        ("3.2", "Transform X_val and X_test using fitted training medians (strict zero-lookahead)", "code"),
        ("3.3", "Extract 42 tabular features; filter to frozen 18-feature candidate list", "code"),
        ("", "PHASE 4: 5-FOLD STRATIFIED CROSS-VALIDATION & FOCAL TRAINING", "header"),
        ("4.1", "Instantiate StratifiedKFold(n_splits=5, shuffle=True, random_state=42)", "code"),
        ("4.2", "Compile custom Focal Loss objective: grad = α*(1-p_t)^γ*(p-y), hess = α*(1-p_t)^γ*p*(1-p)*...", "code"),
        ("4.3", "Train LightGBM across 5 folds with early stopping (stopping_rounds=20)", "code"),
        ("", "PHASE 5: OPERATIONAL DECISION THRESHOLDING & INFERENCE SERVING", "header"),
        ("5.1", "Evaluate probability calibration on Validation set; sweep threshold t ∈ [0.20, 0.80]", "code"),
        ("5.2", "Select optimal threshold t = 0.50; serialize model bundle to models/optimization_v2/", "code"),
        ("5.3", "Evaluate sealed test set once; serve model via FastAPI /api/v1/detect/tampering", "code")
    ],
    ACCENT_PURPLE, "EXECUTION SPEC", ACCENT_PURPLE
)

# PANEL 8: Root Cause Analysis of Performance Ceiling
draw_card(
    50.5, 49.0, 46.5, 23.0,
    "WHY MODEL METRICS PLATEAU: 5 EMPIRICAL ROOT CAUSES",
    "Scientific Diagnostic Audit of the F1 ≈ 0.40 - 0.45 Performance Ceiling",
    [
        ("", "ROOT CAUSE 1: WEAK & OVERLAPPING CONSUMPTION DYNAMICS (~50% IMPACT)", "header"),
        ("•", "Consumption Overlap: Every single consumption feature exhibits 95% - 99.9% histogram overlap", "alert"),
        ("•", "Small Effect Sizes: No feature achieves Cohen's d > 0.40 or Kolmogorov-Smirnov stat > 0.32", "normal"),
        ("•", "Residential Volatility: Normal households with vacations or solar draw mimic theft profiles", "normal"),
        ("", "ROOT CAUSE 2: MEASUREMENT-SIGNAL ENTANGLEMENT (~30% IMPACT)", "header"),
        ("•", "Missingness Dependency: Longest missing streak has lowest overlap (81.8%) and highest importance", "alert"),
        ("•", "Ablation Drop: Excluding missingness drops Val F1 from 0.4415 to 0.3725 (-15.6%)", "alert"),
        ("•", "Operational Dilemma: Models detect telecom packet dropouts rather than genuine energy diversion", "normal"),
        ("", "ROOT CAUSE 3: CLASS IMBALANCE MULTIPLIER (8.53% THEFT PREVALENCE)", "header"),
        ("•", "10.7 Normal Consumers per 1 Thief: A 5% false positive rate produces 457 false alarms per 10k", "normal"),
        ("•", "Precision Cap: Mathematical consequence of the Bayes rate under noisy non-separable features", "normal"),
        ("", "ROOT CAUSE 4: FEATURE & DATA SATURATION (LEARNING CURVE EVIDENCE)", "header"),
        ("•", "Learning Plateau: Scaling train data from 80% to 100% (+5,931 rows) gained only +0.010 CV F1", "normal"),
        ("•", "Feature Saturation: Expanding from 18 to 42 features gained only +0.007 PR-AUC but increased overfit", "normal"),
        ("", "ROOT CAUSE 5: GROUND TRUTH LABEL NOISE (UN-INSPECTED THIEVES)", "header"),
        ("•", "Selection Bias: Label 0 contains undetected thieves who were never audited by inspection teams", "alert"),
        ("•", "Pseudo-Negative Penalty: Model penalizes legitimate theft detections as false positives", "normal")
    ],
    ACCENT_AMBER, "ROOT CAUSE AUDIT", ACCENT_AMBER
)

# ==============================================================================
# ROW 4: PARALLEL PIPELINE & RESEARCH ROADMAP (y: 24.5 down to 2.5, height = 22.0)
# ==============================================================================

# PANEL 9: Parallel UCI Forecasting Pipeline
draw_card(
    3.0, 24.5, 46.5, 22.0,
    "PARALLEL WORKSTREAM: UCI LOAD FORECASTING PIPELINE",
    "Chronological Regressive Architecture for Real-Time Grid Load Prediction",
    [
        ("", "UCI Dataset Overview & Processing:", "header"),
        ("•", "Dataset: 34,896 Hourly System Load Readings (Aggregated from 370 smart meters)", "normal"),
        ("•", "Chronological Splits: Train (26,136h, 75%) → Val (4,344h, 12.5%) → Sealed Test (4,416h, 12.5%)", "normal"),
        ("", "29 Lagged & Rolling Temporal Feature Architecture:", "header"),
        ("•", "Autoregressive Lags: lag_1h, lag_2h, lag_3h, lag_24h, lag_48h, lag_168h (weekly cycle)", "code"),
        ("•", "Rolling Window Aggregations: rolling_mean_24h, rolling_std_24h, rolling_max_168h", "code"),
        ("•", "Calendar & Cyclical Encodings: hour, day_of_week, sin_hour, cos_hour, is_weekend, holiday", "code"),
        ("", "Forecasting Models & Production Benchmarks:", "header"),
        ("•", "Frozen Baseline Regressor (XGBoost): MAE = 210.89 kW | RMSE = 345.12 kW | R² = 0.954", "normal"),
        ("✔", "Tuned LightGBM Regressor: MAE = 194.20 kW (-7.9%) | RMSE = 318.45 kW (-7.7%) | R² = 0.963", "success"),
        ("•", "Multi-Horizon Projections: Next-hour load (1h), Peak-shift load (6h), Day-ahead load (24h)", "normal"),
        ("✔", "FastAPI Production Endpoint: POST /api/v1/forecast/load (<15ms latency)", "success")
    ],
    ACCENT_CYAN, "LOAD FORECASTING", ACCENT_CYAN
)

# PANEL 10: The 3 Recommended Scientific Breakthroughs
draw_card(
    50.5, 24.5, 46.5, 22.0,
    "THE 3 RECOMMENDED NEXT EXPERIMENTS (BREAKING THE 0.45 F1 CEILING)",
    "Actionable Scientific Methodology to Overcome Data & Separability Bottlenecks",
    [
        ("", "RECOMMENDED EXPERIMENT 1: CONTRASTIVE TIMESERIES PRETRAINING (TS2Vec / TCN)", "header"),
        ("•", "Core Problem: Scalar summary stats (mean, std, min, max) destroy sequence micro-patterns", "alert"),
        ("•", "Proposed Solution: Train a 1D-CNN / Temporal Convolutional Network autoencoder on raw days", "normal"),
        ("•", "Contrastive Learning: Captures sudden drop-offs, sudden recovery, and seasonal phase-shifts", "normal"),
        ("✔", "Expected Value: Discovers true diversion shapes without relying on missingness streak counts", "success"),
        ("", "RECOMMENDED EXPERIMENT 2: POSITIVE-UNLABELED (PU) LEARNING FRAMEWORK", "header"),
        ("•", "Core Problem: Class 0 is an un-audited mixture of normal users and undetected thieves", "alert"),
        ("•", "Proposed Solution: Implement Elkan-Noto or Bagging Positive-Unlabeled (PU) classification", "normal"),
        ("•", "Reweighting Scheme: Re-balances loss weights on unlabeled samples to prevent false penalty", "normal"),
        ("✔", "Expected Value: Recovers 15-20% hidden precision by stopping false penalties on true thieves", "success"),
        ("", "RECOMMENDED EXPERIMENT 3: OPERATIONAL COST-SENSITIVE UTILITY MATRIX", "header"),
        ("•", "Core Problem: Symmetric F1-score treats inspection dispatch cost identically to missed theft", "alert"),
        ("•", "Proposed Solution: Define utility function: Utility = (Recovered_kWh * Tariff) - (Inspection_Cost * FP)", "code"),
        ("✔", "Expected Value: Determines optimal threshold based on grid revenue rather than abstract F1", "success")
    ],
    ACCENT_GREEN, "RESEARCH ROADMAP", ACCENT_GREEN
)

# Output directory and save
output_path = os.path.join(r"d:\watt-matic\gridbalance\graphs\optimization_v2", "sgcc_end_to_end_pipeline_mindmap.png")
plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
plt.close(fig)

print(f"High-resolution Mindmap successfully regenerated and saved to: {output_path}")
