import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
import xgboost as xgb
import lightgbm as lgb

base_dir = r"D:\watt-matic\gridbalance"
uci_data_path = os.path.join(base_dir, "data", "processed", "uci_hourly_aggregate_load.csv")
out_graphs = os.path.join(base_dir, "graphs", "optimization_v2", "phase2")
out_results = os.path.join(base_dir, "results", "optimization_v2")
os.makedirs(out_graphs, exist_ok=True)
os.makedirs(out_results, exist_ok=True)

def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_pred - y_true) / np.where(denom == 0, 1e-6, denom)
    return np.mean(diff) * 100.0

print("=== 1. LOADING UCI DATASET AND EXTRACTING FEATURES ===")
df = pd.read_csv(uci_data_path, parse_dates=[0], index_col=0)
df = df.sort_index()

# 29 baseline features
lags = [1, 2, 3, 6, 12, 24, 48, 72, 168]
for l in lags:
    df[f"load_t_minus_{l}"] = df["aggregate_load_kwh"].shift(l)
df["load_t"] = df["aggregate_load_kwh"]

# Rolling features
for r in [3, 6, 12, 24, 48, 168]:
    df[f"rolling_mean_{r}"] = df["aggregate_load_kwh"].rolling(r).mean()
df["rolling_std_24"] = df["aggregate_load_kwh"].rolling(24).std()
df["rolling_std_168"] = df["aggregate_load_kwh"].rolling(168).std()
df["rolling_min_24"] = df["aggregate_load_kwh"].rolling(24).min()
df["rolling_max_24"] = df["aggregate_load_kwh"].rolling(24).max()

# Calendar & cyclic
hour = df.index.hour
dow = df.index.dayofweek
df["hour"] = hour
df["day_of_week"] = dow
df["day_of_month"] = df.index.day
df["month"] = df.index.month
df["week_of_year"] = df.index.isocalendar().week.astype(int)
df["sin_hour"] = np.sin(2 * np.pi * hour / 24.0)
df["cos_hour"] = np.cos(2 * np.pi * hour / 24.0)
df["sin_day_of_week"] = np.sin(2 * np.pi * dow / 7.0)
df["cos_day_of_week"] = np.cos(2 * np.pi * dow / 7.0)

# Target: next hour load
df["target_t_plus_1"] = df["aggregate_load_kwh"].shift(-1)
# Difference target: delta(t+1) = y(t+1) - y(t)
df["target_delta"] = df["target_t_plus_1"] - df["load_t"]

# Additional difference features for difference forecasting
df["delta_lag_1"] = df["load_t"] - df["load_t_minus_1"]
df["delta_lag_24"] = df["load_t"] - df["load_t_minus_24"]
df["delta_lag_168"] = df["load_t"] - df["load_t_minus_168"]

# Drop NaNs created by lag 168
df_clean = df.dropna().copy()
print(f"Cleaned dataset shape: {df_clean.shape}")

feature_cols_29 = [
    'load_t', 'load_t_minus_1', 'load_t_minus_2', 'load_t_minus_3',
    'load_t_minus_6', 'load_t_minus_12', 'load_t_minus_24', 'load_t_minus_48',
    'load_t_minus_72', 'load_t_minus_168', 'rolling_mean_3', 'rolling_mean_6',
    'rolling_mean_12', 'rolling_mean_24', 'rolling_mean_48', 'rolling_mean_168',
    'rolling_std_24', 'rolling_std_168', 'rolling_min_24', 'rolling_max_24',
    'hour', 'day_of_week', 'day_of_month', 'month', 'week_of_year',
    'sin_hour', 'cos_hour', 'sin_day_of_week', 'cos_day_of_week'
]
feature_cols_diff = feature_cols_29 + ["delta_lag_1", "delta_lag_24", "delta_lag_168"]

# Partitions
train_mask = (df_clean.index <= '2013-12-31 23:00:00')
val_mask = (df_clean.index >= '2014-01-01 00:00:00') & (df_clean.index <= '2014-06-30 23:00:00')
test_mask = (df_clean.index >= '2014-07-01 00:00:00') & (df_clean.index <= '2014-12-31 23:00:00')

print(f"Train samples: {train_mask.sum()}, Val samples: {val_mask.sum()}, Test samples: {test_mask.sum()}")

print("\n=== EXPERIMENT 4: UCI CV REGIME AUDIT (ANNUAL STATS & TSCV COMPARISON) ===")
# Annual stats
annual_stats = []
for yr in [2011, 2012, 2013, 2014]:
    s_yr = df.loc[df.index.year == yr, "aggregate_load_kwh"]
    annual_stats.append({
        "Year": yr,
        "Hours_Count": len(s_yr),
        "Mean_Load_kWh": round(s_yr.mean(), 2),
        "Std_Load_kWh": round(s_yr.std(), 2),
        "Min_Load_kWh": round(s_yr.min(), 2),
        "Max_Load_kWh": round(s_yr.max(), 2),
        "Operational_Status": "Partial Onboarding" if yr == 2011 else "Full Network Operational"
    })
df_annual = pd.DataFrame(annual_stats)
print("--- Annual Aggregate Load Statistics ---")
print(df_annual.to_string(index=False))
df_annual.to_csv(os.path.join(out_results, "uci_annual_load_statistics.csv"), index=False)

# Protocol A: Regime-Aware CV (2012-2013 only, stationary period)
mask_cv_a = (df_clean.index >= '2012-01-01 00:00:00') & (df_clean.index <= '2013-12-31 23:00:00')
df_cv_a = df_clean.loc[mask_cv_a]

# Protocol B: Full-History Stress Evaluation (2011-2013, includes onboarding jump)
mask_cv_b = train_mask
df_cv_b = df_clean.loc[mask_cv_b]

def run_tscv_audit(df_subset, protocol_name):
    tscv = TimeSeriesSplit(n_splits=5)
    records = []
    X_sub = df_subset[feature_cols_29]
    y_sub = df_subset["target_t_plus_1"]
    
    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_sub)):
        tr_start, tr_end = X_sub.index[tr_idx].min().date(), X_sub.index[tr_idx].max().date()
        va_start, va_end = X_sub.index[va_idx].min().date(), X_sub.index[va_idx].max().date()
        
        m = xgb.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1)
        m.fit(X_sub.iloc[tr_idx], y_sub.iloc[tr_idx])
        preds = m.predict(X_sub.iloc[va_idx])
        
        y_true = y_sub.iloc[va_idx].values
        mae = mean_absolute_error(y_true, preds)
        rmse = np.sqrt(mean_squared_error(y_true, preds))
        r2 = r2_score(y_true, preds)
        sm = smape(y_true, preds)
        
        records.append({
            "Protocol": protocol_name,
            "Fold": fold + 1,
            "Train_Window": f"{tr_start} to {tr_end}",
            "Val_Window": f"{va_start} to {va_end}",
            "Val_Samples": len(va_idx),
            "MAE_kWh": round(mae, 2),
            "RMSE_kWh": round(rmse, 2),
            "R2": round(r2, 4),
            "sMAPE_pct": round(sm, 2)
        })
    return pd.DataFrame(records)

df_tscv_a = run_tscv_audit(df_cv_a, "Protocol A (Regime-Aware 2012-2013)")
df_tscv_b = run_tscv_audit(df_cv_b, "Protocol B (Full-History Stress 2011-2013)")

df_tscv_all = pd.concat([df_tscv_a, df_tscv_b], ignore_index=True)
print("\n--- TimeSeriesSplit Comparison: Protocol A vs Protocol B ---")
print(df_tscv_all.to_string(index=False))
df_tscv_all.to_csv(os.path.join(out_results, "uci_tscv_regime_comparison.csv"), index=False)

# Plot TSCV fold MAE comparison
plt.figure(figsize=(9, 5), dpi=300)
sns.barplot(data=df_tscv_all, x="Fold", y="MAE_kWh", hue="Protocol", palette="Set2")
plt.title("UCI TimeSeriesSplit Cross-Validation: Regime-Aware vs Full-History", fontsize=13, fontweight="bold")
plt.ylabel("Validation MAE (kWh)", fontsize=11, fontweight="semibold")
plt.xlabel("Chronological CV Fold", fontsize=11, fontweight="semibold")
plt.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(out_graphs, "uci_tscv_regime_comparison.png"))
plt.close()

print("\n=== EXPERIMENT 5: UCI DIFFERENCE FORECASTING ===")
X_train = df_clean.loc[train_mask]
X_val = df_clean.loc[val_mask]
X_test = df_clean.loc[test_mask]

# 1. XGBoost Direct
m_xgb_dir = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1)
m_xgb_dir.fit(X_train[feature_cols_29], X_train["target_t_plus_1"])
val_pred_xgb_dir = m_xgb_dir.predict(X_val[feature_cols_29])
test_pred_xgb_dir = m_xgb_dir.predict(X_test[feature_cols_29])

# 2. XGBoost Difference
m_xgb_diff = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1)
m_xgb_diff.fit(X_train[feature_cols_diff], X_train["target_delta"])
val_delta_xgb = m_xgb_diff.predict(X_val[feature_cols_diff])
test_delta_xgb = m_xgb_diff.predict(X_test[feature_cols_diff])
val_pred_xgb_diff = X_val["load_t"].values + val_delta_xgb
test_pred_xgb_diff = X_test["load_t"].values + test_delta_xgb

# 3. LightGBM Direct
m_lgb_dir = lgb.LGBMRegressor(n_estimators=300, num_leaves=63, learning_rate=0.05, random_state=42, verbose=-1, n_jobs=-1)
m_lgb_dir.fit(X_train[feature_cols_29], X_train["target_t_plus_1"])
val_pred_lgb_dir = m_lgb_dir.predict(X_val[feature_cols_29])
test_pred_lgb_dir = m_lgb_dir.predict(X_test[feature_cols_29])

# 4. LightGBM Difference
m_lgb_diff = lgb.LGBMRegressor(n_estimators=300, num_leaves=63, learning_rate=0.05, random_state=42, verbose=-1, n_jobs=-1)
m_lgb_diff.fit(X_train[feature_cols_diff], X_train["target_delta"])
val_delta_lgb = m_lgb_diff.predict(X_val[feature_cols_diff])
test_delta_lgb = m_lgb_diff.predict(X_test[feature_cols_diff])
val_pred_lgb_diff = X_val["load_t"].values + val_delta_lgb
test_pred_lgb_diff = X_test["load_t"].values + test_delta_lgb

# 5. Ridge Regression Direct
m_ridge = Ridge(alpha=1.0, random_state=42)
m_ridge.fit(X_train[feature_cols_29], X_train["target_t_plus_1"])
val_pred_ridge = m_ridge.predict(X_val[feature_cols_29])
test_pred_ridge = m_ridge.predict(X_test[feature_cols_29])

def get_metrics_row(y_true, y_pred, model_name, split):
    return {
        "Model": model_name,
        "Split": split,
        "MAE_kWh": round(mean_absolute_error(y_true, y_pred), 2),
        "RMSE_kWh": round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
        "R2": round(r2_score(y_true, y_pred), 4),
        "sMAPE_pct": round(smape(y_true, y_pred), 2)
    }

models_eval = [
    ("XGBoost Direct (29 Feat)", val_pred_xgb_dir, test_pred_xgb_dir),
    ("XGBoost Difference (Delta)", val_pred_xgb_diff, test_pred_xgb_diff),
    ("LightGBM Direct (29 Feat)", val_pred_lgb_dir, test_pred_lgb_dir),
    ("LightGBM Difference (Delta)", val_pred_lgb_diff, test_pred_lgb_diff),
    ("Ridge Regression (29 Feat)", val_pred_ridge, test_pred_ridge)
]

exp5_records = []
for name, v_pred, t_pred in models_eval:
    exp5_records.append(get_metrics_row(X_val["target_t_plus_1"].values, v_pred, name, "Validation (H1 2014)"))
    exp5_records.append(get_metrics_row(X_test["target_t_plus_1"].values, t_pred, name, "Sealed Test (H2 2014)"))

df_exp5 = pd.DataFrame(exp5_records)
print("\n--- Experiment 5 Model Performance Comparison ---")
print(df_exp5.to_string(index=False))
df_exp5.to_csv(os.path.join(out_results, "uci_exp5_difference_forecasting.csv"), index=False)

# Error analysis during Rising vs Falling load periods on Test set
actual_delta = X_test["target_delta"].values
rising_mask = actual_delta > 0
falling_mask = actual_delta < 0

period_analysis = []
for name, _, t_pred in models_eval:
    err_rise = np.abs(X_test.loc[rising_mask, "target_t_plus_1"].values - t_pred[rising_mask])
    err_fall = np.abs(X_test.loc[falling_mask, "target_t_plus_1"].values - t_pred[falling_mask])
    period_analysis.append({
        "Model": name,
        "Rising_Period_MAE_kWh": round(np.mean(err_rise), 2),
        "Falling_Period_MAE_kWh": round(np.mean(err_fall), 2),
        "Overall_Test_MAE_kWh": round(mean_absolute_error(X_test["target_t_plus_1"].values, t_pred), 2)
    })
df_periods = pd.DataFrame(period_analysis)
print("\n--- Rising vs Falling Load Error Analysis (Test Set) ---")
print(df_periods.to_string(index=False))
df_periods.to_csv(os.path.join(out_results, "uci_exp5_rising_falling_error.csv"), index=False)

# Plot Forecast comparison on a 2-week slice
sample_slice = slice(500, 836) # 2 weeks (336 hours)
plt.figure(figsize=(14, 5), dpi=300)
plt.plot(X_test.index[sample_slice], X_test["target_t_plus_1"].iloc[sample_slice], label="Actual Demand (kWh)", color="black", lw=1.5)
plt.plot(X_test.index[sample_slice], test_pred_xgb_dir[sample_slice], label="XGBoost Direct", color="blue", lw=1.2, linestyle="--")
plt.plot(X_test.index[sample_slice], test_pred_xgb_diff[sample_slice], label="XGBoost Difference (Delta)", color="crimson", lw=1.2, linestyle=":")
plt.title("UCI Electricity Load Forecast: Direct vs Difference Forecasting (2-Week Sample)", fontsize=13, fontweight="bold")
plt.ylabel("Hourly Grid Load (kWh)", fontsize=11, fontweight="semibold")
plt.legend(loc="upper right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_graphs, "uci_difference_forecasting_comparison.png"))
plt.close()

print("\n=== EXPERIMENT 6: UCI EMA INFERENCE WARMUP AUDIT ===")
# Evaluate warmup error of finite-window EMA against ground-truth continuous EMA
s_full = df["aggregate_load_kwh"]
true_ema_6 = s_full.ewm(span=6, adjust=False).mean()
true_ema_24 = s_full.ewm(span=24, adjust=False).mean()

# Test windows: at each test timestamp, compute finite EMA using W past hours
test_timestamps = df_clean.loc[test_mask].index[::10] # sample every 10 hours for efficiency (440 points)
window_lengths = [24, 48, 72, 120, 168, 336]
ema_warmup_records = []

for w in window_lengths:
    diffs_6, diffs_24 = [], []
    for ts in test_timestamps:
        loc = df.index.get_loc(ts)
        finite_s = s_full.iloc[loc - w + 1 : loc + 1]
        finite_ema_6 = finite_s.ewm(span=6, adjust=False).mean().iloc[-1]
        finite_ema_24 = finite_s.ewm(span=24, adjust=False).mean().iloc[-1]
        
        diffs_6.append(abs(finite_ema_6 - true_ema_6.iloc[loc]))
        diffs_24.append(abs(finite_ema_24 - true_ema_24.iloc[loc]))
        
    ema_warmup_records.append({
        "History_Window_Hours": w,
        "EMA_6_Max_Error_kWh": round(np.max(diffs_6), 4),
        "EMA_6_Mean_Error_kWh": round(np.mean(diffs_6), 4),
        "EMA_24_Max_Error_kWh": round(np.max(diffs_24), 4),
        "EMA_24_Mean_Error_kWh": round(np.mean(diffs_24), 4),
        "Warmup_Status": "Saturated (<0.01 kWh error)" if np.mean(diffs_24) < 0.01 else "Warmup Bias Present"
    })

df_warmup = pd.DataFrame(ema_warmup_records)
print("--- EMA Warmup Error as a Function of Finite History Window ---")
print(df_warmup.to_string(index=False))
df_warmup.to_csv(os.path.join(out_results, "uci_ema_inference_warmup_audit.csv"), index=False)

# Plot EMA Warmup Convergence
plt.figure(figsize=(8, 4.5), dpi=300)
plt.plot(df_warmup["History_Window_Hours"], df_warmup["EMA_6_Mean_Error_kWh"], marker="o", label="EMA 6 (Mean Error kWh)", lw=2, color="blue")
plt.plot(df_warmup["History_Window_Hours"], df_warmup["EMA_24_Mean_Error_kWh"], marker="s", label="EMA 24 (Mean Error kWh)", lw=2, color="crimson")
plt.axvline(168, color="gray", linestyle="--", label="Default API History (168h)")
plt.xlabel("Finite History Window (Hours)", fontsize=11, fontweight="semibold")
plt.ylabel("Mean Discrepancy from Continuous EMA (kWh)", fontsize=11, fontweight="semibold")
plt.title("Convergence of Finite History EMA to Continuous EMA State", fontsize=13, fontweight="bold")
plt.yscale("log")
plt.legend(loc="upper right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_graphs, "uci_ema_warmup_convergence.png"))
plt.close()

print("\n--- FINISHED ALL UCI EXPERIMENTS SUCCESSFULLY ---")
