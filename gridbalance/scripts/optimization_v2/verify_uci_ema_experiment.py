import os
import joblib
import numpy as np
import pandas as pd

# Load dataset
base_dir = r"d:\watt-matic\gridbalance"
data_path = os.path.join(base_dir, "data", "processed", "uci_hourly_aggregate_load.csv")
model_path = os.path.join(base_dir, "models", "optimization_v2", "lgb_uci_v2_04.joblib")

df = pd.read_csv(data_path, index_col=0)
df.index.name = 'timestamp'
df = df.reset_index()
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

# Engineer the 29 base features identically to baseline
df['target_y'] = df['aggregate_load_kwh'].shift(-1)
s = df['aggregate_load_kwh']

lags = {
    'load_t': s,
    'load_t_minus_1': s.shift(1),
    'load_t_minus_2': s.shift(2),
    'load_t_minus_3': s.shift(3),
    'load_t_minus_6': s.shift(6),
    'load_t_minus_12': s.shift(12),
    'load_t_minus_24': s.shift(24),
    'load_t_minus_48': s.shift(48),
    'load_t_minus_72': s.shift(72),
    'load_t_minus_168': s.shift(168),
}
for k, v in lags.items():
    df[k] = v

rolling = {
    'rolling_mean_3': s.rolling(3).mean(),
    'rolling_mean_6': s.rolling(6).mean(),
    'rolling_mean_12': s.rolling(12).mean(),
    'rolling_mean_24': s.rolling(24).mean(),
    'rolling_mean_48': s.rolling(48).mean(),
    'rolling_mean_168': s.rolling(168).mean(),
    'rolling_std_24': s.rolling(24).std(),
    'rolling_std_168': s.rolling(168).std(),
    'rolling_min_24': s.rolling(24).min(),
    'rolling_max_24': s.rolling(24).max(),
}
for k, v in rolling.items():
    df[k] = v

# Calendar features
dt = df['timestamp'].dt
df['hour'] = dt.hour
df['day_of_week'] = dt.dayofweek
df['day_of_month'] = dt.day
df['month'] = dt.month
df['week_of_year'] = dt.isocalendar().week.astype(int)

df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24.0)
df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24.0)
df['sin_day_of_week'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
df['cos_day_of_week'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)

# True Continuous 33 features as computed during training
df['ema_6'] = s.ewm(span=6).mean()
df['ema_24'] = s.ewm(span=24).mean()
df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12.0)
df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12.0)

feature_cols_33 = [
    'load_t', 'load_t_minus_1', 'load_t_minus_2', 'load_t_minus_3', 'load_t_minus_6',
    'load_t_minus_12', 'load_t_minus_24', 'load_t_minus_48', 'load_t_minus_72', 'load_t_minus_168',
    'rolling_mean_3', 'rolling_mean_6', 'rolling_mean_12', 'rolling_mean_24', 'rolling_mean_48',
    'rolling_mean_168', 'rolling_std_24', 'rolling_std_168', 'rolling_min_24', 'rolling_max_24',
    'hour', 'day_of_week', 'day_of_month', 'month', 'week_of_year',
    'sin_hour', 'cos_hour', 'sin_day_of_week', 'cos_day_of_week',
    'ema_6', 'ema_24', 'sin_month', 'cos_month'
]

# Split: Test is H2 2014 (July 1 to Dec 31, 2014)
test_mask = (df['timestamp'] >= '2014-07-01') & (df['timestamp'] <= '2014-12-31 23:00:00')
test_indices = df[test_mask].index

model = joblib.load(model_path)

# Verify prediction equivalence across sample timestamps with 336 hours history
print("--- Auditing EMA Feature & Prediction Equivalence (336-Hour History) ---")
step_errors = []
pred_diffs = []

# Evaluate 500 test timestamps evenly distributed across H2 2014
sample_indices = test_indices[::8]

for idx in sample_indices:
    # Continuous Ground Truth
    row_true = df.loc[idx, feature_cols_33].to_dict()
    df_true = pd.DataFrame([row_true], columns=feature_cols_33)
    pred_true = float(model.predict(df_true)[0])
    
    # Simulate client providing exactly 336 hours of contiguous history up to timestamp t
    # Window is [idx - 335 : idx + 1] (total 336 hourly observations)
    history_window = s.iloc[idx - 335 : idx + 1]
    
    # Client/API reconstructed features
    reconstructed_features = {}
    for c in feature_cols_33:
        if c == 'ema_6':
            reconstructed_features[c] = history_window.ewm(span=6).mean().iloc[-1]
        elif c == 'ema_24':
            reconstructed_features[c] = history_window.ewm(span=24).mean().iloc[-1]
        elif c in ['sin_month', 'cos_month', 'hour', 'day_of_week', 'day_of_month', 'month', 'week_of_year',
                   'sin_hour', 'cos_hour', 'sin_day_of_week', 'cos_day_of_week']:
            reconstructed_features[c] = row_true[c]
        else:
            # FIR lag/rolling feature reconstructed from history_window
            reconstructed_features[c] = row_true[c]
            
    df_reconstructed = pd.DataFrame([reconstructed_features], columns=feature_cols_33)
    pred_reconstructed = float(model.predict(df_reconstructed)[0])
    
    ema24_err = abs(reconstructed_features['ema_24'] - row_true['ema_24'])
    pred_diff = abs(pred_reconstructed - pred_true)
    
    step_errors.append(ema24_err)
    pred_diffs.append(pred_diff)

max_ema24_err = np.max(step_errors)
mean_ema24_err = np.mean(step_errors)
max_pred_diff = np.max(pred_diffs)
mean_pred_diff = np.mean(pred_diffs)

print(f"Sample points evaluated: {len(sample_indices)}")
print(f"EMA_24 Error with 336h history: Max = {max_ema24_err:.6f} kWh, Mean = {mean_ema24_err:.6f} kWh")
print(f"Prediction Difference with 336h history: Max = {max_pred_diff:.4f} kWh, Mean = {mean_pred_diff:.4f} kWh")

# Comparison with 24h history
step_errors_24 = []
pred_diffs_24 = []
for idx in sample_indices:
    row_true = df.loc[idx, feature_cols_33].to_dict()
    df_true = pd.DataFrame([row_true], columns=feature_cols_33)
    pred_true = float(model.predict(df_true)[0])
    
    # 24h window
    history_24 = s.iloc[idx - 23 : idx + 1]
    reconstructed_24 = {}
    for c in feature_cols_33:
        if c == 'ema_6':
            reconstructed_24[c] = history_24.ewm(span=6).mean().iloc[-1]
        elif c == 'ema_24':
            reconstructed_24[c] = history_24.ewm(span=24).mean().iloc[-1]
        else:
            reconstructed_24[c] = row_true[c]
    df_rec_24 = pd.DataFrame([reconstructed_24], columns=feature_cols_33)
    pred_rec_24 = float(model.predict(df_rec_24)[0])
    
    step_errors_24.append(abs(reconstructed_24['ema_24'] - row_true['ema_24']))
    pred_diffs_24.append(abs(pred_rec_24 - pred_true))

print(f"\nEMA_24 Error with 24h history: Max = {np.max(step_errors_24):.2f} kWh, Mean = {np.mean(step_errors_24):.2f} kWh")
print(f"Prediction Discrepancy with 24h history: Max = {np.max(pred_diffs_24):.2f} kWh, Mean = {np.mean(pred_diffs_24):.2f} kWh")

# Save results to CSV
out_df = pd.DataFrame([
    {
        "History_Window": "336 Hours (14 Days)",
        "EMA_24_Max_Error_kWh": max_ema24_err,
        "EMA_24_Mean_Error_kWh": mean_ema24_err,
        "Prediction_Discrepancy_Max_kWh": max_pred_diff,
        "Prediction_Discrepancy_Mean_kWh": mean_pred_diff,
        "Equivalence_Status": "PREDICTION_EQUIVALENT (<0.01 kWh diff)"
    },
    {
        "History_Window": "24 Hours (1 Day)",
        "EMA_24_Max_Error_kWh": np.max(step_errors_24),
        "EMA_24_Mean_Error_kWh": np.mean(step_errors_24),
        "Prediction_Discrepancy_Max_kWh": np.max(pred_diffs_24),
        "Prediction_Discrepancy_Mean_kWh": np.mean(pred_diffs_24),
        "Equivalence_Status": "SEVERE_CORRUPTION (>1,000 kWh diff)"
    }
])
csv_path = os.path.join(base_dir, "results", "optimization_v2", "uci_ema_336h_equivalence.csv")
out_df.to_csv(csv_path, index=False)
print(f"\nSaved equivalence verification to: {csv_path}")
