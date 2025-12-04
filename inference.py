import os
import numpy as np
import pandas as pd
import json
import pickle
import warnings
from datetime import datetime, timedelta
import requests
from tensorflow.keras.models import load_model

warnings.filterwarnings("ignore")

# API Configuration
API_BASE_URL = "https://le3tvo1cgc.execute-api.us-east-1.amazonaws.com/prod/get-data"
DEVICE_TABLES = ["device5", "device6", "device7", "device8"]

# Model paths (directly in current directory)
AUTOENCODER_PATH = "autoencoder_best.h5"
ISOLATION_FOREST_PATH = "isolation_forest.pkl"
SCALER_PATH = "scaler.pkl"
FEATURE_NAMES_PATH = "feature_names.json"
CONFIG_PATH = "config.json"

# Inference parameters
RECENT_DATA_WINDOW_MINUTES = 20
THRESHOLD_MULTIPLIER = 2.5
MIN_ANOMALY_DURATION = 3
ROLL_WIN = 5

# Threshold parameters
CURRENT_IQR_MULTIPLIER = 2.5
CURRENT_STD_MULTIPLIER = 3.0
CURRENT_PERCENTILE_LOW = 1
CURRENT_PERCENTILE_HIGH = 99
CURRENT_RATE_PERCENTILE = 95

# Output directory
OUTPUT_DIR = "inference_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("BATTERY ANOMALY DETECTION - INFERENCE MODE")
print("="*80)

print("\n[1/5] Loading trained models and artifacts...")

# Load autoencoder with custom objects
from tensorflow.keras import losses, metrics

custom_objects = {
    'mse': losses.MeanSquaredError(),
    'mae': metrics.MeanAbsoluteError()
}

if os.path.exists(AUTOENCODER_PATH):
    try:
        autoencoder = load_model(AUTOENCODER_PATH, custom_objects=custom_objects, compile=False)
        # Recompile the model
        from tensorflow.keras.optimizers import Adam
        autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        print(f" Loaded autoencoder from: {AUTOENCODER_PATH}")
    except Exception as e:
        print(f" Error loading with custom_objects: {e}")
        print(f" Trying alternative loading method...")
        try:
            autoencoder = load_model(AUTOENCODER_PATH, compile=False)
            autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
            print(f" Loaded autoencoder (alternative method)")
        except Exception as e2:
            raise Exception(f"Failed to load autoencoder: {e2}")
else:
    raise FileNotFoundError(f"Autoencoder not found: {AUTOENCODER_PATH}")

# Load scaler
if os.path.exists(SCALER_PATH):
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print(f" Loaded scaler from: {SCALER_PATH}")
else:
    raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")

# Load feature names
if os.path.exists(FEATURE_NAMES_PATH):
    with open(FEATURE_NAMES_PATH, 'r') as f:
        feature_names = json.load(f)
    print(f" Loaded feature names ({len(feature_names)} features)")
else:
    raise FileNotFoundError(f"Feature names not found: {FEATURE_NAMES_PATH}")

# Load Isolation Forest
if os.path.exists(ISOLATION_FOREST_PATH):
    with open(ISOLATION_FOREST_PATH, 'rb') as f:
        isolation_forest = pickle.load(f)
    print(f" Loaded Isolation Forest from: {ISOLATION_FOREST_PATH}")
else:
    print(f" Isolation Forest not found (optional)")
    isolation_forest = None

# Load config (optional)
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, 'r') as f:
        training_config = json.load(f)
    print(f" Loaded training config")

    # Use thresholds from training if available
    if 'current_thresholds' in training_config:
        current_thresholds_trained = training_config['current_thresholds']
        print(f" Using pre-computed current thresholds from training")
    else:
        current_thresholds_trained = None

    if 'temperature_thresholds' in training_config:
        temperature_thresholds_trained = training_config['temperature_thresholds']
        print(f" Using pre-computed temperature thresholds from training")
    else:
        temperature_thresholds_trained = None
else:
    print(f" Config not found (will compute thresholds from data)")
    training_config = None
    current_thresholds_trained = None
    temperature_thresholds_trained = None

print("\n All models and artifacts loaded successfully!")

def fetch_device_data(device_table):
    try:
        url = f"{API_BASE_URL}?table={device_table}"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        if isinstance(data, list):
            df = pd.DataFrame(data)
            print(f" Fetched {len(df):,} records from {device_table}")
            return df
        else:
            print(f" Unexpected data format for {device_table}")
            return None
    except Exception as e:
        print(f" Error fetching {device_table}: {e}")
        return None


def clean_numeric_series(s):
    return pd.to_numeric(
        s.astype(str).str.replace("'", "").str.replace(",", "").str.strip(),
        errors="coerce"
    )


def calculate_adaptive_thresholds(data, column_name='current'):
    data_clean = data.dropna()

    # IQR-based
    Q1 = np.percentile(data_clean, 25)
    Q3 = np.percentile(data_clean, 75)
    IQR = Q3 - Q1
    iqr_lower = Q1 - CURRENT_IQR_MULTIPLIER * IQR
    iqr_upper = Q3 + CURRENT_IQR_MULTIPLIER * IQR

    # Mean ± Std
    mean_val = np.mean(data_clean)
    std_val = np.std(data_clean)
    std_lower = mean_val - CURRENT_STD_MULTIPLIER * std_val
    std_upper = mean_val + CURRENT_STD_MULTIPLIER * std_val

    # Percentile-based
    percentile_lower = np.percentile(data_clean, CURRENT_PERCENTILE_LOW)
    percentile_upper = np.percentile(data_clean, CURRENT_PERCENTILE_HIGH)

    # MAD
    median_val = np.median(data_clean)
    mad = np.median(np.abs(data_clean - median_val))
    mad_lower = median_val - 3 * 1.4826 * mad
    mad_upper = median_val + 3 * 1.4826 * mad

    # Combine
    lower_threshold = min(iqr_lower, std_lower, mad_lower)
    upper_threshold = max(iqr_upper, std_upper, mad_upper)

    warning_lower = np.percentile(data_clean, 5)
    warning_upper = np.percentile(data_clean, 95)

    rate_data = np.abs(np.diff(data_clean))
    rate_threshold = np.percentile(rate_data, CURRENT_RATE_PERCENTILE)

    return {
        'lower_critical': lower_threshold,
        'lower_warning': warning_lower,
        'upper_warning': warning_upper,
        'upper_critical': upper_threshold,
        'rate_threshold': rate_threshold,
        'mean': mean_val,
        'median': median_val,
        'std': std_val,
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR
    }


def detect_current_anomalies_adaptive(current, thresholds):
    anomalies = np.zeros(len(current), dtype=int)
    reasons = [''] * len(current)
    severity = ['normal'] * len(current)

    current_rate = np.abs(np.diff(current, prepend=current[0]))

    for i in range(len(current)):
        reason_parts = []
        current_severity = 'normal'

        if current[i] == 0:
            pass
        elif current[i] > thresholds['upper_critical']:
            anomalies[i] = 1
            current_severity = 'critical'
            reason_parts.append(f"Excessive charge: {current[i]:.3f}A")
        elif current[i] > thresholds['upper_warning']:
            anomalies[i] = 1
            current_severity = 'high'
            reason_parts.append(f"High charge: {current[i]:.3f}A")

        if 0 < current[i] < thresholds['lower_critical']:
            anomalies[i] = 1
            if current_severity == 'normal':
                current_severity = 'medium'
            reason_parts.append(f"Unusually low current: {current[i]:.3f}A")

        if current_rate[i] > thresholds['rate_threshold']:
            anomalies[i] = 1
            if current_severity == 'normal':
                current_severity = 'medium'
            reason_parts.append(f"Rapid current change: {current_rate[i]:.3f}A/sample")

        reasons[i] = "; ".join(reason_parts)
        severity[i] = current_severity

    return anomalies, reasons, severity


def detect_temperature_anomalies_adaptive(temperature, thresholds):
    anomalies = np.zeros(len(temperature), dtype=int)
    reasons = [''] * len(temperature)
    severity = ['normal'] * len(temperature)

    temp_rate = np.abs(np.diff(temperature, prepend=temperature[0]))

    for i in range(len(temperature)):
        reason_parts = []
        temp_severity = 'normal'

        if temperature[i] > thresholds['upper_critical']:
            anomalies[i] = 1
            temp_severity = 'critical'
            reason_parts.append(f"Critical high temp: {temperature[i]:.2f}°C")
        elif temperature[i] > thresholds['upper_warning']:
            anomalies[i] = 1
            temp_severity = 'high'
            reason_parts.append(f"High temp: {temperature[i]:.2f}°C")

        if temperature[i] < thresholds['lower_critical']:
            anomalies[i] = 1
            temp_severity = 'critical' if temp_severity != 'critical' else temp_severity
            reason_parts.append(f"Critical low temp: {temperature[i]:.2f}°C")
        elif temperature[i] < thresholds['lower_warning']:
            anomalies[i] = 1
            if temp_severity == 'normal':
                temp_severity = 'medium'
            reason_parts.append(f"Low temp: {temperature[i]:.2f}°C")

        if temp_rate[i] > thresholds['rate_threshold']:
            anomalies[i] = 1
            if temp_severity == 'normal':
                temp_severity = 'medium'
            reason_parts.append(f"Rapid temp change: {temp_rate[i]:.2f}°C/sample")

        reasons[i] = "; ".join(reason_parts)
        severity[i] = temp_severity

    return anomalies, reasons, severity


def smooth_anomalies(anomaly_series, min_duration=3):
    smoothed = anomaly_series.copy()
    changes = np.diff(np.concatenate([[0], anomaly_series, [0]]))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    for start, end in zip(starts, ends):
        if end - start < min_duration:
            smoothed[start:end] = 0

    return smoothed


def engineer_features(df):
    Xf = df[['current', 'temperature']].copy()

    # Basic statistical features
    Xf['current_diff'] = Xf['current'].diff().fillna(0)
    Xf['current_diff_abs'] = Xf['current_diff'].abs()
    Xf['current_pct_change'] = Xf['current'].pct_change().fillna(0).replace([np.inf, -np.inf], 0)

    Xf['temp_diff'] = Xf['temperature'].diff().fillna(0)
    Xf['temp_pct_change'] = Xf['temperature'].pct_change().fillna(0).replace([np.inf, -np.inf], 0)

    # Rolling window features
    roll = Xf['current'].rolling(ROLL_WIN, min_periods=1)
    Xf['current_roll_mean'] = roll.mean()
    Xf['current_roll_std'] = roll.std().fillna(0)
    Xf['current_roll_min'] = roll.min()
    Xf['current_roll_max'] = roll.max()
    Xf['current_deviation'] = (Xf['current'] - Xf['current_roll_mean']).fillna(0)

    temp_roll = Xf['temperature'].rolling(ROLL_WIN, min_periods=1)
    Xf['temp_roll_mean'] = temp_roll.mean()
    Xf['temp_roll_std'] = temp_roll.std().fillna(0)

    # Domain-specific features
    Xf['current_abs'] = Xf['current'].abs()
    Xf['current_rate'] = Xf['current_diff'].abs()
    Xf['current_volatility'] = Xf['current'].rolling(10, min_periods=1).std().fillna(0)

    # Temporal features
    if df['parsed_datetime'].notna().any():
        local_dt = df['parsed_datetime'].dt.tz_convert(None) if df['parsed_datetime'].dt.tz else df['parsed_datetime']
        Xf['hour'] = local_dt.dt.hour.fillna(0)
        Xf['minute'] = local_dt.dt.minute.fillna(0)
        Xf['day_of_week'] = local_dt.dt.dayofweek.fillna(0)
        Xf['is_weekend'] = (local_dt.dt.dayofweek >= 5).astype(int)
        Xf['hour_sin'] = np.sin(2 * np.pi * Xf['hour'] / 24)
        Xf['hour_cos'] = np.cos(2 * np.pi * Xf['hour'] / 24)
    else:
        for col in ['hour', 'minute', 'day_of_week', 'is_weekend', 'hour_sin', 'hour_cos']:
            Xf[col] = 0

    # Clean features
    Xf = Xf.fillna(method='bfill').fillna(method='ffill').fillna(0)
    Xf = Xf.replace([np.inf, -np.inf], 0)

    available_features = [f for f in feature_names if f in Xf.columns]
    missing_features = [f for f in feature_names if f not in Xf.columns]

    if missing_features:
        print(f"\n  ⚠ Warning: {len(missing_features)} features missing, adding as zeros:")
        for feat in missing_features:
            print(f"    - {feat}")
            Xf[feat] = 0

    # Reorder to match training
    Xf = Xf[feature_names]

    return Xf

print("\n[2/5] Fetching data from API...")

all_dfs = []
device_names = []

for device_table in DEVICE_TABLES:
    df_temp = fetch_device_data(device_table)

    if df_temp is not None and len(df_temp) > 0:
        df_temp['device_id'] = device_table
        all_dfs.append(df_temp)
        device_names.append(device_table)

if not all_dfs:
    raise Exception("No data fetched from API!")

df = pd.concat(all_dfs, ignore_index=True)
print(f"\n Total samples: {len(df):,} from {len(device_names)} devices")

print("\n[3/5] Cleaning and preprocessing data...")

# Clean numeric columns
df['current'] = clean_numeric_series(df['current'])
df['temperature'] = clean_numeric_series(df['temperature'])

# Parse datetime
df['parsed_datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
if 'ts' in df.columns and df['parsed_datetime'].isna().all():
    df['parsed_datetime'] = pd.to_datetime(df['ts'], unit='s', errors='coerce')

# Sort
df = df.sort_values(['device_id', 'parsed_datetime']).reset_index(drop=True)

# Handle missing values
df = df.dropna(subset=['current']).reset_index(drop=True)

# Handle negative currents
df['current'] = df['current'].clip(lower=0)

# Handle temperature
temp_median = df['temperature'].median()
df['temperature'].fillna(temp_median, inplace=True)
temp_q1 = df['temperature'].quantile(0.01)
temp_q99 = df['temperature'].quantile(0.99)
df['temperature'] = df['temperature'].clip(lower=temp_q1, upper=temp_q99)

print(f" Data cleaned: {df.shape}")
print(f"  Current range: [{df['current'].min():.4f}, {df['current'].max():.4f}] A")
print(f"  Temperature range: [{df['temperature'].min():.2f}, {df['temperature'].max():.2f}] °C")

print("\n[4/5] Computing adaptive thresholds...")

if current_thresholds_trained:
    current_thresholds = current_thresholds_trained
    print("   Using pre-computed current thresholds from training")
else:
    current_thresholds = calculate_adaptive_thresholds(df['current'], 'current')
    print("  Computed current thresholds from data")

if temperature_thresholds_trained:
    temperature_thresholds = temperature_thresholds_trained
    print(" Using pre-computed temperature thresholds from training")
else:
    temperature_thresholds = calculate_adaptive_thresholds(df['temperature'], 'temperature')
    print(" Computed temperature thresholds from data")

print(f"\nCurrent Thresholds:")
print(f"  Critical Low: <{current_thresholds['lower_critical']:.4f} A")
print(f"  Critical High: >{current_thresholds['upper_critical']:.4f} A")
print(f"\nTemperature Thresholds:")
print(f"  Critical Low: <{temperature_thresholds['lower_critical']:.2f} °C")
print(f"  Critical High: >{temperature_thresholds['upper_critical']:.2f} °C")


print("\n[5/5] Engineering features...")

Xf = engineer_features(df)
X = Xf.values

print(f" Features engineered: {X.shape}")
print(f"  Features: {len(feature_names)}")

# Scale features
X_scaled = scaler.transform(X)
print(f" Features scaled")

print("\n" + "="*80)
print("RUNNING ANOMALY DETECTION")
print("="*80)

# Autoencoder reconstruction
print("\n[1/4] Computing reconstruction error...")
X_recon = autoencoder.predict(X_scaled, batch_size=32, verbose=0)
recon_error = np.mean(np.square(X_scaled - X_recon), axis=1)

threshold = np.mean(recon_error) + THRESHOLD_MULTIPLIER * np.std(recon_error)
percentile_95 = np.percentile(recon_error, 95)

y_threshold = (recon_error > threshold).astype(int)
y_percentile = (recon_error > percentile_95).astype(int)

print(f"  Reconstruction complete")
print(f"  Mean error: {recon_error.mean():.6f}")
print(f"  Threshold: {threshold:.6f}")
print(f"  Anomalies (threshold): {y_threshold.sum():,} ({100*y_threshold.mean():.2f}%)")

# Isolation Forest
print("\n[2/4] Running Isolation Forest...")
if isolation_forest:
    try:
        # Check expected feature count
        expected_features = isolation_forest.n_features_in_
        current_features = X_scaled.shape[1]

        # Try with reconstruction error appended
        if expected_features == current_features + 1:
            X_if = np.hstack([X_scaled, recon_error.reshape(-1, 1)])
            y_if = (isolation_forest.predict(X_if) == -1).astype(int)
            print(f" Anomalies (IF): {y_if.sum():,} ({100*y_if.mean():.2f}%)")
        # Try without reconstruction error
        elif expected_features == current_features:
            y_if = (isolation_forest.predict(X_scaled) == -1).astype(int)
            print(f" Anomalies (IF): {y_if.sum():,} ({100*y_if.mean():.2f}%)")
        else:
            print(f"   Feature mismatch: IF expects {expected_features}, got {current_features}")
            print(f"   Skipping Isolation Forest (feature count mismatch)")
            y_if = np.zeros(len(X_scaled), dtype=int)
    except Exception as e:
        print(f"  Error running Isolation Forest: {e}")
        print(f"  Skipping Isolation Forest")
        y_if = np.zeros(len(X_scaled), dtype=int)
else:
    y_if = np.zeros(len(X_scaled), dtype=int)
    print("  Skipped (model not available)")

# Domain-specific detection
print("\n[3/4] Domain-specific anomaly detection...")

current_anomalies, current_reasons, current_severity = detect_current_anomalies_adaptive(
    df['current'].values, current_thresholds
)

temperature_anomalies, temperature_reasons, temperature_severity = detect_temperature_anomalies_adaptive(
    df['temperature'].values, temperature_thresholds
)

domain_anomalies = np.maximum(current_anomalies, temperature_anomalies)

print(f" Current anomalies: {current_anomalies.sum():,}")
print(f" Temperature anomalies: {temperature_anomalies.sum():,}")
print(f" Combined domain anomalies: {domain_anomalies.sum():,}")

# Ensemble
print("\n[4/4] Ensemble detection...")

votes = y_threshold + y_percentile + y_if + domain_anomalies * 2
y_ensemble = ((votes >= 3) | (domain_anomalies == 1)).astype(int)
y_ensemble_smooth = smooth_anomalies(y_ensemble, MIN_ANOMALY_DURATION)

print(f" Ensemble (raw): {y_ensemble.sum():,}")
print(f" Ensemble (smoothed): {y_ensemble_smooth.sum():,} ({100*y_ensemble_smooth.mean():.2f}%)")

print("\n" + "="*80)
print("COMPILING RESULTS")
print("="*80)

df_out = df.copy()
df_out['recon_error'] = recon_error
df_out['anomaly_score'] = (recon_error - np.mean(recon_error)) / np.std(recon_error)
df_out['anomaly_current'] = current_anomalies
df_out['current_anomaly_reason'] = current_reasons
df_out['current_severity'] = current_severity
df_out['anomaly_temperature'] = temperature_anomalies
df_out['temperature_anomaly_reason'] = temperature_reasons
df_out['temperature_severity'] = temperature_severity
df_out['anomaly_threshold'] = y_threshold
df_out['anomaly_percentile'] = y_percentile
df_out['anomaly_isolation_forest'] = y_if
df_out['anomaly_ensemble'] = y_ensemble
df_out['anomaly_ensemble_smoothed'] = y_ensemble_smooth

# Combined severity
combined_severity = []
for i in range(len(current_severity)):
    if current_severity[i] == 'critical' or temperature_severity[i] == 'critical':
        combined_severity.append('critical')
    elif current_severity[i] == 'high' or temperature_severity[i] == 'high':
        combined_severity.append('high')
    elif current_severity[i] == 'medium' or temperature_severity[i] == 'medium':
        combined_severity.append('medium')
    else:
        combined_severity.append('normal')

df_out['combined_severity'] = combined_severity

print(f" Results compiled: {df_out.shape}")


print("\n" + "="*80)
print(f"ANALYZING RECENT {RECENT_DATA_WINDOW_MINUTES} MINUTES")
print("="*80)

device_verdicts = {}

for device in device_names:
    print(f"\n{device}:")
    device_data = df_out[df_out['device_id'] == device].copy()

    latest_timestamp = device_data['parsed_datetime'].max()
    cutoff_time = latest_timestamp - timedelta(minutes=RECENT_DATA_WINDOW_MINUTES)

    recent_data = device_data[device_data['parsed_datetime'] >= cutoff_time].copy()

    if len(recent_data) == 0:
        print("  No recent data")
        device_verdicts[device] = {
            'verdict': 'INSUFFICIENT_DATA',
            'message': 'No recent data available',
            'samples': 0,
            'anomaly_count': 0,
            'anomaly_rate': 0
        }
        continue

    recent_anomalies = recent_data[recent_data['anomaly_ensemble_smoothed'] == 1]
    recent_anomaly_count = len(recent_anomalies)
    recent_anomaly_rate = recent_anomaly_count / len(recent_data) * 100

    recent_critical = recent_data[recent_data['combined_severity'] == 'critical']
    recent_high = recent_data[recent_data['combined_severity'] == 'high']

    print(f"  Samples: {len(recent_data):,}")
    print(f"  Anomalies: {recent_anomaly_count:,} ({recent_anomaly_rate:.2f}%)")
    print(f"  Critical events: {len(recent_critical):,}")
    print(f"  High severity: {len(recent_high):,}")

    # Decision
    if recent_anomaly_rate > 5.0 or len(recent_critical) > 0 or len(recent_high) > 3:
        verdict = "YES"
        message = "ANOMALY DETECTED"
        print(f" VERDICT: {verdict} - {message}")
    else:
        verdict = "NO"
        message = "NORMAL OPERATION"
        print(f" VERDICT: {verdict} - {message}")

    device_verdicts[device] = {
        'verdict': verdict,
        'message': message,
        'samples': len(recent_data),
        'anomaly_count': recent_anomaly_count,
        'anomaly_rate': recent_anomaly_rate,
        'critical_events': len(recent_critical),
        'high_severity_events': len(recent_high),
        'latest_timestamp': str(latest_timestamp)
    }


print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save full results
output_path = os.path.join(OUTPUT_DIR, 'inference_results.csv')
df_out.to_csv(output_path, index=False)
print(f" Full results: {output_path}")

# Save anomalies only
anomalies_df = df_out[df_out['anomaly_ensemble_smoothed'] == 1].copy()
if len(anomalies_df) > 0:
    anomalies_path = os.path.join(OUTPUT_DIR, 'detected_anomalies.csv')
    anomalies_df.to_csv(anomalies_path, index=False)
    print(f" Anomalies only: {anomalies_path}")

# Save device verdicts
verdicts_path = os.path.join(OUTPUT_DIR, 'device_verdicts.json')
verdicts_summary = {
    'analysis_timestamp': datetime.now().isoformat(),
    'analysis_window_minutes': RECENT_DATA_WINDOW_MINUTES,
    'devices': device_verdicts
}

with open(verdicts_path, 'w') as f:
    json.dump(verdicts_summary, f, indent=4)
print(f" Device verdicts: {verdicts_path}")

# Save per-device results
for device in device_names:
    device_df = df_out[df_out['device_id'] == device]
    device_path = os.path.join(OUTPUT_DIR, f'inference_{device}.csv')
    device_df.to_csv(device_path, index=False)
print(f" Per-device results saved")


print("\n" + "="*80)
print("INFERENCE COMPLETE - SUMMARY")
print("="*80)

print(f"\nTotal Samples: {len(df_out):,}")
print(f"Total Anomalies: {y_ensemble_smooth.sum():,} ({100*y_ensemble_smooth.mean():.2f}%)")

print(f"\nDevice Verdicts (Recent {RECENT_DATA_WINDOW_MINUTES} minutes):")
for device, verdict_info in device_verdicts.items():
    status_icon = "" if verdict_info['verdict'] == 'YES' else " "
    print(f"  {status_icon} {device}: {verdict_info['verdict']} - {verdict_info['message']}")
    if verdict_info['samples'] > 0:
        print(f"     Samples: {verdict_info['samples']:,}, Anomalies: {verdict_info['anomaly_count']:,} ({verdict_info['anomaly_rate']:.2f}%)")

print(f"\n All results saved to: {OUTPUT_DIR}")

import matplotlib.pyplot as plt
import seaborn as sns

print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("\n[1/5] Creating comprehensive analysis plot...")
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

axes[0].plot(df_out.index, df_out['current'], alpha=0.7, linewidth=1,
             color='#1f77b4', label='Current')
anomalies = df_out[df_out['anomaly_ensemble_smoothed'] == 1]
axes[0].scatter(anomalies.index, anomalies['current'],
               color='red', s=30, alpha=0.7, label='Anomaly', zorder=5)

# Plot adaptive thresholds
axes[0].axhline(y=current_thresholds['lower_critical'], color='blue',
               linestyle='--', label=f"Critical Low ({current_thresholds['lower_critical']:.3f}A)",
               linewidth=1.5, alpha=0.7)
axes[0].axhline(y=current_thresholds['upper_warning'], color='orange',
               linestyle='--', label=f"Warning High ({current_thresholds['upper_warning']:.3f}A)",
               linewidth=1.5, alpha=0.7)
axes[0].axhline(y=current_thresholds['upper_critical'], color='red',
               linestyle='--', label=f"Critical High ({current_thresholds['upper_critical']:.3f}A)",
               linewidth=1.5, alpha=0.7)
axes[0].fill_between(df_out.index, current_thresholds['lower_warning'],
                     current_thresholds['upper_warning'],
                     alpha=0.1, color='green', label='Normal Zone')

axes[0].set_ylabel('Current (A)', fontsize=11)
axes[0].set_title('Current with Detected Anomalies and Adaptive Thresholds',
                  fontsize=12, fontweight='bold')
axes[0].legend(loc='best', fontsize=8)
axes[0].grid(True, alpha=0.3)

# Reconstruction error
axes[1].plot(df_out.index, df_out['recon_error'], color='#9467bd',
            linewidth=1, label='Recon Error')
axes[1].axhline(y=threshold, color='red', linestyle='--',
               label=f'Threshold: {threshold:.4f}', linewidth=1.5)
axes[1].fill_between(df_out.index, 0, df_out['recon_error'],
                    where=(df_out['anomaly_ensemble_smoothed'] == 1),
                    color='red', alpha=0.3, label='Anomaly Regions')
axes[1].set_ylabel('Reconstruction Error', fontsize=11)
axes[1].set_title('Autoencoder Reconstruction Error', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_yscale('log')

# Anomaly score
axes[2].plot(df_out.index, df_out['anomaly_score'], color='#9467bd',
            linewidth=1, alpha=0.7, label='Anomaly Score')
axes[2].axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
axes[2].axhline(y=THRESHOLD_MULTIPLIER, color='red', linestyle='--',
               label=f'Threshold ({THRESHOLD_MULTIPLIER}σ)')
axes[2].fill_between(df_out.index, df_out['anomaly_score'].min(),
                    df_out['anomaly_score'],
                    where=(df_out['anomaly_ensemble_smoothed'] == 1),
                    color='red', alpha=0.3)
axes[2].set_xlabel('Sample Index', fontsize=11)
axes[2].set_ylabel('Anomaly Score (σ)', fontsize=11)
axes[2].set_title('Normalized Anomaly Score', fontsize=12, fontweight='bold')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'comprehensive_analysis.png'),
           dpi=150, bbox_inches='tight')
plt.close()
print("  comprehensive_analysis.png")

# 2. Error Distribution
print("\n[2/5] Creating error distribution plot...")
plt.figure(figsize=(10, 5))
sns.histplot(recon_error, bins=80, kde=True, stat='density', color='skyblue')
plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2,
           label=f'Threshold: {threshold:.4f}')
plt.axvline(x=percentile_95, color='orange', linestyle='--', linewidth=2,
           label=f'95th Percentile: {percentile_95:.4f}')
plt.xlabel('Reconstruction Error', fontsize=11)
plt.ylabel('Density', fontsize=11)
plt.title('Reconstruction Error Distribution', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'error_distribution.png'),
           dpi=150, bbox_inches='tight')
plt.close()
print("   error_distribution.png")

# 3. Current Distribution
print("\n[3/5] Creating current distribution plot...")
plt.figure(figsize=(14, 6))
plt.hist(df_out['current'], bins=50, alpha=0.7, label='All Data',
        color='lightgreen', edgecolor='black')
plt.hist(df_out[df_out['anomaly_current'] == 1]['current'],
        bins=30, alpha=0.7, label='Current Anomalies',
        color='red', edgecolor='black')

# Mark thresholds
plt.axvline(x=current_thresholds['lower_critical'], color='blue',
           linestyle='--', label=f"Critical Low ({current_thresholds['lower_critical']:.3f}A)",
           linewidth=2)
plt.axvline(x=current_thresholds['median'], color='green',
           linestyle='-', label=f"Median ({current_thresholds['median']:.3f}A)",
           linewidth=2)
plt.axvline(x=current_thresholds['upper_warning'], color='orange',
           linestyle=':', label=f"Warning High ({current_thresholds['upper_warning']:.3f}A)",
           linewidth=2)
plt.axvline(x=current_thresholds['upper_critical'], color='red',
           linestyle='--', label=f"Critical High ({current_thresholds['upper_critical']:.3f}A)",
           linewidth=2)

# Shade normal zone
plt.axvspan(current_thresholds['lower_warning'],
           current_thresholds['upper_warning'],
           alpha=0.1, color='green', label='Normal Operating Zone')

plt.xlabel('Current (A)', fontsize=11)
plt.ylabel('Frequency', fontsize=11)
plt.title('Current Distribution with Adaptive Thresholds',
         fontsize=12, fontweight='bold')
plt.legend(fontsize=8, loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'current_distribution.png'),
           dpi=150, bbox_inches='tight')
plt.close()
print("  current_distribution.png")

# 4. Temperature Distribution
print("\n[4/5] Creating temperature distribution plot...")
plt.figure(figsize=(14, 6))
plt.hist(df_out['temperature'], bins=50, alpha=0.7, label='All Data',
        color='lightcoral', edgecolor='black')
plt.hist(df_out[df_out['anomaly_temperature'] == 1]['temperature'],
        bins=30, alpha=0.7, label='Temperature Anomalies',
        color='darkred', edgecolor='black')

# Mark thresholds
plt.axvline(x=temperature_thresholds['lower_critical'], color='blue',
           linestyle='--', label=f"Critical Low ({temperature_thresholds['lower_critical']:.2f}°C)",
           linewidth=2)
plt.axvline(x=temperature_thresholds['median'], color='green',
           linestyle='-', label=f"Median ({temperature_thresholds['median']:.2f}°C)",
           linewidth=2)
plt.axvline(x=temperature_thresholds['upper_warning'], color='orange',
           linestyle=':', label=f"Warning High ({temperature_thresholds['upper_warning']:.2f}°C)",
           linewidth=2)
plt.axvline(x=temperature_thresholds['upper_critical'], color='red',
           linestyle='--', label=f"Critical High ({temperature_thresholds['upper_critical']:.2f}°C)",
           linewidth=2)

# Shade normal zone
plt.axvspan(temperature_thresholds['lower_warning'],
           temperature_thresholds['upper_warning'],
           alpha=0.1, color='green', label='Normal Operating Zone')

plt.xlabel('Temperature (°C)', fontsize=11)
plt.ylabel('Frequency', fontsize=11)
plt.title('Temperature Distribution with Adaptive Thresholds',
         fontsize=12, fontweight='bold')
plt.legend(fontsize=8, loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'temperature_distribution.png'),
           dpi=150, bbox_inches='tight')
plt.close()
print(" temperature_distribution.png")

# 5. Per-Device Recent Data Visualizations
print("\n[5/5] Creating per-device recent data plots...")
for device in device_names:
    device_data = df_out[df_out['device_id'] == device].copy()
    latest_timestamp = device_data['parsed_datetime'].max()
    cutoff_time = latest_timestamp - timedelta(minutes=RECENT_DATA_WINDOW_MINUTES)
    recent_data = device_data[device_data['parsed_datetime'] >= cutoff_time].copy()

    if len(recent_data) == 0:
        continue

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Recent current data
    axes[0].plot(recent_data.index, recent_data['current'], alpha=0.7,
                linewidth=2, color='#1f77b4', label='Current',
                marker='o', markersize=4)
    recent_anom = recent_data[recent_data['anomaly_ensemble_smoothed'] == 1]
    if len(recent_anom) > 0:
        axes[0].scatter(recent_anom.index, recent_anom['current'],
                       color='red', s=80, alpha=0.8, label='Anomaly',
                       zorder=5, marker='X')

    axes[0].axhline(y=current_thresholds['upper_critical'], color='red',
                   linestyle='--', label=f"Critical High", linewidth=2, alpha=0.7)
    axes[0].axhline(y=current_thresholds['lower_critical'], color='blue',
                   linestyle='--', label=f"Critical Low", linewidth=2, alpha=0.7)
    axes[0].fill_between(recent_data.index,
                        current_thresholds['lower_warning'],
                        current_thresholds['upper_warning'],
                        alpha=0.15, color='green', label='Normal Zone')

    # Add verdict badge
    verdict_info = device_verdicts[device]
    if verdict_info['verdict'] == 'YES':
        badge_color = 'red'
        badge_text = ' ANOMALY'
    elif verdict_info['verdict'] == 'NO':
        badge_color = 'green'
        badge_text = ' NORMAL'
    else:
        badge_color = 'orange'
        badge_text = ' INSUFFICIENT DATA'

    axes[0].text(0.98, 0.95, badge_text, transform=axes[0].transAxes,
                fontsize=12, fontweight='bold', va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor=badge_color, alpha=0.7))

    axes[0].set_ylabel('Current (A)', fontsize=12, fontweight='bold')
    axes[0].set_title(f'{device} - Recent {RECENT_DATA_WINDOW_MINUTES} Minutes - Current Analysis',
                     fontsize=13, fontweight='bold')
    axes[0].legend(loc='best', fontsize=9)
    axes[0].grid(True, alpha=0.4)

    # Recent temperature data
    axes[1].plot(recent_data.index, recent_data['temperature'], alpha=0.7,
                linewidth=2, color='#ff7f0e', label='Temperature',
                marker='s', markersize=4)
    if len(recent_anom) > 0:
        axes[1].scatter(recent_anom.index, recent_anom['temperature'],
                       color='red', s=80, alpha=0.8, label='Anomaly Point',
                       zorder=5, marker='X')

    axes[1].axhline(y=temperature_thresholds['upper_critical'], color='red',
                   linestyle='--', label=f"Critical High", linewidth=2, alpha=0.7)
    axes[1].axhline(y=temperature_thresholds['lower_critical'], color='blue',
                   linestyle='--', label=f"Critical Low", linewidth=2, alpha=0.7)
    axes[1].fill_between(recent_data.index,
                        temperature_thresholds['lower_warning'],
                        temperature_thresholds['upper_warning'],
                        alpha=0.15, color='orange', label='Normal Zone')

    axes[1].set_xlabel('Sample Index', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
    axes[1].set_title(f'{device} - Recent {RECENT_DATA_WINDOW_MINUTES} Minutes - Temperature Analysis',
                     fontsize=13, fontweight='bold')
    axes[1].legend(loc='best', fontsize=9)
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'recent_data_{device}.png'),
               dpi=150, bbox_inches='tight')
    plt.close()

print(f" Per-device plots created for {len(device_names)} devices")

# 6. Method Comparison Heatmap
print("\n[BONUS] Creating method comparison heatmap...")
method_df = pd.DataFrame({
    'Threshold': y_threshold,
    'Percentile': y_percentile,
    'IsolationForest': y_if,
    'Current': current_anomalies,
    'Temperature': temperature_anomalies,
    'Ensemble': y_ensemble_smooth
})

n_samples = min(500, len(method_df))
sample_indices = np.linspace(0, len(method_df)-1, n_samples).astype(int)

plt.figure(figsize=(12, 6))
sns.heatmap(method_df.iloc[sample_indices].T, cmap='RdYlGn_r',
           cbar_kws={'label': 'Anomaly (1) / Normal (0)'},
           yticklabels=True, xticklabels=False)
plt.title('Anomaly Detection Methods Comparison', fontsize=12, fontweight='bold')
plt.xlabel('Sample Index', fontsize=11)
plt.ylabel('Detection Method', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'method_comparison.png'),
           dpi=150, bbox_inches='tight')
plt.close()
print("  method_comparison.png")

# 7. Anomaly Summary Bar Chart
print("\n[BONUS] Creating anomaly summary bar chart...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Anomalies by device
device_anomaly_counts = []
for device in device_names:
    device_mask = df_out['device_id'] == device
    anomaly_count = df_out[device_mask]['anomaly_ensemble_smoothed'].sum()
    device_anomaly_counts.append(anomaly_count)

axes[0].bar(device_names, device_anomaly_counts, color='#ff7f0e', alpha=0.7)
axes[0].set_xlabel('Device', fontsize=11)
axes[0].set_ylabel('Anomaly Count', fontsize=11)
axes[0].set_title('Anomalies Detected per Device', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(device_anomaly_counts):
    axes[0].text(i, v + max(device_anomaly_counts)*0.02, str(v),
                ha='center', fontweight='bold')

# Severity distribution
severity_counts = df_out['combined_severity'].value_counts()
severity_order = ['critical', 'high', 'medium', 'normal']
severity_colors = {'critical': 'red', 'high': 'orange',
                  'medium': 'yellow', 'normal': 'green'}
colors = [severity_colors.get(s, 'gray') for s in severity_order if s in severity_counts]
counts = [severity_counts.get(s, 0) for s in severity_order if s in severity_counts]
labels = [s for s in severity_order if s in severity_counts]

axes[1].bar(labels, counts, color=colors, alpha=0.7)
axes[1].set_xlabel('Severity', fontsize=11)
axes[1].set_ylabel('Count', fontsize=11)
axes[1].set_title('Severity Distribution', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(counts):
    axes[1].text(i, v + max(counts)*0.02, f'{v:,}',
                ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'anomaly_summary.png'),
           dpi=150, bbox_inches='tight')
plt.close()
print("  anomaly_summary.png")

print("\n ALL VISUALIZATIONS CREATED")
print(f"  Location: {OUTPUT_DIR}/")

print("="*80)