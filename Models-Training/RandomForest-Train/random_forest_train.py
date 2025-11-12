import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# --- Define paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS_DIR = SCRIPT_DIR.parent.parent / "Datasets"
ALIGNED_CSV_PATH = DATASETS_DIR / "aligned_monthly_outer_interp.csv"
print("Looking for file at:", ALIGNED_CSV_PATH)

# --- Load aligned data ---
if not ALIGNED_CSV_PATH.exists():
    raise FileNotFoundError(f"Missing required file: {ALIGNED_CSV_PATH}. Run the 'align_to_monthly.py' script in the Datasets DIR first.")
df = pd.read_csv(ALIGNED_CSV_PATH, index_col=0, parse_dates=True)
print("Loaded dataset shape:", df.shape)

# --- Feature engineering: lags, rolling means, month cyclic ---
df = df.sort_index()
# create lags of pm25 (1..3)
for lag in (1,2,3):
    df[f'pm25_lag{lag}'] = df['pm25'].shift(lag)

# rolling means of pm25
df['pm25_rm_3'] = df['pm25'].rolling(window=3, min_periods=1).mean().shift(1)   # trailing mean (use previous months)
df['pm25_rm_6'] = df['pm25'].rolling(window=6, min_periods=1).mean().shift(1)
df['pm25_rm_12'] = df['pm25'].rolling(window=12, min_periods=1).mean().shift(1)

# month of year as cyclical features
months = df.index.month
df['month_sin'] = np.sin(2 * np.pi * months / 12)
df['month_cos'] = np.cos(2 * np.pi * months / 12)

# drop rows with any NaNs (due to lags)
df = df.dropna()
print("Data prepared. Shape after features:", df.shape)

# --- Prepare X, y ---
y = df['pm25']
X = df.drop(columns=['pm25'])

# --- Time-based train/test split (first 80% months train, last 20% test) ---
n = len(df)
split_idx = int(np.floor(0.8 * n))
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
print(f"Train/Test rows: {len(X_train)}/{len(X_test)}")

# --- Train Random Forest ---
rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=1)
rf.fit(X_train, y_train)

# --- Evaluate ---
y_pred = rf.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
print(f"R2: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

# --- Save artifacts ---
joblib.dump(rf, "rf_with_lags_model.joblib")
pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv("rf_with_lags_preds.csv")
fi = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
fi.to_csv("rf_with_lags_feature_importances.csv")

# --- Quick plots ---
plt.figure(figsize=(10,4))
plt.plot(y_test.index, y_test.values, marker='o', label='actual')
plt.plot(y_test.index, y_pred, marker='x', label='predicted')
plt.title("PM2.5 Actual vs Predicted")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig("rf_with_lags_pred_vs_actual.png")
plt.close()

plt.figure(figsize=(8,4))
fi.plot(kind='bar')
plt.title("Feature importances (with lags & rolls)")
plt.tight_layout()
plt.savefig("rf_with_lags_feature_importances.png")
plt.close()

print("Saved model, predictions, feature importances and plots.")
