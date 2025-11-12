import os, joblib
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

# --- feature engineering (common models pipeline) ---
df = df.sort_index()
for lag in (1,2,3):
    df[f'pm25_lag{lag}'] = df['pm25'].shift(lag)
df['pm25_rm_3'] = df['pm25'].rolling(window=3, min_periods=1).mean().shift(1)
df['pm25_rm_6'] = df['pm25'].rolling(window=6, min_periods=1).mean().shift(1)
df['pm25_rm_12'] = df['pm25'].rolling(window=12, min_periods=1).mean().shift(1)
months = df.index.month
df['month_sin'] = np.sin(2 * np.pi * months / 12)
df['month_cos'] = np.cos(2 * np.pi * months / 12)

df = df.dropna()
print("Prepared dataset shape:", df.shape)

# --- prepare train/test (time-based split: first 80% train, last 20% test) ---
y = df['pm25']
X = df.drop(columns=['pm25'])
n = len(df)
split_idx = int(np.floor(0.8 * n))
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
print("Train/Test rows:", len(X_train), "/", len(X_test))

# --- train GradientBoostingRegressor ---
gbr = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)
gbr.fit(X_train, y_train)

# --- evaluate ---
y_pred = gbr.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = (mean_squared_error(y_test, y_pred))**0.5
mae = mean_absolute_error(y_test, y_pred)
print(f"Gradient Boosting results -- R2: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

# --- save outputs ---
joblib.dump(gbr, "gbr_with_lags_model.joblib")
pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv("gbr_with_lags_preds.csv")
fi = pd.Series(gbr.feature_importances_, index=X.columns).sort_values(ascending=False)
fi.to_csv("gbr_with_lags_feature_importances.csv")

# --- plots ---
plt.figure(figsize=(10,4))
plt.plot(y_test.index, y_test.values, marker='o', label='actual')
plt.plot(y_test.index, y_pred, marker='x', label='predicted')
plt.title("GBR: PM2.5 Actual vs Predicted")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig("gbr_with_lags_pred_vs_actual.png")
plt.close()

plt.figure(figsize=(8,4))
fi.plot(kind='bar')
plt.title("GBR Feature Importances")
plt.tight_layout()
plt.savefig("gbr_with_lags_feature_importances.png")
plt.close()

print("Saved: gbr_with_lags_model.joblib, gbr_with_lags_preds.csv, gbr_with_lags_feature_importances.csv and plots.")
