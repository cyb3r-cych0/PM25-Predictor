import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
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

# --- Feature engineering (same as previous models) ---
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

# --- Prepare features and target ---
y = df["pm25"]
X = df.drop(columns=["pm25"])
n = len(df)
split_idx = int(np.floor(0.8 * n))
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
print("Train/Test rows:", len(X_train), "/", len(X_test))

# --- Train Lasso Regression with cross-validation for best alpha ---
lasso = LassoCV(cv=5, random_state=42, n_jobs=-1)
lasso.fit(X_train, y_train)

# --- Evaluate ---
y_pred = lasso.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
print(f"Lasso Regression results -- R2: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
print(f"Optimal alpha: {lasso.alpha_:.5f}")

# --- Save model, predictions, and coefficients ---
joblib.dump(lasso, "lasso_with_lags_model.joblib")
pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv("lasso_with_lags_preds.csv")

coeffs = pd.Series(lasso.coef_, index=X.columns).sort_values(ascending=False)
coeffs.to_csv("lasso_with_lags_coefficients.csv")

# --- Plot predictions ---
plt.figure(figsize=(10,4))
plt.plot(y_test.index, y_test.values, marker='o', label='actual')
plt.plot(y_test.index, y_pred, marker='x', label='predicted')
plt.title("Lasso Regression: PM2.5 Actual vs Predicted")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig("lasso_with_lags_pred_vs_actual.png")
plt.close()

# --- Plot coefficients ---
plt.figure(figsize=(8,4))
coeffs.plot(kind="bar")
plt.title("Lasso Regression Coefficients")
plt.tight_layout()
plt.savefig("lasso_with_lags_coefficients.png")
plt.close()

print("Saved: lasso_with_lags_model.joblib, lasso_with_lags_preds.csv, lasso_with_lags_coefficients.csv and plots.")
