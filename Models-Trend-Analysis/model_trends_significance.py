import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from pathlib import Path

# --- Define paths & Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
ALIGNED_CSV_PATH = SCRIPT_DIR.parent / "Datasets/aligned_monthly_outer_interp.csv"

MODEL_FILES = {
    # We use SCRIPT_DIR.parent as the base for all model files
    "RandomForest": SCRIPT_DIR.parent / "Models-Training/RandomForest-Train/rf_with_lags_preds.csv",
    "GradientBoosting": SCRIPT_DIR.parent / "Models-Training/GradientBoosting-Train/gbr_with_lags_preds.csv",
    "Lasso": SCRIPT_DIR.parent / "Models-Training/Lasso-Train/lasso_with_lags_preds.csv",
    "Ridge": SCRIPT_DIR.parent / "Models-Training/RidgeRegression-Train/ridge_with_lags_preds.csv",
    "LSTM": SCRIPT_DIR.parent / "Models-Training/LSTM-Train/lstm_preds_vs_actual.csv"
}

print(f"Loading actuals from: {ALIGNED_CSV_PATH}")

if ALIGNED_CSV_PATH.exists():
    df_actuals = pd.read_csv(ALIGNED_CSV_PATH, index_col=0, parse_dates=True)
    print(f"Actuals shape: {df_actuals.shape}\n")
else:
    raise FileNotFoundError(f"Missing actuals file: {ALIGNED_CSV_PATH}")

print("Loading model predictions:")
for model_name, path_obj in MODEL_FILES.items():
    if path_obj.exists():
        df_model = pd.read_csv(path_obj, index_col=0, parse_dates=True)
        print(f"- {model_name}: Loaded with shape {df_model.shape}")
    else:
        print(f"- {model_name}: File not found at {path_obj}")

OUT_DIR = "inc-dec_significance_trends"
os.makedirs(OUT_DIR, exist_ok=True)
DPI = 300
# --------- end config -------------

def detect_ycols(df):
    cols = [c.lower() for c in df.columns]
    if "y_pred" in cols:
        yp = df.columns[cols.index("y_pred")]
    elif "yhat" in cols:
        yp = df.columns[cols.index("yhat")]
    else:
        numeric = df.select_dtypes(include=[float,int]).columns
        yp = numeric[1] if len(numeric) >= 2 else numeric[0]
    if "y_true" in cols:
        yt = df.columns[cols.index("y_true")]
    elif "y" in cols:
        yt = df.columns[cols.index("y")]
    else:
        yt = numeric[0]
    return yt, yp

def compute_trend(series):
    s = series.dropna()
    if len(s) < 3:
        return np.nan, np.nan
    x = np.arange(len(s))
    r = linregress(x, s.values)
    slope_yr = r.slope * 12
    return slope_yr, r.pvalue

def interpret_trend(slope, p):
    if np.isnan(slope):
        return "insufficient data"
    direction = "increasing" if slope > 0 else "decreasing"
    significance = "significant" if p < 0.05 else "not significant"
    return f"{direction}, {significance}"

def load_preds(path, aligned_ref=None):
    df = pd.read_csv(path)
    yt, yp = detect_ycols(df)
    dt_col = None
    for c in df.columns:
        if "date" in c.lower() or "time" in c.lower():
            dt_col = c; break
    if dt_col:
        df[dt_col] = pd.to_datetime(df[dt_col], errors='coerce')
        if df[dt_col].notna().sum() > 0:
            df = df.set_index(dt_col).sort_index()
    if not isinstance(df.index, pd.DatetimeIndex) and aligned_ref is not None:
        ref = aligned_ref
        if len(ref) >= len(df):
            df.index = ref.index[-len(df):]
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.resample("ME").mean()
    df = df.rename(columns={yp: "y_pred", yt: "y_true"}).dropna()
    return df

# load reference for timestamp alignment
ref = pd.read_csv(ALIGNED_CSV_PATH, index_col=0, parse_dates=True)
ref = ref.resample("ME").mean().dropna()

# store results
trend_rows = []

for model, file in MODEL_FILES.items():
    if not os.path.exists(file):
        print(f"⚠️ Skipping {model}: file not found.")
        continue
    df = load_preds(file, aligned_ref=ref)
    y_pred = df["y_pred"]
    slope, pval = compute_trend(y_pred)
    interpretation = interpret_trend(slope, pval)
    trend_rows.append({"Model": model, "Slope_per_year": slope, "p_value": pval, "Interpretation": interpretation})
    print(f"{model} trend: {slope:+.4f} per year, p={pval:.4f} ({interpretation})")

    # Plot per-model trend figure
    plt.figure(figsize=(8,4))
    plt.plot(df.index, y_pred, '-o', label="Predicted PM2.5", alpha=0.7)
    x = np.arange(len(y_pred))
    fit = np.poly1d(np.polyfit(x, y_pred.values, 1))
    plt.plot(df.index, fit(x), '--', color='red', label=f"Trend: {slope:+.4f}/yr, p={pval:.4f}")
    plt.title(f"{model} — PM2.5 Prediction Trend")
    plt.ylabel("PM2.5 (µg/m³)")
    plt.xlabel("Year")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    outpath = os.path.join(OUT_DIR, f"{model}_trend_significance.png")
    plt.savefig(outpath, dpi=DPI)
    plt.close()
    print(f"  → Saved plot: {outpath}")

# save tabular summary
trend_df = pd.DataFrame(trend_rows)
trend_df.to_csv(os.path.join(OUT_DIR, "trend_significance_summary.csv"), index=False)
print("\n✅ Trend significance report saved to:", OUT_DIR)
