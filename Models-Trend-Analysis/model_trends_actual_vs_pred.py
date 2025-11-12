import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import linregress
from pathlib import Path

# --- Define paths & Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
ALIGNED_CSV_PATH = SCRIPT_DIR.parent / "Datasets/aligned_monthly_outer_interp.csv"

# 3. Construct the full absolute paths for the MODEL_FILES dictionary
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

OUT_DIR = "actual_vs_pred_trends"
os.makedirs(OUT_DIR, exist_ok=True)
DPI = 300
# ---------- end config ----------

def detect_ycols(df):
    cols_low = [c.lower() for c in df.columns]
    # y_pred
    if "y_pred" in cols_low:
        ypred = df.columns[cols_low.index("y_pred")]
    elif "yhat" in cols_low:
        ypred = df.columns[cols_low.index("yhat")]
    else:
        numerics = df.select_dtypes(include=[float,int]).columns.tolist()
        ypred = numerics[1] if len(numerics) >= 2 else (numerics[0] if numerics else None)
    # y_true
    if "y_true" in cols_low:
        ytrue = df.columns[cols_low.index("y_true")]
    elif "y" in cols_low:
        ytrue = df.columns[cols_low.index("y")]
    else:
        numerics = df.select_dtypes(include=[float,int]).columns.tolist()
        ytrue = numerics[0] if len(numerics) >= 1 else None
    return ytrue, ypred

def load_preds(path, aligned_ref=None):
    df = pd.read_csv(path)
    ytrue_col, ypred_col = detect_ycols(df)
    if ytrue_col is None or ypred_col is None:
        raise RuntimeError(f"Could not detect y_true/y_pred columns in {path}")
    # attempt to find datetime column
    dt_col = None
    for c in df.columns:
        if "date" in c.lower() or "time" in c.lower() or "datetime" in c.lower():
            dt_col = c; break
    if dt_col:
        df[dt_col] = pd.to_datetime(df[dt_col], errors='coerce')
        if df[dt_col].notna().sum() > 0:
            df = df.set_index(dt_col).sort_index()
    # if no datetime and aligned_ref provided, attach last N timestamps
    df_sub = df[[ytrue_col, ypred_col]].rename(columns={ytrue_col: "y_true", ypred_col: "y_pred"})
    if not isinstance(df_sub.index, pd.DatetimeIndex) and aligned_ref is not None:
        ref = aligned_ref
        if len(ref) >= len(df_sub):
            df_sub.index = ref.index[-len(df_sub):]
    # resample to month-end if datetime
    if isinstance(df_sub.index, pd.DatetimeIndex):
        df_sub = df_sub.resample("ME").mean()
    return df_sub

def slope_p_for_series(series):
    s = series.dropna()
    if len(s) < 3:
        return np.nan, np.nan
    x = np.arange(len(s))  # months
    res = linregress(x, s.values)
    slope_per_year = res.slope * 12.0
    return float(slope_per_year), float(res.pvalue)

def interpret(slope, p):
    if np.isnan(slope):
        return "insufficient data"
    direction = "increasing" if slope > 0 else "decreasing"
    significance = "significant" if p < 0.05 else "not significant"
    return f"{direction}, {significance}"

# Load aligned actuals
if not os.path.exists(ALIGNED_CSV_PATH):
    raise SystemExit(f"Aligned CSV not found: {ALIGNED_CSV_PATH}")
aligned = pd.read_csv(ALIGNED_CSV_PATH, index_col=0, parse_dates=True).sort_index()
aligned = aligned.resample("ME").mean().dropna()

# detect actual PM2.5 column: prefer 'pm25' else first numeric
if "pm25" in aligned.columns:
    actual_series_full = aligned["pm25"].dropna()
else:
    num_cols = aligned.select_dtypes(include=[float,int]).columns.tolist()
    if not num_cols:
        raise SystemExit("No numeric column in aligned CSV for actual PM2.5.")
    actual_series_full = aligned[num_cols[0]].dropna()

summary_rows = []

for model_name, pred_path in MODEL_FILES.items():
    if not os.path.exists(pred_path):
        print(f"⚠️  {model_name}: prediction file not found ({pred_path}), skipping.")
        continue

    # load predictions (monthly)
    preds = load_preds(pred_path, aligned_ref=aligned)
    # drop NaNs
    preds = preds.dropna()
    if preds.empty:
        print(f"⚠️  {model_name}: no valid preds after loading/alignment, skipping.")
        continue

    # find common timestamps between actual and preds
    if isinstance(preds.index, pd.DatetimeIndex) and isinstance(actual_series_full.index, pd.DatetimeIndex):
        common_idx = actual_series_full.index.intersection(preds.index)
    else:
        # positional fallback: use last N months
        n = min(len(preds), len(actual_series_full))
        common_idx = actual_series_full.index[-n:]

    actual = actual_series_full.reindex(common_idx).dropna()
    preds = preds.reindex(common_idx).dropna()

    if len(actual) < 3 or len(preds) < 3:
        print(f"⚠️  {model_name}: insufficient overlapping months ({len(common_idx)}) after alignment — need >=3. Skipping.")
        continue

    # compute trends and p-values
    slope_act, p_act = slope_p_for_series(actual)
    slope_pred, p_pred = slope_p_for_series(preds["y_pred"])
    slope_bias, p_bias = slope_p_for_series(preds["y_pred"] - actual)

    interpretation_act = interpret(slope_act, p_act)
    interpretation_pred = interpret(slope_pred, p_pred)
    interpretation_bias = interpret(slope_bias, p_bias)

    # print professional one-line summaries
    print(f"{model_name} — Actual trend: {slope_act:+.4f}/yr, p={p_act:.4f} ({interpretation_act})")
    print(f"{model_name} — Pred trend:   {slope_pred:+.4f}/yr, p={p_pred:.4f} ({interpretation_pred})")
    print(f"{model_name} — Bias trend:   {slope_bias:+.4f}/yr, p={p_bias:.4f} ({interpretation_bias})")
    print("-" * 80)

    # save numeric summary row
    summary_rows.append({
        "model": model_name,
        "n_obs": len(common_idx),
        "slope_actual_per_year": slope_act,
        "pval_actual": p_act,
        "interpretation_actual": interpretation_act,
        "slope_pred_per_year": slope_pred,
        "pval_pred": p_pred,
        "interpretation_pred": interpretation_pred,
        "slope_bias_per_year": slope_bias,
        "pval_bias": p_bias,
        "interpretation_bias": interpretation_bias
    })

    # Plot: actual vs predicted with trendlines side-by-side in one figure
    fig, axes = plt.subplots(1, 2, figsize=(14,4), gridspec_kw={"width_ratios":[2,1]})
    ax = axes[0]
    ax.plot(actual.index, actual.values, '-o', linewidth=2, markersize=5, label="Actual")
    ax.plot(preds.index, preds["y_pred"].values, '-x', color='tab:orange', linewidth=1.5, markersize=5, label="Predicted")
    # trend lines: compute fitted values by regression on positions
    x = np.arange(len(actual))
    # actual fit
    coef_act = np.polyfit(x, actual.values, 1)
    fit_act = coef_act[0] * x + coef_act[1]
    # pred fit
    coef_pred = np.polyfit(x, preds["y_pred"].values, 1)
    fit_pred = coef_pred[0] * x + coef_pred[1]
    # convert monthly-fit to year-scale label: slope*12 used above
    ax.plot(actual.index, fit_act, '--', color='gray', label=f"Actual trend {slope_act:+.3f}/yr (p={p_act:.3f})")
    ax.plot(preds.index, fit_pred, '--', color='tab:orange', label=f"Pred trend   {slope_pred:+.3f}/yr (p={p_pred:.3f})")
    ax.set_title(f"{model_name}: Actual vs Predicted (monthly) + trends")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(loc='upper left', fontsize=9)

    # Right panel: bias series and its trend
    ax2 = axes[1]
    bias = preds["y_pred"] - actual
    ax2.plot(bias.index, bias.values, '-s', color='tab:red', markersize=5, label='Bias (pred-actual)')
    coef_bias = np.polyfit(x, bias.values, 1)
    fit_bias = coef_bias[0] * x + coef_bias[1]
    ax2.plot(bias.index, fit_bias, '--', color='brown', label=f"Bias trend {slope_bias:+.3f}/yr (p={p_bias:.3f})")
    ax2.axhline(0, color='k', linewidth=0.6)
    ax2.set_title("Bias & bias trend")
    ax2.grid(axis='y', linestyle=':', alpha=0.4)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, f"{model_name}_actual_vs_pred_trend.png")
    plt.savefig(out_png, dpi=DPI)
    plt.close(fig)
    print(f"Saved figure: {out_png}\n")

# save summary CSV
if summary_rows:
    pd.DataFrame(summary_rows).to_csv(os.path.join(OUT_DIR, "models_actual_vs_pred_trend_summary.csv"), index=False)
    print("Saved combined numeric summary to:", os.path.join(OUT_DIR, "models_actual_vs_pred_trend_summary.csv"))
else:
    print("No summaries generated (no overlapping data).")
