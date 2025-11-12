"""
Robust trend computation & diagnostic plotting for model predictions.
This version improves datetime parsing and alignment so CSVs without
explicit date indices will align positionally to the master aligned CSV.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SCRIPT_DIR = Path(__file__).resolve().parent
ALIGNED_CSV_PATH = SCRIPT_DIR.parent / "Datasets" / "aligned_monthly_outer_interp.csv"

MODEL_FILES = {
    "RandomForest": SCRIPT_DIR.parent / "Models-Training" / "RandomForest-Train" / "rf_with_lags_preds.csv",
    "GradientBoosting": SCRIPT_DIR.parent / "Models-Training" / "GradientBoosting-Train" / "gbr_with_lags_preds.csv",
    "Lasso": SCRIPT_DIR.parent / "Models-Training" / "Lasso-Train" / "lasso_with_lags_preds.csv",
    "Ridge": SCRIPT_DIR.parent / "Models-Training" / "RidgeRegression-Train" / "ridge_with_lags_preds.csv",
    "LSTM": SCRIPT_DIR.parent / "Models-Training" / "LSTM-Train" / "lstm_preds_vs_actual.csv"
}

OUT_DIR = SCRIPT_DIR / "computed_trends_diagnostics"
OUT_DIR.mkdir(exist_ok=True)

def try_read_with_index(path):
    """Try to read CSV and keep index as datetime if possible."""
    # attempt 1: read with index_col=0 and parse_dates
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        # if index parsed to datetime with >0 non-null entries, accept
        if isinstance(df.index, pd.DatetimeIndex) and df.index.notna().sum() > 0:
            return df
    except Exception:
        pass
    # fallback: read without index parsing
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to read {path}: {e}")

def detect_target_pred_cols(df):
    """Return (y_true_col, y_pred_col) column names or (None,None)."""
    cols_low = [c.lower() for c in df.columns]
    # candidate names for truth and pred
    truth_candidates = ["y_true", "y", "actual", "observed", "true"]
    pred_candidates = ["y_pred", "yhat", "y_hat", "pred", "prediction", "y_predicted"]
    y_true_col = None
    y_pred_col = None
    for cand in truth_candidates:
        if cand in cols_low:
            y_true_col = df.columns[cols_low.index(cand)]
            break
    for cand in pred_candidates:
        if cand in cols_low:
            y_pred_col = df.columns[cols_low.index(cand)]
            break
    # if still missing, fall back to numeric columns
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if y_true_col is None:
        if len(numeric) >= 1:
            # assume first numeric column is truth if there's a second numeric column for pred
            y_true_col = numeric[0]
    if y_pred_col is None:
        if len(numeric) >= 2:
            y_pred_col = numeric[1]
        elif len(numeric) == 1:
            # only one numeric column present — ambiguous
            y_pred_col = None
    return y_true_col, y_pred_col

def load_preds(path):
    df_raw = try_read_with_index(path)
    df = df_raw.copy()
    # detect y_true/y_pred names
    y_true_col, y_pred_col = detect_target_pred_cols(df)
    if y_true_col is None or y_pred_col is None:
        # provide diagnostics for user
        raise RuntimeError(f"Couldn't detect y_true/y_pred in {path}. Columns: {list(df.columns)}")
    # if a datetime-like column exists as a regular column, try to set it as index
    for c in df.columns:
        if c.lower().count("date") or c.lower().count("time"):
            try:
                s = pd.to_datetime(df[c], errors="coerce")
                if s.notna().sum() > 0:
                    df = df.set_index(s)
                    break
            except Exception:
                pass
    # ensure we only keep the two columns
    out = df[[y_true_col, y_pred_col]].copy()
    out.columns = ["y_true", "y_pred"]
    # if index is datetime-like strings, try parsing
    if not isinstance(out.index, pd.DatetimeIndex):
        try:
            parsed = pd.to_datetime(out.index, errors="coerce")
            if parsed.notna().sum() > 0:
                out.index = parsed
        except Exception:
            pass
    return out

def year_frac_from_index(idx):
    if not isinstance(idx, pd.DatetimeIndex):
        return np.arange(len(idx))
    years = idx.year.astype(float)
    months = idx.month.astype(float)
    days = idx.day.astype(float)
    return years + (months-1)/12.0 + (days-1)/365.0

def linear_trend(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return slope, intercept

def plot_pred_vs_actual(index, y_true, y_pred, model_name, out_dir):
    x_num = year_frac_from_index(index)
    s_true, i_true = linear_trend(x_num, y_true)
    s_pred, i_pred = linear_trend(x_num, y_pred)
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(index, y_true, label="actual", marker='o', linewidth=1)
    ax.plot(index, y_pred, label="predicted", marker='x', linewidth=1)
    ax.plot(index, s_true * x_num + i_true, label=f"actual trend ({s_true:.3f}/yr)", linestyle='--', color='tab:blue')
    ax.plot(index, s_pred * x_num + i_pred, label=f"pred trend ({s_pred:.3f}/yr)", linestyle='--', color='tab:orange')
    ax.set_title(f"{model_name}: Predicted vs Actual with Trendlines")
    ax.set_ylabel("PM2.5")
    ax.legend(); ax.grid(True); fig.tight_layout()
    fpath = os.path.join(out_dir, f"{model_name}_pred_vs_actual.png")
    fig.savefig(fpath); plt.close(fig)
    return s_true, s_pred

def plot_bias_trend(index, bias, model_name, out_dir):
    x_num = year_frac_from_index(index)
    s_bias, i_bias = linear_trend(x_num, bias)
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(index, bias, label="bias (pred - actual)", marker='o')
    ax.plot(index, s_bias * x_num + i_bias, label=f"bias trend ({s_bias:.3f}/yr)", linestyle='--')
    ax.axhline(0, color='k', linewidth=0.6)
    ax.set_title(f"{model_name}: Bias Trend")
    ax.set_ylabel("Bias (µg/m³)")
    ax.legend(); ax.grid(True); fig.tight_layout()
    fpath = os.path.join(out_dir, f"{model_name}_bias_trend.png")
    fig.savefig(fpath); plt.close(fig)
    return s_bias

def summarize_and_save(model_name, index, y_true, y_pred, out_dir):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    s_true, s_pred = plot_pred_vs_actual(index, y_true, y_pred, model_name, out_dir)
    bias = y_pred - y_true
    s_bias = plot_bias_trend(index, bias, model_name, out_dir)
    df_tmp = pd.DataFrame({"actual": y_true, "pred": y_pred}, index=index)
    df_tmp["mae_rolling_12"] = (df_tmp["pred"] - df_tmp["actual"]).abs().rolling(12, min_periods=1).mean()
    fig, ax = plt.subplots(figsize=(10,3))
    ax.plot(df_tmp.index, df_tmp["mae_rolling_12"], label="12-month rolling MAE")
    ax.set_title(f"{model_name}: Rolling MAE (12 months)")
    ax.set_ylabel("MAE"); ax.grid(True); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{model_name}_rolling_mae.png"))
    plt.close(fig)
    summary = {
        "model": model_name,
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "trend_actual_per_year": s_true,
        "trend_pred_per_year": s_pred,
        "trend_bias_per_year": s_bias,
        "n_obs": len(y_true)
    }
    pd.DataFrame([summary]).to_csv(os.path.join(out_dir, f"{model_name}_trend_summary.csv"), index=False)
    print(f"[{model_name}] R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, trend_actual={s_true:.4f}/yr, trend_pred={s_pred:.4f}/yr, bias_trend={s_bias:.4f}/yr")

# ---------------- Main ----------------
if not ALIGNED_CSV_PATH.exists():
    raise FileNotFoundError(f"Missing aligned CSV: {ALIGNED_CSV_PATH}")

aligned_df = pd.read_csv(ALIGNED_CSV_PATH, index_col=0, parse_dates=True)
print(f"Loaded actuals from: {ALIGNED_CSV_PATH}")
print("Actuals shape:", aligned_df.shape)
print("\nLoading model predictions (diagnostics shown):")
for model_name, path in MODEL_FILES.items():
    print(f"- {model_name}: {path} -> exists={path.exists()}")

print("\nProcessing models...\n")
for model_name, pred_path in MODEL_FILES.items():
    if not pred_path.exists():
        print(f"Skipping {model_name}: file not found: {pred_path}")
        continue
    try:
        preds = load_preds(pred_path)
    except Exception as e:
        print(f"Error reading {model_name} predictions: {e}")
        print(" First rows / cols preview:")
        try:
            print(pd.read_csv(pred_path, nrows=5))
        except Exception:
            print("  (preview failed)")
        continue

    print(f"\n[{model_name}] initial preds shape: {preds.shape}, index type: {type(preds.index)}")
    # Align to aligned_df: prefer timestamp intersection, otherwise positional (last N)
    if isinstance(preds.index, pd.DatetimeIndex) and isinstance(aligned_df.index, pd.DatetimeIndex):
        common_idx = aligned_df.index.intersection(preds.index)
        if len(common_idx) > 0:
            preds = preds.reindex(common_idx).dropna()
            print(f"  Using intersection of timestamps: {len(common_idx)} -> kept {len(preds)} rows after dropna")
        else:
            # no overlap: fallback to positional
            print("  No timestamp overlap with aligned data; falling back to positional alignment (last N rows).")
            preds.index = aligned_df.index[-len(preds):]
            preds = preds.dropna()
            print(f"  After positional assignment -> {len(preds)} rows")
    else:
        # positional alignment (set index to last N timestamps)
        preds.index = aligned_df.index[-len(preds):]
        # count NaNs before drop
        n_before = len(preds)
        n_nans = preds.isna().any(axis=1).sum()
        if n_nans > 0:
            print(f"  Positional alignment set index to last {n_before} timestamps; {n_nans} rows contain NaN and will be dropped.")
        preds = preds.dropna()
        print(f"  After dropna -> {len(preds)} rows")

    if len(preds) == 0:
        print(f"After alignment, no rows for {model_name}. Skipping. Preview of preds head:")
        print(pd.read_csv(pred_path, nrows=5))
        continue

    # final arrays
    y_true = preds["y_true"].values
    y_pred = preds["y_pred"].values
    index = preds.index

    summarize_and_save(model_name, index, y_true, y_pred, str(OUT_DIR))

print("\nAll done. Outputs written to:", OUT_DIR)
