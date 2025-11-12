# model_comparison_figures.py (datetime parsing fixed)
"""
Polished model vs actual comparison figures.
- removes deprecated infer_datetime_format
- forces explicit datetime parsing and reports any bad rows
- legend inside top-left, softened actual color, layout improved
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import linregress
import seaborn as sns
import warnings

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.45,
})

# ---------- EDIT THESE PATHS ----------
ROOT = Path(".")
PATHS = {
    "actuals": ROOT / "Datasets" / "aligned_monthly_outer_interp.csv",
    "RandomForest": ROOT / "Models-Training" / "RandomForest-Train" / "rf_with_lags_preds.csv",
    "GradientBoosting": ROOT / "Models-Training" / "GradientBoosting-Train" / "gbr_with_lags_preds.csv",
    "Lasso": ROOT / "Models-Training" / "Lasso-Train" / "lasso_with_lags_preds.csv",
    "Ridge": ROOT / "Models-Training" / "RidgeRegression-Train" / "ridge_with_lags_preds.csv",
    "LSTM": ROOT / "Models-Training" / "LSTM-Train" / "lstm_preds_vs_actual.csv"
}
OUT_DIR = ROOT / "Figures-Comparison"
OUT_DIR.mkdir(exist_ok=True)
# --------------------------------------

MODELS = ["RandomForest", "GradientBoosting", "Lasso", "Ridge", "LSTM"]

def safe_read_csv_with_index(path):
    """Read CSV with first column as index, then coerce index to datetime explicitly."""
    df = pd.read_csv(path, index_col=0, parse_dates=[0])  # no infer_datetime_format
    # coerce index to datetime explicitly, report failures
    idx_parsed = pd.to_datetime(df.index, errors='coerce')
    n_bad = idx_parsed.isna().sum()
    if n_bad > 0:
        warnings.warn(f"{n_bad} datetime index rows could not be parsed in {path}; they will be dropped.")
        df = df.loc[~pd.isna(idx_parsed)]
        idx_parsed = idx_parsed.loc[~pd.isna(idx_parsed)]
    df.index = idx_parsed
    return df

def read_preds(path):
    df = safe_read_csv_with_index(path)
    cols = list(df.columns)
    pred_col = None
    for c in cols:
        if "pred" in c.lower() or "prediction" in c.lower():
            pred_col = c; break
    if pred_col is None:
        pred_col = cols[1] if len(cols) >= 2 else cols[0]
    out = df[[pred_col]].rename(columns={pred_col: "pred"})
    return out

def read_actuals(path):
    df = safe_read_csv_with_index(path)
    col = None
    for c in df.columns:
        if "pm25" in c.lower() or "pm" in c.lower():
            col = c; break
    if col is None:
        col = df.columns[0]
    return df[[col]].rename(columns={col: "pm25"})

def align_by_positional(actuals, preds):
    inter = actuals.index.intersection(preds.index)
    if len(inter) > 0:
        a = actuals.loc[inter]
        p = preds.loc[inter]
        return a, p
    n = len(preds)
    if n == 0:
        return None, None
    a = actuals.iloc[-n:].copy()
    p = preds.copy()
    p.index = a.index
    return a, p

def year_fraction_index(idx):
    yrs = idx.year.astype(float)
    doy = idx.dayofyear.astype(float)
    return yrs + doy / 365.25

def compute_slope_and_p(series):
    x = year_fraction_index(series.index)
    y = series.values.astype(float)
    if len(x) < 2 or np.all(np.isnan(y)):
        return np.nan, np.nan
    res = linregress(x, y)
    return float(res.slope), float(res.pvalue)

def print_summary_tables():
    trend_csv = ROOT / "Models-Trend-Analysis" / "actual_vs_pred_trends" / "models_actual_vs_pred_trend_summary.csv"
    metrics_files = list((ROOT / "Models-Trend-Analysis" / "computed_trends_diagnostics").glob("*_trend_summary.csv"))

    df_trend = pd.read_csv(trend_csv)
    df_trend = df_trend.round(3)
    print("\n=== Trend Summary ===")
    print(df_trend.to_markdown(index=False))

    dfs = []
    for f in metrics_files:
        m = pd.read_csv(f)
        m["model"] = f.stem.split("_")[0]
        dfs.append(m)
    if dfs:
        df_metrics = pd.concat(dfs, ignore_index=True)
        df_metrics = df_metrics[["model", "r2", "rmse", "mae"]].round(3)
        print("\n=== Metrics Summary ===")
        print(df_metrics.to_markdown(index=False))

# ---------- Load data ----------
actuals = read_actuals(PATHS["actuals"])
preds_all = {}
for m in MODELS:
    p = read_preds(PATHS[m])
    a, p_al = align_by_positional(actuals, p)
    if a is None:
        raise RuntimeError(f"No data for model {m}")
    preds_all[m] = p_al["pred"]

# build common index (use union then sort)
common_index = None
for s in preds_all.values():
    common_index = s.index if common_index is None else common_index.union(s.index)
common_index = common_index.sort_values()
actual_ts = actuals.reindex(common_index).ffill().bfill()

# color palette and actual color softened
palette = sns.color_palette("tab10", n_colors=len(MODELS))
actual_color = "#517891" 

# ---------- 1) Timeseries: actual vs predictions ----------
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(12,9))
ax.plot(actual_ts.index, actual_ts["pm25"], color=actual_color, marker="o", lw=2.0, label="Actual", zorder=5, alpha=0.95)
for i, m in enumerate(MODELS):
    s = preds_all[m].reindex(actual_ts.index)
    ax.plot(actual_ts.index, s, label=m, marker="x", linestyle="--", linewidth=1.4, markersize=6, color=palette[i], alpha=0.95)
ax.set_title("Actual vs Model Predictions — Monthly PM2.5", fontsize=14)
ax.set_ylabel("PM2.5 (µg/m³)")
ax.set_xlim(actual_ts.index.min(), actual_ts.index.max())
ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
# legend inside top-left
ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), frameon=True, fontsize=9)
plt.tight_layout(rect=[0,0,0.98,1])
out1 = OUT_DIR / "all_models_vs_actual_timeseries.png"
fig.savefig(out1, dpi=200)
plt.close(fig)

# ---------- 2) Model trend comparison (bar chart) ----------
rows = []
s_act, p_act = compute_slope_and_p(actual_ts["pm25"])
for m in MODELS:
    slope, pval = compute_slope_and_p(preds_all[m])
    rows.append({"model": m, "slope_pred": slope, "pval_pred": pval})
df_trends = pd.DataFrame(rows).set_index("model")
df_trends["slope_actual"] = s_act
df_trends["pval_actual"] = p_act

fig, ax = plt.subplots(figsize=(11,6))
ind = np.arange(len(MODELS))
width = 0.35
ax.bar(ind - width/2, df_trends["slope_actual"].values, width, label="Actual trend (µg/yr)", color="lightgray", edgecolor="gray")
# handle palette list for bars
bar_colors = [palette[i] for i in range(len(MODELS))]
ax.bar(ind + width/2, df_trends["slope_pred"].values, width, label="Predicted trend (µg/yr)", color=bar_colors, edgecolor="black", alpha=0.9)
for i, m in enumerate(MODELS):
    ax.text(ind[i]-width/2, df_trends["slope_actual"].iloc[i] + 0.002, f"p={df_trends['pval_actual'].iloc[i]:.3f}", ha='center', va='bottom', fontsize=8)
    ax.text(ind[i]+width/2, df_trends["slope_pred"].iloc[i] + 0.002, f"p={df_trends['pval_pred'].iloc[i]:.3f}", ha='center', va='bottom', fontsize=8)
ax.set_xticks(ind)
ax.set_xticklabels(MODELS, rotation=30)
ax.axhline(0, color="k", linewidth=0.6)
ax.set_ylabel("Trend (µg/m³ per year)")
ax.set_title("Model trend comparison — Actual vs Predicted (annual slope)")
ax.legend(loc="upper left", bbox_to_anchor=(0.01,0.98), fontsize=9)
plt.tight_layout(rect=[0,0,0.98,1])
out2 = OUT_DIR / "model_trend_comparison_bars.png"
fig.savefig(out2, dpi=200)
plt.close(fig)

# ---------- 3) Predictions + Trends composite ----------
fig = plt.figure(figsize=(14,7))
gs = fig.add_gridspec(1, 2, width_ratios=[2.6,1])
ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1])

ax0.plot(actual_ts.index, actual_ts["pm25"], color=actual_color, lw=2.0, marker="o", label="Actual")
for i, m in enumerate(MODELS):
    s = preds_all[m].reindex(actual_ts.index)
    ax0.plot(actual_ts.index, s, label=m, lw=1.4, linestyle="--", marker=".", markersize=5, color=palette[i])
ax0.set_title("Actual & Model Predictions — Time series")
ax0.set_ylabel("PM2.5 (µg/m³)")
ax0.legend(loc="upper left", bbox_to_anchor=(0.01,0.98), fontsize=9)

years = year_fraction_index(actual_ts.index)
a_slope, a_p = compute_slope_and_p(actual_ts["pm25"])
yfit_act = a_slope * years + (np.nanmean(actual_ts["pm25"]) - a_slope*np.nanmean(years))
ax1.plot(years, yfit_act, color=actual_color, lw=2.2, label=f"Actual {a_slope:+.3f}/yr (p={a_p:.3f})")
for i, m in enumerate(MODELS):
    s_series = preds_all[m].reindex(actual_ts.index)
    slope, pval = compute_slope_and_p(s_series)
    intercept = np.nanmean(s_series.fillna(np.nanmean(s_series))) - slope*np.nanmean(years)
    ax1.plot(years, slope*years + intercept, label=f"{m} {slope:+.3f}/yr (p={pval:.3f})", color=palette[i], lw=1.6)
ax1.set_title("Linear trend lines (annual)")
ax1.set_xlabel("Year (fractional)")
ax1.set_ylabel("PM2.5 (µg/m³)")
ax1.legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.01,0.98))

plt.tight_layout(rect=[0,0,0.98,1])
out3 = OUT_DIR / "predictions_and_trends_comparison.png"
fig.savefig(out3, dpi=200)
plt.close(fig)

print("Saved figures to:", OUT_DIR)
print("-----" * 10)
print("Printing Summary Tables:")
print("-----" * 10)
print_summary_tables()
