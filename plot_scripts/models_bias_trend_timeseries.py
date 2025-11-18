#!/usr/bin/env python3
"""
Create per-model bias (pred - actual) time-series plots and a combined 2x3 figure.
Adds p-value to legend and uses 2-year x-axis tick steps. Titles prefixed (a)-(f).

Expect input files:
  Pipeline-Outputs/<MODEL>_preds_fullrange.csv
Columns required: index = datetime, columns 'y_true' and 'y_pred'

Outputs:
  Bias-Trend-Figures/<model>_bias_trend.png
  Bias-Trend-Figures/bias_trend_combined_2x3.png
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import matplotlib.dates as mdates

# ---------------- CONFIG ----------------
ROOT = Path(".")
PIPE = ROOT / "models_pipeline_data"
OUT = ROOT / "plot_figures" / "bias_trend_plots"
OUT.mkdir(exist_ok=True)

# Order determines (a)-(f) labeling and subplot positions
MODELS = ["GradientBoosting", "LSTM", "Lasso", "MLR", "RandomForest", "Ridge"]
LETTERS = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

# Plot range
YEAR_START = 2000
YEAR_END = 2026

plt.style.use("bmh")
LINE_COLOR = "#ff8c00"   # trend dashed color
BIAS_COLOR = "#1f77b4"   # bias point/line color

# ---------------- helpers ----------------
def decimal_year_from_index(idx):
    s = pd.to_datetime(idx)
    return s.year + s.dayofyear / 365.25

def fit_and_get_trend(x_dec, y):
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return None
    slope, intercept, r, pval, stderr = linregress(x_dec[mask], y[mask])
    return {"slope": slope, "intercept": intercept, "r": r, "p": pval, "stderr": stderr}

def plot_single_bias(model, df, outpath, add_title_prefix=None):
    df = df.sort_index()
    df["bias"] = df["y_pred"] - df["y_true"]
    x_dec = decimal_year_from_index(df.index)
    y = df["bias"].values

    trend = fit_and_get_trend(x_dec, y)
    # build trend line (for plotting) across YEAR_START..YEAR_END
    years_line = np.linspace(YEAR_START, YEAR_END, 200)
    if trend is not None:
        y_line = trend["intercept"] + trend["slope"] * years_line
    else:
        y_line = None

    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.plot(df.index, y, marker="o", ms=4, lw=1, linestyle="-", color=BIAS_COLOR, label="Residual (pred - actual)")

    if y_line is not None:
        # convert years_line to datetimes for plotting (approx mid-year)
        line_dates = pd.to_datetime(np.round(years_line).astype(int).astype(str))
        ax.plot(line_dates, y_line, linestyle="--", color=LINE_COLOR, label=f"Trend = ({trend['slope']:+.3f} µg/m³/yr), p=({trend['p']:.3f})")

    # ax.axhline(0, color="k", lw=0.8)
    title = f"{add_title_prefix + ' ' if add_title_prefix else ''}{model} Model"
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Residual (µg/m³)")
    ax.set_xlabel("Year")
    # x ticks: every 2 years at mid-year
    ax.set_xlim(pd.Timestamp(f"{YEAR_START}"), pd.Timestamp(f"{YEAR_END}"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.get_xticklabels(), rotation=0)
    ax.grid(alpha=0.35)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print("Saved:", outpath)
    return trend

# ---------------- run: individual figures ----------------
trends = {}
for i, model in enumerate(MODELS):
    csv_path = PIPE / f"{model}_FULL_predictions_2000_2025.csv"
    if not csv_path.exists():
        print(f"[skip] missing: {csv_path}")
        continue
    try:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    except Exception as e:
        print(f"[error] reading {csv_path}: {e}")
        continue
    letter = LETTERS[i]
    out_single = OUT / f"{model}_bias_trend.png"
    trend = plot_single_bias(model, df, out_single, add_title_prefix=letter)
    trends[model] = trend

# ---------------- combined 2x3 figure ----------------
# create grid 3 rows x 2 cols in the order of MODELS list (a-f)
nrows, ncols = 3, 2
fig, axes = plt.subplots(nrows, ncols, figsize=(18, 14))
axes = axes.flatten()

for i, model in enumerate(MODELS):
    ax = axes[i]
    csv_path = PIPE / f"{model}_FULL_predictions_2000_2025.csv"
    if not csv_path.exists():
        ax.set_visible(False)
        continue
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True).sort_index()
    df["bias"] = df["y_pred"] - df["y_true"]
    x_dec = decimal_year_from_index(df.index)
    y = df["bias"].values
    ax.plot(df.index, y, marker="o", ms=3.5, lw=1, linestyle="-", color=BIAS_COLOR, label="Residual (pred - actual)")

    trend = fit_and_get_trend(x_dec, y)
    if trend is not None:
        years_line = np.linspace(YEAR_START, YEAR_END, 200)
        y_line = trend["intercept"] + trend["slope"] * years_line
        line_dates = pd.to_datetime(np.round(years_line).astype(int).astype(str))
        ax.plot(line_dates, y_line, linestyle="--", color=LINE_COLOR, label=f"Trend = ({trend['slope']:+.3f} µg/m³/yr), p=({trend['p']:.3f})")

    # ax.axhline(0, color="k", lw=0.8)
    # title with letter prefix
    ax.set_title(f"{LETTERS[i]} {model} Model", fontsize=14)
    ax.set_ylabel("Residual (µg/m³)")
    ax.set_xlabel("Year")
    ax.set_xlim(pd.Timestamp(f"{YEAR_START}"), pd.Timestamp(f"{YEAR_END}"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.get_xticklabels(), rotation=0)
    ax.grid(alpha=0.35)
    ax.legend(fontsize=9, loc="upper left")

# hide any unused axes (if fewer than 6 models present)
for j in range(len(MODELS), nrows * ncols):
    axes[j].set_visible(False)

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
out_combined = OUT / "Grid_models_bias_trend.png"
fig.savefig(out_combined, dpi=300)
plt.close(fig)
print("Saved combined figure:", out_combined)
