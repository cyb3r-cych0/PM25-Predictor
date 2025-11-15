"""
Produce one publication-style PNG per model showing:
 - monthly Actual (thin line + markers)
 - monthly Predicted (thin line + markers)
 - linear trend fit on the monthly Actual and Predicted series (no yearly aggregation)
 - Title prefixed (a), (b), ... ; labeled axes; mid-year ticks with configurable spacing.

Expects CSVs under PIPE_DIR like: <Model>_preds_fullrange.csv
CSV must include two numeric columns (y_true,y_pred or similar).
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import warnings, os
warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from classP import PredictionVisualizer

# ---------- CONFIG ----------
ROOT = Path(".")
PIPE_DIR = ROOT / "models_pipeline_data"
OUT_DIR = ROOT / "plot_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# order of models and letter prefixes for titles
MODELS = ["GradientBoosting", "LSTM", "Lasso", "MLR", "RandomForest", "Ridge"]
LETTERS = [f"({chr(97+i)})" for i in range(len(MODELS))]  # (a), (b), ...

YEAR_START = 2000
YEAR_END = 2026
YEAR_STEP = 2 
FIGSIZE = (14, 4.2)
plt.style.use("bmh")

# colors
ACTUAL_COLOR = "#0b4f6c"
PRED_COLOR = "#ff7f0e"
TREND_ACTUAL_COLOR = "#1f2e3a" 
TREND_PRED_COLOR = "#e91e63"    
MODEL_COLORS = {
    "RandomForest": "#2ca02c",
    "GradientBoosting": "#ff7f0e",
    "Lasso": "#9c27b0",
    "Ridge": "#03a9f4",
    "MLR": "#6d4c41",
    "LSTM": "#e91e63"
}

# ---------- helpers ----------
def find_preds_csv(model_name):
    candidates = [PIPE_DIR / f"{model_name}_FULL_predictions_2000_2025.csv"]
    for p in candidates:
        if p.exists():
            return p
    return None

def read_preds(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    cols = list(df.columns)
    if "y_true" in cols and "y_pred" in cols:
        return df[["y_true","y_pred"]].sort_index()
    # fallback: choose first two numeric columns and rename
    numcols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if len(numcols) >= 2:
        return df[[numcols[0], numcols[1]]].rename(columns={numcols[0]:"y_true", numcols[1]:"y_pred"}).sort_index()
    # final fallback: first two columns
    return df.iloc[:, :2].rename(columns={df.columns[0]:"y_true", df.columns[1]:"y_pred"}).sort_index()

def year_fraction_from_index(idx):
    # idx: DatetimeIndex -> returns array of floats like 2000.083, 2000.167, ...
    dt = pd.to_datetime(idx)
    years = dt.year.astype(float)
    frac = (dt.dayofyear.astype(float) - 1) / 365.25
    return years + frac

def fit_trend_on_monthly(series):
    # series: pandas Series with datetime index
    if series.dropna().shape[0] < 2:
        return np.nan, np.nan, None
    x = year_fraction_from_index(series.index)
    y = series.values.astype(float)
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return np.nan, np.nan, None
    res = linregress(x[mask], y[mask])
    xs = np.linspace(YEAR_START, YEAR_END, 200)
    ys = res.slope * xs + res.intercept
    # convert xs to datetimes (mid-year or year-end); use year-end for plotting
    xs_dt = pd.to_datetime(xs.astype(int).astype(str) + "-12-31")
    return float(res.slope), float(res.pvalue), (xs_dt, ys)

# ---------- main: loop models, produce single PNG each ----------
for i, model in enumerate(MODELS):
    csvp = find_preds_csv(model)
    if csvp is None:
        print(f"[skip] predictions not found for {model} (no CSV in {PIPE_DIR})")
        continue

    try:
        df = read_preds(csvp)
    except Exception as e:
        print(f"[error] reading {csvp}: {e}")
        continue

    # ensure datetime index and clip to desired window
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[(df.index >= f"{YEAR_START}-01-01") & (df.index <= f"{YEAR_END}-12-31")]

    if df.shape[0] == 0:
        print(f"[skip] {model}: no data in {YEAR_START}-{YEAR_END}")
        continue

    ser_actual = df["y_true"].astype(float)
    ser_pred = df["y_pred"].astype(float)

    # compute trend directly on monthly series
    slope_a, p_a, line_a = fit_trend_on_monthly(ser_actual)
    slope_p, p_p, line_p = fit_trend_on_monthly(ser_pred)

    # plot
    fig, ax = plt.subplots(figsize=FIGSIZE)

    # monthly series (thin lines + small markers)
    ax.plot(ser_actual.index, ser_actual.values,
            marker="o", ms=3, lw=1.1, color=ACTUAL_COLOR, alpha=0.95, label="Actual")
    ax.plot(ser_pred.index, ser_pred.values,
            marker="o", ms=3, lw=1.0, color=PRED_COLOR, alpha=0.9, label="Predicted")

    # trend lines (over full year span)
    if line_a is not None:
        ax.plot(line_a[0], line_a[1], color=TREND_ACTUAL_COLOR, lw=2.0, label=f"Actual trend ({slope_a:.3f}/yr, p={p_a:.3f})")
    if line_p is not None:
        ax.plot(line_p[0], line_p[1], color=TREND_PRED_COLOR, lw=1.6, ls="--", label=f"Pred trend ({slope_p:.3f}/yr, p={p_p:.3f})")

    # title with prefix
    prefix = LETTERS[i] if i < len(LETTERS) else ""
    ax.set_title(f"{prefix} {model}", fontsize=14, fontweight="bold")

    # labels
    ax.set_xlabel("Year")
    ax.set_ylabel("PM2.5 (µg/m³)")

    # x ticks: mid-year ticks with YEAR_STEP spacing
    years = np.arange(YEAR_START, YEAR_END + 1, YEAR_STEP)
    ax.set_xticks(pd.to_datetime([f"{y}-06-30" for y in years]))
    ax.set_xlim(pd.Timestamp(f"{YEAR_START}-01-01"), pd.Timestamp(f"{YEAR_END}-12-31"))
    ax.tick_params(axis="x", rotation=25)

    ax.grid(True, linestyle=":", alpha=0.8)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    out_png = OUT_DIR / f"{i+1:02d}_{model}_actual_vs_pred_trends.png"
    plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print("Saved:", out_png)

print("Done. Individual publication-style Actual vs Predicted trend PNGs in:", OUT_DIR)

# ---------- combined plot of all models (6 per page) ----------
plot_generator = PredictionVisualizer(root_dir=ROOT, output_dir_name=OUT_DIR)
plot_generator.generate_plot()
# ---- single plot function (for reference) ----
