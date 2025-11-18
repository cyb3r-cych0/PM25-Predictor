"""
Generate supplementary-style predicted vs observed scatter plots (per model + combined grid).
Looks for prediction CSVs in Pipeline-Outputs/ with columns y_true,y_pred (or numeric first two).
Outputs:
  - Pipeline-Outputs/supplementary_<Model>.png  (one per model)
  - Pipeline-Outputs/supplementary_combined_grid.png  (2 cols x 3 rows)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import math

ROOT = Path(".")
PIPE = ROOT / "models_pipeline_data"
OUT = ROOT / "plot_figures" / "pred_vs_obs_plots"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["Ridge", "Lasso", "RandomForest", "GradientBoosting", "MLR", "LSTM"]
PANEL_LABELS = {
    "Ridge": "(a)",
    "Lasso": "(b)",
    "RandomForest": "(c)",
    "GradientBoosting": "(d)",
    "LSTM": "(e)",
    "MLR": "(f)"
}
GRID_ORDER = ["Ridge", "Lasso", "RandomForest", "GradientBoosting", "LSTM", "MLR"] 

# candidate filenames for a model
def find_pred_file(model):
    cand = [PIPE / f"{model}_FULL_predictions_2000_2025.csv",]
    for p in cand:
        if p.exists():
            return p
    return None

def load_preds(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    cols = list(df.columns)
    if "y_true" in cols and "y_pred" in cols:
        return df[["y_true","y_pred"]].dropna()
    # fallback: take first two numeric columns
    numcols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if len(numcols) >= 2:
        return df[[numcols[0], numcols[1]]].rename(columns={numcols[0]:"y_true", numcols[1]:"y_pred"}).dropna()
    raise ValueError(f"Could not interpret columns in {path}")

def fit_through_origin(x, y):
    """
    Fit y = k * x (no intercept).
    Returns k (slope), stderr_k (approx), and residuals.
    stderr_k computed as sqrt(sigma2 / sum(x^2)) where sigma2 = SSR/(n-1).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (~np.isnan(x)) & (~np.isnan(y))
    x = x[mask]; y = y[mask]
    n = len(x)
    if n < 2:
        return np.nan, np.nan, None, n
    denom = np.sum(x * x)
    if denom == 0:
        return np.nan, np.nan, None, n
    k = float(np.sum(x * y) / denom)
    residuals = y - k * x
    SSR = np.sum(residuals**2)
    # unbiased estimator with dof = n-1 (since intercept fixed to 0)
    sigma2 = SSR / max(n - 1, 1)
    var_k = sigma2 / denom
    stderr_k = float(math.sqrt(var_k)) if var_k >= 0 else np.nan
    return k, stderr_k, residuals, n

def stats_from_series(y_true, y_pred):
    mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
    if mask.sum() == 0:
        return {}
    y_true_a = np.asarray(y_true)[mask]
    y_pred_a = np.asarray(y_pred)[mask]
    r2 = float(r2_score(y_true_a, y_pred_a))
    rmse = float(np.sqrt(mean_squared_error(y_true_a, y_pred_a)))
    mae = float(mean_absolute_error(y_true_a, y_pred_a))
    mbe = float(np.mean(y_pred_a - y_true_a))
    return {"r2": r2, "rmse": rmse, "mae": mae, "mbe": mbe, "n": int(mask.sum())}

# plotting helper for single model
def plot_single_model(model, df, outpath, title_extra=None, xlim=(4.5,25.5), ylim=(4,26)):
    y_true = df["y_true"].values
    y_pred = df["y_pred"].values
    # compute fit through origin
    k, k_se, residuals, n = fit_through_origin(y_true, y_pred)  # note: fit y = k*x but example fits predicted vs obs -> use x = observed
    # We want slope of predicted vs observed: fit y_pred = k * y_true
    # If user wants reverse, adjust accordingly. Above call uses x=y_true, y=y_pred.

    stats = stats_from_series(y_true, y_pred)
    # slope stderr formatting
    slope_txt = f"Slope = {k:+.2f} ± {k_se:.2f}" if not np.isnan(k) else "Slope = n/a"

    # plotting
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(y_true, y_pred, s=30, alpha=0.8, edgecolor='none')
    # 1:1 dashed
    xr = np.array(xlim)
    ax.plot(xr, xr, ls='--', color='k', lw=1.2, label="1:1 ref")
    # weighted fit (no intercept): plot y = k * x
    if not np.isnan(k):
        xs = np.linspace(xlim[0], xlim[1], 200)
        ax.plot(xs, k * xs, color='#d62728', lw=2.2, label="Weighted fit (no intercept)")

    # annotate metrics box in top-left
    left = 0.02; top = 0.98
    box_lines = []
    box_lines.append(slope_txt)
    if "r2" in stats:
        box_lines.append(f"R² = {stats['r2']:.2f}")
        box_lines.append(f"RMSE = {stats['rmse']:.2f}")
        box_lines.append(f"MAE = {stats['mae']:.2f}")
        box_lines.append(f"MBE = {stats['mbe']:.2f}")
    box_text = "\n".join(box_lines)
    bbox = dict(boxstyle="round,pad=0.4", fc="white", ec="black", lw=0.8)
    ax.text(0.02, 0.98, box_text, transform=ax.transAxes, fontsize=10,
            va='top', ha='left', bbox=bbox)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("MERRA-2 PM₂.₅ (µg/m³)")
    ax.set_ylabel("Predicted PM₂.₅ (µg/m³)")
    panel = PANEL_LABELS.get(model, "")
    ax.set_title(f"{panel} {model} Model")
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print("Saved:", outpath)

# ---- combined 2x3 grid ----
def plot_combined_grid(model_dfs, outpath, order=GRID_ORDER):
    n = 6
    cols = 2
    rows = 3
    fig, axes = plt.subplots(rows, cols, figsize=(10,14))
    axes = axes.flatten()
    xlim=(4.5,25.5)
    ylim=(4,26)
    for i, model in enumerate(order):
        ax = axes[i]
        df = model_dfs.get(model)
        if df is None:
            ax.set_visible(False)
            continue
        y_true = df["y_true"].values
        y_pred = df["y_pred"].values
        k, k_se, residuals, nobs = fit_through_origin(y_true, y_pred)
        stats = stats_from_series(y_true, y_pred)
        ax.scatter(y_true, y_pred, s=20, alpha=0.8, edgecolor='none')
        xr = np.array(xlim)
        ax.plot(xr, xr, ls='--', color='k', lw=1.2, label="1:1 ref")
        if not np.isnan(k):
            xs = np.linspace(xlim[0], xlim[1], 200)
            ax.plot(xs, k*xs, color='#d62728', lw=1.8, label="Weighted fit (no intercept)")
        # small box
        slope_txt = f"Slope = {k:+.2f} ± {k_se:.2f}" if not np.isnan(k) else "Slope = n/a"
        lines = [slope_txt]
        if "r2" in stats:
            lines += [f"R² = {stats['r2']:.2f}", f"RMSE = {stats['rmse']:.2f}", f"MAE = {stats['mae']:.2f}", f"MBE = {stats['mbe']:.2f}"]
        txt = "\n".join(lines)
        ax.text(0.02, 0.98, txt, transform=ax.transAxes, va='top', ha='left', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.6))
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        panel = PANEL_LABELS.get(model, "")
        ax.set_title(f"{panel} {model} Model", fontsize=11)
        ax.set_xlabel("MERRA-2 PM₂.₅ (µg/m³)")
        ax.set_ylabel("Predicted PM₂.₅ (µg/m³)")
        ax.grid(alpha=0.2, linestyle=":")
        ax.legend(loc='lower right', fontsize=8)
    # hide any extra axes
    for j in range(len(order), rows*cols):
        axes[j].set_visible(False)
    plt.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print("Saved combined grid:", outpath)

# ---- main ----
model_dfs = {}
for model in MODELS:
    p = find_pred_file(model)
    if p is None:
        print(f"[skip] no preds found for {model}")
        continue
    try:
        df = load_preds(p)
    except Exception as e:
        print(f"[error] loading {p}: {e}")
        continue
    # ensure numeric
    df = df.astype(float)
    model_dfs[model] = df
    # single model file output (like example)
    outfile = OUT / f"{model}_pred_vs_obs_(MERRA-2).png"
    plot_single_model(model, df, outfile)
    
# combined grid
combined_out = OUT / "Grid_models_pred_vs_obs_(MERR-2).png"
plot_combined_grid(model_dfs, combined_out, order=GRID_ORDER)

print("All done. Files saved to:", OUT)
