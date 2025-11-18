#!/usr/bin/env python3
"""
    - Rebuilds full-range preds from trained models (sklearn joblib & LSTM .h5 + scalers).
    - Recreates Pipeline-Outputs/models_actual_vs_pred_trend_summary.csv
    - Writes publication-style figures:
        * Combined grid: Full-Range publication (3x2)
        * Individual per-model publication PNGs
    - Expects an aligned monthly dataset (month-end index) at:
        Datasets/aligned_monthly_inner.csv
    Adjust PATH constants below if your files are elsewhere.
"""
import os
# suppress TF / oneDNN info spam early
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import json
import pandas as pd
import numpy as np
from scipy.stats import linregress
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------------- CONFIG ----------------
ROOT = Path(".")
PIPE = ROOT / "models_pipeline_data"
PIPE.mkdir(parents=True, exist_ok=True)
DATASETS = ROOT / "meteo_data"
ALIGNED_CSV = DATASETS / "aligned_monthly_inner.csv"   # canonical aligned monthly (month-end)
OUT_TRND = PIPE / "models_actual_vs_pred_trend_summary.csv"

OUT_FIG_DIR = ROOT / "plot_figures" / "actual_vs_pred_trends_plots"
OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
INDIV_DIR = OUT_FIG_DIR

# models to consider and candidate filenames for robustness
MODEL_KEYS = ["RandomForest", "GradientBoosting", "Lasso", "Ridge", "MLR", "LSTM"]
SKLEARN_CANDIDATES = {
    "RandomForest": ["rf_model.joblib", "RandomForest_model.joblib", "RandomForest.joblib", "RandomForest_model.pkl"],
    "GradientBoosting": ["gbr_model.joblib", "GradientBoosting_model.joblib", "GradientBoosting.joblib"],
    "Lasso": ["lasso_model.joblib","Lasso_model.joblib","lasso.joblib"],
    "Ridge": ["ridge_model.joblib","Ridge_model.joblib","ridge.joblib"],
    "MLR": ["mlr_model.joblib","MLR_model.joblib","mlr.joblib","linear_model.joblib"]
}
LSTM_CANDIDATES = ["lstm_pm25_model.h5", "lstm_best.h5", "lstm_model.h5", "lstm_pm25_model.keras"]
LSTM_SCALER_X_CAND = ["lstm_scaler_X.joblib", "scaler_X.joblib"]
LSTM_SCALER_Y_CAND = ["lstm_scaler_y.joblib", "scaler_y.joblib"]

# lag / target config (must match training)
LAGS = [1,2,3,12]
TARGET = "pm25"
LSTM_LOOKBACK = 12

# plotting / publication config
YEAR_START = 2000
YEAR_END = 2026
plt.style.use("bmh")
MONTHLY_COLOR = "#0b4f6c"
MODEL_COLORS = {
    "RandomForest": "#2ca02c",
    "GradientBoosting": "#ff7f0e",
    "Lasso": "#9c27b0",
    "Ridge": "#03a9f4",
    "MLR": "#6d4c41",
    "LSTM": "#e91e63"
}
PANEL_ORDER = ["GradientBoosting","LSTM","Lasso","MLR","RandomForest","Ridge"]  # (a)->(f) order for titles

# ---------------- helpers ----------------
def find_first_existing(parent: Path, candidates):
    for c in candidates:
        p = parent / c
        if p.exists():
            return p
    return None

def try_load_sklearn_model(model_name):
    cand = SKLEARN_CANDIDATES.get(model_name, [])
    p = find_first_existing(PIPE, cand)
    if p:
        try:
            return joblib.load(p), p
        except Exception as e:
            print(f"[error] loading {p}: {e}")
    # fallback: try common filename {model}_preds_fullrange.csv presence indicates we already have preds
    return None, None

def try_find_lstm():
    p = None
    for c in LSTM_CANDIDATES:
        t = PIPE / c
        if t.exists():
            p = t
            break
    return p

def try_find_scaler(name_cands):
    return find_first_existing(PIPE, name_cands)

def make_lags(df, lags=LAGS):
    out = df.copy()
    for lag in lags:
        for c in df.columns:
            out[f"{c}_lag{lag}"] = out[c].shift(lag)
    out = out.dropna()
    return out

def compute_yearly_means(series):
    yr = series.resample("YE").mean()
    years = yr.index.year.astype(float)
    return yr, years

def fit_trend(years, values):
    if len(values) < 2 or np.all(np.isnan(values)):
        return np.nan, np.nan, None
    mask = ~np.isnan(values)
    if mask.sum() < 2:
        return np.nan, np.nan, None
    res = linregress(years[mask], values[mask])
    line_x = np.linspace(YEAR_START, YEAR_END, 200)
    line_y = res.slope * line_x + res.intercept
    return float(res.slope), float(res.pvalue), (line_x, line_y)

# --- ADD: bootstrap CI for yearly slope (paired bootstrap)
def compute_trend_with_ci_bootstrap(years, values, n_boot=2000, alpha=0.05):
    years = np.asarray(years)
    values = np.asarray(values)
    mask = ~np.isnan(values)
    years = years[mask]; values = values[mask]
    if len(values) < 3:
        return np.nan, np.nan, np.nan, np.nan
    res = linregress(years, values)
    rng = np.random.default_rng(seed=42)
    slopes = []
    n = len(values)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yrs_b = years[idx]; vals_b = values[idx]
        order = np.argsort(yrs_b)
        r = linregress(yrs_b[order], vals_b[order])
        slopes.append(r.slope)
    slopes = np.array(slopes)
    ci_low = np.percentile(slopes, 100*alpha/2)
    ci_high = np.percentile(slopes, 100*(1-alpha/2))
    return ci_low, ci_high


# ---------------- load aligned data ----------------
if not ALIGNED_CSV.exists():
    raise FileNotFoundError(f"Aligned monthly CSV not found: {ALIGNED_CSV}")

print("Loading aligned data:", ALIGNED_CSV)
df = pd.read_csv(ALIGNED_CSV, index_col=0, parse_dates=True).sort_index()
# coerce to month-end timestamps if needed
if not (df.index.is_month_end).all():
    df.index = pd.to_datetime(df.index)
    df.index = df.index.to_period("M").to_timestamp("M")
    df = df[~df.index.duplicated(keep="first")].sort_index()

print("Aligned shape:", df.shape)
df_lag = make_lags(df, LAGS)
print("After lag creation (usable rows):", df_lag.shape)
X_full = df_lag.drop(columns=[TARGET])
y_full = df_lag[TARGET]
full_index = df_lag.index

# ---------------- produce sklearn predictions ----------------
preds = {}   # name -> DataFrame(index, y_true, y_pred)
print("\nProducing sklearn model full-range predictions (if model files present)...")
for name in ["RandomForest","GradientBoosting","Lasso","Ridge","MLR"]:
    model, ppath = try_load_sklearn_model(name)
    if model is None:
        # fallback: try to load existing preds file
        cand_csv = PIPE / f"{name}_preds_fullrange.csv"
        cand_csv2 = PIPE / f"{name}_FULL_predictions_2000_2025.csv"
        if cand_csv.exists():
            try:
                dfp = pd.read_csv(cand_csv, index_col=0, parse_dates=True)
                preds[name] = dfp.sort_index()
                print(f"Loaded existing preds for {name} from {cand_csv.name}")
                continue
            except Exception:
                pass
        if cand_csv2.exists():
            try:
                dfp = pd.read_csv(cand_csv2, index_col=0, parse_dates=True)
                preds[name] = dfp.sort_index()
                print(f"Loaded existing preds for {name} from {cand_csv2.name}")
                continue
            except Exception:
                pass
        print(f"[skip] no model file or preds for {name}")
        continue
    try:
        yhat = model.predict(X_full)
        dfp = pd.DataFrame({"y_true": y_full.values, "y_pred": np.asarray(yhat).ravel()}, index=full_index)
        outcsv = PIPE / f"{name}_preds_fullrange.csv"
        dfp.to_csv(outcsv)
        preds[name] = dfp
        print("Saved full-range preds:", outcsv.name)
    except Exception as e:
        print(f"[error] predicting with {name}: {e}")

# ---------------- LSTM full-range sliding-window ----------------
print("\nAttempting LSTM full-range predictions...")
lstm_path = try_find_lstm()
if lstm_path:
    try:
        # import lazily so script still works without TF if not needed
        from tensorflow.keras.models import load_model
        scaler_x_path = try_find_scaler(LSTM_SCALER_X_CAND)
        scaler_y_path = try_find_scaler(LSTM_SCALER_Y_CAND)
        if scaler_x_path and scaler_y_path:
            scaler_X = joblib.load(scaler_x_path)
            scaler_y = joblib.load(scaler_y_path)
            lstm = load_model(str(lstm_path))
            Xmat = X_full.values.astype(float)
            Xs = scaler_X.transform(Xmat)
            seqs = []
            idxs = []
            for i in range(LSTM_LOOKBACK, len(Xs)):
                seqs.append(Xs[i-LSTM_LOOKBACK:i])
                idxs.append(X_full.index[i])
            if len(seqs) > 0:
                import numpy as _np
                Xseq = _np.stack(seqs)
                yhat_scaled = lstm.predict(Xseq, verbose=0).ravel()
                yhat = scaler_y.inverse_transform(yhat_scaled.reshape(-1,1)).ravel()
                ytrue = y_full.loc[idxs]
                df_lstm = pd.DataFrame({"y_true": ytrue.values, "y_pred": yhat}, index=idxs)
                outcsv = PIPE / "LSTM_FULL_predictions_2000_2025.csv"
                df_lstm.to_csv(outcsv)
                preds["LSTM"] = df_lstm
                print("Saved full-range LSTM preds ->", outcsv.name)
            else:
                print("Not enough rows to build LSTM sequences for full-range.")
        else:
            print("LSTM scaler files not found; skipping LSTM full-range predictions.")
    except Exception as e:
        print("LSTM prediction failed:", e)
else:
    # fallback to existing preds files
    alt = PIPE / "LSTM_FULL_predictions_2000_2025.csv"
    if alt.exists():
        try:
            preds["LSTM"] = pd.read_csv(alt, index_col=0, parse_dates=True)
            print("Loaded existing LSTM full-range preds:", alt.name)
        except Exception:
            pass
    else:
        print("No LSTM model present; skipping.")

# ---------------- fallback: if any model no preds but preds_vs_actual (test-only) exist, load them ----------------
for m in ["RandomForest","GradientBoosting","Lasso","Ridge","MLR","LSTM"]:
    if m not in preds:
        f = PIPE / f"{m}_preds_fullrange.csv"
        f2 = PIPE / f"{m}_FULL_predictions_2000_2025.csv"
        for cand in (f, f2):
            if cand.exists():
                try:
                    dfp = pd.read_csv(cand, index_col=0, parse_dates=True)
                    preds[m] = dfp.sort_index()
                    print(f"Loaded fallback preds for {m} from {cand.name}")
                    break
                except Exception:
                    continue

# ---------------- build trend summaries CSV ----------------
trend_rows = []
for name, dfp in preds.items():
    try:
        dfp = dfp.sort_index()
        ser_true = pd.Series(dfp["y_true"].values, index=pd.to_datetime(dfp.index)).sort_index()
        ser_pred = pd.Series(dfp["y_pred"].values, index=pd.to_datetime(dfp.index)).sort_index()
        yr_true, yrs = compute_yearly_means(ser_true)
        yr_pred, _ = compute_yearly_means(ser_pred)
        slope_true, p_true, _ = fit_trend(yrs, yr_true.values)
        slope_pred, p_pred, _ = fit_trend(yrs, yr_pred.values)
        slope_bias, p_bias, _ = fit_trend(yrs, (yr_pred - yr_true).values)
        trend_rows.append({
            "model": name,
            "n_obs": int(ser_true.notna().sum()),
            "slope_actual_per_year": slope_true,
            "pval_actual": p_true,
            "slope_pred_per_year": slope_pred,
            "pval_pred": p_pred,
            "slope_bias_per_year": slope_bias,
            "pval_bias": p_bias
        })
    except Exception as e:
        print(f"[warn] trend summary failed for {name}: {e}")

if len(trend_rows) == 0:
    print("No trend rows computed; exiting.")
else:
    trend_df = pd.DataFrame(trend_rows).set_index("model").sort_index()
    trend_df.to_csv(OUT_TRND)
    print("\nSaved trend summary CSV ->", OUT_TRND)
    print(trend_df)

# ---------------- plotting functions (publication style) ----------------
def plot_per_model_publication(name, dfp, outpath, panel_label=None):
    dfp = dfp.sort_index()
    idx = pd.to_datetime(dfp.index)
    ser_true = pd.Series(dfp["y_true"].values, index=idx)
    ser_pred = pd.Series(dfp["y_pred"].values, index=idx)

    yr_true, yrs = compute_yearly_means(ser_true)
    yr_pred, _ = compute_yearly_means(ser_pred)
    slope_t, p_t, line_t = fit_trend(yrs, yr_true.values)
    slope_p, p_p, line_p = fit_trend(yrs, yr_pred.values)
    ci_t_low, ci_t_high = compute_trend_with_ci_bootstrap(yrs.values, yr_pred.values)

    fig, ax = plt.subplots(figsize=(10,4.2))
    # monthly lines + markers
    ax.plot(ser_true.index, ser_true.values, marker="o", ms=4, linewidth=1.0, color=MONTHLY_COLOR, label="Actual")
    ax.plot(ser_pred.index, ser_pred.values, marker="o", ms=4, linewidth=1.1, color=MODEL_COLORS.get(name,"#777777"), label="Predicted")
    # trend lines (from seasonal-yearly fit) - draw as full-range line_x -> convert to datetimes
    if line_t is not None:
        line_x_dt = pd.to_datetime(np.round(line_t[0]).astype(int).astype(str))
        ax.plot(line_x_dt, line_t[1], color="gray", lw=1.0, label=f"Actual Trend = ({slope_t:.3f} µg/m³/yr), p=({p_t:.3f})")
    if line_p is not None:
        line_x_dt = pd.to_datetime(np.round(line_p[0]).astype(int).astype(str))
        ci = (f"95% CI = ({ci_t_low:.3f}, {ci_t_high:.3f} µg/m³/yr)")
        ax.plot(line_x_dt, line_p[1], color=MODEL_COLORS.get(name,"#777777"), lw=1.5, ls="--", label=f"Pred Trend = ({slope_p:.3f} µg/m³/yr), {ci}, p=({p_p:.3f})")

    # axis & ticks
    ax.set_xlim(pd.Timestamp(f"{YEAR_START}"), pd.Timestamp(f"{YEAR_END}"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))           # tick every 3 years
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))    # format as 'YYYY'
    plt.setp(ax.get_xticklabels(), rotation=0)                  # optional: rotate labels
    ax.set_xlabel("Year")
    ax.set_ylabel("PM₂.₅ (µg/m³)")
    title_prefix = f"({panel_label}) " if panel_label else ""
    ax.set_title(f"{title_prefix}{name} Model")
    ax.grid(True, linestyle=":", alpha=0.8)
    ax.legend(loc='upper left', fontsize=8)

    plt.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print("Saved:", outpath.name)

def plot_combined_grid(preds_dict, outpath_grid):
    # create 3x2 grid in PANEL_ORDER sequence, label (a)-(f)
    panels = PANEL_ORDER
    fig, axes = plt.subplots(3,2, figsize=(18,14))
    axes = axes.flatten()
    for i, model_name in enumerate(panels):
        ax = axes[i]
        if model_name not in preds_dict:
            ax.axis("off")
            continue
        dfp = preds_dict[model_name].sort_index()
        idx = pd.to_datetime(dfp.index)
        ser_true = pd.Series(dfp["y_true"].values, index=idx)
        ser_pred = pd.Series(dfp["y_pred"].values, index=idx)
        # yearly trend fits
        yr_true, yrs = compute_yearly_means(ser_true)
        yr_pred, _ = compute_yearly_means(ser_pred)
        slope_t, p_t, line_t = fit_trend(yrs, yr_true.values)
        slope_p, p_p, line_p = fit_trend(yrs, yr_pred.values)
        ci_t_low, ci_t_high = compute_trend_with_ci_bootstrap(yrs.values, yr_pred.values)

        # plot monthly actual + predicted
        ax.plot(ser_true.index, ser_true.values, marker="o", ms=3, lw=1.0, color=MONTHLY_COLOR, label="Actual")
        ax.plot(ser_pred.index, ser_pred.values, marker="o", ms=3, lw=1.1, color=MODEL_COLORS.get(model_name,"#777777"), label="Predicted")
        # trend lines
        if line_t is not None:
            line_x_dt = pd.to_datetime(np.round(line_t[0]).astype(int).astype(str))
            ax.plot(line_x_dt, line_t[1], color="gray", lw=1.0, label=f"Actual Trend = ({slope_t:.3f} µg/m³/yr), p=({p_t:.3f})")
        if line_p is not None:
            line_x_dt = pd.to_datetime(np.round(line_p[0]).astype(int).astype(str))
            ci = f"95% CI = ({ci_t_low:.3f}, {ci_t_high:.3f} µg/m³/yr)"
            ax.plot(line_x_dt, line_p[1], color=MODEL_COLORS.get(model_name,"#777777"), lw=1.0, ls="--", label=f"Pred Trend = ({slope_p:.3f} µg/m³/yr), {ci}, p=({p_p:.3f})")

        # ticks & labels
        ax.set_xlim(pd.Timestamp(f"{YEAR_START}"), pd.Timestamp(f"{YEAR_END}"))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.setp(ax.get_xticklabels(), rotation=0)
        ax.set_xlabel("Year")
        ax.set_ylabel("PM₂.₅ (µg/m³)")
        panel_letter = chr(ord("a") + i)
        ax.set_title(f"({panel_letter}) {model_name} Model")
        ax.grid(True, linestyle=":", alpha=0.8)
        ax.legend(fontsize=8, loc="upper left")
    # hide any extra axes
    for j in range(len(panels), len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(outpath_grid, dpi=300)
    plt.close(fig)
    print("Saved combined grid:", outpath_grid.name)

# ---------------- write per-model and combined figures ----------------
if len(preds) == 0:
    print("No predictions available to plot. Exiting.")
else:
    # write individual per-model publication pngs
    for idx, model_name in enumerate(PANEL_ORDER):
        if model_name in preds:
            panel_label = chr(ord("a") + idx)  # a,b,c...
            outpng = INDIV_DIR / f"{model_name}_actual_vs_pred_trends.png"
            plot_per_model_publication(model_name, preds[model_name], outpng, panel_label=panel_label)
    # combined grid
    grid_out = OUT_FIG_DIR / "Grid_models_actual_vs_pred_trends.png"
    plot_combined_grid(preds, grid_out)

print("\nDone. Check outputs in:", OUT_FIG_DIR)
