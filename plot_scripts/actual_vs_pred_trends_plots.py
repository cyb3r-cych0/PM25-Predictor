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

# -------------------------- CLEAN LOGGING --------------------------
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # hide TF INFO/WARNING messages
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
# ------------------------------------------------------------------

from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import linregress
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from main_scripts.dir_config import DirConfig

class ActualVsPredTrendsPlotter(DirConfig):
    def __init__(self):
        super().__init__()
        self.models_dir = super().models_dir_path()
        self.aligned_csv = super().get_meteo_path("aligned_monthly_inner.csv")
        self.summary_csv_path = self.models_dir / "models_actual_vs_pred_trend_summary.csv"
        self.plots_dir = super().plots_dir_path() / "actual_vs_pred_trends_plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.model_keys = ["RandomForest", "GradientBoosting", "Lasso", "Ridge", "MLR", "LSTM"]
        self.sklearn_candidates = {
            "RandomForest": ["rf_model.joblib", "RandomForest_model.joblib", "RandomForest.joblib", "RandomForest_model.pkl"],
            "GradientBoosting": ["gbr_model.joblib", "GradientBoosting_model.joblib", "GradientBoosting.joblib"],
            "Lasso": ["lasso_model.joblib","Lasso_model.joblib","lasso.joblib"],
            "Ridge": ["ridge_model.joblib","Ridge_model.joblib","ridge.joblib"],
            "MLR": ["mlr_model.joblib","MLR_model.joblib","mlr.joblib","linear_model.joblib"]
        }
        self.lstm_canditates = ["lstm_pm25_model.h5", "lstm_best.h5", "lstm_model.h5", "lstm_pm25_model.keras"]
        self.lstm_scaler_x_cand = ["lstm_scaler_X.joblib", "scaler_X.joblib"]
        self.lastm_scaler_y_cand = ["lstm_scaler_y.joblib", "scaler_y.joblib"]

        self.lags = [1,2,3,12]
        self.target = "pm25"
        self.lstm_lookback = 12

        self.year_start = 2000
        self.year_end = 2026
        plt.style.use("bmh")
        self.monthly_color = "#0b4f6c"
        self.model_colors = {
            "RandomForest": "#2ca02c",
            "GradientBoosting": "#ff7f0e",
            "Lasso": "#9c27b0",
            "Ridge": "#03a9f4",
            "MLR": "#6d4c41",
            "LSTM": "#e91e63"
        }
        self.panel_order = ["GradientBoosting","LSTM","Lasso","MLR","RandomForest","Ridge"]

    def find_first_existing(self, parent: Path, candidates):
        for c in candidates:
            p = parent / c
            if p.exists():
                return p
        return None

    def try_load_sklearn_model(self, model_name):
        cand = self.sklearn_candidates.get(model_name, [])
        p = self.find_first_existing(self.models_dir, cand)
        if p:
            try:
                return joblib.load(p), p
            except Exception as e:
                print(f"[error] loading {p}: {e}")
        return None, None

    def try_find_lstm(self):
        p = None
        for c in self.lstm_canditates:
            t = self.models_dir / c
            if t.exists():
                p = t
                break
        return p

    def try_find_scaler(self, name_cands):
        return self.find_first_existing(self.models_dir, name_cands)

    def make_lags(self, df, lags=None):
        if lags is None:
            lags = self.lags
        out = df.copy()
        for lag in lags:
            for c in df.columns:
                out[f"{c}_lag{lag}"] = out[c].shift(lag)
        return out.dropna()

    def compute_yearly_means(self, series):
        yr = series.resample("YE").mean()
        years = yr.index.year.astype(float)
        return yr, years

    def fit_trend(self, years, values):
        if len(values) < 2 or np.all(np.isnan(values)):
            return np.nan, np.nan, None
        mask = ~np.isnan(values)
        if mask.sum() < 2:
            return np.nan, np.nan, None
        res = linregress(years[mask], values[mask])
        line_x = np.linspace(self.year_start, self.year_end, 200)
        line_y = res.slope * line_x + res.intercept
        return float(res.slope), float(res.pvalue), (line_x, line_y)

    def compute_trend_with_ci_bootstrap(self, years, values, n_boot=2000, alpha=0.05):
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

    def load_and_prepare_data(self):
        if not self.aligned_csv.exists():
            raise FileNotFoundError(f"Aligned monthly CSV not found: {self.aligned_csv}")

        print("Loading aligned data:", self.aligned_csv)
        df = pd.read_csv(self.aligned_csv, index_col=0, parse_dates=True).sort_index()
        if not (df.index.is_month_end).all():
            df.index = pd.to_datetime(df.index)
            df.index = df.index.to_period("M").to_timestamp("M")
            df = df[~df.index.duplicated(keep="first")].sort_index()

        print("Aligned shape:", df.shape)
        df_lag = self.make_lags(df, self.lags)
        print("After lag creation (usable rows):", df_lag.shape)
        X_full = df_lag.drop(columns=[self.target])
        y_full = df_lag[self.target]
        full_index = df_lag.index
        return X_full, y_full, full_index

    def produce_sklearn_predictions(self, X_full, y_full, full_index):
        preds = {}
        print("\nProducing sklearn model full-range predictions (if model files present)...")
        for name in ["RandomForest","GradientBoosting","Lasso","Ridge","MLR"]:
            model, ppath = self.try_load_sklearn_model(name)
            if model is None:
                cand_csvs = [self.models_dir / f"{name}_preds_fullrange.csv", self.models_dir / f"{name}_FULL_predictions_2000_2025.csv"]
                loaded = False
                for cand_csv in cand_csvs:
                    if cand_csv.exists():
                        try:
                            dfp = pd.read_csv(cand_csv, index_col=0, parse_dates=True)
                            preds[name] = dfp.sort_index()
                            print(f"Loaded existing preds for {name} from {cand_csv.name}")
                            loaded = True
                            break
                        except Exception:
                            continue
                if not loaded:
                    print(f"[skip] no model file or preds for {name}")
                continue
            try:
                yhat = model.predict(X_full)
                dfp = pd.DataFrame({"y_true": y_full.values, "y_pred": np.asarray(yhat).ravel()}, index=full_index)
                outcsv = self.models_dir / f"{name}_preds_fullrange.csv"
                dfp.to_csv(outcsv)
                preds[name] = dfp
                print("Saved full-range preds:", outcsv.name)
            except Exception as e:
                print(f"[error] predicting with {name}: {e}")
        return preds

    def produce_lstm_predictions(self, X_full, y_full, full_index, preds):
        print("\nAttempting LSTM full-range predictions...")
        lstm_path = self.try_find_lstm()
        if lstm_path:
            try:
                from tensorflow.keras.models import load_model
                scaler_x_path = self.try_find_scaler(self.lstm_scaler_x_cand)
                scaler_y_path = self.try_find_scaler(self.lastm_scaler_y_cand)
                if scaler_x_path and scaler_y_path:
                    scaler_X = joblib.load(scaler_x_path)
                    scaler_y = joblib.load(scaler_y_path)
                    lstm = load_model(str(lstm_path))
                    Xmat = X_full.values.astype(float)
                    Xs = scaler_X.transform(Xmat)
                    seqs = []
                    idxs = []
                    for i in range(self.lstm_lookback, len(Xs)):
                        seqs.append(Xs[i-self.lstm_lookback:i])
                        idxs.append(X_full.index[i])
                    if len(seqs) > 0:
                        import numpy as _np
                        Xseq = _np.stack(seqs)
                        yhat_scaled = lstm.predict(Xseq, verbose=0).ravel()
                        yhat = scaler_y.inverse_transform(yhat_scaled.reshape(-1,1)).ravel()
                        ytrue = y_full.loc[idxs]
                        df_lstm = pd.DataFrame({"y_true": ytrue.values, "y_pred": yhat}, index=idxs)
                        outcsv = self.models_dir / "LSTM_FULL_predictions_2000_2025.csv"
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
            alt = self.models_dir / "LSTM_FULL_predictions_2000_2025.csv"
            if alt.exists():
                try:
                    preds["LSTM"] = pd.read_csv(alt, index_col=0, parse_dates=True)
                    print("Loaded existing LSTM full-range preds:", alt.name)
                except Exception:
                    pass
            else:
                print("No LSTM model present; skipping.")
        return preds

    def load_fallback_preds(self, preds):
        for m in self.model_keys:
            if m not in preds:
                f1 = self.models_dir / f"{m}_preds_fullrange.csv"
                f2 = self.models_dir / f"{m}_FULL_predictions_2000_2025.csv"
                for cand in (f1, f2):
                    if cand.exists():
                        try:
                            dfp = pd.read_csv(cand, index_col=0, parse_dates=True)
                            preds[m] = dfp.sort_index()
                            print(f"Loaded fallback preds for {m} from {cand.name}")
                            break
                        except Exception:
                            continue
        return preds

    def plot_per_model_publication(self, name, dfp, outpath, panel_label=None):
        dfp = dfp.sort_index()
        idx = pd.to_datetime(dfp.index)
        ser_true = pd.Series(dfp["y_true"].values, index=idx)
        ser_pred = pd.Series(dfp["y_pred"].values, index=idx)

        yr_true, yrs = self.compute_yearly_means(ser_true)
        yr_pred, _ = self.compute_yearly_means(ser_pred)
        slope_t, p_t, line_t = self.fit_trend(yrs, yr_true.values)
        slope_p, p_p, line_p = self.fit_trend(yrs, yr_pred.values)
        ci_t_low, ci_t_high = self.compute_trend_with_ci_bootstrap(yrs.values, yr_pred.values)

        fig, ax = plt.subplots(figsize=(10,4.2))
        ax.plot(ser_true.index, ser_true.values, marker="o", ms=4, linewidth=1.0, color=self.monthly_color, label="Actual")
        ax.plot(ser_pred.index, ser_pred.values, marker="o", ms=4, linewidth=1.1, color=self.model_colors.get(name,"#777777"), label="Predicted")

        if line_t is not None:
            line_x_dt = pd.to_datetime(np.round(line_t[0]).astype(int).astype(str))
            ax.plot(line_x_dt, line_t[1], color="gray", lw=1.0, label=f"Actual Trend = ({slope_t:.3f} \u03BCg/m\u00b3/yr), p=({p_t:.3f})")
        if line_p is not None:
            line_x_dt = pd.to_datetime(np.round(line_p[0]).astype(int).astype(str))
            ci = (f"95% CI = ({ci_t_low:.3f}, {ci_t_high:.3f} \u03BCg/m\u00b3/yr)")
            ax.plot(line_x_dt, line_p[1], color=self.model_colors.get(name,"#777777"), lw=1.5, ls="--", label=f"Pred Trend = ({slope_p:.3f} \u03BCg/m\u00b3/yr), {ci}, p=({p_p:.3f})")

        ax.set_xlim(pd.Timestamp(f"{self.year_start}"), pd.Timestamp(f"{self.year_end}"))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.setp(ax.get_xticklabels(), rotation=0)
        ax.set_xlabel("Year")
        ax.set_ylabel("PM\u2082.\u2085 (\u03BCg/m\u00b3)")
        title_prefix = f"({panel_label}) " if panel_label else ""
        ax.set_title(f"{title_prefix}{name} Model")
        ax.grid(True, linestyle=":", alpha=0.8)
        ax.legend(loc='upper left', fontsize=8)

        plt.tight_layout()
        fig.savefig(outpath, dpi=300)
        plt.close(fig)
        print("Saved:", outpath.name)

    def plot_combined_grid(self, preds_dict, outpath_grid):
        panels = self.panel_order
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
            yr_true, yrs = self.compute_yearly_means(ser_true)
            yr_pred, _ = self.compute_yearly_means(ser_pred)
            slope_t, p_t, line_t = self.fit_trend(yrs, yr_true.values)
            slope_p, p_p, line_p = self.fit_trend(yrs, yr_pred.values)
            ci_t_low, ci_t_high = self.compute_trend_with_ci_bootstrap(yrs.values, yr_pred.values)

            ax.plot(ser_true.index, ser_true.values, marker="o", ms=3, lw=1.0, color=self.monthly_color, label="Actual")
            ax.plot(ser_pred.index, ser_pred.values, marker="o", ms=3, lw=1.1, color=self.model_colors.get(model_name,"#777777"), label="Predicted")

            if line_t is not None:
                line_x_dt = pd.to_datetime(np.round(line_t[0]).astype(int).astype(str))
                ax.plot(line_x_dt, line_t[1], color="gray", lw=1.0, label=f"Actual Trend = ({slope_t:.3f} \u03BCg/m\u00b3/yr), p=({p_t:.3f})")
            if line_p is not None:
                line_x_dt = pd.to_datetime(np.round(line_p[0]).astype(int).astype(str))
                ci = f"95% CI = ({ci_t_low:.3f}, {ci_t_high:.3f} \u03BCg/m\u00b3/yr)"
                ax.plot(line_x_dt, line_p[1], color=self.model_colors.get(model_name,"#777777"), lw=1.0, ls="--", label=f"Pred Trend = ({slope_p:.3f} \u03BCg/m\u00b3/yr), {ci}, p=({p_p:.3f})")

            ax.set_xlim(pd.Timestamp(f"{self.year_start}"), pd.Timestamp(f"{self.year_end}"))
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            plt.setp(ax.get_xticklabels(), rotation=0)
            ax.set_xlabel("Year")
            ax.set_ylabel("PM\u2082.\u2085 (\u03BCg/m\u00b3)")
            panel_letter = chr(ord("a") + i)
            ax.set_title(f"({panel_letter}) {model_name} Model")
            ax.grid(True, linestyle=":", alpha=0.8)
            ax.legend(fontsize=8, loc="upper left")

        for j in range(len(panels), len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout(rect=[0,0,1,0.96])
        fig.savefig(outpath_grid, dpi=300)
        plt.close(fig)
        print("Saved combined grid:", outpath_grid.name)

    def run(self):
        X_full, y_full, full_index = self.load_and_prepare_data()
        preds = self.produce_sklearn_predictions(X_full, y_full, full_index)
        preds = self.produce_lstm_predictions(X_full, y_full, full_index, preds)
        preds = self.load_fallback_preds(preds)

        if len(preds) == 0:
            print("No predictions available to plot. Exiting.")
            return

        for idx, model_name in enumerate(self.panel_order):
            if model_name in preds:
                panel_label = chr(ord("a") + idx)
                outpng = self.plots_dir / f"{model_name}_actual_vs_pred_trends.png"
                self.plot_per_model_publication(model_name, preds[model_name], outpng, panel_label=panel_label)

        grid_out = self.plots_dir / "Grid_models_actual_vs_pred_trends.png"
        self.plot_combined_grid(preds, grid_out)

        print("\nDone. Check outputs in:", self.plots_dir)
