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

from main_scripts.dir_config import DirConfig

class BiasTrendTimeseriesPlotter(DirConfig):
    def __init__(self):
        super().__init__()
        self.models_dir = super().models_dir_path()
        self.plots_dir = super().plots_dir_path() / "bias_trend_plots"
        self.plots_dir.mkdir(exist_ok=True)
        self.models = ["GradientBoosting", "LSTM", "Lasso", "MLR", "RandomForest", "Ridge"]
        self.letters = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
        self.year_start = 2000
        self.year_end = 2026
        plt.style.use("bmh")
        self.line_color = "#ff8c00"
        self.bias_color = "#1f77b4"

    def decimal_year_from_index(self, idx):
        s = pd.to_datetime(idx)
        return s.year + s.dayofyear / 365.25

    def fit_and_get_trend(self, x_dec, y):
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return None
        slope, intercept, r, pval, stderr = linregress(x_dec[mask], y[mask])
        return {"slope": slope, "intercept": intercept, "r": r, "p": pval, "stderr": stderr}

    def plot_single_bias(self, model, df, outpath, add_title_prefix=None):
        df = df.sort_index()
        df["bias"] = df["y_pred"] - df["y_true"]
        x_dec = self.decimal_year_from_index(df.index)
        y = df["bias"].values

        trend = self.fit_and_get_trend(x_dec, y)
        years_line = np.linspace(self.year_start, self.year_end, 200)
        if trend is not None:
            y_line = trend["intercept"] + trend["slope"] * years_line
        else:
            y_line = None

        fig, ax = plt.subplots(figsize=(12, 3.8))
        ax.plot(df.index, y, marker="o", ms=4, lw=1, linestyle="-", color=self.bias_color, label="Residual (pred - actual)")

        if y_line is not None:
            line_dates = pd.to_datetime(np.round(years_line).astype(int).astype(str))
            ax.plot(line_dates, y_line, linestyle="--", color=self.line_color, label=f"Trend = ({trend['slope']:+.3f} \u03BCg/m\u00b3/yr), p=({trend['p']:.3f})")

        title = f"{add_title_prefix + ' ' if add_title_prefix else ''}{model} Model"
        ax.set_title(title, fontsize=14)
        ax.set_ylabel("Residual (\u03BCg/m\u00b3)")
        ax.set_xlabel("Year")
        ax.set_xlim(pd.Timestamp(f"{self.year_start}"), pd.Timestamp(f"{self.year_end}"))
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

    def run(self):
        trends = {}
        for i, model in enumerate(self.models):
            csv_path = self.models_dir / f"{model}_FULL_predictions_2000_2025.csv"
            if not csv_path.exists():
                print(f"[skip] missing: {csv_path}")
                continue
            try:
                df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            except Exception as e:
                print(f"[error] reading {csv_path}: {e}")
                continue
            letter = self.letters[i]
            out_single = self.plots_dir / f"{model}_bias_trend.png"
            trend = self.plot_single_bias(model, df, out_single, add_title_prefix=letter)
            trends[model] = trend

        nrows, ncols = 3, 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 14))
        axes = axes.flatten()

        for i, model in enumerate(self.models):
            ax = axes[i]
            csv_path = self.models_dir / f"{model}_FULL_predictions_2000_2025.csv"
            if not csv_path.exists():
                ax.set_visible(False)
                continue
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True).sort_index()
            df["bias"] = df["y_pred"] - df["y_true"]
            x_dec = self.decimal_year_from_index(df.index)
            y = df["bias"].values
            ax.plot(df.index, y, marker="o", ms=3.5, lw=1, linestyle="-", color=self.bias_color, label="Residual (pred - actual)")

            trend = self.fit_and_get_trend(x_dec, y)
            if trend is not None:
                years_line = np.linspace(self.year_start, self.year_end, 200)
                y_line = trend["intercept"] + trend["slope"] * years_line
                line_dates = pd.to_datetime(np.round(years_line).astype(int).astype(str))
                ax.plot(line_dates, y_line, linestyle="--", color=self.line_color, label=f"Trend = ({trend['slope']:+.3f} \u03BCg/m\u00b3/yr), p=({trend['p']:.3f})")

            ax.set_title(f"{self.letters[i]} {model} Model", fontsize=14)
            ax.set_ylabel("Residual (\u03BCg/m\u00b3)")
            ax.set_xlabel("Year")
            ax.set_xlim(pd.Timestamp(f"{self.year_start}"), pd.Timestamp(f"{self.year_end}"))
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            plt.setp(ax.get_xticklabels(), rotation=0)
            ax.grid(alpha=0.35)
            ax.legend(fontsize=9, loc="upper left")

        for j in range(len(self.models), nrows * ncols):
            axes[j].set_visible(False)

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        out_combined = self.plots_dir / "Grid_models_bias_trend.png"
        fig.savefig(out_combined, dpi=300)
        plt.close(fig)
        print("Saved combined figure:", out_combined)
