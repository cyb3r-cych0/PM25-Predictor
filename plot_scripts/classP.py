from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import warnings
import os

# Suppress warnings and set TensorFlow log level quietly if it's imported elsewhere
warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


class PredictionVisualizer:
    """
    A class to generate a combined publication-style figure comparing 
    actual vs predicted PM2.5 trends for multiple models.
    """
    def __init__(self, root_dir=Path("."), output_dir_name="plot_figures"):
        self.ROOT = root_dir
        self.PIPE_DIR = self.ROOT / "models_pipeline_data"
        self.OUT_DIR = self.ROOT / output_dir_name
        self.OUT_DIR.mkdir(parents=True, exist_ok=True)

        self.MODELS = ["GradientBoosting", "LSTM", "Lasso", "MLR", "RandomForest", "Ridge"]
        self.LETTERS = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

        self.YEAR_START = 2000
        self.YEAR_END = 2026
        self.YEAR_STEP = 3
        self.PANEL_FIGSIZE = (16, 12)
        
        # Colors
        self.ACTUAL_COLOR = "#0b4f6c"
        self.PRED_COLOR = "#ff7f0e"
        self.TREND_ACTUAL_COLOR = "#1f2e3a"
        self.TREND_PRED_COLOR = "#e91e63"

        # Apply plot style
        plt.style.use("bmh")

    def find_preds_csv(self, model_name):
        """Locates the prediction CSV file for a given model name."""
        candidates = [self.PIPE_DIR / f"{model_name}_FULL_predictions_2000_2025.csv"]
        for c in candidates:
            if c.exists():
                return c
        return None

    def load_preds(self, path):
        """Loads and formats the predictions DataFrame."""
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        cols = df.columns
        if "y_true" in cols and "y_pred" in cols:
            return df[["y_true", "y_pred"]].sort_index()
        # fallback to first 2 numeric columns if needed
        nums = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        if len(nums) >= 2:
            return df[[nums[0], nums[1]]].rename(columns={nums[0]:"y_true", nums[1]:"y_pred"}).sort_index()
        return None

    @staticmethod
    def year_fraction(idx):
        """Converts a DatetimeIndex to fractional years."""
        dt = pd.to_datetime(idx)
        return dt.year + (dt.dayofyear - 1) / 365.25

    def fit_trend(self, series):
        """Fits a linear regression trend line to the series."""
        if series.dropna().shape[0] < 2:
            return np.nan, np.nan, None
        x = self.year_fraction(series.index)
        y = series.values
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return np.nan, np.nan, None
        lr = linregress(x[mask], y[mask])
        xs = np.linspace(self.YEAR_START, self.YEAR_END, 200)
        ys = lr.slope * xs + lr.intercept
        # Create datetime objects for plotting the trend line
        xs_dt = pd.to_datetime([f"{int(round(y))}-12-31" for y in xs])
        return lr.slope, lr.pvalue, (xs_dt, ys)

    def generate_plot(self):
        """Main method to generate and save the combined figure."""
        fig, axes = plt.subplots(3, 2, figsize=self.PANEL_FIGSIZE)
        axes = axes.flatten()

        for i, model in enumerate(self.MODELS):
            ax = axes[i]
            preds_path = self.find_preds_csv(model)
            if preds_path is None:
                print(f"Could not find predictions file for {model}")
                ax.set_visible(False)
                continue

            df = self.load_preds(preds_path)
            if df is None:
                print(f"Could not load data from {preds_path}")
                ax.set_visible(False)
                continue

            df.index = pd.to_datetime(df.index)
            df = df[(df.index >= f"{self.YEAR_START}-01-01") & (df.index <= f"{self.YEAR_END}-12-31")]

            ser_a = df["y_true"]
            ser_p = df["y_pred"]

            # trend lines
            slope_a, p_a, trend_a = self.fit_trend(ser_a)
            slope_p, p_p, trend_p = self.fit_trend(ser_p)

            # plot actual
            ax.plot(ser_a.index, ser_a.values, marker="o", ms=3, lw=1.0,
                    color=self.ACTUAL_COLOR, label="Actual")

            # plot predicted
            ax.plot(ser_p.index, ser_p.values, marker="o", ms=3, lw=1.0,
                    color=self.PRED_COLOR, label="Predicted")

            # trend lines
            if trend_a:
                ax.plot(trend_a[0], trend_a[1],
                        color=self.TREND_ACTUAL_COLOR, lw=2,
                        label=f"Actual trend ({slope_a:.3f}/yr)")
            if trend_p:
                ax.plot(trend_p[0], trend_p[1],
                        color=self.TREND_PRED_COLOR, lw=2, ls="--",
                        label=f"Pred trend ({slope_p:.3f}/yr)")

            # title with prefix
            ax.set_title(f"{self.LETTERS[i]} {model} Model", fontsize=12, fontweight="bold")

            # labels
            ax.set_xlabel("Year")
            ax.set_ylabel("PM2.5 (µg/m³)")

            # x-axis ticks
            years = np.arange(self.YEAR_START, self.YEAR_END + 1, self.YEAR_STEP)
            xticks = pd.to_datetime([f"{y}-06-30" for y in years])
            ax.set_xticks(xticks)
            ax.set_xlim(pd.Timestamp(f"{self.YEAR_START}-01-01"), pd.Timestamp(f"{self.YEAR_END}-12-31"))
            ax.tick_params(axis="x", rotation=30)

            ax.grid(True, linestyle=":", alpha=0.8)
            ax.legend(fontsize=7, framealpha=0.9)

        # hide unused axes (if any)
        for j in range(len(self.MODELS), 6):
            axes[j].set_visible(False)

        plt.suptitle("Actual vs Predicted PM2.5 Trends Per Model (2000–2025)", fontsize=16, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        outfile = self.OUT_DIR / "combined_models_actual_vs_pred_trends.png"
        fig.savefig(outfile, dpi=300)
        plt.close(fig)

        print("Saved combined multi-panel PNG:", outfile)

# If you run this file directly, it will execute the plot generation
if __name__ == "__main__":
    visualizer = PredictionVisualizer()
    visualizer.generate_plot()
