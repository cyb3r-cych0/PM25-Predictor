# ============================================================
# TREND COMPUTATION
# ============================================================
import pandas as pd
from scipy.stats import linregress
from main_scripts.dir_config import DirConfig

class TrendComputation:
    """A class to compute trends in time series data."""
    def __init__(self, pipe_path=DirConfig().get_models_pipe_path()):
        self.pipe_path = pipe_path

    def compute_trend(self, series):
        idx = pd.to_datetime(series.index)
        x = idx.year + idx.dayofyear/365.25
        res = linregress(x, series.values)
        return res.slope, res.pvalue
    
    def save_trend_summary(self, name, df_full):
        slope_t, p_t = self.compute_trend(df_full["y_true"])
        slope_p, p_p = self.compute_trend(df_full["y_pred"])
        slope_b, p_b = self.compute_trend(df_full["y_pred"] - df_full["y_true"])

        out = {
            "model": name,
            "slope_actual": slope_t,
            "pval_actual": p_t,
            "slope_pred": slope_p,
            "pval_pred": p_p,
            "slope_bias": slope_b,
            "pval_bias": p_b,
            "n_obs": len(df_full)
        }
        pd.DataFrame([out]).to_csv(self.pipe_path / f"{name}_trend_summary_FULL.csv", index=False)
        return out
    