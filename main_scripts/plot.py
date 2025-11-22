# ============================================================
# FULL RANGE PLOTTING
# ============================================================
from matplotlib import pyplot as plt
from main_scripts.dir_config import DirConfig

class PlotPredVsActual:
    """A class to plot predicted vs actual values."""
    def __init__(self, pipe_path=DirConfig().get_models_pipe_path()):
        self.pipe_path = pipe_path

    def plot_pred_vs_actual(self, df, name):
        fig, ax = plt.subplots(figsize=(12,4))
        ax.plot(df.index, df["y_true"], label="Actual", lw=1.5)
        ax.plot(df.index, df["y_pred"], label="Predicted", lw=1.2)
        ax.legend()
        ax.set_title(f"{name} (Full Range 2000–2025)")
        ax.set_ylabel("PM2.5")
        plt.tight_layout()
        fig.savefig(self.pipe_path / f"{name}.png", dpi=200)
        plt.close()
        