# ============================================================
# FULL RANGE PLOTTING
# ============================================================
from matplotlib import pyplot as plt
from main_scripts.dir_config import DirConfig

class PlotPredVsActual(DirConfig):
    def __init__(self):
        super().__init__()
        self.models_dir = super().models_dir_path()

    def plot_pred_vs_actual(self, df, name):
        fig, ax = plt.subplots(figsize=(12,4))
        ax.plot(df.index, df["y_true"], label="Actual", lw=1.5)
        ax.plot(df.index, df["y_pred"], label="Predicted", lw=1.2)
        ax.legend()
        ax.set_title(f"{name} (Full Range 2000–2025)")
        ax.set_ylabel("PM2.5")
        plt.tight_layout()
        fig.savefig(self.models_dir / f"{name}.png", dpi=200)
        plt.close()
        