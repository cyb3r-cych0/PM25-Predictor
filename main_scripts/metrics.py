# ============================================================
# METRICS
# ============================================================
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import json
from main_scripts.dir_config import DirConfig

class Metrics:
    def __init__(self, pipe_path=DirConfig().get_models_pipe_path()):
        self.pipe_path = pipe_path

    def save_preds_and_metrics(self, name, y_test, y_pred):
        dfp = pd.DataFrame({"y_true": y_test, "y_pred": y_pred}, index=y_test.index)
        dfp.to_csv(self.pipe_path / f"{name}_TEST_predictions.csv")
        r2 = r2_score(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        with open(self.pipe_path / f"{name}_metrics.json", "w") as f:
            json.dump({"model": name, "r2": r2, "rmse": rmse, "mae": mae}, f, indent=2)

        print(f"{name}: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
        return dfp