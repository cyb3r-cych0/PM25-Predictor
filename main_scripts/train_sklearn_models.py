# ============================================================
# SKLEARN MODELS
# ============================================================
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV, RidgeCV, LinearRegression
import joblib
import numpy as np
import pandas as pd

from main_scripts.metrics import Metrics
from main_scripts.trend_computation import TrendComputation
from main_scripts.dir_config import DirConfig
from main_scripts.plot import PlotPredVsActual

class TrainSklearnModels(DirConfig):
    def __init__(self, seed=42, results={}):
        super().__init__()
        
        self.models_dir = super().models_dir_path()
        self.seed = seed
        np.random.seed(self.seed)
        self.results = results

    def train_standard_models(self, X_train, y_train, X_test, y_test, df_lag):
        self.results = {}

        def train_model(model, name):
            model.fit(X_train, y_train)

            # test predictions
            print("saving test predictions and metrics...")
            df_test = Metrics().save_preds_and_metrics(name, y_test, model.predict(X_test))

            # full-range predictions
            full_pred = model.predict(df_lag.drop(columns=["pm25"]))
            df_full = pd.DataFrame({
                "y_true": df_lag["pm25"],
                "y_pred": full_pred
            }, index=df_lag.index)

            df_full.to_csv(self.models_dir / f"{name}_FULL_predictions_2000_2025.csv")
            PlotPredVsActual().plot_pred_vs_actual(df_full, f"{name}_FULL")
            TrendComputation().save_trend_summary(name, df_full)

            joblib.dump(model, self.models_dir / f"{name}_model.joblib")
            return df_full
        
        self.results["RandomForest"] = train_model(
            RandomForestRegressor(n_estimators=200, random_state=self.seed),
            "RandomForest"
        )
        self.results["GradientBoosting"] = train_model(
            GradientBoostingRegressor(random_state=self.seed),
            "GradientBoosting"
        )
        self.results["Lasso"] = train_model(
            LassoCV(cv=5, random_state=self.seed), "Lasso"
        )
        self.results["Ridge"] = train_model(
            RidgeCV(cv=5), "Ridge"
        )
        self.results["MLR"] = train_model(LinearRegression(), "MLR")
        return self.results
    