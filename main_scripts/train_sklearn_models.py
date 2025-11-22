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

class TrainSklearnModels:
    def __init__(self, 
                 trend_obj=TrendComputation(), 
                 dir_obj=DirConfig().get_models_pipe_path(), 
                 metrics_obj=Metrics(),
                 plot_obj=PlotPredVsActual(),
                 seed=42,
                 results={}):
        
        self.trend_obj = trend_obj
        self.pipe_path = dir_obj
        self.metrics_obj = metrics_obj
        self.plot_obj = plot_obj
        self.seed = seed
        np.random.seed(self.seed)
        self.results = results

    def train_standard_models(self, X_train, y_train, X_test, y_test, df_lag):
        self.results = {}

        def train_model(model, name):
            model.fit(X_train, y_train)

            # test predictions
            print("saving test predictions and metrics...")
            df_test = self.metrics_obj.save_preds_and_metrics(name, y_test, model.predict(X_test))

            # full-range predictions
            full_pred = model.predict(df_lag.drop(columns=["pm25"]))
            df_full = pd.DataFrame({
                "y_true": df_lag["pm25"],
                "y_pred": full_pred
            }, index=df_lag.index)

            df_full.to_csv(self.pipe_path / f"{name}_FULL_predictions_2000_2025.csv")
            self.plot_obj.plot_pred_vs_actual(df_full, f"{name}_FULL")
            self.trend_obj.save_trend_summary(name, df_full)

            joblib.dump(model, self.pipe_path / f"{name}_model.joblib")
            return df_full

        # 1 Random Forest
        self.results["RandomForest"] = train_model(
            RandomForestRegressor(n_estimators=200, random_state=self.seed),
            "RandomForest"
        )

        # 2 Gradient Boosting
        self.results["GradientBoosting"] = train_model(
            GradientBoostingRegressor(random_state=self.seed),
            "GradientBoosting"
        )

        # 3 Lasso
        self.results["Lasso"] = train_model(
            LassoCV(cv=5, random_state=self.seed), "Lasso"
        )

        # 4 Ridge
        self.results["Ridge"] = train_model(
            RidgeCV(cv=5), "Ridge"
        )

        # 5 MLR
        self.results["MLR"] = train_model(LinearRegression(), "MLR")

        return self.results
    