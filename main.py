# ============================================================
# FINAL PATCHED VERSION-2.0
# FULL RANGE PREDICTIONS (2000–2025)
# TEST PREDICTIONS FOR LSTM  ONLY (2023-2025)
# CLEAN LOGS + UPDATED TREND + UPDATED PLOTS
# ============================================================

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

from main_scripts.dir_config import DirConfig
from main_scripts.date_parser import DateParser
from main_scripts.train_test_split import TrainTestSplit
from main_scripts.train_sklearn_models import TrainSklearnModels
from main_scripts.train_lstm_model import TrainLSTMModel


class RunMain(DirConfig):
    def __init__(self):
        super().__init__()
        self.models_dir = super().models_dir_path()
        self.meteo_dir = super().load_meteo_data()
        self.parser = DateParser()
        self.train_test_split = TrainTestSplit()
        self.train_sklearn = TrainSklearnModels()
        self.train_lstm = TrainLSTMModel()

    def run(self):
        print("Loading files...")
        load_meteo_csv_files = self.meteo_dir

        print("Loading and normalizing monthly data...")
        df = self.parser.load_all_monthly(load_meteo_csv_files)
        print("Inner join shape:", df.shape)
        
        print("Creating lag features...")
        df_lag = self.parser.make_lags_monthly(df)
        print("Lags created:", df_lag.shape)

        print("Performing time-based train-test split...")
        X_train, X_test, y_train, y_test, train_df, test_df = self.train_test_split.time_split(df_lag)

        # ============================================================
        # SKLEARN MODELS
        # ============================================================
        print("Training standard models(sklearn)...")
        self.train_sklearn.train_standard_models(X_train, y_train, X_test, y_test, df_lag)

        # ============================================================
        # LSTM MODEL (TEST ONLY + FULL RANGE SEQUENCE)
        # ============================================================
        print("Training LSTM model...")
        self.train_lstm.train_lstm(train_df, test_df, df_lag)

        print(f"Pipeline done. Outputs in: '{self.models_dir}' directory")


        
if __name__ == "__main__":
    run = RunMain()
    run.run()
