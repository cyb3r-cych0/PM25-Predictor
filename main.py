# ============================================================
# FINAL PATCHED VERSION-2.0
# FULL RANGE PREDICTIONS (2000–2025)
# TEST PREDICTIONS FOR LSTM  ONLY (2023-2025)
# CLEAN LOGS + UPDATED TREND + UPDATED PLOTS
# ============================================================

# -------------------------- CLEAN LOGGING --------------------------
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # hide TF INFO/WARNING messages

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


# ============================================================
# DIRECTORY CONFIG
# ============================================================
dir_obj = DirConfig()
OUT_DIR = dir_obj.get_models_pipe_path()
# ============================================================


# ============================================================
# PIPELINE RUN
# ============================================================
def run():
    """ Creating and Instatiating Objects """
    parser_obj = DateParser()
    split_obj = TrainTestSplit()
    train_sklearn_obj = TrainSklearnModels()
    train_lstm_obj = TrainLSTMModel()

    print("Loading files...")
    load_meteo_csv_files = dir_obj.load_meteo_data()

    print("Loading and normalizing monthly data...")
    df = parser_obj.load_all_monthly(load_meteo_csv_files)
    print("Inner join shape:", df.shape)
    
    print("Creating lag features...")
    df_lag = parser_obj.make_lags_monthly(df)
    print("Lags created:", df_lag.shape)

    print("Performing time-based train-test split...")
    X_train, X_test, y_train, y_test, train_df, test_df = split_obj.time_split(df_lag)

    # ============================================================
    # SKLEARN MODELS
    # ============================================================
    print("Training standard models(sklearn)...")
    train_sklearn_obj.train_standard_models(X_train, y_train, X_test, y_test, df_lag)

    # ============================================================
    # LSTM MODEL (TEST ONLY + FULL RANGE SEQUENCE)
    # ============================================================
    print("Training LSTM model...")
    train_lstm_obj.train_lstm(train_df, test_df, df_lag)

    print(f"Pipeline done. Outputs in: '{OUT_DIR}' directory")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    run()
