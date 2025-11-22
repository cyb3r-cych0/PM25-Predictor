# ============================================================
# LSTM MODEL (TEST ONLY + FULL RANGE SEQUENCE)
# ============================================================
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, InputLayer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import joblib

from main_scripts.dir_config import DirConfig
from main_scripts.plot import PlotPredVsActual
from main_scripts.trend_computation import TrendComputation

class TrainLSTMModel:
    def __init__(self, 
                 trend_obj=TrendComputation(), 
                 dir_obj=DirConfig().get_models_pipe_path(), 
                 plot_obj=PlotPredVsActual(),
                 seed=42,
                 LSTM_LOOKBACK = 12,
                 LSTM_EPOCHS = 150,
                 LSTM_BATCH = 16):
        
        self.trend_obj = trend_obj
        self.pipe_path = dir_obj
        self.plot_obj = plot_obj
        self.seed = seed
        self.LSTM_LOOKBACK = LSTM_LOOKBACK
        self.LSTM_EPOCHS = LSTM_EPOCHS
        self.LSTM_BATCH = LSTM_BATCH
        tf.random.set_seed(self.seed)

    def prepare_lstm_sequences(self, df):
        X = df.drop(columns=["pm25"]).values
        y = df["pm25"].values

        scX = StandardScaler()
        scY = StandardScaler()
        Xs = scX.fit_transform(X)
        ys = scY.fit_transform(y.reshape(-1,1)).ravel()

        seq, tgt, idx = [], [], []
        for i in range(self.LSTM_LOOKBACK, len(Xs)):
            seq.append(Xs[i-self.LSTM_LOOKBACK:i])
            tgt.append(ys[i])
            idx.append(df.index[i])
        return np.array(seq), np.array(tgt), np.array(idx), scX, scY

    def train_lstm(self, train_df, test_df, df_lag):
        full_df = pd.concat([train_df, test_df])

        X_seq, y_seq, idxs, scX, scY = self.prepare_lstm_sequences(full_df)
        n_train = len(train_df) - self.LSTM_LOOKBACK
        if n_train <= 0:
            print("Not enough data for LSTM")
            return None

        X_tr, X_te = X_seq[:n_train], X_seq[n_train:]
        y_tr, y_te = y_seq[:n_train], y_seq[n_train:]
        idx_te = idxs[n_train:]

        model = Sequential([
            InputLayer(input_shape=(self.LSTM_LOOKBACK, X_tr.shape[2])),
            LSTM(64), Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")

        ckpt = self.pipe_path / "lstm_best.h5"
        es = EarlyStopping(patience=12, restore_best_weights=True)
        mc = ModelCheckpoint(ckpt, save_best_only=True)

        model.fit(X_tr, y_tr, validation_data=(X_te, y_te),
                epochs=self.LSTM_EPOCHS, batch_size=self.LSTM_BATCH,
                callbacks=[es, mc], verbose=0)
        
        joblib.dump(scX, self.pipe_path / "lstm_scaler_X.joblib")
        joblib.dump(scY, self.pipe_path / "lstm_scaler_y.joblib")

        # ---- TEST predictions ----
        preds_scaled = model.predict(X_te).ravel()
        preds = scY.inverse_transform(preds_scaled.reshape(-1,1)).ravel()
        y_test_real = scY.inverse_transform(y_te.reshape(-1,1)).ravel()

        df_test = pd.DataFrame({"y_true": y_test_real, "y_pred": preds}, index=idx_te)
        df_test.to_csv(self.pipe_path / "LSTM_TEST_predictions.csv")
        self.plot_obj.plot_pred_vs_actual(df_test, "LSTM_TEST")

        # ---- FULL RANGE predictions ----
        X_full, _, idx_full, scX, scY = self.prepare_lstm_sequences(df_lag)
        full_scaled = model.predict(X_full).ravel()
        full_pred = scY.inverse_transform(full_scaled.reshape(-1,1)).ravel()
        y_full = df_lag["pm25"].iloc[self.LSTM_LOOKBACK:]

        df_full = pd.DataFrame({"y_true": y_full.values, "y_pred": full_pred}, index=idx_full)
        df_full.to_csv(self.pipe_path / "LSTM_FULL_predictions_2000_2025.csv")
        self.plot_obj.plot_pred_vs_actual(df_full, "LSTM_FULL")
        self.trend_obj.save_trend_summary("LSTM", df_full)
        return df_full
    