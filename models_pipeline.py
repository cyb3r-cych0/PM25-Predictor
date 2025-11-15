# ============================================================
# pipeline_full_run.py — FINAL PATCHED VERSION (ALL 6 MODELS)
# FULL RANGE PREDICTIONS (2000–2025) + TEST PREDICTIONS
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
# -------------------------------------------------------------------

from pathlib import Path
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LassoCV, RidgeCV, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, InputLayer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from scipy.stats import linregress


# ============================================================
# DIRECTORY CONFIG
# ============================================================
ROOT = Path(".")
DATA_DIR = ROOT / "meteo_data"
OUT_DIR = ROOT / "models_pipeline_data"
OUT_DIR.mkdir(exist_ok=True)

PATHS = {
    "pm25": DATA_DIR / "Total-Surface-Mass-Concentration-PM2.5.csv",
    "dew_temp": DATA_DIR / "2-meter dew point temperature.csv",
    "pbl": DATA_DIR / "Planetary boundary layer height.csv",
    "surface_air_temp": DATA_DIR / "Surface air temperature.csv",
    "surface_pressure": DATA_DIR / "Surface pressure.csv",
    "surface_skin_temp": DATA_DIR / "Surface skin temperature.csv",
    "surface_wind_temp": DATA_DIR / "Surface wind speed.csv",
    "surface_precipitation": DATA_DIR / "Total surface precipitation.csv"
}

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

TEST_FRAC = 0.2
LAGS = [1,2,3,12]
LSTM_LOOKBACK = 12
LSTM_EPOCHS = 150
LSTM_BATCH = 16


# ============================================================
# DATE PARSER
# ============================================================
COMMON_FORMATS = [
    "%Y-%m-%d", "%Y-%m-%d %H:%M:%S",
    "%d/%m/%Y", "%d/%m/%Y %H:%M",
    "%d/%m/%Y %H:%M:%S"
]

def try_parse_dates(s):
    s = s.astype(str)
    for fmt in COMMON_FORMATS:
        parsed = pd.to_datetime(s, format=fmt, errors="coerce")
        if parsed.notna().sum() >= 0.8 * len(parsed):
            return parsed
    return pd.to_datetime(s, errors="coerce")

def read_and_normalize_monthly(path, col_name):
    df = pd.read_csv(path)
    if df.shape[1] == 1:
        df = df.reset_index()

    date_col = df.columns[0]
    val_col = "Values" if "Values" in df.columns else df.columns[1]

    parsed = try_parse_dates(df[date_col])
    df = df.assign(Datetime_raw=parsed)
    df = df.dropna(subset=["Datetime_raw", val_col])

    df["month"] = df["Datetime_raw"].dt.to_period("M").dt.to_timestamp("M")
    df = df.set_index("month").sort_index()

    return df[[val_col]].rename(columns={val_col: col_name})

def load_all_monthly(paths_dict):
    merged = None
    for key, pth in paths_dict.items():
        dfk = read_and_normalize_monthly(pth, key)
        merged = dfk if merged is None else merged.join(dfk, how="inner")
    return merged.sort_index()


# ============================================================
# LAGS
# ============================================================
def make_lags_monthly(df, lags=LAGS):
    df_lag = df.copy()
    for lag in lags:
        for col in df.columns:
            df_lag[f"{col}_lag{lag}"] = df_lag[col].shift(lag)
    return df_lag.dropna()


# ============================================================
# TRAIN/TEST SPLIT
# ============================================================
def time_split(df, target="pm25", test_frac=TEST_FRAC):
    n = len(df)
    n_test = int(np.ceil(n * test_frac))

    train = df.iloc[:-n_test]
    test = df.iloc[-n_test:]

    X_train = train.drop(columns=[target])
    y_train = train[target]
    X_test = test.drop(columns=[target])
    y_test = test[target]

    print(f"Train rows: {len(train)}, Test rows: {len(test)}")
    return X_train, X_test, y_train, y_test, train, test


# ============================================================
# METRICS
# ============================================================
def save_preds_and_metrics(name, y_test, y_pred):
    dfp = pd.DataFrame({"y_true": y_test, "y_pred": y_pred}, index=y_test.index)
    dfp.to_csv(OUT_DIR / f"{name}_TEST_predictions.csv")

    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    with open(OUT_DIR / f"{name}_metrics.json", "w") as f:
        json.dump({"model": name, "r2": r2, "rmse": rmse, "mae": mae}, f, indent=2)

    print(f"{name}: R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
    return dfp


# ============================================================
# TREND COMPUTATION
# ============================================================
def compute_trend(series):
    idx = pd.to_datetime(series.index)
    x = idx.year + idx.dayofyear/365.25
    res = linregress(x, series.values)
    return res.slope, res.pvalue


def save_trend_summary(name, df_full):
    slope_t, p_t = compute_trend(df_full["y_true"])
    slope_p, p_p = compute_trend(df_full["y_pred"])
    slope_b, p_b = compute_trend(df_full["y_pred"] - df_full["y_true"])

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
    pd.DataFrame([out]).to_csv(OUT_DIR / f"{name}_trend_summary_FULL.csv", index=False)
    return out


# ============================================================
# FULL RANGE PLOTTING
# ============================================================
def plot_pred_vs_actual(df, name):
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(df.index, df["y_true"], label="Actual", lw=1.5)
    ax.plot(df.index, df["y_pred"], label="Predicted", lw=1.2)
    ax.legend()
    ax.set_title(f"{name} (Full Range 2000–2025)")
    ax.set_ylabel("PM2.5")
    plt.tight_layout()
    fig.savefig(OUT_DIR / f"{name}.png", dpi=200)
    plt.close()


# ============================================================
# SKLEARN MODELS
# ============================================================
def train_standard_models(X_train, y_train, X_test, y_test, df_lag):
    results = {}

    def train_model(model, name):
        model.fit(X_train, y_train)

        # test predictions
        df_test = save_preds_and_metrics(name, y_test, model.predict(X_test))

        # full-range predictions
        full_pred = model.predict(df_lag.drop(columns=["pm25"]))
        df_full = pd.DataFrame({
            "y_true": df_lag["pm25"],
            "y_pred": full_pred
        }, index=df_lag.index)

        df_full.to_csv(OUT_DIR / f"{name}_FULL_predictions_2000_2025.csv")
        plot_pred_vs_actual(df_full, f"{name}_FULL")
        save_trend_summary(name, df_full)

        joblib.dump(model, OUT_DIR / f"{name}_model.joblib")
        return df_full

    # 1 Random Forest
    results["RandomForest"] = train_model(
        RandomForestRegressor(n_estimators=200, random_state=SEED),
        "RandomForest"
    )

    # 2 Gradient Boosting
    results["GradientBoosting"] = train_model(
        GradientBoostingRegressor(random_state=SEED),
        "GradientBoosting"
    )

    # 3 Lasso
    results["Lasso"] = train_model(
        LassoCV(cv=5, random_state=SEED), "Lasso"
    )

    # 4 Ridge
    results["Ridge"] = train_model(
        RidgeCV(cv=5), "Ridge"
    )

    # 5 MLR
    results["MLR"] = train_model(LinearRegression(), "MLR")

    return results


# ============================================================
# LSTM MODEL (TEST ONLY + FULL RANGE SEQUENCE)
# ============================================================
def prepare_lstm_sequences(df):
    X = df.drop(columns=["pm25"]).values
    y = df["pm25"].values

    scX = StandardScaler()
    scY = StandardScaler()
    Xs = scX.fit_transform(X)
    ys = scY.fit_transform(y.reshape(-1,1)).ravel()

    seq, tgt, idx = [], [], []
    for i in range(LSTM_LOOKBACK, len(Xs)):
        seq.append(Xs[i-LSTM_LOOKBACK:i])
        tgt.append(ys[i])
        idx.append(df.index[i])

    return np.array(seq), np.array(tgt), np.array(idx), scX, scY


def train_lstm(train_df, test_df, df_lag):
    full_df = pd.concat([train_df, test_df])

    X_seq, y_seq, idxs, scX, scY = prepare_lstm_sequences(full_df)
    n_train = len(train_df) - LSTM_LOOKBACK
    if n_train <= 0:
        print("Not enough data for LSTM")
        return None

    X_tr, X_te = X_seq[:n_train], X_seq[n_train:]
    y_tr, y_te = y_seq[:n_train], y_seq[n_train:]
    idx_te = idxs[n_train:]

    model = Sequential([
        InputLayer(input_shape=(LSTM_LOOKBACK, X_tr.shape[2])),
        LSTM(64), Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    ckpt = OUT_DIR / "lstm_best.h5"
    es = EarlyStopping(patience=12, restore_best_weights=True)
    mc = ModelCheckpoint(ckpt, save_best_only=True)

    model.fit(X_tr, y_tr, validation_data=(X_te, y_te),
              epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH,
              callbacks=[es, mc], verbose=0)
    
    joblib.dump(scX, OUT_DIR / "lstm_scaler_X.joblib")
    joblib.dump(scY, OUT_DIR / "lstm_scaler_y.joblib")


    # ---- TEST predictions ----
    preds_scaled = model.predict(X_te).ravel()
    preds = scY.inverse_transform(preds_scaled.reshape(-1,1)).ravel()
    y_test_real = scY.inverse_transform(y_te.reshape(-1,1)).ravel()

    df_test = pd.DataFrame({"y_true": y_test_real, "y_pred": preds}, index=idx_te)
    df_test.to_csv(OUT_DIR / "LSTM_TEST_predictions.csv")
    plot_pred_vs_actual(df_test, "LSTM_TEST")

    # ---- FULL RANGE predictions ----
    X_full, _, idx_full, scX, scY = prepare_lstm_sequences(df_lag)
    full_scaled = model.predict(X_full).ravel()
    full_pred = scY.inverse_transform(full_scaled.reshape(-1,1)).ravel()
    y_full = df_lag["pm25"].iloc[LSTM_LOOKBACK:]

    df_full = pd.DataFrame({"y_true": y_full.values, "y_pred": full_pred}, index=idx_full)
    df_full.to_csv(OUT_DIR / "LSTM_FULL_predictions_2000_2025.csv")
    plot_pred_vs_actual(df_full, "LSTM_FULL")
    save_trend_summary("LSTM", df_full)

    return df_full


# ============================================================
# PIPELINE RUN
# ============================================================
def run():
    print("Loading files...")
    df = load_all_monthly(PATHS)
    print("Inner join shape:", df.shape)

    df_lag = make_lags_monthly(df)
    X_train, X_test, y_train, y_test, train_df, test_df = time_split(df_lag)

    # sklearn models (5)
    sklearn_results = train_standard_models(X_train, y_train, X_test, y_test, df_lag)

    # LSTM (full range)
    train_lstm(train_df, test_df, df_lag)

    print("Pipeline done. Outputs in:", OUT_DIR)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    run()
