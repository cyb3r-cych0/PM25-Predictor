import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Disable oneDNN optimizations (hides the startup message)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TensorFlow info and warning messages
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path

# --- Define paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
DATASETS_DIR = SCRIPT_DIR.parent.parent / "Datasets"
ALIGNED_CSV_PATH = DATASETS_DIR / "aligned_monthly_outer_interp.csv"
print("Looking for file at:", ALIGNED_CSV_PATH)

# ---------- User params ----------
ALIGNED_CSV_PATH = ALIGNED_CSV_PATH
SEQ_LEN = 12              # length of input sequence in months (use 12 for yearly context)
TEST_FRAC = 0.2           # last 20% months as test set
BATCH_SIZE = 16
EPOCHS = 200
MODEL_PATH = "lstm_pm25_model.h5"
SCALER_X_PATH = "lstm_scaler_X.save"
SCALER_y_PATH = "lstm_scaler_y.save"
RANDOM_SEED = 42
# ---------------------------------
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# --- Load aligned data ---
if not ALIGNED_CSV_PATH.exists():
    raise FileNotFoundError(f"Missing required file: {ALIGNED_CSV_PATH}. Run the 'align_to_monthly.py' script in the Datasets DIR first.")
df = pd.read_csv(ALIGNED_CSV_PATH, index_col=0, parse_dates=True)
print("Loaded dataset shape:", df.shape)

# ----- Feature engineering (same as previous scripts) -----
df = df.sort_index()
# create pm25 lags and rolling features (these become part of inputs)
for lag in (1,2,3):
    df[f'pm25_lag{lag}'] = df['pm25'].shift(lag)
df['pm25_rm_3'] = df['pm25'].rolling(window=3, min_periods=1).mean().shift(1)
df['pm25_rm_6'] = df['pm25'].rolling(window=6, min_periods=1).mean().shift(1)
df['pm25_rm_12'] = df['pm25'].rolling(window=12, min_periods=1).mean().shift(1)
months = df.index.month
df['month_sin'] = np.sin(2 * np.pi * months / 12)
df['month_cos'] = np.cos(2 * np.pi * months / 12)

df = df.dropna()  # drop head rows lost to lags
print("Prepared dataframe shape (after features):", df.shape)

# ---------- Create sequences ----------
# We'll use multivariate sequences of length SEQ_LEN to predict next-month pm25
feature_cols = df.columns.tolist()
feature_cols.remove("pm25")   # pm25 is target (we keep lagged pm25 as inputs since they exist)
X_all = df[feature_cols].values
y_all = df["pm25"].values.reshape(-1, 1)

def create_sequences(X, y, seq_len):
    X_seq, y_seq, idxs = [], [], []
    for i in range(seq_len, len(X)):
        X_seq.append(X[i-seq_len:i, :])    # rows [i-seq_len ... i-1]
        y_seq.append(y[i, 0])              # target at time i
        idxs.append(i)
    return np.array(X_seq), np.array(y_seq), np.array(idxs)

X_seq, y_seq, idxs = create_sequences(X_all, y_all, SEQ_LEN)
print("Sequences shape:", X_seq.shape, y_seq.shape)

# ---------- Time-based train/test split ----------
n_samples = len(X_seq)
test_size = int(np.ceil(TEST_FRAC * n_samples))
train_size = n_samples - test_size
X_train_seq = X_seq[:train_size]
X_test_seq  = X_seq[train_size:]
y_train = y_seq[:train_size].reshape(-1,1)
y_test  = y_seq[train_size:].reshape(-1,1)

print("Train sequences:", X_train_seq.shape[0], "Test sequences:", X_test_seq.shape[0])

# ---------- Scale data (fit scalers on training data only) ----------
# Flatten training X to fit scaler per feature
n_features = X_train_seq.shape[2]
X_train_flat = X_train_seq.reshape(-1, n_features)
X_test_flat  = X_test_seq.reshape(-1, n_features)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

scaler_X.fit(X_train_flat)
scaler_y.fit(y_train)

# transform
X_train_scaled = scaler_X.transform(X_train_flat).reshape(X_train_seq.shape)
X_test_scaled  = scaler_X.transform(X_test_flat).reshape(X_test_seq.shape)
y_train_scaled = scaler_y.transform(y_train)
y_test_scaled  = scaler_y.transform(y_test)

# Save scalers
joblib.dump(scaler_X, SCALER_X_PATH)
joblib.dump(scaler_y, SCALER_y_PATH)

# ---------- Build LSTM model ----------
tf.keras.backend.clear_session()
model = Sequential([
    LSTM(64, input_shape=(SEQ_LEN, n_features), return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# Callbacks
es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
mc = ModelCheckpoint(MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1)

# ---------- Train ----------
history = model.fit(
    X_train_scaled, y_train_scaled,
    validation_data=(X_test_scaled, y_test_scaled),
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    callbacks=[es, mc], verbose=2
)

# ---------- Predict (inverse scale) ----------
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1,1)).flatten()
y_true = scaler_y.inverse_transform(y_test_scaled).flatten()

# ---------- Evaluate ----------
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
print(f"LSTM results -- R2: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

# Save artifacts
model.save(MODEL_PATH)
pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv("lstm_preds_vs_actual.csv", index=False)
# Save losses plot
plt.figure(figsize=(8,4)); plt.plot(history.history['loss'], label='train'); plt.plot(history.history['val_loss'], label='val')
plt.title("LSTM training/validation loss"); plt.legend(); plt.tight_layout(); plt.savefig("lstm_loss.png"); plt.close()
# Pred vs actual plot
plt.figure(figsize=(10,4)); plt.plot(y_true, marker='o', label='actual'); plt.plot(y_pred, marker='x', label='predicted')
plt.title("LSTM: PM2.5 Actual vs Predicted (test)"); plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig("lstm_pred_vs_actual.png"); plt.close()

print("Saved model, scalers, predictions and plots.")
