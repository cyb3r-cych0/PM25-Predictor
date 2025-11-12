# Model: LSTM

**Auto-generated report** — LSTM

## 1. Training & model info

**Model parameters (best-effort):**
```json
{
  "model_file": "lstm_pm25_model.h5"
}
```

- **Training data (features):** dew_temp, pbl, surface_air_temp, surface_pressure, surface_skin_temp, surface_wind_temp, surface_precipitation (monthly).

## 2. Metrics

| Metric | Value |
|---:|---:|
| n_obs | 59 |
| r2 | 0.6264894433225041 |
| rmse | 2.8318543283084967 |
| mae | 2.030142254237288 |


## 3. Trends (actual vs predicted & bias)

| Series | Slope (µg/m³·yr⁻¹) | p-value | Interpretation |
|---|---:|---:|---|
| Actual | +0.3444 | 0.4290 | increasing, not significant |
| Predicted | +0.0450 | 0.8868 | increasing, not significant |
| Bias (pred-actual) | -0.2994 | 0.3867 | decreasing, not significant |


The observed PM₂.₅ exhibits a increasing, not significant trend of **+0.3444 µg·m⁻³·yr⁻¹** (p=0.429).

The LSTM prediction shows a increasing, not significant trend of **+0.0450 µg·m⁻³·yr⁻¹** (p=0.887).

This results in a bias trend (pred−actual) of **-0.2994 µg·m⁻³·yr⁻¹** (p=0.387).

## 4. Figure (Actual vs Predicted + bias)

![LSTM trends](..\actual_vs_pred_trends\LSTM_actual_vs_pred_trend.png)

## 5. Notes / recommended actions

- Check residuals and seasonal decomposition if bias trend is non-zero.
- If bias trend is significant, consider recalibration (e.g., time-varying intercept) or stacking with stronger models.
- Archive scalers and feature order with model files for reproducibility.
