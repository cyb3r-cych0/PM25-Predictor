# Model: Lasso

**Auto-generated report** — Lasso

## 1. Training & model info

**Model parameters (best-effort):**
```json
{
  "alphas": "warn",
  "copy_X": true,
  "cv": 5,
  "eps": 0.001,
  "fit_intercept": true,
  "max_iter": 1000,
  "n_alphas": "deprecated",
  "n_jobs": -1,
  "positive": false,
  "precompute": "auto",
  "random_state": 42,
  "selection": "cyclic",
  "tol": 0.0001,
  "verbose": false
}
```

- **Training data (features):** dew_temp, pbl, surface_air_temp, surface_pressure, surface_skin_temp, surface_wind_temp, surface_precipitation (monthly).

## 2. Metrics

| Metric | Value |
|---:|---:|
| n_obs | 61 |
| r2 | 0.5694827481547038 |
| rmse | 3.0818557625644964 |
| mae | 2.3249138181832665 |


## 3. Trends (actual vs predicted & bias)

| Series | Slope (µg/m³·yr⁻¹) | p-value | Interpretation |
|---|---:|---:|---|
| Actual | -0.2412 | 0.5638 | decreasing, not significant |
| Predicted | -0.1237 | 0.6334 | decreasing, not significant |
| Bias (pred-actual) | +0.1175 | 0.6602 | increasing, not significant |


The observed PM₂.₅ exhibits a decreasing, not significant trend of **-0.2412 µg·m⁻³·yr⁻¹** (p=0.564).

The Lasso prediction shows a decreasing, not significant trend of **-0.1237 µg·m⁻³·yr⁻¹** (p=0.633).

This results in a bias trend (pred−actual) of **+0.1175 µg·m⁻³·yr⁻¹** (p=0.660).

## 4. Figure (Actual vs Predicted + bias)

![Lasso trends](..\actual_vs_pred_trends\Lasso_actual_vs_pred_trend.png)

## 5. Notes / recommended actions

- Check residuals and seasonal decomposition if bias trend is non-zero.
- If bias trend is significant, consider recalibration (e.g., time-varying intercept) or stacking with stronger models.
- Archive scalers and feature order with model files for reproducibility.
