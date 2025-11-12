# Model: Ridge

**Auto-generated report** — Ridge

## 1. Training & model info

**Model parameters (best-effort):**
```json
{
  "alpha_per_target": false,
  "alphas": "<class 'numpy.ndarray'>",
  "cv": 5,
  "fit_intercept": true,
  "gcv_mode": "<class 'NoneType'>",
  "scoring": "neg_mean_squared_error",
  "store_cv_results": false
}
```

- **Training data (features):** dew_temp, pbl, surface_air_temp, surface_pressure, surface_skin_temp, surface_wind_temp, surface_precipitation (monthly).

## 2. Metrics

| Metric | Value |
|---:|---:|
| n_obs | 61 |
| r2 | 0.560409677104245 |
| rmse | 3.1141612108259418 |
| mae | 2.346404255099084 |


## 3. Trends (actual vs predicted & bias)

| Series | Slope (µg/m³·yr⁻¹) | p-value | Interpretation |
|---|---:|---:|---|
| Actual | -0.2412 | 0.5638 | decreasing, not significant |
| Predicted | -0.0882 | 0.7328 | decreasing, not significant |
| Bias (pred-actual) | +0.1530 | 0.5697 | increasing, not significant |


The observed PM₂.₅ exhibits a decreasing, not significant trend of **-0.2412 µg·m⁻³·yr⁻¹** (p=0.564).

The Ridge prediction shows a decreasing, not significant trend of **-0.0882 µg·m⁻³·yr⁻¹** (p=0.733).

This results in a bias trend (pred−actual) of **+0.1530 µg·m⁻³·yr⁻¹** (p=0.570).

## 4. Figure (Actual vs Predicted + bias)

![Ridge trends](..\actual_vs_pred_trends\Ridge_actual_vs_pred_trend.png)

## 5. Notes / recommended actions

- Check residuals and seasonal decomposition if bias trend is non-zero.
- If bias trend is significant, consider recalibration (e.g., time-varying intercept) or stacking with stronger models.
- Archive scalers and feature order with model files for reproducibility.
