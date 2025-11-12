# Model: RandomForest

**Auto-generated report** — RandomForest

## 1. Training & model info

**Model parameters (best-effort):**
```json
{
  "bootstrap": true,
  "ccp_alpha": 0.0,
  "criterion": "squared_error",
  "max_depth": "<class 'NoneType'>",
  "max_features": 1.0,
  "max_leaf_nodes": "<class 'NoneType'>",
  "max_samples": "<class 'NoneType'>",
  "min_impurity_decrease": 0.0,
  "min_samples_leaf": 1,
  "min_samples_split": 2,
  "min_weight_fraction_leaf": 0.0,
  "monotonic_cst": "<class 'NoneType'>",
  "n_estimators": 200,
  "n_jobs": 1,
  "oob_score": false,
  "random_state": 42,
  "verbose": 0,
  "warm_start": false
}
```

- **Training data (features):** dew_temp, pbl, surface_air_temp, surface_pressure, surface_skin_temp, surface_wind_temp, surface_precipitation (monthly).

## 2. Metrics

| Metric | Value |
|---:|---:|
| n_obs | 61 |
| r2 | 0.6250968062452222 |
| rmse | 2.875918664438696 |
| mae | 2.104525802067211 |


## 3. Trends (actual vs predicted & bias)

| Series | Slope (µg/m³·yr⁻¹) | p-value | Interpretation |
|---|---:|---:|---|
| Actual | -0.2412 | 0.5638 | decreasing, not significant |
| Predicted | +0.0052 | 0.9838 | increasing, not significant |
| Bias (pred-actual) | +0.2465 | 0.3256 | increasing, not significant |


The observed PM₂.₅ exhibits a decreasing, not significant trend of **-0.2412 µg·m⁻³·yr⁻¹** (p=0.564).

The RandomForest prediction shows a increasing, not significant trend of **+0.0052 µg·m⁻³·yr⁻¹** (p=0.984).

This results in a bias trend (pred−actual) of **+0.2465 µg·m⁻³·yr⁻¹** (p=0.326).

## 4. Figure (Actual vs Predicted + bias)

![RandomForest trends](..\actual_vs_pred_trends\RandomForest_actual_vs_pred_trend.png)

## 5. Notes / recommended actions

- Check residuals and seasonal decomposition if bias trend is non-zero.
- If bias trend is significant, consider recalibration (e.g., time-varying intercept) or stacking with stronger models.
- Archive scalers and feature order with model files for reproducibility.
