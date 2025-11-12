# Model: GradientBoosting

**Auto-generated report** — GradientBoosting

## 1. Training & model info

**Model parameters (best-effort):**
```json
{
  "alpha": 0.9,
  "ccp_alpha": 0.0,
  "criterion": "friedman_mse",
  "init": "<class 'NoneType'>",
  "learning_rate": 0.05,
  "loss": "squared_error",
  "max_depth": 4,
  "max_features": "<class 'NoneType'>",
  "max_leaf_nodes": "<class 'NoneType'>",
  "min_impurity_decrease": 0.0,
  "min_samples_leaf": 1,
  "min_samples_split": 2,
  "min_weight_fraction_leaf": 0.0,
  "n_estimators": 300,
  "n_iter_no_change": "<class 'NoneType'>",
  "random_state": 42,
  "subsample": 1.0,
  "tol": 0.0001,
  "validation_fraction": 0.1,
  "verbose": 0,
  "warm_start": false
}
```

- **Training data (features):** dew_temp, pbl, surface_air_temp, surface_pressure, surface_skin_temp, surface_wind_temp, surface_precipitation (monthly).

## 2. Metrics

| Metric | Value |
|---:|---:|
| n_obs | 61 |
| r2 | 0.568876556426364 |
| rmse | 3.0840247096804925 |
| mae | 2.405666628425432 |


## 3. Trends (actual vs predicted & bias)

| Series | Slope (µg/m³·yr⁻¹) | p-value | Interpretation |
|---|---:|---:|---|
| Actual | -0.2412 | 0.5638 | decreasing, not significant |
| Predicted | +0.3241 | 0.2583 | increasing, not significant |
| Bias (pred-actual) | +0.5653 | 0.0361 | increasing, significant |


The observed PM₂.₅ exhibits a decreasing, not significant trend of **-0.2412 µg·m⁻³·yr⁻¹** (p=0.564).

The GradientBoosting prediction shows a increasing, not significant trend of **+0.3241 µg·m⁻³·yr⁻¹** (p=0.258).

This results in a bias trend (pred−actual) of **+0.5653 µg·m⁻³·yr⁻¹** (p=0.036).

## 4. Figure (Actual vs Predicted + bias)

![GradientBoosting trends](..\actual_vs_pred_trends\GradientBoosting_actual_vs_pred_trend.png)

## 5. Notes / recommended actions

- Check residuals and seasonal decomposition if bias trend is non-zero.
- If bias trend is significant, consider recalibration (e.g., time-varying intercept) or stacking with stronger models.
- Archive scalers and feature order with model files for reproducibility.
