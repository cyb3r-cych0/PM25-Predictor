### 📊 Trend Analysis — PM₂.₅ Predictions vs Observations

This folder contains all scripts and tools for **trend computation, significance testing, and visualization** of model predictions versus observed PM₂.₅ concentrations.

Trend analysis is a key step in the AirPoll project — it quantifies whether predicted PM₂.₅ values increase, decrease, or remain stable over time, and compares those trends to the actual measured trends.

---

## 📘 Purpose

Once all models have been trained and their predictions saved, the scripts in this folder are used to:
1. Compute **actual vs predicted trends** for each model.
2. Test the **statistical significance** (p-values) of those trends.
3. Create **multi-model trend comparison figures** and diagnostics.

Each script outputs results to a specific directory, making it easy to track, visualize, and report model behaviors.

---

## 🗂️ Folder structure

Model-trend-Analysis/
│
├── model_trends_actual_vs_pred.py # Computes and plots actual vs predicted & bias trends
├── model_trends_significance.py # Performs significance tests for increasing/decreasing trends
├── model_trend_multi_plots.py # Combines multiple model trends into unified visuals
│
├── actual_vs_pred_trends/ # Output of script 1 — per-model PNGs & CSV summary
├── inc-dec_significance_trends/ # Output of script 2 — significance results & plots
├── computed_trends_diagnostics/ # Output of script 3 — combined multi-model trend plots
└── README.md # (This file)


---

## ⚙️ Script 1 — `model_trends_actual_vs_pred.py`

# 🎯 Purpose
Computes and visualizes **actual vs predicted PM₂.₅ trends** for each trained model.

# 🔍 What it does
- Loads predictions and actual PM₂.₅ from the `models-training` outputs.  
- Fits linear trends (using `scipy.stats.linregress`) to:
  - **Actual PM₂.₅**  
  - **Predicted PM₂.₅**  
  - **Bias (predicted − actual)**  
- Calculates slope (µg·m⁻³·yr⁻¹) and p-value for each series.  
- Saves visual comparisons as dual-panel plots:
  - **Left:** Actual vs predicted with trendlines  
  - **Right:** Bias vs time with its trendline

# ▶️ Usage
```bash
python model_trends_actual_vs_pred.py
```

# 📁 Output — actual_vs_pred_trends/
File	Description
<Model>_actual_vs_pred_trend.png	Trend comparison plots per model
models_actual_vs_pred_trend_summary.csv	Summary table with slope, p-values, and interpretations

# 🧮 Example console output
RandomForest — Actual trend: -0.2412/yr, p=0.5638 (decreasing, not significant)
GradientBoosting — Bias trend: +0.5653/yr, p=0.0361 (increasing, significant)
...
Saved combined summary: models_actual_vs_pred_trend_summary.csv

## ⚙️ Script 2 — model_trends_significance.py
🎯 Purpose

Determines whether each variable or model trend is increasing or decreasing and whether that change is statistically significant.

# 🔍 What it does

Reads output from actual_vs_pred_trends/
Applies trend significance testing (p < 0.05 threshold)
Classifies trends as:
- “Increasing, significant”
- “Increasing, not significant”
- “Decreasing, significant”
- “Decreasing, not significant”

Creates bar or scatter plots visualizing the direction and magnitude of trends across models.

# ▶️ Usage
python model_trends_significance.py

# 📁 Output — inc-dec_significance_trends/
File	Description
<Model>_significance_plot.png	Per-model visual showing trend significance
trend_significance_summary.csv	Summary table with trend direction, slope, and p-values
trend_significance_report.txt	Human-readable interpretation summary

# 🧮 Example console output
- RandomForest — Decreasing, not significant
- GradientBoosting — Increasing, significant
- LSTM — Stable, not significant
- Summary saved: trend_significance_summary.csv

## ⚙️ Script 3 — model_trend_multi_plots.py
🎯 Purpose

Generates composite multi-model trend diagnostics — side-by-side plots comparing slopes and bias trends for all models in a single figure.

# 🔍 What it does
Loads combined outputs from the previous scripts
- Creates multi-panel plots summarizing:
- Actual vs predicted trends (all models)
- Bias trend magnitudes
- Model performance metrics (R², RMSE, MAE)
- Adds annotations (trend slopes, significance labels)

# ▶️ Usage
python model_trend_multi_plots.py

# 📁 Output — computed_trends_diagnostics/
File	Description
multi_model_trend_comparison.png	Combined slope/bias comparison for all models
trend_diagnostics_table.csv	Consolidated numeric diagnostics
multi_model_summary_plot.png	Compact summary visualization (R² vs bias trend)

## 📊 Typical example workflow
# Step 1 — Compute actual vs predicted trends
python model_trends_actual_vs_pred.py

# Step 2 — Assess significance of increasing/decreasing trends
python model_trends_significance.py

# Step 3 — Generate multi-model comparison visuals
python model_trends_compute_diag.py

Resulting folders:
- actual_vs_pred_trends/
- inc-dec_significance_trends/
- computed_trends_diagnostics/

Each contains publication-ready CSVs, summaries, and figures.

| Folder                         | Output                                  | Purpose                                       |
| ------------------------------ | --------------------------------------- | --------------------------------------------- |
| `actual_vs_pred_trends/`       | `RandomForest_actual_vs_pred_trend.png` | Shows actual vs predicted vs bias trend       |
| `inc-dec_significance_trends/` | `trend_significance_summary.csv`        | Summarizes increasing/decreasing significance |
| `computed_trends_diagnostics/` | `multi_model_trend_comparison.png`      | Compares all models in one figure             |

# 🧩 Integration in the workflow

| Stage                    | Input              | Output                         | Next Step              |
| ------------------------ | ------------------ | ------------------------------ | ---------------------- |
| 1️⃣ Trend computation    | Model predictions  | `actual_vs_pred_trends/`       | Significance testing   |
| 2️⃣ Significance testing | Trend summary CSVs | `inc-dec_significance_trends/` | Multi-model comparison |
| 3️⃣ Multi-model plotting | All summaries      | `computed_trends_diagnostics/` | Reporting / paper      |

# 🧠 Interpretation notes

- Slopes are expressed in µg·m⁻³·yr⁻¹
- p-values represent the probability that the slope = 0
- Significance threshold: p < 0.05 (95% confidence)
- Directional interpretation:
- Positive slope → Increasing trend
- Negative slope → Decreasing trend

# 🧾 Typical outputs summary (example)

| Model            | Actual trend | p(actual) | Pred trend | p(pred) | Bias trend |    p(bias) |       Significance |
| :--------------- | -----------: | --------: | ---------: | ------: | ---------: | ---------: | -----------------: |
| RandomForest     |      −0.2412 |    0.5638 |    +0.0052 |  0.9838 |    +0.2465 |     0.3256 |    Not significant |
| GradientBoosting |      −0.2412 |    0.5638 |    +0.3241 |  0.2583 |    +0.5653 | **0.0361** | Bias ↑ significant |
| Lasso            |      −0.2412 |    0.5638 |    −0.1237 |  0.6334 |    +0.1175 |     0.6602 |    Not significant |
| Ridge            |      −0.2412 |    0.5638 |    −0.0882 |  0.7328 |    +0.1530 |     0.5697 |    Not significant |
| LSTM             |      −0.0439 |    0.9195 |    +0.0450 |  0.8868 |    +0.0889 |     0.7369 |    Not significant |
