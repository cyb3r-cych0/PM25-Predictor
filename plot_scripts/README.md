# Plot Scripts

This directory contains Python scripts for generating visualizations and plots related to model performance, trends, and comparisons in the PM₂.₅ prediction project.

## 📘 Purpose

These scripts produce **diagnostic and comparative figures** to evaluate model accuracy, trends, ci and biases. They create plots for time series, actual vs. predicted values, trend analyses, and supplementary visualizations, aiding in the interpretation of model results.

## 🗂️ Contents

`models_actual_vs_pred_trends_plots.py` | Main script for generating trend plots comparing actual vs. predicted PM₂.₅ over time | Produces PNG files in \plot_figures/\ (e.g., actual vs. pred trends for each model) |
`models_pred_vs_obs_supp_plots.py`| Supplementary plotting script for additional diagnostics | Generates supplementary figures like grid plots and individual model plots |


## ⚙️ Usage

1. **Trend Plots**:
   `python models_actual_vs_pred_trends_plots.py`
   
   - Generates time series plots showing actual PM₂.₅ vs. model predictions
   - Outputs to \plot_figures/\

2. **Supplementary Plots**:

   `python models_pred_vs_obs_supp_plots.py`
   
   - Creates additional diagnostic plots (e.g., combined grids, individual model visuals)

3. **Using classP.py**:
   - Import and use plotting classes for custom visualizations

## 🧠 Integration with Pipeline

- Run after model training and evaluation (after \models_pipeline.py\)
- Consumes data from \models_pipeline_data/\ (predictions, metrics)
- Outputs figures to \plot_figures/\ for reporting and analysis
- Supports trend analysis by visualizing long-term biases and performance

Ensure model outputs are available before running these scripts.
