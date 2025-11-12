# 🌍 AirPoll — PM₂.₅ Modeling & Trend Analysis

Predict and analyze **PM₂.₅ concentration trends** using meteorological data. This project integrates multiple machine learning models (Random Forest, Gradient Boosting, Lasso, Ridge, LSTM) to forecast PM₂.₅ levels and evaluates their long-term bias and predictive performance through comprehensive trend analysis.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## 📘 Overview

The goal of this project is to **predict monthly PM₂.₅ concentrations** from meteorological variables and assess model accuracy through trend analysis. The workflow includes:

1. **Data Preparation**: Aligning 8 meteorological and PM₂.₅ CSV files into a unified monthly dataset.
2. **Model Training**: Training five predictive models on the aligned data.
3. **Trend & Bias Analysis**: Computing long-term slopes (µg·m⁻³·yr⁻¹) for actual, predicted, and bias (predicted−actual) values, including statistical significance testing.
4. **Model Comparison**: Generating comparative figures, metrics tables, and automated Markdown reports for each model.

This enables robust evaluation of model performance in capturing PM₂.₅ trends over time (2000–2025).

---

## ✨ Features

- **Multi-Model Comparison**: Trains and compares 5 ML models (Random Forest, Gradient Boosting, Lasso, Ridge, LSTM) on the same dataset.
- **Trend Analysis**: Computes linear trends for actual vs. predicted PM₂.₅ and bias, with p-value significance testing.
- **Automated Reporting**: Generates per-model Markdown reports with metrics, trends, and figures.
- **Visualization**: Produces time series plots, trend comparison bars, and diagnostic figures.
- **Data Alignment**: Handles temporal alignment and interpolation of meteorological data for consistent modeling.
- **Reproducibility**: Modular scripts with saved models, predictions, and evaluation metrics.

---

## 🛠️ Installation

### Prerequisites
- Python 3.12 - tensorflow compatible
- Git

### Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd AirPollution/Final-Work
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv airpoll_env
   airpoll_env\Scripts\activate  # On Windows
   # source airpoll_env/bin/activate  # On macOS/Linux
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Dependencies
Key packages include:
- **pandas, numpy**: Data manipulation
- **scikit-learn**: ML models (Random Forest, Gradient Boosting, Lasso, Ridge)
- **tensorflow/keras**: LSTM model
- **matplotlib, seaborn**: Visualization
- **scipy**: Statistical analysis (trend computation)
- **joblib**: Model serialization

---

## 🚀 Usage

### Quick Start Workflow

1. **Prepare Data**:
   ```bash
   cd Datasets
   python align_to_monthly.py
   ```
   This merges 8 CSV files into `aligned_monthly_outer_interp.csv` (307 monthly observations).

2. **Train Models** (run individually or in sequence):
   ```bash
   cd ../Models-Training

   # Random Forest
   cd RandomForest-Train
   python random_forest_train.py

   # Gradient Boosting
   cd ../GradientBoosting-Train
   python gradient_boosting_train.py

   # Lasso Regression
   cd ../Lasso-Train
   python lasso_train.py

   # Ridge Regression
   cd ../RidgeRegression-Train
   python ridge_train.py

   # LSTM (requires TensorFlow)
   cd ../LSTM-Train
   python lstm_train.py
   ```

3. **Analyze Trends**:
   ```bash
   cd ../Models-Trend-Analysis
   python model_trends_actual_vs_pred.py    # Compute trends
   python model_trends_significance.py       # Test significance
   python model_trends_compute_diag.py         # Generate diagnostics
   ```

4. **Compare Models & Generate Figures**:
   ```bash
   cd ..
   python model_comparison_figures.py
   ```

5. **Generate Reports**:
   ```bash
   cd Models-Trend-Analysis
   python generate_reports.py
   ```

### Output Locations
- **Figures**: `Figures-Comparison/` (time series, trend bars, composites)
- **Model Outputs**: `Models-Training/{Model}-Train/` (saved models, predictions, feature importances)
- **Trend Analysis**: `Models-Trend-Analysis/` subfolders (CSVs, PNGs, reports)
- **Reports**: `Models-Trend-Analysis/reports/` (per-model Markdown files)

---

## 📁 Project Structure

```
AirPollution/Final-Work/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── model_comparison_figures.py        # Main comparison script
│
├── Datasets/                          # Raw data and alignment
│   ├── align_to_monthly.py            # Data merging script
│   ├── aligned_monthly_outer_interp.csv  # Processed dataset (output)
│   ├── *.csv                          # 8 meteorological + PM₂.₅ files
│   └── README.md                      # Dataset documentation
│
├── Models-Training/                   # Model training scripts
│   ├── README.md                      # Training overview
│   ├── RandomForest-Train/            # RF model & outputs
│   ├── GradientBoosting-Train/        # GB model & outputs
│   ├── Lasso-Train/                   # Lasso model & outputs
│   ├── RidgeRegression-Train/         # Ridge model & outputs
│   └── LSTM-Train/                    # LSTM model & outputs
│
├── Models-Trend-Analysis/             # Trend computation & reporting
│   ├── README.md                      # Analysis overview
│   ├── generate_reports.py            # Report generator
│   ├── model_trends_*.py              # Trend analysis scripts
│   ├── actual_vs_pred_trends/         # Trend plots & summaries
│   ├── inc-dec_significance_trends/   # Significance tests
│   ├── computed_trends_diagnostics/   # Multi-model diagnostics
│   └── reports/                       # Generated Markdown reports
│
└── Figures-Comparison/                # Comparative visualizations
    ├── all_models_vs_actual_timeseries.png
    ├── model_trend_comparison_bars.png
    └── predictions_and_trends_comparison.png
```

---

## 📊 Results

### Key Metrics (Example)
| Model          | R²    | RMSE  | MAE   | Bias Trend (µg·m⁻³·yr⁻¹) |
|----------------|-------|-------|-------|--------------------------|
| Random Forest  | 0.625 | 2.876 | 2.105 | +0.25                    |
| Gradient Boosting | 0.569 | 3.084 | 2.406 | +0.56                    |
| Lasso          | 0.569 | 3.082 | 2.325 | +0.12                    |
| Ridge          | 0.560 | 3.114 | 2.346 | +0.15                    |
| LSTM           | 0.627 | 2.832 | 2.030 | +0.09                    |

### Outputs
- **Time Series Plots**: Actual PM₂.₅ vs. model predictions over 2000–2025.
- **Trend Comparisons**: Bar charts showing annual slopes for actual vs. predicted trends.
- **Diagnostic Figures**: Bias trends, rolling MAE, and multi-model summaries.
- **Reports**: Auto-generated Markdown files per model with metrics, trends, and interpretations.
- **CSVs**: Prediction files, trend summaries, and significance tables.

Trends are expressed in µg·m⁻³·yr⁻¹ with p-values indicating statistical significance (p < 0.05).

---

## 🤝 Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-name`.
3. Make changes and test thoroughly.
4. Submit a pull request with a clear description.

### Guidelines
- Follow PEP 8 for Python code.
- Add docstrings to new functions.
- Update README for significant changes.
- Test scripts on the provided dataset before committing.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

*For questions or issues, please open a GitHub issue or contact the maintainers.*

👨‍💻 Author

[cyb3r-cych0]
M.Sc. Computer Science
Email: [minigates21@gmail.com]
