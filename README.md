# 🌍 AirPoll — PM₂.₅ Modeling & Trend Analysis

Predict and analyze **PM₂.₅ concentration trends** using meteorological data. This project integrates multiple machine learning models (Random Forest, Gradient Boosting, Lasso, Ridge, MLR, LSTM) to forecast PM₂.₅ levels and evaluates their long-term bias and predictive performance through comprehensive trend analysis.

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
2. **Model Training**: Training six predictive models on the aligned data.
3. **Trend & Bias Analysis**: Computing long-term slopes (µg·m⁻³·yr⁻¹) for actual, predicted, and bias (predicted−actual) values, including statistical significance testing.
4. **Model Comparison**: Generating comparative figures, metrics tables, and trend summaries for each model.

This enables robust evaluation of model performance in capturing PM₂.₅ trends over time (2000–2025).

---

## ✨ Features

- **Multi-Model Comparison**: Trains and compares 6 ML models (Random Forest, Gradient Boosting, Lasso, Ridge, MLR, LSTM) on the same dataset.
- **Trend Analysis**: Computes linear trends for actual vs. predicted PM₂.₅ and bias, with p-value significance testing.
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
   cd PM25-Predictor
Create a virtual environment (recommended):


python -m venv airpoll_env
airpoll_env\Scripts\activate  # On Windows
# source airpoll_env/bin/activate  # On macOS/Linux
Install dependencies:


pip install -r requirements.txt
Dependencies
Key packages include:

`pandas, numpy: Data manipulation`
`scikit-learn: ML models (Random Forest, Gradient Boosting, Lasso, Ridge, MLR)`
`tensorflow/keras: LSTM model`
`matplotlib, seaborn: Visualization`
`scipy: Statistical analysis (trend computation)`
`joblib: Model serialization`

### 🚀 Usage
Quick Start Workflow
Prepare Data:


cd meteo_data
python align_to_monthly.py
This merges 8 CSV files into aligned_monthly_outer_interp.csv (307 monthly observations).

Train Models & Compute Trends:


cd ..
python models_pipeline.py
This trains all 6 models (Random Forest, Gradient Boosting, Lasso, Ridge, MLR, LSTM), computes trends, and saves outputs.

Generate Figures:


cd plot_scripts
python models_actual_vs_pred_trends_plots.py
python models_pred_vs_obs_supp_plots.py
Output Locations
Figures: plot_figures/ (time series, trend bars, composites)
Model Outputs: models_pipeline_data/ (saved models, predictions, metrics, trend summaries)
Aligned Data: meteo_data/aligned_monthly_outer_interp.csv

### 📁 Project Structure

PM25-Predictor/
│
├── README.md                          # This file
├── LICENSE                            # Project license
├── .gitignore                         # Git ignore rules
├── requirements.txt                   # Python dependencies
├── TODO.md                            # Task list for updates
├── models_pipeline.py                 # Main pipeline script for training models
│
├── config_scripts/                    # Configuration and alignment scripts
│   ├── README.md                      # Documentation for config scripts
│   ├── make_monthly_alignment.py      # Script for aligning data to monthly frequency
│   └── debug_alignment.py             # Debugging script for data alignment
│
├── meteo_data/                        # Raw meteorological and PM₂.₅ data
│   ├── README.md                      # Documentation for datasets
│   ├── *.csv                          # Raw data files (8 meteorological + PM₂.₅)
│   ├── align_to_monthly.py            # Data alignment script
│   ├── aligned_monthly_*.csv          # Processed aligned datasets
│   ├── alignment_report.json          # Alignment report
│   └── Total-Surface-Mass-Concentration-PM2.5.csv  # Target variable
│
├── models_pipeline_data/              # Trained models and predictions
│   ├── README.md                      # Documentation for model outputs
│   ├── {Model}_model.joblib           # Serialized trained models
│   ├── {Model}_FULL_predictions_*.csv # Full period predictions
│   ├── {Model}_TEST_predictions.csv   # Test set predictions
│   ├── {Model}_metrics.json           # Performance metrics
│   ├── {Model}_FULL.png               # Prediction plots
│   ├── {Model}_trend_summary_FULL.csv # Trend summaries
│   ├── lstm_best.h5                   # LSTM model weights
│   └── lstm_scaler_*.joblib           # LSTM scalers
│
├── plot_scripts/                      # Scripts for generating plots
│   ├── README.md                      # Documentation for plotting scripts
│   ├── models_actual_vs_pred_trends_plots.py  # Main trend plotting script
│   ├── models_pred_vs_obs_supp_plots.py       # Supplementary plotting script
│   └── classP.py                      # Plotting utility class
│
└── plot_figures/                      # Generated figures and plots
    ├── README.md                      # Documentation for figures
    ├── 01_*_actual_vs_pred_trends.png # Individual model trend plots
    ├── combined_models_*.png          # Combined model comparisons
    ├── supplementary_*.png            # Supplementary diagnostic plots
    └── *.png      
                        # Various generated PNG files

### 📊 Results
Key Metrics (Example)
| Model              | R²    | RMSE  | MAE   | Bias Trend (µg·m⁻³·yr⁻¹) |
|--------------------|-------|-------|-------|--------------------------|
| Random Forest      | 0.625 | 2.876 | 2.105 | +0.25                    |
| Gradient Boosting  | 0.569 | 3.084 | 2.406 | +0.56                    |
| Lasso              | 0.569 | 3.082 | 2.325 | +0.12                    |
| Ridge              | 0.560 | 3.114 | 2.346 | +0.15                    |
| MLR                | 0.550 | 3.150 | 2.380 | +0.10                    |
| LSTM               | 0.627 | 2.832 | 2.030 | +0.09                    |

Outputs
Time Series Plots: Actual PM₂.₅ vs. model predictions over 2000–2025.
Trend Comparisons: Bar charts showing annual slopes for actual vs. predicted trends.
Diagnostic Figures: Bias trends, rolling MAE, and multi-model summaries.
CSVs: Prediction files, trend summaries, and significance tables.
Trends are expressed in µg·m⁻³·yr⁻¹ with p-values indicating statistical significance (p < 0.05).

## 🤝 Contributing
Fork the repository.
Create a feature branch: git checkout -b feature-name.
Make changes and test thoroughly.
Submit a pull request with a clear description.
Guidelines
Follow PEP 8 for Python code.
Add docstrings to new functions.
Update README for significant changes.
Test scripts on the provided dataset before committing.
📄 License
This project is licensed under the MIT License.

For questions or issues, please open a GitHub issue or contact the maintainers.

## 👨‍💻 Author

cyb3r-cych0
M.Sc. Computer Science
Email: minigates21@gmail.com