# Configuration Scripts

This directory contains Python scripts for data configuration, alignment, and debugging tasks related to preparing meteorological and PM₂.₅ datasets for modeling.

## 📘 Purpose

These scripts handle the **preprocessing and alignment** of raw data to ensure consistency and completeness before model training. They are essential for creating a unified dataset from multiple sources, fixing alignment issues, and validating data integrity.

## 🗂️ Contents

| File | Description | Purpose |
|------|-------------|---------|
| `make_monthly_alignment.py` | Main script for aligning meteorological data to monthly frequency | Converts irregular timestamps to month-end, merges datasets, and interpolates missing values |
| `debug_alignment.py` | Debugging script for data alignment issues | Helps identify and resolve problems in data merging, timestamp conversion, or interpolation |

## ⚙️ Usage

1. **Monthly Alignment**:
   ```bash
   python make_monthly_alignment.py
   ```
   - Processes raw CSVs in `meteo_data/`
   - Outputs aligned dataset (e.g., `aligned_monthly_inner` & `aligned_monthly_outer`)

2. **Debugging**:
   ```bash
   python debug_alignment.py
   ```
   - Run if alignment fails or to inspect intermediate steps
   - Provides logs and reports on data issues

## 🧠 Integration with Pipeline

- Run these scripts before training models to prepare `meteo_data/aligned_monthly_inner.csv` / `aligned_monthly_outer`.
- Outputs are consumed by `models_pipeline.py` for training.
- Ensures temporal consistency for accurate trend analysis.

If data sources change, re-run alignment scripts to update the master dataset.
