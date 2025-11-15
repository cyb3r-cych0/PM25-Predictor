### Summary

All input meteorological variables and PM₂.₅ concentrations were merged into a single, monthly-aligned dataset (`aligned_monthly_outer_interp.csv`).
To achieve this, convertion of all timestamps to month-end frequency, performed an outer join on the time index, and linearly interpolated missing values.
The resulting dataset contains 307 monthly observations (2000–2025) with complete coverage across all predictors.
This alignment ensured temporal consistency for model training and enabled valid trend estimation on a uniform monthly basis.

## 📦 Dataset — PM₂.₅ and Meteorological Variables

This folder contains all **raw input datasets** and the **data alignment script** used to prepare the monthly-resolved master dataset for model training and trend analysis.

---

## 📘 Purpose

The datasets in this folder represent the **base data layer** of the project.  
They are used to generate a single, continuous, and temporally aligned file —  
`aligned_monthly_outer_interp.csv` — which serves as the **input for all predictive models**.

---

## 🗂️ Contents

| File | Description | Frequency | Units / Notes |
|------|--------------|------------|----------------|
| `Total-Surface-Mass-Concentration-PM2.5.csv` | Ground truth PM₂.₅ concentrations (target variable) | Irregular → Monthly | µg·m⁻³ |
| `2-meter dew point temperature.csv` | Dew point temperature at 2 m | Monthly | °C |
| `Planetary boundary layer height.csv` | Planetary boundary layer (PBL) height | Monthly | m |
| `Surface air temperature.csv` | Surface air temperature | Monthly | K |
| `Surface pressure.csv` | Surface pressure | Monthly | Pa |
| `Surface skin temperature.csv` | Surface skin temperature | Monthly | K |
| `Surface wind speed.csv` | Surface wind speed | Monthly | m s⁻¹ |
| `Total surface precipitation.csv` | Total surface precipitation | Monthly | mm or kg m⁻² |
| `align_to_monthly.py` | Python script to align and merge all CSVs into one dataset |
| *(output)* `aligned_monthly_outer_interp.csv` | Final monthly-aligned dataset with interpolation (used for all models) |

---

## ⚙️ Data preparation script — `align_to_monthly.py`

This script combines the 8 input CSVs into a single, synchronized monthly dataset by:

1. **Parsing timestamps** — converts `Datetime` / `Date` columns to month-end format.  
2. **Resampling** — ensures every variable is monthly (uses mean if multiple records per month).  
3. **Outer join** — merges all variables on a shared monthly timeline.  
4. **Linear interpolation** — fills any missing values to achieve complete coverage.  
5. **Export** — saves `aligned_monthly_outer_interp.csv` with all variables aligned.

---

### ▶️ Usage

Run the script from the dataset folder:

```bash
python align_to_monthly.py
```

## 📊 Output description — aligned_monthly_outer_interp.csv

Rows: 307 monthly observations (2000 – 2025)
Columns: 8 (PM₂.₅ + 7 meteorological predictors)
Index: Monthly timestamp (Datetime)
No missing values — all variables linearly interpolated

Column	Description	Example Value
pm25	Monthly average PM₂.₅	34.21
dew_temp	2 m dew point temperature	16.7
pbl	Planetary boundary layer height	322.4
surface_air_temp	Surface air temperature	294.8
surface_pressure	Surface pressure	101210.0
surface_skin_temp	Surface skin temperature	298.1
surface_wind_temp	Surface wind speed	3.25
surface_precipitation	Total surface precipitation	1.82

## 🧠 Why this step matters

All machine learning models require a consistent and complete feature matrix.
This preprocessing step ensures:
- Each month aligns perfectly across all variables.
- No month has missing predictor values.
- The time series is evenly spaced (essential for trend analysis).
Without alignment and interpolation, only 26 overlapping months would exist across all files — insufficient for modeling.

## 🧩 Integration with the pipeline

Run this script → produces aligned_monthly_outer_interp.csv.
Move or reference that file in the ModelTraining/ folder to train models.
Downstream scripts (trend analysis, bias evaluation, report generation) all depend on this unified dataset.