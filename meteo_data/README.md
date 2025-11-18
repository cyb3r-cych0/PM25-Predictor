## 📦 Dataset — PM₂.₅ and Meteorological Variables

This folder contains all **raw input datasets** for model training and trend analysis.

---

## 📘 Purpose

The datasets in this folder represent the **base data layer** of the project.  

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

---
