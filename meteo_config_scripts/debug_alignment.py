
# debug_alignment.py
from pathlib import Path
import pandas as pd

ROOT = Path(".")
DATA_DIR = ROOT / "meteo_data"

FILES = {
    "pm25": DATA_DIR / "Total-Surface-Mass-Concentration-PM2.5.csv",
    "dew_temp": DATA_DIR / "2-meter dew point temperature.csv",
    "pbl": DATA_DIR / "Planetary boundary layer height.csv",
    "surface_air_temp": DATA_DIR / "Surface air temperature.csv",
    "surface_pressure": DATA_DIR / "Surface pressure.csv",
    "surface_skin_temp": DATA_DIR / "Surface skin temperature.csv",
    "surface_wind_temp": DATA_DIR / "Surface wind speed.csv",
    "surface_precipitation": DATA_DIR / "Total surface precipitation.csv",
}

def try_read(p):
    df = pd.read_csv(p)
    # try to parse index-as-first-column to datetime
    idx = pd.to_datetime(df.iloc[:,0].astype(str), errors='coerce')
    return df, idx

print("Diagnostic report")
indexes = {}
for k,p in FILES.items():
    pth = Path(p)
    if not pth.exists():
        print(f" MISSING: {p}")
        continue
    df, idx = try_read(pth)
    print(f"\nFile: {p}  shape={df.shape}")
    print(" Columns:", list(df.columns[:5]))
    print(" First 5 index-strings:", df.iloc[:5,0].astype(str).tolist())
    print(" Parsed index head:", idx[:5].tolist())
    print(" Parsed index tail:", idx[-5:].tolist())
    n_valid = idx.notna().sum()
    print(" Parsed datetime valid rows:", n_valid, " / ", len(idx))
    # store valid set of datetimes for intersection
    indexes[k] = set(idx.dropna().astype(str).tolist())

# intersection diagnostics (string compare)
all_keys = list(indexes.keys())
for i,k in enumerate(all_keys):
    for j in range(i+1, len(all_keys)):
        k2 = all_keys[j]
        if indexes.get(k) is None or indexes.get(k2) is None: continue
        inter = len(indexes[k].intersection(indexes[k2]))
        print(f"Overlap {k} <-> {k2}: {inter} timestamps")

print("\nIf overlaps are low or zero, you have a datetime/format mismatch or missing header naming issues.")
