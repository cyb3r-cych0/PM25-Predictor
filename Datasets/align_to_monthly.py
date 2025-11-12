# align_to_monthly.py
"""
Create a single monthly-aligned dataset with outer-join + interpolation.

Edit INPUT_FILES if you want to specify exact file paths or keys.
If INPUT_FILES is empty, the script will try to auto-discover CSVs in cwd.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np

OUT_CSV = "aligned_monthly_outer_interp.csv"

# ----- CONFIG: Edit these paths or leave empty for auto-discovery -----
# Format: "friendly_key": "path/to/file.csv"
INPUT_FILES = {
    # example defaults (adjust or replace with your exact filenames)
    "pm25": "Total-Surface-Mass-Concentration-PM2.5.csv",
    "dew_temp": "2-meter dew point temperature.csv",
    "pbl": "Planetary boundary layer height.csv",
    "surface_air_temp": "Surface air temperature.csv",
    "surface_pressure": "Surface pressure.csv",
    "surface_skin_temp": "Surface skin temperature.csv",
    "surface_wind_temp": "Surface wind speed.csv",
    "surface_precipitation": "Total surface precipitation.csv",
}
# ---------------------------------------------------------------------

def discover_csvs():
    # find csv files in cwd excluding the output file
    p = Path(".")
    csvs = [str(x) for x in p.glob("*.csv") if x.name != OUT_CSV]
    return csvs

def infer_datetime_and_value_columns(df):
    # common names we expect
    datetime_cols = [c for c in df.columns if c.lower() in ("datetime", "date", "time")]
    value_cols = [c for c in df.columns if c.lower() in ("values", "value", "val", "pm25", "pm2.5", "pm_2_5")]
    # fallback to first datetime-like column by dtype
    if not datetime_cols:
        for c in df.columns:
            if np.issubdtype(df[c].dtype, np.datetime64):
                datetime_cols = [c]; break
            # try parsing strings that look like datetimes later (we'll coerce anyway)
    # fallback to second column as values if not found
    if not value_cols:
        cand = [c for c in df.columns if c not in datetime_cols]
        if cand:
            value_cols = [cand[0]]
    return (datetime_cols[0] if datetime_cols else None, value_cols[0] if value_cols else None)

def load_and_resample(path, to_month_end="ME"):
    df = pd.read_csv(path)
    dt_col, val_col = infer_datetime_and_value_columns(df)
    if dt_col is None or val_col is None:
        # try heuristics: assume first column is datetime, second is value
        if df.shape[1] >= 2:
            dt_col, val_col = df.columns[0], df.columns[1]
        else:
            raise RuntimeError(f"Cannot infer columns for {path}")
    # parse datetimes
    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    # drop rows with bad datetimes
    df = df.dropna(subset=[dt_col])
    # set month-end index
    df = df.set_index(dt_col).sort_index()
    # resample to month-end and take mean (handles multiple samples per month)
    out = df[val_col].resample(to_month_end).mean().rename(path).to_frame()
    return out

def main():
    # if INPUT_FILES left with defaults that don't exist, try discover
    use_auto = False
    missing = [k for k,v in INPUT_FILES.items() if not os.path.exists(v)]
    if len(INPUT_FILES) == 0 or (missing and all(not os.path.exists(v) for v in INPUT_FILES.values())):
        print("No valid INPUT_FILES found — auto-discovering CSV files in current directory.")
        csvs = discover_csvs()
        if not csvs:
            raise SystemExit("No CSV files found to process.")
        INPUTS = {Path(p).stem: p for p in csvs}
        use_auto = True
    else:
        # use provided mapping but filter out non-existent paths
        INPUTS = {k: v for k,v in INPUT_FILES.items() if os.path.exists(v)}
        if not INPUTS:
            raise SystemExit("Provided INPUT_FILES paths do not exist. Edit the script and try again.")

    print("Loading files and converting to monthly timestamps...")
    series_list = []
    for key, path in INPUTS.items():
        try:
            s = load_and_resample(path)
            # rename column to friendly key if using manual mapping; otherwise keep filename
            if use_auto:
                s.columns = [key]
            else:
                s.columns = [key]
            rows = len(s.dropna())
            rng = (s.index.min(), s.index.max()) if rows>0 else (None, None)
            print(f" - {path} -> {key}\n   rows: {rows}, range: {rng[0]} -> {rng[1]}")
            series_list.append(s)
        except Exception as e:
            print(f"Skipping {path}: {e}")

    if not series_list:
        raise SystemExit("No time series loaded successfully.")

    # outer join on index
    print("\nMerging (outer join on month timestamps)...")
    df_outer = pd.concat(series_list, axis=1, join="outer").sort_index()
    print("Outer-joined shape (before interp):", df_outer.shape)

    # interpolate time-wise (linear), limit_direction both
    df_interp = df_outer.interpolate(method="time", limit_direction="both")
    na_per_col = df_interp.isna().sum()
    print("After interpolation, missing per column:\n", na_per_col)

    # drop rows still containing NaNs (if any)
    df_final = df_interp.dropna()
    print("Final aligned shape (after dropna):", df_final.shape)

    # save
    df_final.to_csv(OUT_CSV, index=True)
    print("\nSaved aligned dataset:", OUT_CSV)

    # print small summary
    print("\nSample head:\n", df_final.head())
    print("\nSample tail:\n", df_final.tail())

if __name__ == "__main__":
    main()
