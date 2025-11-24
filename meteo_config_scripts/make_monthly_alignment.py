#!/usr/bin/env python3
"""
make_monthly_alignment.py

Read raw CSV files (various datetime formats), convert to month-end, aggregate monthly,
and produce aligned CSVs:
  - aligned_monthly_inner.csv   (intersection months present in all sources)
  - aligned_monthly_outer.csv   (union of months, NaNs where missing; no interpolation)

Also writes a small JSON diagnostic report: alignment_report.json

Usage:
  python scripts/make_monthly_alignment.py

Edit INPUT_FILES mapping below if your CSV filenames differ or you want to include/exclude variables.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import json
from main_scripts.dir_config import DirConfig


class MakeMonthlyAlignment(DirConfig):
    def __init__(self, common_formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%d/%m/%Y %H:%M:%S",
            "%d/%m/%Y %H:%M",
            "%d/%m/%Y",
            "%Y/%m/%d",
            "%Y-%m",
            "%m/%d/%Y %H:%M:%S",
            "%m/%d/%Y",
            "%d-%b-%Y",         # e.g., 01-Jan-2000
        ]):
        super().__init__()

        self.meteo_data = super().load_meteo_data()
        self.out_dir = super().meteo_config_scripts_path()
        self.common_formats = common_formats

    def try_parse_dates(self, s: pd.Series):
        s = s.astype(str).copy()
        # try explicit formats (stop early if success)
        for fmt in self.common_formats:
            parsed = pd.to_datetime(s, format=fmt, errors="coerce")
            if parsed.notna().sum() >= 0.9 * len(parsed):   # if 90% parse success, accept
                return parsed
        # fallback to pandas auto-parse
        return pd.to_datetime(s, errors="coerce")

    def read_and_monthly(self, path: Path, colname: str):
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        df = pd.read_csv(path)
        if df.shape[1] == 1:
            # single-column CSV — not expected; raise to avoid silent mistakes
            raise ValueError(f"File {path} appears to contain only one column. Expected datetime + value.")
        date_col = df.columns[0]
        # prefer a column named 'Values' or 'value' or second column otherwise
        if "Values" in df.columns:
            value_col = "Values"
        elif "value" in df.columns:
            value_col = "value"
        else:
            value_col = df.columns[1]

        # parse dates robustly
        parsed = self.try_parse_dates(df[date_col].astype(str))
        n_bad = parsed.isna().sum()
        if n_bad > 0:
            print(f"[warn] {n_bad} rows in {path.name} could not be parsed as datetimes and will be dropped.")
        # assemble series, drop rows missing datetime or value
        ser = pd.DataFrame({
            "dt": parsed,
            colname: pd.to_numeric(df[value_col], errors="coerce")
        }).dropna(subset=["dt", colname])

        # normalize to month-end timestamp (period -> to_timestamp('M'))
        ser["month"] = ser["dt"].dt.to_period("M").dt.to_timestamp("M")
        monthly = ser.groupby("month")[colname].mean().to_frame()   # aggregate via mean for the month
        monthly.index.name = "month"
        monthly = monthly.sort_index()
        return monthly

    def align_monthly(self, input_files: dict, how_outer=False):
        dfs = {}
        for key, p in input_files.items():
            print(f"Reading {key} <- {p.name}")
            df_month = self.read_and_monthly(Path(p), key)
            dfs[key] = df_month

        # join all dataframes on index (month)
        keys = list(dfs.keys())
        base = dfs[keys[0]].copy()
        for k in keys[1:]:
            base = base.join(dfs[k], how="outer" if how_outer else "inner")

        # ensure index sorted and named
        base = base.sort_index()
        base.index.name = "month"
        return base, dfs

    def compute_overlap_matrix(self, dfs):
        keys = list(dfs.keys())
        n = len(keys)
        mat = np.zeros((n,n), dtype=int)
        for i,k1 in enumerate(keys):
            for j,k2 in enumerate(keys):
                mat[i,j] = len(dfs[k1].index.intersection(dfs[k2].index))
        return keys, mat

    def main(self):
        print("==== Monthly alignment utility ====")
        print("Input files to read (edit script if filenames differ):")
        for k,p in self.meteo_data.items():
            print(f"  - {k}: {p}")

        # produce outer (union) and inner (intersection) aligned tables
        print("\nProducing OUTER (union) alignment (no interpolation)...")
        outer_df, dfs = self.align_monthly(self.meteo_data, how_outer=True)
        outer_out = self.out_dir / "aligned_monthly_outer.csv"
        outer_df.to_csv(outer_out, index=True)
        print(f"Saved outer-aligned CSV: {outer_out} (shape={outer_df.shape})")

        print("\nProducing INNER (intersection) alignment (months present in all files)...")
        inner_df, _ = self.align_monthly(self.meteo_data, how_outer=False)
        inner_out = self.out_dir / "aligned_monthly_inner.csv"
        inner_df.to_csv(inner_out, index=True)
        print(f"Saved inner-aligned CSV: {inner_out} (shape={inner_df.shape})")

        # diagnostics
        report = {}
        report["outer_shape"] = tuple(map(int, outer_df.shape))
        report["inner_shape"] = tuple(map(int, inner_df.shape))

        # first/last timestamps (outer)
        if len(outer_df.index):
            report["outer_first"] = str(outer_df.index.min())
            report["outer_last"] = str(outer_df.index.max())
        if len(inner_df.index):
            report["inner_first"] = str(inner_df.index.min())
            report["inner_last"] = str(inner_df.index.max())

        # count full-observed months vs partial in outer
        report["outer_total_months"] = int(len(outer_df))
        report["outer_full_months"] = int((~outer_df.isna()).all(axis=1).sum())
        report["outer_partial_months"] = int(((~outer_df.isna()).any(axis=1) & outer_df.isna().any(axis=1)).sum())

        # overlap matrix
        keys, mat = self.compute_overlap_matrix(dfs)
        report["overlap_keys"] = keys
        report["overlap_matrix"] = mat.tolist()

        # show a quick summary to console
        print("\n=== Alignment diagnostic summary ===")
        print(f"Outer (union) shape: {report['outer_shape']}")
        print(f"Inner (intersection) shape: {report['inner_shape']}")
        print(f"Outer months: {report['outer_total_months']}  fully-observed: {report['outer_full_months']}  partial: {report['outer_partial_months']}")
        if "inner_first" in report:
            print(f"Inner months cover: {report['inner_first']} -> {report['inner_last']}")
        if "outer_first" in report:
            print(f"Outer months cover: {report['outer_first']} -> {report['outer_last']}")
        print("\nOverlap counts (rows/columns = variables):")
        print(pd.DataFrame(mat, index=keys, columns=keys))

        # write json diagnostic
        rpt = self.out_dir / "alignment_report.json"
        with open(rpt, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nWrote diagnostic report to: {rpt}")

        # quick sanity check for pm25 presence
        if "pm25" in inner_df.columns:
            print("\nSanity: pm25 column present in inner-aligned CSV (good).")
        else:
            print("\nWarning: pm25 column NOT present in inner alignment — check input filenames/columns.")

        print("\nDone. If you want inner-only months for modeling, use:")
        print(f"  {inner_out}")
        print("If you want full timeline for plotting / inspection (with NaNs), use:")
        print(f"  {outer_out}\n\n")

