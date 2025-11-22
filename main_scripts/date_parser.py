# ============================================================
# DATE PARSER
# ============================================================
import pandas as pd

class DateParser:
    def __init__(self, 
                 date_formats=
                 [
                     "%Y-%m-%d", 
                     "%Y-%m-%d %H:%M:%S", 
                     "%d/%m/%Y", 
                     "%d/%m/%Y %H:%M", 
                     "%d/%m/%Y %H:%M:%S"
                ],
                 lags=[1,2,3,12]
                 ):
        self.date_formats = date_formats
        self.lags = lags

    def try_parse_dates(self, s):
        s = s.astype(str)
        for fmt in self.date_formats:
            parsed = pd.to_datetime(s, format=fmt, errors="coerce")
            if parsed.notna().sum() >= 0.8 * len(parsed):
                return parsed
        return pd.to_datetime(s, errors="coerce")

    def read_and_normalize_monthly(self, path, col_name):
        print(f"Loading data from: {path}")
        df = pd.read_csv(path)
        if df.shape[1] == 1:
            df = df.reset_index()
        date_col = df.columns[0]
        val_col = "Values" if "Values" in df.columns else df.columns[1]

        parsed = self.try_parse_dates(df[date_col])
        df = df.assign(Datetime_raw=parsed)
        df = df.dropna(subset=["Datetime_raw", val_col])
        df["month"] = df["Datetime_raw"].dt.to_period("M").dt.to_timestamp("M")
        df = df.set_index("month").sort_index()
        return df[[val_col]].rename(columns={val_col: col_name})

    def load_all_monthly(self, paths_dict):
        merged = None
        for key, pth in paths_dict.items():
            dfk = self.read_and_normalize_monthly(pth, key)
            merged = dfk if merged is None else merged.join(dfk, how="inner")
        return merged.sort_index()
    
    def make_lags_monthly(self, df=None, lags=None):
        if lags is None:
            lags = self.lags
        if df is None:
            df = self.read_and_normalize_monthly()
        df_lag = df.copy()
        for lag in lags:
            for col in df.columns:
                df_lag[f"{col}_lag{lag}"] = df_lag[col].shift(lag)
        return df_lag.dropna()
    