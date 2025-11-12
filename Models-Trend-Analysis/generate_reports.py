# #!/usr/bin/env python3
"""
Hardcoded report generator with robust trend-row selection and column mapping.
Edit MODEL_CONFIG to point to exact files. Outputs go to ./reports/
"""
import os, json, textwrap
from pathlib import Path
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# ----------------- REQUIRED: Fill these with exact paths -----------------
MODEL_CONFIG = {
    "RandomForest": {
        "metrics_csv": ROOT / "computed_trends_diagnostics" / "RandomForest_trend_summary.csv",
        "trend_csv":   ROOT / "actual_vs_pred_trends" / "models_actual_vs_pred_trend_summary.csv",
        "pred_fig":    ROOT / "actual_vs_pred_trends" / "RandomForest_actual_vs_pred_trend.png",
        "model_file":  ROOT.parent / "Models-Training" / "RandomForest-Train" / "rf_with_lags_model.joblib"
    },
    "GradientBoosting": {
        "metrics_csv": ROOT / "computed_trends_diagnostics" / "GradientBoosting_trend_summary.csv",
        "trend_csv":   ROOT / "actual_vs_pred_trends" / "models_actual_vs_pred_trend_summary.csv",
        "pred_fig":    ROOT / "actual_vs_pred_trends" / "GradientBoosting_actual_vs_pred_trend.png",
        "model_file":  ROOT.parent / "Models-Training" / "GradientBoosting-Train" / "gbr_with_lags_model.joblib"
    },
    "Lasso": {
        "metrics_csv": ROOT / "computed_trends_diagnostics" / "Lasso_trend_summary.csv",
        "trend_csv":   ROOT / "actual_vs_pred_trends" / "models_actual_vs_pred_trend_summary.csv",
        "pred_fig":    ROOT / "actual_vs_pred_trends" / "Lasso_actual_vs_pred_trend.png",
        "model_file":  ROOT.parent / "Models-Training" / "Lasso-Train" / "lasso_with_lags_model.joblib"
    },
    "Ridge": {
        "metrics_csv": ROOT / "computed_trends_diagnostics" / "Ridge_trend_summary.csv",
        "trend_csv":   ROOT / "actual_vs_pred_trends" / "models_actual_vs_pred_trend_summary.csv",
        "pred_fig":    ROOT / "actual_vs_pred_trends" / "Ridge_actual_vs_pred_trend.png",
        "model_file":  ROOT.parent / "Models-Training" / "RidgeRegression-Train" / "ridge_with_lags_model.joblib"
    },
    "LSTM": {
        "metrics_csv": ROOT / "computed_trends_diagnostics" / "LSTM_trend_summary.csv",
        "trend_csv":   ROOT / "actual_vs_pred_trends" / "models_actual_vs_pred_trend_summary.csv",
        "pred_fig":    ROOT / "actual_vs_pred_trends" / "LSTM_actual_vs_pred_trend.png",
        "model_file":  ROOT.parent / "Models-Training" / "LSTM-Train" / "lstm_pm25_model.h5"
    }
}
# ------------------------------------------------------------------------

def safe_load_csv(path):
    try:
        if path is None: return None
        p = Path(path)
        if not p.exists(): return None
        return pd.read_csv(p)
    except Exception:
        return None

def load_model_params(pth):
    if pth is None: return None
    p = Path(pth)
    if not p.exists(): return None
    try:
        if str(p).lower().endswith((".joblib", ".pkl")):
            m = joblib.load(p)
            if hasattr(m, "get_params"):
                params = m.get_params()
                small = {k: (v if isinstance(v, (int, float, str, bool)) else str(type(v))) for k, v in params.items()}
                return small
            return {"model_type": str(type(m))}
        else:
            return {"model_file": str(p.name)}
    except Exception as e:
        return {"error": str(e), "path": str(p)}

def pretty_p(p):
    try:
        if pd.isna(p): return "n/a"
        return ("p<0.001" if p < 0.001 else f"p={p:.3f}")
    except Exception:
        return "n/a"

def compose_markdown(model, metrics_df, trend_row, figure_path, params):
    lines = []
    lines.append(f"# Model: {model}\n")
    lines.append(f"**Auto-generated report** — {model}\n")
    lines.append("## 1. Training & model info\n")
    if params:
        lines.append("**Model parameters (best-effort):**\n```json")
        try:
            lines.append(json.dumps(params, indent=2))
        except Exception:
            lines.append(str(params))
        lines.append("```\n")
    else:
        lines.append("- Model parameters not found or could not be loaded.\n")
    lines.append("- **Training data (features):** dew_temp, pbl, surface_air_temp, surface_pressure, surface_skin_temp, surface_wind_temp, surface_precipitation (monthly).\n")
    lines.append("## 2. Metrics\n")
    if metrics_df is not None and not metrics_df.empty:
        try:
            m = metrics_df.iloc[0].to_dict()
            lines.append("| Metric | Value |")
            lines.append("|---:|---:|")
            for k in ["n_obs", "r2", "rmse", "mae"]:
                lines.append(f"| {k} | {m.get(k,'')} |")
            lines.append("\n")
        except Exception:
            lines.append("_Metrics file unreadable or has unexpected format._\n")
    else:
        lines.append("_Per-model metric summary not provided or file missing._\n")
    lines.append("## 3. Trends (actual vs predicted & bias)\n")
    if trend_row is not None:
        # canonical keys should be numbers or strings parseable
        sa = trend_row.get("trend_actual_per_year") or trend_row.get("slope_actual_per_year") or trend_row.get("slope_actual") or trend_row.get("slope_actu") or trend_row.get("slope_actual")
        pa = trend_row.get("pval_actual") or trend_row.get("p_actual") or trend_row.get("pval_act")
        sp = trend_row.get("trend_pred_per_year") or trend_row.get("slope_pred_per_year") or trend_row.get("slope_pred")
        pp = trend_row.get("pval_pred") or trend_row.get("p_pred")
        sb = trend_row.get("trend_bias_per_year") or trend_row.get("slope_bias_per_year") or trend_row.get("slope_bias")
        pb = trend_row.get("pval_bias") or trend_row.get("p_bias")
        def cell(s,p):
            if s is None or (isinstance(s, float) and pd.isna(s)):
                return ("n/a","n/a","insufficient data")
            try:
                sf = float(s)
            except Exception:
                return ("n/a","n/a","insufficient data")
            direction = "increasing" if sf > 0 else "decreasing"
            sig = "significant" if (p is not None and float(p) < 0.05) else "not significant"
            return (f"{sf:+.4f}", f"{float(p):.4f}" if p is not None else "n/a", f"{direction}, {sig}")
        a_s,a_p,a_i = cell(sa, pa)
        p_s,p_p,p_i = cell(sp, pp)
        b_s,b_p,b_i = cell(sb, pb)
        lines.append("| Series | Slope (µg/m³·yr⁻¹) | p-value | Interpretation |")
        lines.append("|---|---:|---:|---|")
        lines.append(f"| Actual | {a_s} | {a_p} | {a_i} |")
        lines.append(f"| Predicted | {p_s} | {p_p} | {p_i} |")
        lines.append(f"| Bias (pred-actual) | {b_s} | {b_p} | {b_i} |")
        lines.append("\n")
        if a_s != "n/a":
            lines.append(f"The observed PM₂.₅ exhibits a {a_i} trend of **{a_s} µg·m⁻³·yr⁻¹** ({pretty_p(pa)}).\n")
        else:
            lines.append("Actual trend: insufficient data.\n")
        if p_s != "n/a":
            lines.append(f"The {model} prediction shows a {p_i} trend of **{p_s} µg·m⁻³·yr⁻¹** ({pretty_p(pp)}).\n")
        else:
            lines.append("Predicted trend: insufficient data.\n")
        if b_s != "n/a":
            lines.append(f"This results in a bias trend (pred−actual) of **{b_s} µg·m⁻³·yr⁻¹** ({pretty_p(pb)}).\n")
        else:
            lines.append("Bias trend: insufficient data.\n")
    else:
        lines.append("_No trend numeric summary provided for this model (file missing or unreadable)._ \n")
    lines.append("## 4. Figure (Actual vs Predicted + bias)\n")
    figp = Path(figure_path) if figure_path is not None else None
    if figp and figp.exists():
        rel = os.path.relpath(str(figp), str(REPORTS_DIR))
        lines.append(f"![{model} trends]({rel})\n")
    else:
        lines.append(f"- Figure not found (expected at {figure_path}). Place the model PNG here manually.\n")
    lines.append("## 5. Notes / recommended actions\n")
    lines.append(textwrap.dedent("""\
        - Check residuals and seasonal decomposition if bias trend is non-zero.
        - If bias trend is significant, consider recalibration (e.g., time-varying intercept) or stacking with stronger models.
        - Archive scalers and feature order with model files for reproducibility.
    """))
    return "\n".join(lines)

def select_trend_row_from_df(dftr, model):
    """
    Robust selection:
     - normalize column names (strip, lower)
     - try 'model' column exact/contains match
     - search any string columns for token match
     - map column names to canonical keys by returning a dict with normalized keys
    """
    # normalize columns
    df = dftr.copy()
    df.columns = [str(c).strip() for c in df.columns]
    cols_l = [c.lower().strip() for c in df.columns]

    # debug: show columns and first row (compact)
    # print("  Trend CSV columns:", df.columns.tolist())
    # try:
    #     print("  Trend CSV sample row0:", df.iloc[0].to_dict())
    # except Exception:
    #     pass

    token = model.lower()
    # 1) model column match
    if "model" in cols_l:
        col = df.columns[cols_l.index("model")]
        hits = df[df[col].astype(str).str.lower().str.contains(token, na=False)]
        if not hits.empty:
            return hits.iloc[0].to_dict()
    # 2) search string/object columns
    for c in df.columns:
        if df[c].dtype == object:
            hits = df[df[c].astype(str).str.lower().str.contains(token, na=False)]
            if not hits.empty:
                return hits.iloc[0].to_dict()
    # 3) fallback: if only one row, return it
    if len(df) == 1:
        return df.iloc[0].to_dict()
    # 4) last resort: None
    return None

def main():
    compiled_rows = []
    for model, conf in MODEL_CONFIG.items():
        print("Processing:", model)
        metrics_path = conf.get("metrics_csv")
        trend_path = conf.get("trend_csv")
        fig = conf.get("pred_fig")
        mfile = conf.get("model_file")

        metrics_df = safe_load_csv(metrics_path) if metrics_path is not None else None

        trend_row = None
        if trend_path is not None:
            dftr = safe_load_csv(trend_path)
            if dftr is not None and not dftr.empty:
                selected = select_trend_row_from_df(dftr, model)
                if selected is not None:
                    trend_row = selected
                else:
                    print(f"⚠️ No matching row found for model '{model}' in {trend_path}. Using first row as fallback.")
                    trend_row = dftr.iloc[0].to_dict()

        params = load_model_params(mfile) if mfile is not None else None

        # print("  metrics_csv exists:", (Path(metrics_path).exists() if metrics_path is not None else False), metrics_path)
        # print("  trend_csv exists:  ", (Path(trend_path).exists() if trend_path is not None else False), trend_path)
        # print("  pred_fig exists:   ", (Path(fig).exists() if fig is not None else False), fig)
        # print("  model_file exists: ", (Path(mfile).exists() if mfile is not None else False), mfile)

        md = compose_markdown(model, metrics_df, trend_row, fig, params)
        out_md = REPORTS_DIR / f"{model.replace(' ','_')}.md"
        out_md.write_text(md, encoding="utf8")
        print(" Wrote:", out_md)

        compiled_rows.append({
            "model": model,
            "metrics_csv": str(metrics_path) if metrics_path else "",
            "trend_csv": str(trend_path) if trend_path else "",
            "figure": str(fig) if fig else ""
        })

    # compiled summary
    compiled_md = REPORTS_DIR / "compiled_for_paper.md"
    with open(compiled_md, "w", encoding="utf8") as f:
        f.write("# Compiled model reports\n\n")
        f.write("## Per-model artifacts\n\n")
        for r in compiled_rows:
            f.write(f"- **{r['model']}**\n")
            f.write(f"  - metrics_csv: `{r['metrics_csv']}`\n")
            f.write(f"  - trend_csv: `{r['trend_csv']}`\n")
            f.write(f"  - figure: `{r['figure']}`\n\n")
        f.write("\n## How to use\n1. Open each per-model Markdown file in the `reports/` folder.\n2. Insert the PNG figure referenced if missing.\n3. Copy Markdown into Word/LaTeX and edit as needed.\n")
    print("Compiled summary written to:", compiled_md)
    print("All reports written in:", REPORTS_DIR)

if __name__ == "__main__":
    main()

