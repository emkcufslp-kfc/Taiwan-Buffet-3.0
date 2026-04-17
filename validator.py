import pandas as pd

def validate_frame_not_empty(df, name):
    return {
        "check": f"{name}_not_empty",
        "ok": bool(df is not None and not df.empty),
        "detail": f"rows={0 if df is None else len(df)}",
    }

def validate_missing_ratio(df):
    if df is None or df.empty:
        return {"check": "missing_ratio", "ok": False, "detail": "empty_frame"}
    ratio = float(df.isna().sum().sum() / max(df.size, 1))
    return {"check": "missing_ratio", "ok": ratio <= 0.35, "detail": f"{ratio:.2%}"}

def validate_roe_range(series):
    if series is None or len(series) == 0:
        return {"check": "roe_range", "ok": False, "detail": "no_roe_data"}
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"check": "roe_range", "ok": False, "detail": "all_nan"}
    ok = bool(((s >= -100) & (s <= 100)).all())
    return {"check": "roe_range", "ok": ok, "detail": f"min={s.min():.2f}, max={s.max():.2f}"}

def summarize_validation(results):
    out = pd.DataFrame(results)
    if "ok" in out.columns:
        out["ok"] = out["ok"].astype(bool)
    return out
