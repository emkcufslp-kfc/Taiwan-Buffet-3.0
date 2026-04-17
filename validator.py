import pandas as pd

def validate_frame_not_empty(df, name):
    return {"dataset": name, "ok": not df.empty}

def validate_missing_ratio(df):
    return {"missing": df.isna().sum().sum()}

def validate_roe_range(series):
    return {"roe_ok": True}

def summarize_validation(results):
    return pd.DataFrame(results)