from __future__ import annotations

import pandas as pd
import streamlit as st
import yfinance as yf

from tw_data import TwseHybridLoader, compute_snapshot
from validator import (
    summarize_validation,
    validate_frame_not_empty,
    validate_missing_ratio,
    validate_roe_range,
)

st.set_page_config(page_title="TW Buffett Hybrid Dashboard", layout="wide")
st.title("TW Buffett Hybrid Dashboard")
st.caption("Official TWSE/MOPS fundamentals + Yahoo price history, with explicit validation gates.")

@st.cache_resource(show_spinner=False)
def get_loader() -> TwseHybridLoader:
    return TwseHybridLoader()

@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def load_core_data():
    loader = get_loader()
    basic = loader.load_basic_info()
    fundamentals = loader.build_fundamentals()
    return basic, fundamentals

@st.cache_data(ttl=60 * 60, show_spinner=False)
def market_regime():
    df = yf.download("^TWII", start="2006-01-01", progress=False, auto_adjust=False)
    if df is None or df.empty:
        return False, pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    close_col = "Close"
    df = df.copy()
    df["MA200"] = df[close_col].rolling(200).mean()
    ok = bool(df[close_col].iloc[-1] > df["MA200"].iloc[-1]) if len(df) >= 200 else False
    return ok, df[[close_col, "MA200"]].rename(columns={close_col: "Close"}).tail(300)

with st.sidebar:
    st.header("Controls")
    top_n = st.number_input("Stocks to evaluate", min_value=5, max_value=300, value=40, step=5)
    sector_filter = st.text_input("Sector contains", "")
    name_filter = st.text_input("Stock ID or name contains", "")
    only_pass = st.checkbox("Only show entry-pass", value=False)
    run = st.button("Run scan", type="primary")

regime_ok, regime_df = market_regime()

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Market regime", "Risk-on" if regime_ok else "Risk-off")
with c2:
    if not regime_df.empty:
        st.metric("TWII latest", f"{float(regime_df['Close'].iloc[-1]):,.2f}")
with c3:
    if not regime_df.empty:
        st.metric("TWII 200DMA", f"{float(regime_df['MA200'].iloc[-1]):,.2f}")

if not regime_df.empty:
    st.line_chart(regime_df)

basic, fundamentals = load_core_data()

validation_rows = [
    validate_frame_not_empty(basic, "basic_info"),
    validate_frame_not_empty(fundamentals, "fundamentals"),
    validate_missing_ratio(fundamentals),
    validate_roe_range(fundamentals["roe"] if "roe" in fundamentals.columns else pd.Series(dtype="float64")),
]

st.subheader("Validation status")
st.dataframe(summarize_validation(validation_rows), use_container_width=True, height=180)

if run:
    work = basic.copy()

    if sector_filter.strip():
        work = work[work["sector"].astype(str).str.contains(sector_filter.strip(), case=False, na=False)]
    if name_filter.strip():
        q = name_filter.strip()
        work = work[
            work["stock_id"].astype(str).str.contains(q, case=False, na=False)
            | work["stock_name"].astype(str).str.contains(q, case=False, na=False)
        ]

    work = work.head(int(top_n)).copy()

    rows = []
    prog = st.progress(0)
    txt = st.empty()

    for i, (_, r) in enumerate(work.iterrows(), start=1):
        txt.text(f"Evaluating {r['stock_id']} {r['stock_name']} ({i}/{len(work)})")
        snap = compute_snapshot(r, fundamentals, get_loader())
        row = snap.__dict__
        row["regime_ok"] = regime_ok
        row["entry_pass"] = bool(row["entry_pass"] and regime_ok)
        rows.append(row)
        prog.progress(i / max(len(work), 1))

    txt.empty()
    prog.empty()

    df = pd.DataFrame(rows).sort_values(["entry_pass", "score", "stock_id"], ascending=[False, False, True])

    if only_pass:
        df = df[df["entry_pass"]].copy()

    st.subheader("Signal monitor")
    st.dataframe(df, use_container_width=True, height=520)

    passed = df[df["entry_pass"]].copy()
    if passed.empty:
        st.info("No stocks pass under the current filters and market regime.")
    else:
        passed["target_weight"] = 0.0
        sector_weights = {}
        for idx, row in passed.iterrows():
            sector = row["sector"]
            sector_weights.setdefault(sector, 0.0)
            remaining = 0.40 - sector_weights[sector]
            if remaining <= 0:
                continue
            w = min(0.10, remaining)
            passed.at[idx, "target_weight"] = w
            sector_weights[sector] += w

        passed = passed[passed["target_weight"] > 0].copy()
        if not passed.empty and passed["target_weight"].sum() > 1.0:
            passed["target_weight"] = passed["target_weight"] / passed["target_weight"].sum()

        st.subheader("Suggested portfolio")
        st.dataframe(
            passed[["stock_id", "stock_name", "sector", "score", "target_weight"]],
            use_container_width=True,
            height=280,
        )

        st.bar_chart(passed.groupby("sector")["target_weight"].sum())

    st.download_button(
        "Download signals CSV",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="tw_buffett_hybrid_signals.csv",
        mime="text/csv",
    )

with st.expander("What is validated here"):
    st.write("- Official listed-company basic information from TWSE/MOPS bulk CSV")
    st.write("- Official quarterly income statement and balance sheet bulk CSVs")
    st.write("- Yahoo price history is used for long-horizon backtest price series")
    st.write("- Validation rejects empty or excessively sparse fundamentals")
    st.write("- ROE sanity range is checked before ranking results")
    st.write("- Cross-source price-gap validation hook is built in, but needs an official same-day close input source")
