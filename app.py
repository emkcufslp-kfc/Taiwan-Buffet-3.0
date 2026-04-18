from __future__ import annotations

import pandas as pd
import streamlit as st
import yfinance as yf

from tw_data import TwseHybridLoader, compute_snapshot, get_default_criteria
from validator import (
    summarize_validation,
    validate_frame_not_empty,
    validate_missing_ratio,
    validate_roe_range,
)

st.set_page_config(page_title="TW Buffett Hybrid Dashboard", layout="wide")
st.title("TW Buffett Hybrid Dashboard")
st.caption("Live TWSE data + latest published financial reports + Yahoo price history.")

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
    st.subheader("Criteria")
    defaults = get_default_criteria()
    with st.expander("Entry thresholds", expanded=False):
        threshold_roe_min = st.number_input("ROE min (%)", value=float(defaults["threshold_roe_min"]), step=0.5)
        threshold_pe_min = st.number_input("PE min", value=float(defaults["threshold_pe_min"]), step=0.5)
        threshold_pe_max = st.number_input("PE max", value=float(defaults["threshold_pe_max"]), step=0.5)
        threshold_pb_max = st.number_input("PB max", value=float(defaults["threshold_pb_max"]), step=0.1)
        threshold_revenue_yoy_min = st.number_input(
            "Revenue YoY min (%)", value=float(defaults["threshold_revenue_yoy_min"]), step=1.0
        )
        threshold_revenue_mom_min = st.number_input(
            "Revenue MoM min (%)", value=float(defaults["threshold_revenue_mom_min"]), step=1.0
        )
        threshold_momentum_6m_min = st.number_input(
            "6M momentum min", value=float(defaults["threshold_momentum_6m_min"]), step=0.01, format="%.2f"
        )
        threshold_drawdown_1y_min = st.number_input(
            "1Y drawdown min", value=float(defaults["threshold_drawdown_1y_min"]), step=0.01, format="%.2f"
        )

    with st.expander("Scoring weights", expanded=False):
        score_roe_cap = st.number_input("ROE score cap", value=float(defaults["score_roe_cap"]), step=1.0)
        score_pe_cap = st.number_input("PE score cap", value=float(defaults["score_pe_cap"]), step=1.0)
        score_pe_anchor = st.number_input("PE anchor", value=float(defaults["score_pe_anchor"]), step=1.0)
        score_pb_cap = st.number_input("PB score cap", value=float(defaults["score_pb_cap"]), step=1.0)
        score_pb_anchor = st.number_input("PB anchor", value=float(defaults["score_pb_anchor"]), step=0.1)
        score_pb_multiplier = st.number_input(
            "PB multiplier", value=float(defaults["score_pb_multiplier"]), step=0.5
        )
        score_dividend_yield_cap = st.number_input(
            "Dividend yield score cap", value=float(defaults["score_dividend_yield_cap"]), step=1.0
        )
        score_revenue_yoy_cap = st.number_input(
            "Revenue YoY score cap", value=float(defaults["score_revenue_yoy_cap"]), step=1.0
        )
        score_revenue_yoy_scale = st.number_input(
            "Revenue YoY scale", value=float(defaults["score_revenue_yoy_scale"]), step=0.5
        )
        score_revenue_mom_cap = st.number_input(
            "Revenue MoM score cap", value=float(defaults["score_revenue_mom_cap"]), step=1.0
        )
        score_revenue_mom_scale = st.number_input(
            "Revenue MoM scale", value=float(defaults["score_revenue_mom_scale"]), step=0.5
        )
        score_momentum_6m_cap = st.number_input(
            "Momentum score cap", value=float(defaults["score_momentum_6m_cap"]), step=1.0
        )
        score_momentum_6m_scale = st.number_input(
            "Momentum scale", value=float(defaults["score_momentum_6m_scale"]), step=5.0
        )
        score_drawdown_1y_cap = st.number_input(
            "Drawdown score cap", value=float(defaults["score_drawdown_1y_cap"]), step=1.0
        )
        score_drawdown_1y_anchor = st.number_input(
            "Drawdown anchor", value=float(defaults["score_drawdown_1y_anchor"]), step=0.05, format="%.2f"
        )
        score_drawdown_1y_scale = st.number_input(
            "Drawdown scale", value=float(defaults["score_drawdown_1y_scale"]), step=1.0
        )

    criteria = {
        "threshold_roe_min": threshold_roe_min,
        "threshold_pe_min": threshold_pe_min,
        "threshold_pe_max": threshold_pe_max,
        "threshold_pb_max": threshold_pb_max,
        "threshold_revenue_yoy_min": threshold_revenue_yoy_min,
        "threshold_revenue_mom_min": threshold_revenue_mom_min,
        "threshold_momentum_6m_min": threshold_momentum_6m_min,
        "threshold_drawdown_1y_min": threshold_drawdown_1y_min,
        "score_roe_cap": score_roe_cap,
        "score_pe_cap": score_pe_cap,
        "score_pe_anchor": score_pe_anchor,
        "score_pb_cap": score_pb_cap,
        "score_pb_anchor": score_pb_anchor,
        "score_pb_multiplier": score_pb_multiplier,
        "score_dividend_yield_cap": score_dividend_yield_cap,
        "score_revenue_yoy_cap": score_revenue_yoy_cap,
        "score_revenue_yoy_scale": score_revenue_yoy_scale,
        "score_revenue_mom_cap": score_revenue_mom_cap,
        "score_revenue_mom_scale": score_revenue_mom_scale,
        "score_momentum_6m_cap": score_momentum_6m_cap,
        "score_momentum_6m_scale": score_momentum_6m_scale,
        "score_drawdown_1y_cap": score_drawdown_1y_cap,
        "score_drawdown_1y_anchor": score_drawdown_1y_anchor,
        "score_drawdown_1y_scale": score_drawdown_1y_scale,
    }
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
st.subheader("Active criteria")
st.dataframe(
    pd.DataFrame({"criterion": list(criteria.keys()), "value": list(criteria.values())}),
    use_container_width=True,
    height=220,
)

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
    if work.empty:
        st.warning("No stocks match current filters.")
        st.stop()

    rows = []
    prog = st.progress(0)
    txt = st.empty()

    for i, (_, r) in enumerate(work.iterrows(), start=1):
        txt.text(f"Evaluating {r['stock_id']} {r['stock_name']} ({i}/{len(work)})")
        snap = compute_snapshot(r, fundamentals, get_loader(), criteria=criteria)
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
            passed[
                [
                    "stock_id",
                    "stock_name",
                    "sector",
                    "score",
                    "pe_basis",
                    "pb_basis",
                    "revenue_basis",
                    "revenue_yoy_pct",
                    "revenue_mom_pct",
                    "target_weight",
                ]
            ],
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
    st.write("- Live TWSE listed-company quote and valuation datasets")
    st.write("- EPS/BVPS are from latest published financial reports; ROE/PE/PB are computed from those values")
    st.write("- Revenue YoY/MoM come from latest published monthly revenue report and are included in ranking")
    st.write("- Yahoo price history is used for momentum and drawdown calculations")
    st.write("- Validation rejects empty or excessively sparse fundamentals")
    st.write("- ROE sanity range is checked before ranking results")
