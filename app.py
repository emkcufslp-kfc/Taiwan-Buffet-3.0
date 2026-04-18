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

CRITERIA_LABELS = {
    "threshold_roe_min": "ROE 最低門檻（%）",
    "threshold_pe_min": "PE 最低門檻",
    "threshold_pe_max": "PE 最高門檻",
    "threshold_pb_max": "PB 最高門檻",
    "threshold_revenue_yoy_min": "營收年增率最低門檻（%）",
    "threshold_revenue_mom_min": "營收月增率最低門檻（%）",
    "threshold_momentum_6m_min": "六個月動能最低門檻",
    "threshold_drawdown_1y_min": "一年回撤最低門檻",
    "score_roe_cap": "ROE 分數上限",
    "score_pe_cap": "PE 分數上限",
    "score_pe_anchor": "PE 錨點",
    "score_pb_cap": "PB 分數上限",
    "score_pb_anchor": "PB 錨點",
    "score_pb_multiplier": "PB 斜率",
    "score_dividend_yield_cap": "殖利率分數上限",
    "score_revenue_yoy_cap": "營收年增分數上限",
    "score_revenue_yoy_scale": "營收年增縮放",
    "score_revenue_mom_cap": "營收月增分數上限",
    "score_revenue_mom_scale": "營收月增縮放",
    "score_momentum_6m_cap": "六個月動能分數上限",
    "score_momentum_6m_scale": "六個月動能縮放",
    "score_drawdown_1y_cap": "一年回撤分數上限",
    "score_drawdown_1y_anchor": "一年回撤錨點",
    "score_drawdown_1y_scale": "一年回撤縮放",
}

VALIDATION_LABELS = {
    "basic_info_not_empty": "基本資料非空",
    "fundamentals_not_empty": "基本面資料非空",
    "missing_ratio": "缺值比例檢查",
    "roe_range": "ROE 合理範圍檢查",
}

DISPLAY_COLS = {
    "stock_id": "股號",
    "stock_name": "股名",
    "sector": "產業別",
    "report_year": "財報年度",
    "report_quarter": "財報季別",
    "revenue_month": "營收月份",
    "close": "收盤價",
    "pe": "PE",
    "pb": "PB",
    "dividend_yield": "殖利率(%)",
    "roe": "ROE(%)",
    "eps": "EPS",
    "bvps": "每股淨值",
    "revenue_yoy_pct": "營收年增(%)",
    "revenue_mom_pct": "營收月增(%)",
    "momentum_6m": "六個月動能",
    "drawdown_1y": "一年回撤",
    "score": "總分",
    "entry_pass": "進場通過",
    "reason": "判定說明",
    "regime_ok": "大盤條件通過",
    "target_weight": "建議權重",
    "pe_basis": "PE 依據",
    "pb_basis": "PB 依據",
    "revenue_basis": "營收依據",
}

st.set_page_config(page_title="台股巴菲特混合策略儀表板", layout="wide")
st.title("台股巴菲特混合策略儀表板")
st.caption("資料來源：TWSE 即時資料 + 最新已公告財報 + Yahoo 歷史價格")


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


def render_criteria_table(criteria: dict[str, float]) -> None:
    rows = [{"參數": CRITERIA_LABELS.get(k, k), "值": v} for k, v in criteria.items()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=260)


def render_strategy_reference(criteria: dict[str, float]) -> None:
    st.subheader("策略與邏輯參考")
    with st.expander("查看目前策略規則與公式", expanded=True):
        st.markdown(
            f"""
### 進場條件（全部都要通過）
- `ROE >= {criteria['threshold_roe_min']:.2f}`
- `{criteria['threshold_pe_min']:.2f} < PE <= {criteria['threshold_pe_max']:.2f}`
- `PB <= {criteria['threshold_pb_max']:.2f}`
- `營收年增(%) >= {criteria['threshold_revenue_yoy_min']:.2f}`
- `營收月增(%) >= {criteria['threshold_revenue_mom_min']:.2f}`
- `六個月動能 > {criteria['threshold_momentum_6m_min']:.2f}`
- `一年回撤 >= {criteria['threshold_drawdown_1y_min']:.2f}`
- 最後還會套用「大盤風險開關」：加權指數站上 200 日均線才允許進場

### 分數模型（越高越優先）
- `ROE 分數 = clip(ROE, 0, {criteria['score_roe_cap']:.2f})`
- `PE 分數 = clip({criteria['score_pe_anchor']:.2f} - PE, 0, {criteria['score_pe_cap']:.2f})`
- `PB 分數 = clip(({criteria['score_pb_anchor']:.2f} - PB) * {criteria['score_pb_multiplier']:.2f}, 0, {criteria['score_pb_cap']:.2f})`
- `殖利率分數 = clip(殖利率, 0, {criteria['score_dividend_yield_cap']:.2f})`
- `營收年增分數 = clip(營收年增 / {criteria['score_revenue_yoy_scale']:.2f}, 0, {criteria['score_revenue_yoy_cap']:.2f})`
- `營收月增分數 = clip(營收月增 / {criteria['score_revenue_mom_scale']:.2f}, 0, {criteria['score_revenue_mom_cap']:.2f})`
- `動能分數 = clip(六個月動能 * {criteria['score_momentum_6m_scale']:.2f}, 0, {criteria['score_momentum_6m_cap']:.2f})`
- `回撤分數 = clip((一年回撤 + {criteria['score_drawdown_1y_anchor']:.2f}) * {criteria['score_drawdown_1y_scale']:.2f}, 0, {criteria['score_drawdown_1y_cap']:.2f})`
            """
        )


with st.sidebar:
    st.header("掃描設定")
    top_n = st.number_input("掃描檔數", min_value=5, max_value=300, value=40, step=5)
    sector_filter = st.text_input("產業關鍵字", "")
    name_filter = st.text_input("股號或股名關鍵字", "")
    only_pass = st.checkbox("只顯示通過進場", value=False)

    st.subheader("策略參數")
    defaults = get_default_criteria()

    with st.expander("進場門檻", expanded=False):
        threshold_roe_min = st.number_input("ROE 最低（%）", value=float(defaults["threshold_roe_min"]), step=0.5)
        threshold_pe_min = st.number_input("PE 最低", value=float(defaults["threshold_pe_min"]), step=0.5)
        threshold_pe_max = st.number_input("PE 最高", value=float(defaults["threshold_pe_max"]), step=0.5)
        threshold_pb_max = st.number_input("PB 最高", value=float(defaults["threshold_pb_max"]), step=0.1)
        threshold_revenue_yoy_min = st.number_input(
            "營收年增最低（%）", value=float(defaults["threshold_revenue_yoy_min"]), step=1.0
        )
        threshold_revenue_mom_min = st.number_input(
            "營收月增最低（%）", value=float(defaults["threshold_revenue_mom_min"]), step=1.0
        )
        threshold_momentum_6m_min = st.number_input(
            "六個月動能最低", value=float(defaults["threshold_momentum_6m_min"]), step=0.01, format="%.2f"
        )
        threshold_drawdown_1y_min = st.number_input(
            "一年回撤最低", value=float(defaults["threshold_drawdown_1y_min"]), step=0.01, format="%.2f"
        )

    with st.expander("分數權重", expanded=False):
        score_roe_cap = st.number_input("ROE 分數上限", value=float(defaults["score_roe_cap"]), step=1.0)
        score_pe_cap = st.number_input("PE 分數上限", value=float(defaults["score_pe_cap"]), step=1.0)
        score_pe_anchor = st.number_input("PE 錨點", value=float(defaults["score_pe_anchor"]), step=1.0)
        score_pb_cap = st.number_input("PB 分數上限", value=float(defaults["score_pb_cap"]), step=1.0)
        score_pb_anchor = st.number_input("PB 錨點", value=float(defaults["score_pb_anchor"]), step=0.1)
        score_pb_multiplier = st.number_input("PB 斜率", value=float(defaults["score_pb_multiplier"]), step=0.5)
        score_dividend_yield_cap = st.number_input(
            "殖利率分數上限", value=float(defaults["score_dividend_yield_cap"]), step=1.0
        )
        score_revenue_yoy_cap = st.number_input(
            "營收年增分數上限", value=float(defaults["score_revenue_yoy_cap"]), step=1.0
        )
        score_revenue_yoy_scale = st.number_input(
            "營收年增縮放", value=float(defaults["score_revenue_yoy_scale"]), step=0.5
        )
        score_revenue_mom_cap = st.number_input(
            "營收月增分數上限", value=float(defaults["score_revenue_mom_cap"]), step=1.0
        )
        score_revenue_mom_scale = st.number_input(
            "營收月增縮放", value=float(defaults["score_revenue_mom_scale"]), step=0.5
        )
        score_momentum_6m_cap = st.number_input(
            "六個月動能分數上限", value=float(defaults["score_momentum_6m_cap"]), step=1.0
        )
        score_momentum_6m_scale = st.number_input(
            "六個月動能縮放", value=float(defaults["score_momentum_6m_scale"]), step=5.0
        )
        score_drawdown_1y_cap = st.number_input(
            "一年回撤分數上限", value=float(defaults["score_drawdown_1y_cap"]), step=1.0
        )
        score_drawdown_1y_anchor = st.number_input(
            "一年回撤錨點", value=float(defaults["score_drawdown_1y_anchor"]), step=0.05, format="%.2f"
        )
        score_drawdown_1y_scale = st.number_input(
            "一年回撤縮放", value=float(defaults["score_drawdown_1y_scale"]), step=1.0
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

    run = st.button("執行掃描", type="primary")

regime_ok, regime_df = market_regime()

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("大盤狀態", "風險開啟" if regime_ok else "風險關閉")
with c2:
    if not regime_df.empty:
        st.metric("加權指數最新", f"{float(regime_df['Close'].iloc[-1]):,.2f}")
with c3:
    if not regime_df.empty:
        st.metric("加權 200 日均", f"{float(regime_df['MA200'].iloc[-1]):,.2f}")

if not regime_df.empty:
    st.line_chart(regime_df)

basic, fundamentals = load_core_data()

validation_rows = [
    validate_frame_not_empty(basic, "basic_info"),
    validate_frame_not_empty(fundamentals, "fundamentals"),
    validate_missing_ratio(fundamentals),
    validate_roe_range(fundamentals["roe"] if "roe" in fundamentals.columns else pd.Series(dtype="float64")),
]

validation_df = summarize_validation(validation_rows).copy()
if "check" in validation_df.columns:
    validation_df["檢查項目"] = validation_df["check"].map(lambda x: VALIDATION_LABELS.get(str(x), str(x)))
if "ok" in validation_df.columns:
    validation_df["是否通過"] = validation_df["ok"].map({True: "通過", False: "未通過"})
if "detail" in validation_df.columns:
    validation_df = validation_df.rename(columns={"detail": "說明"})
validation_display = validation_df[[c for c in ["檢查項目", "是否通過", "說明"] if c in validation_df.columns]]

st.subheader("資料驗證狀態")
st.dataframe(validation_display, use_container_width=True, height=180)

st.subheader("目前生效參數")
render_criteria_table(criteria)
render_strategy_reference(criteria)

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
        st.warning("目前篩選條件下沒有符合的標的。")
        st.stop()

    rows = []
    prog = st.progress(0)
    txt = st.empty()

    for i, (_, r) in enumerate(work.iterrows(), start=1):
        txt.text(f"正在評估 {r['stock_id']} {r['stock_name']}（{i}/{len(work)}）")
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

    st.subheader("選股監控表")
    df_display = df.rename(columns=DISPLAY_COLS)
    st.dataframe(df_display, use_container_width=True, height=520)

    passed = df[df["entry_pass"]].copy()
    if passed.empty:
        st.info("目前條件與大盤狀態下，沒有標的通過進場。")
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

        st.subheader("建議投組")
        passed_view = passed[
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
        ].rename(columns=DISPLAY_COLS)
        st.dataframe(passed_view, use_container_width=True, height=280)

        st.bar_chart(passed.groupby("sector")["target_weight"].sum())

    st.download_button(
        "下載掃描結果 CSV",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="tw_buffett_hybrid_signals.csv",
        mime="text/csv",
    )

with st.expander("資料與驗證說明"):
    st.write("- 上市公司報價與估值資料來自 TWSE")
    st.write("- EPS / BVPS 來自最新已公告財報，ROE / PE / PB 由此推導")
    st.write("- 營收年增與月增來自最新月營收公告")
    st.write("- 六個月動能與一年回撤來自 Yahoo 歷史價格")
    st.write("- 系統會檢查資料非空、缺值比例與 ROE 合理範圍")
