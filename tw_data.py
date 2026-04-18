from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable

import pandas as pd
import requests
import urllib3
import yfinance as yf

# TWSE OpenAPI currently ships with a certificate chain that may fail strict
# verification on some Python/OpenSSL combinations. We keep requests scoped to
# read-only public market data and disable warnings for a stable dashboard UX.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DEFAULT_CRITERIA = {
    "threshold_roe_min": 12.0,
    "threshold_pe_min": 0.0,
    "threshold_pe_max": 20.0,
    "threshold_pb_max": 3.0,
    "threshold_revenue_yoy_min": 0.0,
    "threshold_revenue_mom_min": -20.0,
    "threshold_momentum_6m_min": 0.0,
    "threshold_drawdown_1y_min": -0.30,
    "score_roe_cap": 20.0,
    "score_pe_cap": 20.0,
    "score_pe_anchor": 25.0,
    "score_pb_cap": 15.0,
    "score_pb_anchor": 4.0,
    "score_pb_multiplier": 5.0,
    "score_dividend_yield_cap": 15.0,
    "score_revenue_yoy_cap": 10.0,
    "score_revenue_yoy_scale": 2.0,
    "score_revenue_mom_cap": 5.0,
    "score_revenue_mom_scale": 4.0,
    "score_momentum_6m_cap": 20.0,
    "score_momentum_6m_scale": 100.0,
    "score_drawdown_1y_cap": 10.0,
    "score_drawdown_1y_anchor": 0.30,
    "score_drawdown_1y_scale": 100.0 / 3.0,
}


def get_default_criteria() -> dict[str, float]:
    return dict(DEFAULT_CRITERIA)


def _to_float(raw: object) -> float | None:
    if raw is None:
        return None
    txt = str(raw).strip().replace(",", "")
    if txt in {"", "-", "--", "X", "除權息", "N/A"}:
        return None
    try:
        return float(txt)
    except ValueError:
        return None


def _to_int(raw: object) -> int | None:
    val = _to_float(raw)
    if val is None:
        return None
    return int(val)


@dataclass
class StockSnapshot:
    stock_id: str
    stock_name: str
    sector: str
    report_year: int | None
    report_quarter: int | None
    revenue_month: int | None
    close: float
    pe: float
    pb: float
    pe_basis: str
    pb_basis: str
    revenue_basis: str
    dividend_yield: float
    roe: float
    eps: float
    bvps: float
    revenue_yoy_pct: float
    revenue_mom_pct: float
    momentum_6m: float
    drawdown_1y: float
    score: float
    entry_pass: bool
    reason: str


class TwseHybridLoader:
    MI_INDEX_URL = "https://www.twse.com.tw/rwd/zh/afterTrading/MI_INDEX"
    BWIBBU_URL = "https://www.twse.com.tw/rwd/zh/afterTrading/BWIBBU_d"
    LISTED_COMPANY_URL = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
    INCOME_ENDPOINTS = [
        "t187ap06_L_ci",
        "t187ap06_L_basi",
        "t187ap06_L_bd",
        "t187ap06_L_fh",
        "t187ap06_L_ins",
        "t187ap06_L_mim",
    ]
    BALANCE_ENDPOINTS = [
        "t187ap07_L_ci",
        "t187ap07_L_basi",
        "t187ap07_L_bd",
        "t187ap07_L_fh",
        "t187ap07_L_ins",
        "t187ap07_L_mim",
    ]
    MONTHLY_REVENUE_URL = "https://openapi.twse.com.tw/v1/opendata/t187ap05_L"

    def __init__(self, lookback_days: int = 14) -> None:
        self.lookback_days = lookback_days
        self._latest_trade_date: str | None = None

    def load_basic_info(self) -> pd.DataFrame:
        listed = self._load_listed_companies()
        quotes = self._load_latest_quotes()

        if listed.empty:
            work = quotes.copy()
            work["stock_name"] = work["stock_name_quote"]
            work["sector"] = "Unknown"
            work["listing_years"] = pd.NA
            return work[["stock_id", "stock_name", "sector", "listing_years"]]

        merged = quotes.merge(listed, on="stock_id", how="left")
        merged["stock_name"] = merged["stock_name"].fillna(merged["stock_name_quote"])
        merged["sector"] = merged["sector"].fillna("Unknown")
        return merged[["stock_id", "stock_name", "sector", "listing_years"]]

    def build_fundamentals(self) -> pd.DataFrame:
        quotes = self._load_latest_quotes()[["stock_id", "close"]]
        finance = self._load_latest_financial_report_metrics()
        valuation = self._load_latest_valuation()[["stock_id", "dividend_yield"]]
        monthly_revenue = self._load_latest_monthly_revenue()

        fundamentals = finance.merge(quotes, on="stock_id", how="left")
        fundamentals = fundamentals.merge(valuation, on="stock_id", how="left")
        fundamentals = fundamentals.merge(monthly_revenue, on="stock_id", how="left")

        fundamentals["pe"] = fundamentals.apply(
            lambda r: (r["close"] / r["eps"])
            if pd.notna(r["close"]) and pd.notna(r["eps"]) and r["eps"] > 0
            else pd.NA,
            axis=1,
        )
        fundamentals["pb"] = fundamentals.apply(
            lambda r: (r["close"] / r["bvps"])
            if pd.notna(r["close"]) and pd.notna(r["bvps"]) and r["bvps"] > 0
            else pd.NA,
            axis=1,
        )
        fundamentals["roe"] = fundamentals.apply(
            lambda r: (r["eps"] / r["bvps"] * 100)
            if pd.notna(r["eps"]) and pd.notna(r["bvps"]) and r["bvps"] > 0
            else pd.NA,
            axis=1,
        )
        fundamentals["year"] = fundamentals["report_year"]

        return fundamentals[
            [
                "stock_id",
                "year",
                "report_year",
                "report_quarter",
                "report_date",
                "revenue_month",
                "roe",
                "eps",
                "bvps",
                "pe",
                "pb",
                "dividend_yield",
                "revenue_yoy_pct",
                "revenue_mom_pct",
            ]
        ].copy()

    def get_price_history(self, stock_id: str) -> pd.DataFrame:
        ticker = f"{stock_id}.TW"
        df = yf.download(ticker, period="5y", progress=False, auto_adjust=False)
        if df is None or df.empty:
            return pd.DataFrame(columns=["date", "close"])
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        close_col = "Close"
        if close_col not in df.columns:
            return pd.DataFrame(columns=["date", "close"])
        out = df[[close_col]].reset_index().rename(columns={"Date": "date", close_col: "close"})
        out["close"] = pd.to_numeric(out["close"], errors="coerce")
        out = out.dropna(subset=["close"]).reset_index(drop=True)
        return out

    def _request_json(self, url: str, params: dict[str, object] | None = None) -> object:
        resp = requests.get(url, params=params, timeout=30, verify=False)
        resp.raise_for_status()
        return resp.json()

    def _iter_recent_dates(self) -> Iterable[str]:
        today = date.today()
        for i in range(self.lookback_days):
            d = today - timedelta(days=i)
            yield d.strftime("%Y%m%d")

    def _load_latest_quotes(self) -> pd.DataFrame:
        for d in self._iter_recent_dates():
            payload = self._request_json(
                self.MI_INDEX_URL,
                params={"date": d, "type": "ALLBUT0999", "response": "json"},
            )
            tables = payload.get("tables", []) if isinstance(payload, dict) else []
            table = None
            for candidate in tables:
                fields = candidate.get("fields", [])
                if "證券代號" in fields and "收盤價" in fields:
                    table = candidate
                    break
            if table is None:
                continue

            fields = table.get("fields", [])
            rows = table.get("data", [])
            stock_idx = fields.index("證券代號")
            name_idx = fields.index("證券名稱")
            close_idx = fields.index("收盤價")
            records = []
            for row in rows:
                if len(row) <= max(stock_idx, name_idx, close_idx):
                    continue
                sid = str(row[stock_idx]).strip()
                if not sid.isdigit() or len(sid) != 4:
                    continue
                close = _to_float(row[close_idx])
                if close is None:
                    continue
                records.append(
                    {
                        "stock_id": sid,
                        "stock_name_quote": str(row[name_idx]).strip(),
                        "close": close,
                    }
                )
            if records:
                self._latest_trade_date = d
                return pd.DataFrame(records)
        return pd.DataFrame(columns=["stock_id", "stock_name_quote", "close"])

    def _load_latest_valuation(self) -> pd.DataFrame:
        for d in self._iter_recent_dates():
            payload = self._request_json(
                self.BWIBBU_URL,
                params={"date": d, "selectType": "ALL", "response": "json"},
            )
            if not isinstance(payload, dict):
                continue
            fields = payload.get("fields", [])
            rows = payload.get("data", [])
            if not fields or not rows:
                continue
            if "證券代號" not in fields:
                continue

            stock_idx = fields.index("證券代號")
            pe_idx = fields.index("本益比") if "本益比" in fields else None
            pb_idx = fields.index("股價淨值比") if "股價淨值比" in fields else None
            dy_idx = fields.index("殖利率(%)") if "殖利率(%)" in fields else None

            records = []
            for row in rows:
                if len(row) <= stock_idx:
                    continue
                sid = str(row[stock_idx]).strip()
                if not sid.isdigit() or len(sid) != 4:
                    continue
                records.append(
                    {
                        "stock_id": sid,
                        "pe": _to_float(row[pe_idx]) if pe_idx is not None else None,
                        "pb": _to_float(row[pb_idx]) if pb_idx is not None else None,
                        "dividend_yield": _to_float(row[dy_idx]) if dy_idx is not None else None,
                    }
                )
            if records:
                self._latest_trade_date = d
                return pd.DataFrame(records)

        return pd.DataFrame(columns=["stock_id", "pe", "pb", "dividend_yield"])

    def _load_listed_companies(self) -> pd.DataFrame:
        data = self._request_json(self.LISTED_COMPANY_URL)
        if not isinstance(data, list) or not data:
            return pd.DataFrame(columns=["stock_id", "stock_name", "sector", "listing_years"])

        rows = []
        this_year = date.today().year
        for item in data:
            if not isinstance(item, dict):
                continue
            stock_id = str(item.get("公司代號", "")).strip()
            if not stock_id.isdigit():
                continue

            listed_date = str(item.get("上市日期", "")).strip()
            listed_year = _to_int(listed_date[:4]) if len(listed_date) >= 4 else None
            listing_years = (this_year - listed_year) if listed_year else pd.NA

            rows.append(
                {
                    "stock_id": stock_id,
                    "stock_name": str(item.get("公司簡稱") or item.get("公司名稱") or "").strip(),
                    "sector": str(item.get("產業別") or "Unknown").strip(),
                    "listing_years": listing_years,
                }
            )
        return pd.DataFrame(rows)

    def _load_latest_financial_report_metrics(self) -> pd.DataFrame:
        income = self._load_income_latest()
        balance = self._load_balance_latest()
        merged = income.merge(
            balance,
            on=["stock_id", "report_year", "report_quarter"],
            how="inner",
            suffixes=("_income", "_balance"),
        )
        if merged.empty:
            return pd.DataFrame(
                columns=[
                    "stock_id",
                    "report_year",
                    "report_quarter",
                    "report_date",
                    "eps",
                    "bvps",
                ]
            )

        merged["report_date"] = merged[["report_date_income", "report_date_balance"]].max(axis=1)
        return merged[
            ["stock_id", "report_year", "report_quarter", "report_date", "eps", "bvps"]
        ].copy()

    def _load_latest_monthly_revenue(self) -> pd.DataFrame:
        data = self._request_json(self.MONTHLY_REVENUE_URL)
        if not isinstance(data, list) or not data:
            return pd.DataFrame(
                columns=["stock_id", "revenue_month", "revenue_yoy_pct", "revenue_mom_pct"]
            )

        df = pd.DataFrame(data)
        required = {"公司代號", "資料年月", "營業收入-去年同月增減(%)", "營業收入-上月比較增減(%)"}
        if not required.issubset(set(df.columns)):
            return pd.DataFrame(
                columns=["stock_id", "revenue_month", "revenue_yoy_pct", "revenue_mom_pct"]
            )

        out = pd.DataFrame(
            {
                "stock_id": df["公司代號"].astype(str).str.strip(),
                "revenue_month": pd.to_numeric(df["資料年月"], errors="coerce"),
                "revenue_yoy_pct": pd.to_numeric(df["營業收入-去年同月增減(%)"], errors="coerce"),
                "revenue_mom_pct": pd.to_numeric(df["營業收入-上月比較增減(%)"], errors="coerce"),
            }
        )
        out = out[out["stock_id"].str.fullmatch(r"\d{4}", na=False)].copy()
        out = out.dropna(subset=["revenue_month"]).copy()
        out = out.sort_values(["stock_id", "revenue_month"], ascending=[True, False])
        out = out.groupby("stock_id", as_index=False).head(1).copy()
        return out

    def _load_income_latest(self) -> pd.DataFrame:
        frames = []
        for endpoint in self.INCOME_ENDPOINTS:
            data = self._request_json(f"https://openapi.twse.com.tw/v1/opendata/{endpoint}")
            if not isinstance(data, list) or not data:
                continue
            df = pd.DataFrame(data)
            if "公司代號" not in df.columns:
                continue
            eps_col = "基本每股盈餘（元）" if "基本每股盈餘（元）" in df.columns else None
            if eps_col is None:
                continue
            sub = pd.DataFrame(
                {
                    "stock_id": df["公司代號"].astype(str).str.strip(),
                    "report_year": pd.to_numeric(df.get("年度"), errors="coerce"),
                    "report_quarter": pd.to_numeric(df.get("季別"), errors="coerce"),
                    "report_date": pd.to_numeric(df.get("出表日期"), errors="coerce"),
                    "eps": pd.to_numeric(df[eps_col], errors="coerce"),
                }
            )
            sub = sub[sub["stock_id"].str.fullmatch(r"\d{4}", na=False)].copy()
            frames.append(sub)
        if not frames:
            return pd.DataFrame(
                columns=["stock_id", "report_year", "report_quarter", "report_date", "eps"]
            )
        all_income = pd.concat(frames, ignore_index=True)
        all_income = all_income.dropna(subset=["report_year", "report_quarter"]).copy()
        all_income["rk"] = all_income["report_year"] * 10 + all_income["report_quarter"]
        all_income = all_income.sort_values(["stock_id", "rk", "report_date"], ascending=[True, False, False])
        latest = all_income.groupby("stock_id", as_index=False).head(1).copy()
        return latest.drop(columns=["rk"])

    def _load_balance_latest(self) -> pd.DataFrame:
        frames = []
        for endpoint in self.BALANCE_ENDPOINTS:
            data = self._request_json(f"https://openapi.twse.com.tw/v1/opendata/{endpoint}")
            if not isinstance(data, list) or not data:
                continue
            df = pd.DataFrame(data)
            if "公司代號" not in df.columns or "每股參考淨值" not in df.columns:
                continue
            sub = pd.DataFrame(
                {
                    "stock_id": df["公司代號"].astype(str).str.strip(),
                    "report_year": pd.to_numeric(df.get("年度"), errors="coerce"),
                    "report_quarter": pd.to_numeric(df.get("季別"), errors="coerce"),
                    "report_date": pd.to_numeric(df.get("出表日期"), errors="coerce"),
                    "bvps": pd.to_numeric(df["每股參考淨值"], errors="coerce"),
                }
            )
            sub = sub[sub["stock_id"].str.fullmatch(r"\d{4}", na=False)].copy()
            frames.append(sub)
        if not frames:
            return pd.DataFrame(
                columns=["stock_id", "report_year", "report_quarter", "report_date", "bvps"]
            )
        all_balance = pd.concat(frames, ignore_index=True)
        all_balance = all_balance.dropna(subset=["report_year", "report_quarter"]).copy()
        all_balance["rk"] = all_balance["report_year"] * 10 + all_balance["report_quarter"]
        all_balance = all_balance.sort_values(
            ["stock_id", "rk", "report_date"], ascending=[True, False, False]
        )
        latest = all_balance.groupby("stock_id", as_index=False).head(1).copy()
        return latest.drop(columns=["rk"])


def compute_snapshot(
    stock_row: pd.Series,
    fundamentals: pd.DataFrame,
    loader: TwseHybridLoader,
    criteria: dict[str, float] | None = None,
) -> StockSnapshot:
    c = get_default_criteria()
    if criteria:
        c.update(criteria)

    stock_id = str(stock_row.get("stock_id", "")).strip()
    stock_name = str(stock_row.get("stock_name", "")).strip()
    sector = str(stock_row.get("sector", "Unknown")).strip() or "Unknown"

    funda = fundamentals[fundamentals["stock_id"] == stock_id].copy()
    if not funda.empty and "year" in funda.columns:
        funda = funda.sort_values("year", ascending=False)
    f = funda.iloc[0] if not funda.empty else pd.Series(dtype="object")

    pe = float(f.get("pe")) if pd.notna(f.get("pe", pd.NA)) else float("nan")
    pb = float(f.get("pb")) if pd.notna(f.get("pb", pd.NA)) else float("nan")
    dy = float(f.get("dividend_yield")) if pd.notna(f.get("dividend_yield", pd.NA)) else 0.0
    roe = float(f.get("roe")) if pd.notna(f.get("roe", pd.NA)) else float("nan")
    eps = float(f.get("eps")) if pd.notna(f.get("eps", pd.NA)) else float("nan")
    bvps = float(f.get("bvps")) if pd.notna(f.get("bvps", pd.NA)) else float("nan")
    report_year = int(f.get("report_year")) if pd.notna(f.get("report_year", pd.NA)) else None
    report_quarter = int(f.get("report_quarter")) if pd.notna(f.get("report_quarter", pd.NA)) else None
    revenue_month = int(f.get("revenue_month")) if pd.notna(f.get("revenue_month", pd.NA)) else None
    revenue_yoy_pct = (
        float(f.get("revenue_yoy_pct")) if pd.notna(f.get("revenue_yoy_pct", pd.NA)) else float("nan")
    )
    revenue_mom_pct = (
        float(f.get("revenue_mom_pct")) if pd.notna(f.get("revenue_mom_pct", pd.NA)) else float("nan")
    )

    px = loader.get_price_history(stock_id)
    if px.empty:
        close = float("nan")
        momentum_6m = float("nan")
        drawdown_1y = float("nan")
    else:
        close = float(px["close"].iloc[-1])
        momentum_6m = float(close / px["close"].iloc[-126] - 1.0) if len(px) > 126 else float("nan")
        max_1y = float(px["close"].tail(252).max()) if len(px) >= 20 else close
        drawdown_1y = float(close / max_1y - 1.0) if max_1y > 0 else float("nan")

    score = 0.0
    reasons = []

    if pd.notna(roe):
        score += max(0.0, min(c["score_roe_cap"], roe))
    else:
        reasons.append("missing_roe")
    if pd.notna(pe) and pe > 0:
        score += max(0.0, min(c["score_pe_cap"], (c["score_pe_anchor"] - pe)))
    else:
        reasons.append("invalid_pe")
    if pd.notna(pb) and pb > 0:
        score += max(0.0, min(c["score_pb_cap"], (c["score_pb_anchor"] - pb) * c["score_pb_multiplier"]))
    else:
        reasons.append("invalid_pb")
    score += max(0.0, min(c["score_dividend_yield_cap"], dy))

    if pd.notna(revenue_yoy_pct):
        score += max(0.0, min(c["score_revenue_yoy_cap"], revenue_yoy_pct / c["score_revenue_yoy_scale"]))
    else:
        reasons.append("missing_revenue_yoy")
    if pd.notna(revenue_mom_pct):
        score += max(0.0, min(c["score_revenue_mom_cap"], revenue_mom_pct / c["score_revenue_mom_scale"]))
    else:
        reasons.append("missing_revenue_mom")

    if pd.notna(momentum_6m):
        score += max(0.0, min(c["score_momentum_6m_cap"], momentum_6m * c["score_momentum_6m_scale"]))
    else:
        reasons.append("missing_momentum")
    if pd.notna(drawdown_1y):
        score += max(
            0.0,
            min(
                c["score_drawdown_1y_cap"],
                (drawdown_1y + c["score_drawdown_1y_anchor"]) * c["score_drawdown_1y_scale"],
            ),
        )
    else:
        reasons.append("missing_drawdown")

    rule_failures = []
    if not (pd.notna(roe) and roe >= c["threshold_roe_min"]):
        rule_failures.append("rule_roe")
    if not (pd.notna(pe) and c["threshold_pe_min"] < pe <= c["threshold_pe_max"]):
        rule_failures.append("rule_pe")
    if not (pd.notna(pb) and pb <= c["threshold_pb_max"]):
        rule_failures.append("rule_pb")
    if not (pd.notna(revenue_yoy_pct) and revenue_yoy_pct >= c["threshold_revenue_yoy_min"]):
        rule_failures.append("rule_revenue_yoy")
    if not (pd.notna(revenue_mom_pct) and revenue_mom_pct >= c["threshold_revenue_mom_min"]):
        rule_failures.append("rule_revenue_mom")
    if not (pd.notna(momentum_6m) and momentum_6m > c["threshold_momentum_6m_min"]):
        rule_failures.append("rule_momentum_6m")
    if not (pd.notna(drawdown_1y) and drawdown_1y >= c["threshold_drawdown_1y_min"]):
        rule_failures.append("rule_drawdown_1y")

    entry_pass = bool(not rule_failures)

    reason = "ok" if entry_pass else ",".join(reasons + rule_failures) or "rule_not_met"
    return StockSnapshot(
        stock_id=stock_id,
        stock_name=stock_name,
        sector=sector,
        report_year=report_year,
        report_quarter=report_quarter,
        revenue_month=revenue_month,
        close=close,
        pe=pe,
        pb=pb,
        pe_basis="latest_published_report_eps",
        pb_basis="latest_published_report_bvps",
        revenue_basis="latest_monthly_revenue_report",
        dividend_yield=dy,
        roe=roe,
        eps=eps,
        bvps=bvps,
        revenue_yoy_pct=revenue_yoy_pct,
        revenue_mom_pct=revenue_mom_pct,
        momentum_6m=momentum_6m,
        drawdown_1y=drawdown_1y,
        score=round(float(score), 2),
        entry_pass=entry_pass,
        reason=reason,
    )
