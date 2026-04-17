import pandas as pd
import yfinance as yf

class TwseHybridLoader:

    def load_basic_info(self):
        return pd.DataFrame({
            "stock_id": ["2330", "2317", "2882"],
            "stock_name": ["TSMC", "Hon Hai", "Cathay"],
            "sector": ["半導體", "電子", "金融"],
            "listing_years": [20, 20, 20]
        })

    def build_fundamentals(self):
        data = []
        for stock in ["2330", "2317", "2882"]:
            for year in range(2015, 2025):
                data.append({
                    "stock_id": stock,
                    "year": year,
                    "roe": 15 + (year % 5),
                    "eps": 5 + (year % 3),
                    "bvps": 20 + (year % 4)
                })
        return pd.DataFrame(data)

    def get_price_history(self, stock_id):
        df = yf.download(stock_id + ".TW", start="2020-01-01", progress=False)
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        return df.rename(columns={"Close": "close", "Date": "date"})