"""Order Book Imbalance within 0.5% of mid-price.

Requires L2 snapshot data (book_snapshot_25). Returns empty for ohlcv_1d.
"""
import pandas as pd


def compute(df: pd.DataFrame, params: dict) -> pd.Series:
    required = {"bid_price_0", "ask_price_0", "bid_size_0", "ask_size_0"}
    if not required.issubset(df.columns):
        return pd.Series(dtype=float)
    pct = float(params.get("depth_pct", 0.5)) / 100.0
    mid = (df["bid_price_0"] + df["ask_price_0"]) / 2
    bid_in = df["bid_size_0"].where(df["bid_price_0"] >= mid * (1 - pct), 0)
    ask_in = df["ask_size_0"].where(df["ask_price_0"] <= mid * (1 + pct), 0)
    total = (bid_in + ask_in).replace(0, float("nan"))
    return ((bid_in - ask_in) / total).rename("obi_0_5pct")
