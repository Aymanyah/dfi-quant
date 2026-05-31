"""Z-scored taker buy volume ratio.

Computes the ratio of taker-initiated buy volume to total volume, then
z-scores it over a 20-day rolling window. A high value means unusual
buying pressure relative to recent history — a bullish microstructure signal.

Returns empty series if taker_buy_vol or volume columns are absent.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def compute(df: pd.DataFrame, params: dict) -> pd.Series:
    if "taker_buy_vol" not in df.columns or "volume" not in df.columns:
        return pd.Series(dtype=float)

    window: int = params.get("window_d", 20)

    vol    = df["volume"].astype(float).replace(0.0, np.nan)
    taker  = df["taker_buy_vol"].astype(float)
    ratio  = taker / vol

    mu    = ratio.rolling(window, min_periods=window).mean()
    sigma = ratio.rolling(window, min_periods=window).std()

    zscore = (ratio - mu) / sigma.replace(0.0, np.nan)
    zscore.name = "value"
    return zscore
