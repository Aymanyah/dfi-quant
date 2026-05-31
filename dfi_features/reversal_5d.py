"""5-day short-term reversal.

Short-term mean-reversion signal: assets that have fallen the most over
the past 5 days tend to rebound, and vice versa. The signal is the
negative of the 5-day simple return.

Positive value = asset has sold off recently (expected to rebound).
Negative value = asset has rallied recently (expected to correct).
"""
from __future__ import annotations
import pandas as pd


def compute(df: pd.DataFrame, params: dict) -> pd.Series:
    if "close" not in df.columns:
        return pd.Series(dtype=float)

    window: int = params.get("window_d", 5)

    close  = df["close"].astype(float)
    ret_nd = close / close.shift(window) - 1
    signal = -ret_nd
    signal.name = "value"
    return signal
