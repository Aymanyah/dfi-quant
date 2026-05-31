"""Distance from 20-day rolling high.

Measures how far the current close is below its 20-day high:
    signal(t) = close(t) / max(close, 20d) - 1

Value is always <= 0. Assets near their high = 0, assets far below = negative.
A strongly negative value may signal oversold conditions (mean-reversion).
"""
from __future__ import annotations
import pandas as pd


def compute(df: pd.DataFrame, params: dict) -> pd.Series:
    if "close" not in df.columns:
        return pd.Series(dtype=float)

    window: int = params.get("window_d", 20)

    close      = df["close"].astype(float)
    rolling_max = close.rolling(window, min_periods=window).max()

    signal = close / rolling_max - 1
    signal.name = "value"
    return signal
