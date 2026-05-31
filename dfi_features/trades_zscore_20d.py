"""Z-scored number of trades over 20 days.

Normalises the daily trade count by its 20-day rolling mean and std.
High values indicate unusual market participation — more traders
active than usual, which can signal increased conviction or stress.

Returns empty series if n_trades column is absent.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def compute(df: pd.DataFrame, params: dict) -> pd.Series:
    if "n_trades" not in df.columns:
        return pd.Series(dtype=float)

    window: int = params.get("window_d", 20)

    trades = df["n_trades"].astype(float)
    mu     = trades.rolling(window, min_periods=window).mean()
    sigma  = trades.rolling(window, min_periods=window).std()

    zscore = (trades - mu) / sigma.replace(0.0, np.nan)
    zscore.name = "value"
    return zscore
