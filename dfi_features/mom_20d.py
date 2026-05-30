"""Momentum 20 jours: somme des log-rendements journaliers sur 20 jours.

Équivalent à log(close_t / close_{t-20}). Positif = trend haussier récent.
"""
import numpy as np
import pandas as pd


def compute(df: pd.DataFrame, params: dict) -> pd.Series:
    window_d = int(params.get("window_d", 20))
    if "close" not in df.columns:
        return pd.Series(dtype=float)
    close = df["close"].astype(float)
    log_ret = np.log(close / close.shift(1))
    mom = log_ret.rolling(window_d, min_periods=window_d).sum()
    return mom.rename("mom_20d")
