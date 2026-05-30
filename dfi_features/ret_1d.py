"""Log-rendement journalier: log(close_t / close_{t-1}).

Brique de base pour toutes les features construites sur les rendements.
Premier bar toujours NaN (pas de close précédent).
"""
import numpy as np
import pandas as pd


def compute(df: pd.DataFrame, params: dict) -> pd.Series:
    if "close" not in df.columns:
        return pd.Series(dtype=float)
    close = df["close"].astype(float)
    return np.log(close / close.shift(1)).rename("ret_1d")
