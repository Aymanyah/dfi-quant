"""Core IC / ICIR / decay / regime computation.

All functions are pure: they take DataFrames and return DataFrames.
No file I/O here — that belongs in runner.py.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


# ── IC helpers ────────────────────────────────────────────────────────────────

def compute_ic_series(signal: pd.Series, fwd_ret: pd.Series) -> float:
    """Spearman IC between signal and one forward-return series."""
    both = pd.concat([signal, fwd_ret], axis=1).dropna()
    if len(both) < 30:
        return float("nan")
    return both.iloc[:, 0].corr(both.iloc[:, 1], method="spearman")


def compute_ic_table(
    signals: pd.DataFrame,      # columns = feature names, index = date × asset (MultiIndex or per-asset)
    rets: pd.DataFrame,         # columns = assets, index = date
    horizons: list[int],        # e.g. [1, 3, 5, 7]
    feature_col: str,
    assets: list[str],
    prices: dict[str, pd.DataFrame],
    params_by_feature: dict,
) -> pd.DataFrame:
    """
    Compute rolling IC for one feature across all assets and horizons.

    Returns DataFrame with columns = horizons, index = date.
    IC at date t = cross-asset Spearman corr between feature(t) and ret(t→t+h).
    """
    rows = []
    all_dates = sorted(set.union(*[set(prices[a].index) for a in assets]))
    all_dates = pd.DatetimeIndex(all_dates)

    for h in horizons:
        ic_series = []
        for t in all_dates:
            scores, fwds = [], []
            for asset in assets:
                df = prices[asset]
                if t not in df.index:
                    continue
                feat_val = signals[asset].get(t, float("nan"))
                if pd.isna(feat_val):
                    continue
                # forward return over h days
                loc = df.index.get_loc(t)
                if loc + h >= len(df):
                    continue
                fwd = float(df["log_ret"].iloc[loc + 1: loc + h + 1].sum())
                scores.append(feat_val)
                fwds.append(fwd)
            if len(scores) < 3:
                ic_series.append((t, float("nan")))
            else:
                s = pd.Series(scores)
                f = pd.Series(fwds)
                ic = s.corr(f, method="spearman")
                ic_series.append((t, ic))

        s_idx, s_vals = zip(*ic_series)
        rows.append(pd.Series(s_vals, index=pd.DatetimeIndex(s_idx), name=h))

    return pd.DataFrame(rows).T


def compute_icir(ic_df: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Rolling ICIR = rolling_mean(IC) / rolling_std(IC)."""
    return ic_df.rolling(window, min_periods=20).mean() / \
           ic_df.rolling(window, min_periods=20).std()


# ── Orthogonalisation ─────────────────────────────────────────────────────────

def orthogonalize(target: pd.Series, controls: pd.DataFrame) -> pd.Series:
    """
    Remove the linear influence of `controls` from `target`.

    Returns the residual series — the part of `target` that is not
    explained by any of the control features.

    Purpose: measure the *unique* predictive power of a feature,
    above and beyond what momentum or volatility already explain.
    """
    both = pd.concat([target, controls], axis=1).dropna()
    if len(both) < 30:
        return target

    y = both.iloc[:, 0]
    X = both.iloc[:, 1:]
    # add constant
    X = X.assign(_const=1.0)
    # OLS via numpy
    coeffs, *_ = np.linalg.lstsq(X.values, y.values, rcond=None)
    fitted = X.values @ coeffs
    residual = y - pd.Series(fitted, index=y.index)
    return residual.reindex(target.index)


# ── Regime split ──────────────────────────────────────────────────────────────

def split_regimes(
    btc_rets: pd.Series,
    window: int = 30,
    threshold_quantile: float = 0.5,
) -> pd.Series:
    """
    Label each date as 'high_vol' or 'low_vol' based on BTC realized volatility.

    Purpose: check if a signal works only in stressed markets or in calm ones.
    A signal that only works in one regime is fragile — it may stop working
    when the regime changes.
    """
    rv = btc_rets.rolling(window, min_periods=window).std() * np.sqrt(365)
    threshold = rv.quantile(threshold_quantile)
    labels = pd.Series(
        np.where(rv >= threshold, "high_vol", "low_vol"),
        index=rv.index,
    )
    return labels.where(rv.notna())


def ic_by_regime(
    ic_series: pd.Series,
    regime_labels: pd.Series,
) -> dict[str, dict]:
    """
    Split IC series by regime and return mean / std / ICIR per regime.
    """
    results = {}
    for regime in ["high_vol", "low_vol"]:
        mask = regime_labels == regime
        sub = ic_series[mask].dropna()
        if len(sub) < 10:
            results[regime] = {"mean_ic": float("nan"), "icir": float("nan"), "n": 0}
        else:
            results[regime] = {
                "mean_ic": round(float(sub.mean()), 4),
                "std_ic":  round(float(sub.std()),  4),
                "icir":    round(float(sub.mean() / sub.std()), 3) if sub.std() > 0 else float("nan"),
                "n":       len(sub),
            }
    return results
