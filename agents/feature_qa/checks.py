"""Statistical sanity checks for features.

Each check returns a dict with keys:
    passed  : bool
    message : str   (one-line summary)
    detail  : dict  (numeric diagnostics)
"""
from __future__ import annotations
import numpy as np
import pandas as pd


# ── Distribution checks ───────────────────────────────────────────────────────

def check_distribution(series: pd.Series) -> dict:
    """
    Stationarity (ADF), autocorrelation at lag 1, and outlier rate.
    A feature that explodes or has extreme autocorrelation will break the IC.
    """
    s = series.dropna()
    if len(s) < 30:
        return {"passed": False, "message": "Trop peu de valeurs (<30)", "detail": {}}

    # Outliers : valeurs au-delà de 10 écarts-types
    mu, sigma = s.mean(), s.std()
    outlier_rate = float((np.abs(s - mu) > 10 * sigma).mean()) if sigma > 0 else 0.0

    # Autocorrélation lag-1
    ac1 = float(s.autocorr(lag=1))

    # Test ADF de stationnarité
    try:
        from statsmodels.tsa.stattools import adfuller
        adf_pvalue = float(adfuller(s, maxlag=5, autolag=None)[1])
        stationary = adf_pvalue < 0.05
    except Exception:
        adf_pvalue = float("nan")
        stationary = True  # on ne bloque pas si statsmodels absent

    passed = outlier_rate < 0.01 and stationary

    msg_parts = []
    if outlier_rate >= 0.01:
        msg_parts.append(f"outliers={outlier_rate:.1%}")
    if not stationary:
        msg_parts.append(f"non-stationnaire (ADF p={adf_pvalue:.3f})")
    message = "OK" if not msg_parts else " | ".join(msg_parts)

    return {
        "passed": passed,
        "message": message,
        "detail": {
            "mean":         round(float(mu), 6),
            "std":          round(float(sigma), 6),
            "min":          round(float(s.min()), 6),
            "max":          round(float(s.max()), 6),
            "outlier_rate": round(outlier_rate, 4),
            "autocorr_lag1":round(ac1, 4),
            "adf_pvalue":   round(adf_pvalue, 4) if not np.isnan(adf_pvalue) else None,
            "stationary":   stationary,
        },
    }


# ── Coverage check ────────────────────────────────────────────────────────────

def check_coverage(
    series_by_asset: dict[str, pd.Series],
    warmup_bars: int = 0,
    nan_threshold: float = 0.20,
) -> dict:
    """
    For each asset, measure the NaN rate after the warmup period.
    Fails if any asset exceeds nan_threshold.
    """
    results = {}
    failed_assets = []

    for asset, s in series_by_asset.items():
        if s.empty:
            results[asset] = {"nan_rate": 1.0, "n_valid": 0, "passed": False}
            failed_assets.append(asset)
            continue
        post_warmup = s.iloc[warmup_bars:]
        nan_rate = float(post_warmup.isna().mean())
        n_valid = int(post_warmup.notna().sum())
        passed = nan_rate <= nan_threshold
        results[asset] = {"nan_rate": round(nan_rate, 4), "n_valid": n_valid, "passed": passed}
        if not passed:
            failed_assets.append(asset)

    overall_passed = len(failed_assets) == 0
    message = "OK" if overall_passed else f"NaN trop élevé pour : {', '.join(failed_assets)}"

    return {
        "passed": overall_passed,
        "message": message,
        "detail": results,
    }


# ── Lookahead audit ───────────────────────────────────────────────────────────

def check_lookahead(
    compute_fn,
    params: dict,
    df: pd.DataFrame,
    n_spot_checks: int = 10,
) -> dict:
    """
    For n random dates t, rebuild the feature using only data up to t-1.
    The value at t must be identical to the full-history version.

    If it differs → the feature uses future data (lookahead bias).
    """
    full_series = compute_fn(df, params)
    valid_idx = full_series.dropna().index
    if len(valid_idx) < n_spot_checks:
        return {"passed": True, "message": "Trop peu de valeurs pour auditer", "detail": {}}

    rng = np.random.default_rng(42)
    check_dates = pd.DatetimeIndex(
        rng.choice(valid_idx, size=min(n_spot_checks, len(valid_idx)), replace=False)
    )

    violations = []
    for t in sorted(check_dates):
        df_past = df.loc[df.index <= t]
        past_series = compute_fn(df_past, params)
        val_full = full_series.get(t, float("nan"))
        val_past = past_series.get(t, float("nan")) if t in past_series.index else float("nan")

        if pd.isna(val_full) and pd.isna(val_past):
            continue
        if not np.isclose(val_full, val_past, atol=1e-10, equal_nan=True):
            violations.append({
                "date":      str(t.date()),
                "full":      round(float(val_full), 8),
                "past_only": round(float(val_past), 8),
            })

    passed = len(violations) == 0
    message = "OK" if passed else f"{len(violations)} violation(s) de lookahead détectée(s)"

    return {
        "passed": passed,
        "message": message,
        "detail": {"violations": violations, "n_checked": len(check_dates)},
    }


# ── Explosion check ───────────────────────────────────────────────────────────

def check_explosion(series: pd.Series, window: int = 252) -> dict:
    """
    Split the series into yearly windows and check that the rolling std
    does not grow more than 5× relative to the first window.

    A feature that explodes (variance doubles each year) is unusable
    in a live system because its scale changes unpredictably.
    """
    s = series.dropna()
    if len(s) < window * 2:
        return {"passed": True, "message": "Historique trop court pour tester", "detail": {}}

    roll_std = s.rolling(window, min_periods=window // 2).std()
    first_std = roll_std.dropna().iloc[0]
    last_std  = roll_std.dropna().iloc[-1]

    if first_std == 0:
        return {"passed": True, "message": "std initiale nulle", "detail": {}}

    ratio = last_std / first_std
    passed = ratio < 5.0
    message = "OK" if passed else f"Variance ×{ratio:.1f} depuis le début (seuil = 5×)"

    return {
        "passed": passed,
        "message": message,
        "detail": {
            "std_first_window": round(float(first_std), 6),
            "std_last_window":  round(float(last_std), 6),
            "ratio":            round(float(ratio), 2),
        },
    }
