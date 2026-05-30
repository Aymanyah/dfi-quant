"""L/S backtest engine for the research synthesis agent.

Takes a signal Series and price DataFrame, runs a long/short strategy,
and returns performance metrics + saves a plot bundle.

All functions are pure except save_backtest_plots() which writes to disk.
"""
from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ── Core backtest ─────────────────────────────────────────────────────────────

def run_backtest(
    signal: pd.Series,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Simulate a daily L/S strategy on one asset.

    signal : z-score or any continuous signal, index = DatetimeIndex
    prices : DataFrame with column 'close', same index

    Returns a DataFrame with columns:
        signal, ret_1d, ret_fwd, strat_ret, portf_value, bh_value
    """
    close = prices["close"].astype(float)
    ret_1d = close / close.shift(1) - 1

    df = pd.DataFrame(index=prices.index)
    df["signal"]      = signal.reindex(prices.index)
    df["ret_1d"]      = ret_1d
    df["position"]    = np.where(df["signal"] > 0, 1.0, -1.0)
    df["ret_fwd"]     = df["ret_1d"].shift(-1)
    df["strat_ret"]   = df["position"] * df["ret_fwd"]
    df["portf_value"] = (1 + df["strat_ret"].fillna(0)).cumprod()
    df["bh_value"]    = (1 + df["ret_1d"].fillna(0)).cumprod()
    return df


def compute_metrics(bt: pd.DataFrame) -> dict:
    """Compute annualised Sharpe, max drawdown, win rate, total return."""
    sr = bt["strat_ret"].dropna()
    if sr.empty or sr.std() == 0:
        return {"sharpe": float("nan"), "max_dd": float("nan"),
                "win_rate": float("nan"), "total_ret": float("nan"), "n_days": 0}

    sharpe     = (sr.mean() / sr.std()) * np.sqrt(252)
    rolling_max = bt["portf_value"].cummax()
    max_dd      = ((bt["portf_value"] - rolling_max) / rolling_max).min()
    win_rate    = (sr > 0).sum() / len(sr)
    total_ret   = bt["portf_value"].iloc[-1] - 1

    return {
        "sharpe":    round(float(sharpe),    3),
        "max_dd":    round(float(max_dd),    3),
        "win_rate":  round(float(win_rate),  3),
        "total_ret": round(float(total_ret), 3),
        "n_days":    len(sr),
    }


def run_cross_asset_backtest(
    signals: dict[str, pd.Series],
    prices:  dict[str, pd.DataFrame],
) -> dict[str, dict]:
    """Run backtest for each asset and return metrics dict."""
    results = {}
    for asset, sig in signals.items():
        if asset not in prices or prices[asset].empty:
            continue
        bt = run_backtest(sig, prices[asset])
        results[asset] = {
            "bt":      bt,
            "metrics": compute_metrics(bt),
        }
    return results


def aggregate_metrics(results: dict[str, dict]) -> pd.DataFrame:
    """Build a summary DataFrame from cross-asset backtest results."""
    rows = []
    for asset, r in results.items():
        m = r["metrics"]
        rows.append({"asset": asset, **m})
    return pd.DataFrame(rows).set_index("asset")


# ── Plots ─────────────────────────────────────────────────────────────────────

def save_backtest_plots(
    feature_id: str,
    results: dict[str, dict],
    metrics_df: pd.DataFrame,
    out_dir: pathlib.Path,
) -> pathlib.Path:
    """
    Four-panel plot per feature:
    (0,0) Cumulative returns — strategy vs buy & hold, all assets
    (0,1) Sharpe bar chart per asset
    (1,0) Rolling Sharpe 63d — BTCUSDT
    (1,1) Strategy returns distribution — BTCUSDT
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"L/S Backtest — {feature_id}", fontsize=13, fontweight="bold")

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    # ── (0,0) Cumulative returns ──────────────────────────────────────────────
    ax = axes[0, 0]
    for (asset, r), col in zip(results.items(), colors):
        bt = r["bt"][["portf_value", "bh_value"]].dropna()
        ax.plot(bt.index, bt["portf_value"], lw=1.4, color=col, label=asset)
    ax.axhline(1, color="k", lw=0.6, ls="--")
    ax.set_title("Valeur cumulée — stratégie L/S")
    ax.set_ylabel("Valeur (base 1)")
    ax.legend(fontsize=7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    # ── (0,1) Sharpe par asset ────────────────────────────────────────────────
    ax = axes[0, 1]
    sharpes = metrics_df["sharpe"].fillna(0)
    bar_colors = ["tab:green" if s >= 0 else "tab:red" for s in sharpes]
    ax.bar(sharpes.index, sharpes.values, color=bar_colors, alpha=0.8)
    ax.axhline(0, color="k", lw=0.6)
    ax.axhline(1, color="tab:green", lw=0.8, ls="--", alpha=0.5, label="Sharpe=1")
    ax.set_title("Sharpe annualisé par asset")
    ax.set_ylabel("Sharpe")
    ax.legend(fontsize=7)
    ax.tick_params(axis="x", labelsize=8)

    # ── (1,0) Rolling Sharpe 63d — BTC ───────────────────────────────────────
    ax = axes[1, 0]
    anchor = "BTCUSDT" if "BTCUSDT" in results else next(iter(results))
    sr = results[anchor]["bt"]["strat_ret"].dropna()
    roll = sr.rolling(63, min_periods=20).apply(
        lambda x: (x.mean() / x.std()) * np.sqrt(252) if x.std() > 0 else np.nan
    )
    ax.plot(roll.index, roll.values, lw=1.2, color="steelblue")
    ax.fill_between(roll.index, 0, roll.values,
                    where=roll.values >= 0, color="tab:green", alpha=0.15)
    ax.fill_between(roll.index, 0, roll.values,
                    where=roll.values < 0,  color="tab:red",   alpha=0.15)
    ax.axhline(0, color="k", lw=0.6)
    ax.axhline(1, color="tab:green", lw=0.8, ls="--", alpha=0.5)
    ax.set_title(f"Rolling Sharpe 63d — {anchor}")
    ax.set_ylabel("Sharpe")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    # ── (1,1) Distribution des rendements stratégie — BTC ────────────────────
    ax = axes[1, 1]
    sr_clean = results[anchor]["bt"]["strat_ret"].dropna()
    ax.hist(sr_clean.values * 100, bins=50,
            color="steelblue", alpha=0.7, edgecolor="white", lw=0.3)
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline(sr_clean.mean() * 100, color="tab:orange", lw=1.5, ls="--",
               label=f"mean={sr_clean.mean()*100:.3f}%")
    ax.set_title(f"Distribution des rendements stratégie — {anchor}")
    ax.set_xlabel("Rendement journalier (%)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"backtest_{feature_id}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path
