"""Generate JSON summary and matplotlib plots for one feature's IC analysis."""
from __future__ import annotations
import json
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def _fmt(x) -> float | str:
    if isinstance(x, float) and np.isnan(x):
        return "nan"
    if isinstance(x, float):
        return round(x, 4)
    return x


def save_json(
    feature_id: str,
    assets: list[str],
    horizons: list[int],
    ic_by_horizon: dict[int, float],
    icir_by_horizon: dict[int, float],
    ic_by_horizon_ortho: dict[int, float],
    icir_by_horizon_ortho: dict[int, float],
    ic_by_horizon_regime: dict,
    out_dir: pathlib.Path,
) -> pathlib.Path:
    payload = {
        "feature_id": feature_id,
        "assets":     assets,
        "horizons":   horizons,
        "ic_mean":        {str(h): _fmt(v) for h, v in ic_by_horizon.items()},
        "icir":           {str(h): _fmt(v) for h, v in icir_by_horizon.items()},
        "ic_mean_ortho":  {str(h): _fmt(v) for h, v in ic_by_horizon_ortho.items()},
        "icir_ortho":     {str(h): _fmt(v) for h, v in icir_by_horizon_ortho.items()},
        "regime": {
            str(h): {
                reg: {k: _fmt(v2) for k, v2 in stats.items()}
                for reg, stats in regimes.items()
            }
            for h, regimes in ic_by_horizon_regime.items()
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{feature_id}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def save_plots(
    feature_id: str,
    ic_ts: dict[int, pd.Series],
    ic_ts_ortho: dict[int, pd.Series],
    regime_labels: pd.Series,
    out_dir: pathlib.Path,
) -> pathlib.Path:
    """Six-panel plot:
    (0,0) Decay bar — raw
    (0,1) Rolling IC over time — 1d horizon
    (0,2) IC heatmap — horizon × month
    (1,0) Regime split — high vol vs low vol BTC
    (1,1) Raw vs Ortho IC — grouped bar by horizon
    (1,2) IC distribution — 1d horizon
    """
    horizons = sorted(ic_ts.keys())
    h1 = horizons[0]

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle(f"IC Analysis — {feature_id}", fontsize=13, fontweight="bold")

    # ── (0,0) Decay bar chart ─────────────────────────────────────────────────
    ax = axes[0, 0]
    means   = [ic_ts[h].mean() for h in horizons]
    stderrs = [ic_ts[h].std() / np.sqrt(ic_ts[h].notna().sum()) for h in horizons]
    colors  = ["tab:green" if m >= 0 else "tab:red" for m in means]
    ax.bar([f"{h}d" for h in horizons], means, color=colors, alpha=0.8,
           yerr=stderrs, capsize=4, ecolor="gray")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("Decay curve — IC moyen par horizon (raw)")
    ax.set_ylabel("IC (Spearman)")
    ax.set_xlabel("Horizon")

    # ── (0,1) Rolling IC — 1d ────────────────────────────────────────────────
    ax = axes[0, 1]
    ic1  = ic_ts[h1].dropna()
    roll = ic1.rolling(63, min_periods=20).mean()
    ax.plot(ic1.index, ic1.values,   color="lightsteelblue", lw=0.5, alpha=0.6, label="IC daily")
    ax.plot(roll.index, roll.values, color="steelblue",      lw=1.5,            label="IC rolling 63d")
    ax.fill_between(roll.index, 0, roll.values,
                    where=roll.values >= 0, color="tab:green", alpha=0.15)
    ax.fill_between(roll.index, 0, roll.values,
                    where=roll.values < 0,  color="tab:red",   alpha=0.15)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title(f"IC rolling 63d — horizon {h1}d")
    ax.set_ylabel("IC")
    ax.legend(fontsize=7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.tick_params(axis="x", labelsize=7)

    # ── (0,2) IC heatmap — horizon × mois ────────────────────────────────────
    ax = axes[0, 2]
    monthly_series = []
    for h in horizons:
        m = ic_ts[h].resample("M").mean().rename(h)
        monthly_series.append(m)
    mat = pd.concat(monthly_series, axis=1).T        # shape: (horizons, months)
    mat_vals = mat.values.astype(float)
    valid    = mat_vals[~np.isnan(mat_vals)]
    vmax     = max(np.abs(valid).max(), 0.05) if len(valid) else 0.1

    im = ax.pcolormesh(
        np.arange(mat.shape[1] + 1),
        np.arange(len(horizons) + 1),
        mat_vals,
        cmap="RdYlGn",
        vmin=-vmax, vmax=vmax,
    )
    ax.set_yticks(np.arange(len(horizons)) + 0.5)
    ax.set_yticklabels([f"{h}d" for h in horizons], fontsize=8)
    n_months = mat.shape[1]
    step = max(1, n_months // 12)
    tick_pos = np.arange(0, n_months, step)
    ax.set_xticks(tick_pos + 0.5)
    ax.set_xticklabels(
        [mat.columns[i].strftime("%Y-%m") for i in tick_pos],
        rotation=45, fontsize=6,
    )
    plt.colorbar(im, ax=ax, label="IC moyen mensuel")
    ax.set_title("IC heatmap (horizon × mois)")

    # ── (1,0) Regime split ────────────────────────────────────────────────────
    ax = axes[1, 0]
    if not regime_labels.empty:
        regime_means: dict[str, list] = {"high_vol": [], "low_vol": []}
        for h in horizons:
            ic_h = ic_ts[h]
            for reg in ["high_vol", "low_vol"]:
                mask = regime_labels == reg
                regime_means[reg].append(ic_h[mask].mean())

        x = np.arange(len(horizons))
        w = 0.35
        ax.bar(x - w/2, regime_means["high_vol"], w,
               label="Haute vol", color="tab:orange", alpha=0.8)
        ax.bar(x + w/2, regime_means["low_vol"],  w,
               label="Basse vol",  color="tab:blue",   alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{h}d" for h in horizons])
        ax.legend(fontsize=8)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("IC par régime volatilité BTC (raw)")
    ax.set_ylabel("IC moyen")

    # ── (1,1) Raw vs Ortho comparison ─────────────────────────────────────────
    ax = axes[1, 1]
    raw_means   = [ic_ts[h].mean() for h in horizons]
    ortho_means = (
        [ic_ts_ortho[h].mean() for h in horizons]
        if ic_ts_ortho else [float("nan")] * len(horizons)
    )
    x = np.arange(len(horizons))
    w = 0.35
    ax.bar(x - w/2, raw_means,   w, label="Raw",   color="steelblue",  alpha=0.8)
    ax.bar(x + w/2, ortho_means, w, label="Ortho", color="darkorange", alpha=0.8)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}d" for h in horizons])
    ax.set_title("Raw vs Ortho IC (mom_20d + rv_30d retirés)")
    ax.set_ylabel("IC moyen")
    ax.legend(fontsize=8)

    # ── (1,2) IC distribution — 1d ───────────────────────────────────────────
    ax = axes[1, 2]
    ic1_clean = ic_ts[h1].dropna()
    ax.hist(ic1_clean.values, bins=40, color="tab:purple",
            alpha=0.7, edgecolor="white", lw=0.3)
    ax.axvline(ic1_clean.mean(), color="k",    lw=1.2, ls="--",
               label=f"mean={ic1_clean.mean():.3f}")
    ax.axvline(0,                color="gray", lw=0.8)
    ax.set_title(f"Distribution IC — horizon {h1}d")
    ax.set_xlabel("IC")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{feature_id}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path
