"""Research Synthesis agent — CLI entry point.

Reads QA reports and IC results, ranks features, flags collinearity,
suggests composites, and drafts a research note.

Usage:
    python -m agents.research_synthesis.runner
"""
from __future__ import annotations
import argparse
import datetime as dt
import importlib
import pathlib
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agents.research_synthesis.analyser import (
    load_ic_results,
    load_qa_results,
    rank_features,
    check_collinearity,
    suggest_composites,
)
from agents.research_synthesis.backtest import (
    run_cross_asset_backtest,
    aggregate_metrics,
    save_backtest_plots,
)

RAW     = ROOT / "data" / "raw"
IC_DIR  = ROOT / "reports" / "ic"
QA_DIR  = ROOT / "reports" / "qa"
OUT_DIR = ROOT / "reports" / "synthesis"


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_ohlcv(symbol: str, exchange: str = "binance-futures") -> pd.DataFrame:
    base = RAW / f"exchange={exchange}" / "data_type=ohlcv_1d" / f"symbol={symbol}"
    parts = sorted(base.rglob("part-*.parquet"))
    if not parts:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df["date"] = pd.to_datetime(df["ts_open"], unit="us", utc=True).dt.normalize()
    return df.sort_values("date").set_index("date")


def load_universe() -> list[str]:
    with open(ROOT / "configs" / "universe.yaml") as f:
        return [a["symbol"] for a in yaml.safe_load(f)["assets"]]


def load_registry() -> list[dict]:
    with open(ROOT / "configs" / "features.yaml") as f:
        return yaml.safe_load(f)["features"]


# ── Report generation ─────────────────────────────────────────────────────────

def write_research_note(
    ranking: pd.DataFrame,
    qa_results: dict[str, bool],
    collinear_pairs: list[dict],
    composites: list[dict],
    ic_results: dict[str, dict],
    backtest_summaries: dict[str, pd.DataFrame],
    out_dir: pathlib.Path,
) -> pathlib.Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    today = dt.date.today().isoformat()

    lines = [
        f"# Research Note — Feature Analysis",
        f"",
        f"**Date:** {today}  ",
        f"**Universe:** {', '.join(ic_results[next(iter(ic_results))]['assets'])}  ",
        f"**Période:** 2020 → 2026",
        f"",
        f"---",
        f"",
        f"## 1. Classement des features (horizon 1j)",
        f"",
        f"| Rang | Feature | IC | ICIR | QA | Meilleur régime |",
        f"|---|---|---|---|---|---|",
    ]

    for i, row in ranking.iterrows():
        fid = row["feature"]
        qa_ok = qa_results.get(fid, False)
        qa_flag = "✅" if qa_ok else "❌"
        ic_fmt   = f"{row['IC']:+.4f}"
        icir_fmt = f"{row['ICIR']:+.3f}"
        exploit  = "⭐" if abs(row["ICIR"]) >= 0.10 else ""
        lines.append(
            f"| {i+1} | `{fid}` | {ic_fmt} | {icir_fmt} {exploit} | {qa_flag} | {row['best_regime']} |"
        )

    lines += [
        f"",
        f"> ⭐ = ICIR ≥ 0.10, considéré exploitable en production.",
        f"",
        f"---",
        f"",
        f"## 2. IC par horizon (decay curves)",
        f"",
        f"| Feature | 1j | 3j | 5j | 7j |",
        f"|---|---|---|---|---|",
    ]

    for fid, data in ic_results.items():
        ic_h = [data["ic_mean"].get(str(h), "—") for h in [1, 3, 5, 7]]
        ic_fmt = [f"{float(v):+.4f}" if v != "nan" and v != "—" else "—" for v in ic_h]
        lines.append(f"| `{fid}` | {' | '.join(ic_fmt)} |")

    lines += [
        f"",
        f"---",
        f"",
        f"## 3. Collinéarité entre features",
        f"",
    ]

    if collinear_pairs:
        lines += [
            f"| Feature 1 | Feature 2 | Corrélation | Flag |",
            f"|---|---|---|---|",
        ]
        for pair in collinear_pairs:
            lines.append(
                f"| `{pair['feature_1']}` | `{pair['feature_2']}` "
                f"| {pair['correlation']:+.3f} | {pair['flag']} |"
            )
    else:
        lines.append("Aucune paire collinéaire détectée (seuil = 0.70).")

    lines += [
        f"",
        f"---",
        f"",
        f"## 4. Signaux composites suggérés",
        f"",
    ]

    if composites:
        for comp in composites:
            lines += [
                f"### `{comp['name']}`",
                f"",
                f"**Features :** {', '.join(f'`{f}`' for f in comp['features'])}  ",
                f"**Poids :** {comp['weights']}  ",
                f"**Rationale :** {comp['rationale']}",
                f"",
            ]
    else:
        lines.append("Aucun composite suggéré — revoir les features individuelles.")

    lines += [
        f"",
        f"---",
        f"",
        f"## 5. Backtest L/S — métriques par feature",
        f"",
        f"Signal : +1 si feature > 0, -1 sinon. Entrée clôture t, capture ret t+1.",
        f"",
    ]

    if backtest_summaries:
        lines += [
            f"| Feature | Asset | Sharpe | Max DD | Win rate | Return total |",
            f"|---|---|---|---|---|---|",
        ]
        for fid, df in backtest_summaries.items():
            for asset, row in df.iterrows():
                sh  = f"{row['sharpe']:+.3f}"  if not np.isnan(row['sharpe'])    else "—"
                dd  = f"{row['max_dd']*100:.1f}%" if not np.isnan(row['max_dd']) else "—"
                wr  = f"{row['win_rate']*100:.1f}%" if not np.isnan(row['win_rate']) else "—"
                ret = f"{row['total_ret']*100:.1f}%" if not np.isnan(row['total_ret']) else "—"
                lines.append(f"| `{fid}` | {asset} | {sh} | {dd} | {wr} | {ret} |")
    else:
        lines.append("Aucune feature exploitable — backtest non exécuté.")

    lines += [
        f"",
        f"---",
        f"",
        f"## 6. Conclusions et recommandations",
        f"",
    ]

    # Auto-generate conclusions based on results
    best = ranking.iloc[0] if len(ranking) > 0 else None
    exploitable = ranking[ranking["ICIR"].abs() >= 0.10]["feature"].tolist()

    if best is not None:
        lines.append(
            f"- **Signal le plus fort :** `{best['feature']}` "
            f"(ICIR = {best['ICIR']:+.3f})"
        )

    if exploitable:
        lines.append(
            f"- **Features exploitables (|ICIR| ≥ 0.10) :** "
            f"{', '.join(f'`{f}`' for f in exploitable)}"
        )
    else:
        lines.append(
            "- **Aucune feature n'atteint le seuil ICIR = 0.10** avec 5 assets. "
            "L'univers doit être élargi (objectif : 40 assets) pour réduire le bruit."
        )

    if collinear_pairs:
        pairs_str = ", ".join(f"`{p['feature_1']}` / `{p['feature_2']}`" for p in collinear_pairs)
        lines.append(
            f"- **Collinéarité détectée** entre {pairs_str}. "
            f"Utiliser le composite ICIR-pondéré plutôt que les features brutes."
        )

    lines += [
        f"- **Prochaine étape :** construire le signal composite et simuler le P&L "
        f"sur la période 2020–2026.",
        f"",
    ]

    path = out_dir / "research_note.md"
    path.write_text("\n".join(lines))
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=1)
    args = ap.parse_args()

    print("Chargement des résultats IC et QA...")
    ic_results = load_ic_results(IC_DIR)
    qa_results = load_qa_results(QA_DIR)

    if not ic_results:
        print("Aucun résultat IC trouvé dans reports/ic/. Lance d'abord le backtest IC agent.")
        sys.exit(1)

    print(f"  {len(ic_results)} features IC chargées")
    print(f"  {len(qa_results)} rapports QA chargés")

    # Ranking
    print("\nClassement des features...")
    ranking = rank_features(ic_results, horizon=args.horizon)
    print(ranking[["feature", "IC", "ICIR"]].to_string(index=False))

    # Collinearity — recompute signals
    print("\nVérification de la collinéarité...")
    assets = load_universe()
    registry = load_registry()
    prices = {s: load_ohlcv(s) for s in assets}

    signals: dict[str, dict[str, pd.Series]] = {}
    for feat_cfg in registry:
        if feat_cfg.get("source") != "ohlcv_1d":
            continue
        fid = feat_cfg["id"]
        if fid not in ic_results:
            continue
        params = feat_cfg.get("params", {})
        try:
            mod = importlib.import_module(f"dfi_features.{fid}")
            signals[fid] = {
                sym: mod.compute(df, params)
                for sym, df in prices.items()
                if not df.empty
            }
        except Exception:
            pass

    collinear_pairs = check_collinearity(signals, threshold=0.70)
    if collinear_pairs:
        for pair in collinear_pairs:
            print(f"  ⚠️  {pair['feature_1']} ↔ {pair['feature_2']} : corr = {pair['correlation']}")
    else:
        print("  Aucune paire collinéaire.")

    # Composite suggestions
    composites = suggest_composites(ranking, collinear_pairs)
    print(f"\n{len(composites)} composite(s) suggéré(s).")

    # ── L/S backtest on exploitable features ─────────────────────────────────
    print("\nL/S backtest des features exploitables (|ICIR| ≥ 0.05)...")
    exploitable = ranking[ranking["ICIR"].abs() >= 0.05]["feature"].tolist()

    backtest_summaries: dict[str, pd.DataFrame] = {}
    for fid in exploitable:
        feat_cfg = next((f for f in registry if f["id"] == fid), None)
        if feat_cfg is None or feat_cfg.get("source") != "ohlcv_1d":
            continue
        if fid not in signals:
            continue
        print(f"\n  {fid}")
        bt_results = run_cross_asset_backtest(signals[fid], prices)
        if not bt_results:
            continue
        metrics_df = aggregate_metrics(bt_results)
        backtest_summaries[fid] = metrics_df

        mean_sharpe = metrics_df["sharpe"].mean()
        print(f"    Sharpe moyen cross-asset : {mean_sharpe:+.3f}")
        print(metrics_df[["sharpe", "max_dd", "win_rate", "total_ret"]].to_string())

        plot_path = save_backtest_plots(fid, bt_results, metrics_df, OUT_DIR)
        print(f"    → {plot_path.relative_to(ROOT)}")

    # ── Write research note ───────────────────────────────────────────────────
    path = write_research_note(
        ranking, qa_results, collinear_pairs, composites,
        ic_results, backtest_summaries, OUT_DIR,
    )
    print(f"\n→ {path.relative_to(ROOT)}")
    print("\n── Terminé ──")


if __name__ == "__main__":
    main()
