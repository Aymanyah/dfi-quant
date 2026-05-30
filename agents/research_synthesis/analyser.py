"""Core analysis: ranking, collinearity, composite suggestions.

All functions are pure — no file I/O.
"""
from __future__ import annotations
import json
import pathlib
import numpy as np
import pandas as pd


# ── Load IC results ───────────────────────────────────────────────────────────

def load_ic_results(ic_dir: pathlib.Path) -> dict[str, dict]:
    """Load all JSON files from reports/ic/."""
    results = {}
    for path in sorted(ic_dir.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        results[data["feature_id"]] = data
    return results


def load_qa_results(qa_dir: pathlib.Path) -> dict[str, bool]:
    """Parse QA Markdown reports and return {feature_id: overall_passed}."""
    results = {}
    for path in sorted(qa_dir.glob("*.md")):
        fid = path.stem
        text = path.read_text()
        results[fid] = "Overall: ✅ PASS" in text
    return results


# ── Feature ranking ───────────────────────────────────────────────────────────

def rank_features(ic_results: dict[str, dict], horizon: int = 1) -> pd.DataFrame:
    """
    Rank features by ICIR at a given horizon.
    Higher |ICIR| = more consistent signal.
    """
    rows = []
    for fid, data in ic_results.items():
        h = str(horizon)
        ic_mean = data["ic_mean"].get(h, float("nan"))
        icir    = data["icir"].get(h, float("nan"))

        # Best regime
        regime_data = data.get("regime", {}).get(h, {})
        best_regime = max(
            regime_data.items(),
            key=lambda kv: abs(kv[1].get("mean_ic", 0)),
            default=("?", {}),
        )

        rows.append({
            "feature":     fid,
            "IC":          round(float(ic_mean), 4) if ic_mean != "nan" else float("nan"),
            "ICIR":        round(float(icir), 3)    if icir    != "nan" else float("nan"),
            "|ICIR|":      abs(float(icir))          if icir    != "nan" else 0.0,
            "best_regime": best_regime[0],
            "regime_IC":   round(best_regime[1].get("mean_ic", float("nan")), 4),
        })

    df = pd.DataFrame(rows).sort_values("|ICIR|", ascending=False).drop(columns=["|ICIR|"])
    return df.reset_index(drop=True)


# ── Collinearity check ────────────────────────────────────────────────────────

def check_collinearity(
    signals: dict[str, dict[str, pd.Series]],
    threshold: float = 0.70,
) -> list[dict]:
    """
    Compute pairwise Spearman correlation between features (pooled across assets).
    Flag pairs with |corr| > threshold as collinear.

    Collinear features carry redundant information — including both in a
    composite signal doesn't improve prediction and wastes risk budget.
    """
    feature_ids = list(signals.keys())
    # Pool all assets into one long series per feature
    pooled: dict[str, pd.Series] = {}
    for fid, asset_signals in signals.items():
        parts = [s.rename(fid) for s in asset_signals.values() if not s.empty]
        if parts:
            pooled[fid] = pd.concat(parts)

    flagged = []
    for i, f1 in enumerate(feature_ids):
        for f2 in feature_ids[i + 1:]:
            if f1 not in pooled or f2 not in pooled:
                continue
            both = pd.concat([pooled[f1], pooled[f2]], axis=1).dropna()
            if len(both) < 30:
                continue
            corr = float(both.iloc[:, 0].corr(both.iloc[:, 1], method="spearman"))
            if abs(corr) >= threshold:
                flagged.append({
                    "feature_1": f1,
                    "feature_2": f2,
                    "correlation": round(corr, 3),
                    "flag": "⚠️ collinéaire" if abs(corr) >= threshold else "ok",
                })

    return flagged


# ── Composite suggestions ─────────────────────────────────────────────────────

def suggest_composites(
    ranking: pd.DataFrame,
    collinear_pairs: list[dict],
    icir_threshold: float = 0.05,
) -> list[dict]:
    """
    Suggest composite signals based on:
    1. Keep features with |ICIR| > threshold
    2. Among collinear pairs, keep only the one with higher |ICIR|
    3. Suggest equal-weight and ICIR-weight composites
    """
    # Features that pass the ICIR threshold
    good = ranking[ranking["ICIR"].abs() >= icir_threshold]["feature"].tolist()

    # Remove the weaker feature from each collinear pair
    to_drop = set()
    for pair in collinear_pairs:
        f1, f2 = pair["feature_1"], pair["feature_2"]
        icir_f1 = ranking.loc[ranking["feature"] == f1, "ICIR"].values
        icir_f2 = ranking.loc[ranking["feature"] == f2, "ICIR"].values
        if len(icir_f1) and len(icir_f2):
            weaker = f2 if abs(icir_f1[0]) >= abs(icir_f2[0]) else f1
            to_drop.add(weaker)

    selected = [f for f in good if f not in to_drop]

    composites = []

    if len(selected) >= 2:
        composites.append({
            "name":     "composite_equal",
            "features": selected,
            "weights":  "égaux (1/N)",
            "rationale": "Diversification simple — chaque feature contribue également.",
        })

    # ICIR-weighted composite
    icir_vals = {
        row["feature"]: abs(row["ICIR"])
        for _, row in ranking.iterrows()
        if row["feature"] in selected and not np.isnan(row["ICIR"])
    }
    total = sum(icir_vals.values())
    if total > 0 and len(icir_vals) >= 2:
        weights = {f: round(v / total, 3) for f, v in icir_vals.items()}
        composites.append({
            "name":     "composite_icir_weighted",
            "features": list(weights.keys()),
            "weights":  weights,
            "rationale": "Pondération par ICIR — les features les plus régulières pèsent plus.",
        })

    return composites
