# SKILL: Backtest / IC agent

## Job
Measure the predictive power of each feature via Information Coefficient (IC),
ICIR, decay curves, IC heatmap, regime splits, and orthogonalization.
Produces a standardized JSON report and a 6-panel plot per feature.
Does not simulate portfolio P&L — that belongs to the research synthesis agent.

## Inputs (read-only)
- `configs/features.yaml`        -> feature registry (id, family, source, params)
- `configs/universe.yaml`        -> assets to analyse
- `data/raw/exchange=.../...`    -> OHLCV partitions written by daily_ingestion

## Outputs (write-only)
```
reports/ic/
  <feature_id>.json    -> IC / ICIR / regime stats, raw + ortho
  <feature_id>.png     -> 6-panel plot
```

### JSON schema (per feature)
```json
{
  "feature_id": "zscore_ret_composite",
  "assets": ["BTCUSDT", "ETHUSDT", ...],
  "horizons": [1, 3, 5, 7],
  "ic_mean":        {"1": 0.031, "3": 0.028, "5": 0.025, "7": 0.021},
  "icir":           {"1": 0.080, "3": 0.072, "5": 0.060, "7": 0.049},
  "ic_mean_ortho":  {"1": 0.020, "3": 0.017, "5": 0.015, "7": 0.012},
  "icir_ortho":     {"1": 0.050, "3": 0.042, "5": 0.036, "7": 0.029},
  "regime": {
    "1": {
      "high_vol": {"mean_ic": 0.04, "std_ic": 0.12, "icir": 0.33, "n": 1146},
      "low_vol":  {"mean_ic": 0.02, "std_ic": 0.10, "icir": 0.20, "n": 1146}
    }
  }
}
```

`ic_mean_ortho` / `icir_ortho` are empty dicts `{}` when `--no-ortho` is passed.

## Metrics computed

| Metric | Definition | Threshold |
|---|---|---|
| IC | Spearman rank corr between feature_t and ret_{t+h}, cross-asset | \|IC\| > 0.02 = non-trivial |
| ICIR | mean(IC) / std(IC) over full history | \|ICIR\| > 0.10 = exploitable |
| Decay curve | IC mean at h = 1, 3, 5, 7 days | Signal should decay smoothly |
| IC heatmap | Monthly mean IC, horizon × time (2D colour map) | Stable colour = robust signal |
| Regime split | IC in high-vol vs low-vol BTC regimes (raw IC) | Both regimes should be same sign |
| Orthogonalized IC | IC after regressing out mom_20d + rv_30d | Tests uniqueness of information |

## Plot panels (6-panel 2×3 grid)

| Panel | Content |
|---|---|
| (0,0) | Decay bar chart — raw IC mean ± stderr per horizon |
| (0,1) | Rolling IC line (63d window) — 1d horizon |
| (0,2) | **IC heatmap** — horizon (y) × month (x), RdYlGn colormap |
| (1,0) | Regime split — IC in high-vol vs low-vol BTC, all horizons |
| (1,1) | **Raw vs Ortho** — grouped bar per horizon |
| (1,2) | IC distribution histogram — 1d horizon |

## Horizons by feature family
```
returns, volatility, momentum  -> [1, 3, 5, 7]
derivatives                    -> [1, 3, 5]
microstructure                 -> [1, 3]
```

## Contract
1. **No lookahead.** Feature values at time t use only data ≤ t.
   IC is computed as corr(feature_t, ret_{t→t+h}) — forward return starts
   strictly AFTER t.
2. **Cross-asset IC.** At each date t, rank all available assets by their
   feature value and correlate against their forward returns.
3. **Dual ortho pass.** The runner always computes IC twice in one invocation:
   once on raw signals, once on signals orthogonalized against mom_20d + rv_30d.
   Both sets are saved in the JSON. Use `--no-ortho` to skip the second pass.
4. **Regime labels.** High/low volatility split based on 30-day BTC realized
   volatility, median threshold. Labels derived from the same raw data.
5. **Atomic writes.** JSON and PNG are written to `reports/ic/` only on
   successful completion of all metric computations for that feature.
6. **Empty features.** Features that return empty Series for all assets
   are skipped with a warning — not an error.

## CLI
```bash
# All ohlcv_1d features
python -m agents.backtest_ic.runner

# Specific features
python -m agents.backtest_ic.runner --features zscore_ret_composite cvd_20d

# Custom horizons
python -m agents.backtest_ic.runner --features rv_30d --horizons 1 5 10 21

# Raw IC only (skip ortho pass)
python -m agents.backtest_ic.runner --no-ortho
```

## Tools the agent may use
- `data/raw/` (read-only)
- `configs/` (read-only)
- `dfi_features/` package (to recompute features on the fly)
- `reports/ic/` (write)
- `agents/backtest_ic/ic_engine.py`, `report.py`
- `numpy`, `pandas`, `scipy`, `matplotlib`

## Tools the agent must NOT use
- `data/features/` write access
- Any live data feed or external API
- Mutations of `configs/`

## Definition of done
- `reports/ic/<feature_id>.json` exists for every requested feature.
- `reports/ic/<feature_id>.png` exists for every requested feature.
- JSON contains both `ic_mean` (raw) and `ic_mean_ortho` (orthogonalized).
- No feature with available data produces an empty JSON.
- ICIR values in JSON match notebook output within 0.001.
