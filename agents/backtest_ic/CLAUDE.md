# Claude Code — Backtest / IC Agent

You are working in `agents/backtest_ic/`. Your job is to measure the
predictive power of features — not to build trading strategies.

## What this agent does

For each feature, computes:
- **IC** — Spearman correlation between signal at t and forward return at t+h
- **ICIR** — mean(IC) / std(IC) — consistency of the signal over time
- **Decay curve** — IC at horizons [1, 3, 5, 7] days
- **Regime split** — IC in high-vol vs low-vol BTC regimes
- **Orthogonalized IC** — IC after removing mom_20d + rv_30d influence

Outputs go to `reports/ic/<feature_id>.json` and `reports/ic/<feature_id>.png`.

## How to run

```bash
python -m agents.backtest_ic.runner                          # toutes les features
python -m agents.backtest_ic.runner --features cvd_20d       # une seule
python -m agents.backtest_ic.runner --features cvd_20d --no-ortho
python -m agents.backtest_ic.runner --features cvd_20d --horizons 1 5 10 21
```

## IC is computed cross-asset

At each date t, we rank all assets by their signal value and correlate
against their forward returns. This gives one IC number per day.
The daily IC series is then averaged → mean IC reported.

**Not** a time-series autocorrelation on a single asset.

## Thresholds

| Metric | Threshold | Meaning |
|---|---|---|
| \|IC\| | > 0.02 | Non-trivial signal |
| \|ICIR\| | > 0.10 | Exploitable in production |
| \|ICIR\| | > 0.30 | Strong signal (rare) |

## When modifying ic_engine.py

- IC computation must be strictly point-in-time: forward return starts at t+1
- Orthogonalization uses OLS residuals — do not use ridge or lasso
- Regime split threshold is median BTC rv_30d — do not hardcode a value
- All functions must be pure (no file I/O, no global state)
