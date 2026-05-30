# Claude Code — Research Synthesis Agent

You are working in `agents/research_synthesis/`. This is the final agent
in the pipeline — it translates IC numbers into actionable research decisions
and validates them via L/S backtest.

## What this agent does

1. Reads `reports/ic/*.json` — IC/ICIR results per feature
2. Reads `reports/qa/*.md`   — QA pass/fail per feature
3. Ranks features by |ICIR| at horizon 1d
4. Flags collinear feature pairs (Spearman corr > 0.70)
5. Suggests composite signals (equal-weight + ICIR-weighted)
6. **Runs L/S backtest** on exploitable features (|ICIR| ≥ 0.05)
7. Writes a research note to `reports/synthesis/research_note.md`
8. Saves backtest plots to `reports/synthesis/backtest_<feature_id>.png`

## How to run

```bash
python -m agents.research_synthesis.runner             # horizon 1j (défaut)
python -m agents.research_synthesis.runner --horizon 5 # horizon 5j
```

Or via the orchestrator:
```bash
python orchestration/flows.py --research
```

## Full pipeline output

```
reports/
  ic/
    <feature_id>.json       ← backtest_ic agent
    <feature_id>.png
  synthesis/
    research_note.md        ← classement + IC + backtest L/S + recommandations
    backtest_<feature>.png  ← 4-panel plot par feature exploitable
```

## Interpreting results

- **ICIR ≥ 0.10** → feature exploitable, inclure dans le composite
- **ICIR < 0.05** → signal trop bruité avec 5 assets
- **Sharpe > 1.0** → stratégie L/S viable sur cette feature
- **Collinéarité > 0.70** → garder uniquement la feature avec le |ICIR| le plus élevé
- **IC change de signe entre régimes** → signal fragile

## Backtest L/S logic

- Signal : `+1` si feature(t) > 0, `-1` sinon
- Entrée : clôture du jour t
- Capture : rendement simple du jour t+1
- `strat_ret(t) = signal(t) × ret(t+1)`
- Sharpe annualisé : `mean(strat_ret) / std(strat_ret) × √252`

## What this agent must NOT do

- Modify `data/` or `configs/`
- Implement features directly (that's `dfi_features/`)
- Invent IC numbers — always read from `reports/ic/*.json`
- Connect to live trading — research only
