# SKILL: Research Synthesis Agent

## Job
Translate raw IC numbers into actionable research decisions. Reads QA reports
and IC results, ranks features by predictive power, flags collinearity, suggests
composites, and drafts a human-readable research note. This is the final agent
in the pipeline and the one you interact with most directly.

## Inputs (read-only)
- `reports/ic/*.json`         → IC/ICIR results per feature (from backtest_ic)
- `reports/qa/*.md`           → QA pass/fail per feature (from feature_qa)
- `configs/features.yaml`     → feature registry (source, params)
- `configs/universe.yaml`     → asset universe
- `data/raw/exchange=.../...` → OHLCV data (for collinearity computation)

## Outputs (write-only)
```
reports/synthesis/
  research_note.md    → ranking + IC decay + collinearity + composites + reco
```

## What it does (in order)
1. Load all `reports/ic/*.json` and `reports/qa/*.md`
2. Rank features by |ICIR| at the requested horizon
3. Flag collinear feature pairs (Spearman corr > 0.70, pooled cross-asset)
4. Suggest composite signals (equal-weight + ICIR-weighted), pruning collinear duplicates
5. Write `reports/synthesis/research_note.md` with all findings

## Contract
1. **IC numbers are never recomputed** — always read from `reports/ic/*.json`.
2. **Collinearity** uses Spearman correlation pooled cross-asset; threshold = 0.70.
3. **Composite pruning** — among collinear pairs, drop the feature with lower |ICIR|.
4. **Never modifies** `data/` or `configs/`.

## Interpreting results
- **ICIR ≥ 0.10** → feature exploitable, inclure dans le composite
- **ICIR < 0.05** → signal trop bruité
- **Collinéarité > 0.70** → garder uniquement la feature avec le |ICIR| le plus élevé
- **IC change de signe entre régimes** → signal fragile

## CLI
```bash
python -m agents.research_synthesis.runner             # horizon 1j (défaut)
python -m agents.research_synthesis.runner --horizon 5 # horizon 5j
```

## Definition of done
- `reports/synthesis/research_note.md` existe avec les sections : ranking, IC decay,
  collinéarité, composites, conclusions
- Les IC dans la note correspondent exactement à `reports/ic/*.json`
- Aucune donnée feature n'est recalculée from scratch

## What this agent must NOT do
- Modifier `data/` ou `configs/`
- Implémenter des features (c'est `dfi_features/`)
- Recalculer des IC — toujours lire depuis `reports/ic/*.json`
- Se connecter au live ou lancer des backtests
