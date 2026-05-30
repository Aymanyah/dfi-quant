# SKILL: Feature QA agent

## Job
Run statistical sanity checks on every feature after construction.
Produces a Markdown report per feature with pass/fail flags.
Does NOT modify features — read-only audit only.

## Inputs (read-only)
- `configs/features.yaml`      -> feature registry
- `configs/universe.yaml`      -> assets to check
- `data/raw/exchange=.../...`  -> OHLCV partitions (recomputes features on the fly)

## Outputs (write-only)
```
reports/qa/
  <feature_id>.md    -> Markdown report with pass/fail per check
```

## Checks

| Check | Question | Fail condition |
|---|---|---|
| Distribution | La feature est-elle stationnaire ? A-t-elle des outliers ? | ADF p > 0.05 ou outlier rate > 1% |
| Coverage | La feature se calcule sur tous les assets avec peu de NaN ? | NaN rate > 20% après warmup |
| Lookahead | La valeur à t change-t-elle si on masque les données après t ? | Toute différence > 1e-10 |
| Explosion | La variance reste-t-elle stable dans le temps ? | std finale > 5× std initiale |

## Contract
1. Re-running produces the same report for unchanged features.
2. Checks are run on BTCUSDT as reference asset for single-series checks.
3. Lookahead check uses 10 random spot-checks with seed=42 (reproducible).
4. A FAIL does not stop the pipeline — it flags the feature for review.
5. Never modifies `data/` or `configs/`.

## CLI
```bash
python -m agents.feature_qa.runner                        # toutes les features
python -m agents.feature_qa.runner --features mom_20d     # une seule feature
```

## Definition of done
- `reports/qa/<feature_id>.md` exists for every requested feature.
- Exit code 0 even if checks fail (failures are in the report, not exceptions).
