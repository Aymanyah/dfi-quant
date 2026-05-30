# dfi-quant

Multi-agent crypto research stack. Step 1 = deterministic data layer.
The agents (Claude Code instances) are non-deterministic; this layer is not.

## Agent chain
1. Tardis ingestion agent  -> raw parquet on S3 (hive-partitioned)
2. Feature construction agent -> versioned feature parquet
3. Feature QA agent -> stationarity, leakage, NaN audits
4. Backtest / IC agent -> IC, decile spreads, turnover
5. Research synthesis agent -> markdown reports + dashboards

Each agent has its own `agents/<name>/SKILL.md` describing job, inputs,
outputs, and tool access. Agents do not own scheduling. Orchestration
(Prefect / cron) lives in `orchestration/`.

## Layout
- `configs/`        single source of truth (settings, features, universe)
- `data/`           local cache (mirrors S3 layout)
- `agents/`         SKILL.md + agent-owned code per agent
- `orchestration/`  DAGs, schedules, Prefect flows
- `scripts/`        CLI entrypoints (idempotent, deterministic)
- `tests/`          pytest, must pass before any agent code is merged
- `notebooks/`      exploratory only, never imported

## Determinism contract
- Every feature is keyed by (feature_id, version, asset, ts).
- A feature version is immutable. Code changes -> new version.
- Agents read `configs/features.yaml` as the registry.
