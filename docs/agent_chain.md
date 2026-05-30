# Agent chain contracts

Each arrow represents a strict input/output contract. If the contract
breaks, the downstream gate fails and the orchestrator halts the chain.

```
Tardis ingestion  --[ raw parquet, _SUCCESS markers ]-->
Feature construction  --[ feature parquet, (id, version) ]-->
Feature QA  --[ qa report + pass/fail flag ]-->
Backtest / IC  --[ ic table, decile pnl ]-->
Research synthesis  --[ markdown report + dashboard json ]
```

## Invariants every agent must respect
1. **Read-only configs.** `configs/*` is the single source of truth.
   Agents may propose changes via PR but must not edit at runtime.
2. **Versioned outputs.** `(feature_id, version)` is immutable. New
   logic = new version. Old data stays addressable for reproducibility.
3. **Pure subprocess invocation.** The orchestrator runs each agent as
   a subprocess. No shared Python state, no shared in-memory caches.
4. **Atomic writes.** Every writer uses tmp-then-rename plus a
   `_SUCCESS` marker. Half-written outputs are invisible to readers.
5. **No NaN imputation upstream.** Imputation is a feature decision.
   Raw and feature layers preserve NaNs so the QA agent can audit.
6. **No cross-agent network calls.** An agent only reads from the layer
   directly upstream and only writes to its own layer.

## Step status
- [x] Tardis ingestion agent: SKILL + schemas + writer + CLI scaffold
- [ ] Feature construction agent
- [ ] Feature QA agent
- [ ] Backtest / IC agent
- [ ] Research synthesis agent
