# Project Bible: Self-Healing ML Pipeline

> **If the interviewer remembers only one thing, it should be:**
> *"Execution isn't the finish line — verification is. I built the layer
> that checks whether a fix actually worked, not just whether it ran."*

---

## 1. Elevator Pitch (30 sec)

"AI models don't fail loudly — they drift quietly, and by the time someone notices, bad decisions have already shipped. I built a system that watches a live credit-risk model, detects when it's degrading (drift, bad data, latency, cost overruns), and picks the right fix automatically — adjust a threshold, retrain, roll back, or fall back to a simpler rule — based on cost, risk, and speed tradeoffs, not one fixed rule. It then verifies the fix actually worked and rolls back if it didn't. Tested against 4 naive strategies across 12 failure scenarios, it resolved 75% of incidents vs. 33–58% for the naive baselines."

---

## 2. Problem — why it existed

Production ML models degrade silently. Causes: input data drifts from training data, upstream data pipelines break, a bad model gets deployed, traffic patterns shift latency/cost. Nobody watches this in real time by hand at 3am. Most teams find out from a downstream business metric (loan defaults, fraud losses) weeks later.

## 3. Why existing approaches weren't enough

- **Static monitoring dashboards** — someone has to look at them and manually decide. No automatic action.
- **Single fixed response** (e.g. "always retrain on drift") — wrong tool for the wrong problem. Retraining doesn't fix a bad deployment; rollback does. Retraining is slow and costly when a threshold tweak would do.
- **Confidence-based auto-remediation** — picking whichever automated fix "sounds most sure of itself" ignores cost, tenant SLA, and risk. Tested this as a baseline — underperforms.
- **No verification loop** — most auto-remediation demos never check if the fix worked. This one does, and rolls back automatically if it didn't.

This is the framing recruiters respond to: not "I built agents," but "I identified that fixed rules fail because incidents aren't uniform, and built something that reasons per-incident instead."

---

## 4. Architecture (high level)

```
Historical dataset replayed as live traffic
        ↓
FastAPI prediction endpoint (real inference, real latency)
        ↓
Prometheus (scrapes real metrics)
        ↓
Incident Detector (compares live metrics to tenant baselines/SLAs)
        ↓
5 remediation agents each propose a fix (threshold / retrain / rollback / fallback / data-repair)
        ↓
Commander scores each proposal by tenant-specific cost/risk/quality weights, picks one
        ↓
Executor runs it, waits, re-measures
        ↓
Guardrail check: did it actually resolve? → accept or auto-rollback
        ↓
Result stored → meta-harness periodically re-tunes the commander's scoring weights offline
```

## 5. Tech stack (everything)

| Layer | Tech |
|---|---|
| Language | Python 3.12 |
| ML model | LightGBM (`lightgbm`) |
| ML utilities | scikit-learn, scipy, numpy, pandas |
| API | FastAPI + uvicorn |
| Metrics | prometheus-client (instrumentation) + Prometheus server (storage/query) |
| Dashboards | Grafana |
| Database | SQLite (dev) / PostgreSQL (`psycopg`, docker-compose) |
| ORM | SQLAlchemy |
| Config | pydantic-settings |
| Multi-agent reconciliation | LangGraph + langchain-anthropic (Claude) — only used for tie-break debates, optional |
| Testing | pytest, pytest-asyncio |
| Lint/type-check | ruff, mypy |
| Package/env manager | uv |
| Containers | Docker + docker-compose |

## 6. Data — where it came from

- **Source**: UCI "Default of Credit Card Clients" dataset, fetched live at train time via the `ucimlrepo` Python package (`fetch_uci_credit_default()` in `src/self_healing_pipeline/pipeline.py`) — not a static file checked into the repo.
- **Size**: ~30,000 rows, 23 features.
- **Task**: binary classification — will this customer default on their credit card payment.
- **Split**: training set, validation/baseline-calibration set, and a replay set. The replay set is what gets sent through the live `/predict` API to generate real runtime telemetry (not mocked).
- **Tenants**: rows are split into 3 synthetic tenants (enterprise/standard/free tier) by tertile, to simulate a multi-tenant SaaS model with different SLAs and cost sensitivities on the same underlying model.

## 7. Models — exactly which ones

- **One real trained model**: LightGBM binary classifier (`lgbm_credit_default.joblib`), predicting default probability. Baseline validation ROC-AUC ≈ 0.77.
- **No other ML models.** The 5 "remediation agents" are not separate trained models — they are rule-based/heuristic decision logic (eligibility conditions + weighted confidence formulas over live metrics). Important to say this precisely in an interview: the "multi-agent" part is decision-making agents, not an ensemble of ML models.
- **Optional LLM use**: Claude (via `langchain-anthropic` + LangGraph) is invoked only for reconciliation — when two remediation proposals score within 10% of each other, an LLM-mediated debate breaks the tie. The system runs fully without an API key (heuristic-only mode).

## 8. Infrastructure — everything deployed

From `docker-compose.yml`, 4 services:

| Service | Image/Build | Purpose |
|---|---|---|
| `postgres` | `postgres:15-alpine` | production-mode control-plane DB (SQLite used for local/dev) |
| `api` | built from `docker/api/Dockerfile` | FastAPI prediction + control-plane service |
| `prometheus` | `prom/prometheus:latest` | metric scraping/storage |
| `grafana` | `grafana/grafana:latest` | dashboards |

No Kubernetes, no cloud deployment — runs locally via `docker-compose up`. Be upfront about this if asked; it's a demonstrated pattern, not a production deployment.

## 9. Evaluation — metrics

- **Resolution rate**: % of scenarios where all guardrails pass after the action (AUC ≥ min, latency ≤ SLA, missing-rate ≤ limit).
- **Mean reward**: outcome-based score computed only after verification, from real before/after metric deltas (quality gain, reliability gain, cost gain, minus execution cost/time/regression penalties).
- **Guardrail violation rate**: % of scenarios with at least one guardrail breach.
- **Unnecessary retrain rate**: % of times retrain got picked for a non-drift incident (waste indicator).

Evaluated across 12 deterministic incident scenarios (severe drift, mild drift, deployment regression, latency breach, missing-data, schema corruption, cost overrun, etc.) × 5 selection policies = 60 trials. Deterministic, fixed seeds — not random noise from one lucky run.

## 10. Results — numbers

| Policy | Resolution | Mean Reward | Guardrail Violations |
|---|---:|---:|---:|
| **Adaptive commander (this project)** | **75%** | **0.346** | **50%** |
| Highest confidence | 58% | 0.285 | 50% |
| Always retrain | 50% | 0.186 | 75% |
| Fixed priority | 33% | 0.143 | 67% |
| Cheapest eligible action | 33% | 0.143 | 67% |

465 tests passing (`uv run pytest tests/`), 1 skipped (documented reason: shared on-disk DB engine causes cross-test pollution — see below).

## 11. Challenges — engineering

- **Severity isn't one formula.** Drift, data-quality, latency, and cost incidents need different severity math (different signals, different units). Had to build a per-incident-type severity calculator instead of one generic score, and store the components so severity is auditable, not a black box.
- **Confidence ≠ what matters.** Early version picked the agent most "sure of itself." Had to replace that with a multi-objective utility function (quality/reliability/cost/speed weighted per tenant) — this was the highest-leverage design change, proven by the baseline comparison.
- **Verifying without live production traffic.** No real user base to check "did this actually help." Solved by replaying held-out dataset rows through the real inference path so metrics are computed, not mocked, even though the traffic source is historical.
- **Avoiding double-counting in reward.** Agents predict their own expected effect; naively trusting that would let an agent "grade its own homework." Reward is computed strictly from measured before/after state, agent predictions are only used for initial ranking.
- **Meta-harness confound.** Agent historical-success is tracked globally per agent type, not conditioned on incident subtype/severity — so a boost from "retrain wins on severe drift" leaks into "retrain vs mild drift" ranking. Eligibility gates prevent it from causing outright wrong-type selection, and canary rollout catches regressions before full rollout, but the root cause isn't fixed. Still open — listed in "what you'd improve" below.
- **The meta-harness never reached the live commander — fixed, twice.** Traced the full path: `commander_v3.py` scores proposals using `UtilityWeights` loaded from a tenant's `TenantTierConfig` DB row. The meta-harness (`analyzer.py` → `tuner.py` → `canary.py`) computed and version-controlled new `ScoringWeights` correctly, and the DB columns already had matching names (`business_value_weight`, `confidence_weight`, etc.) — but nothing ever wrote the tuned values back into that row. First fix added `meta_harness/apply.py::sync_tuned_weights()`, verified with `tests/test_meta_harness_apply.py` (tune → apply → confirm `UtilityScorer` reads the new values) — but its only caller, `demo_meta_harness.py`, ran it against a throwaway in-memory SQLite engine, so the wiring existed in code but never touched production. Also found a second, deeper gap while closing this: `EvidenceBundleAnalyzer` reads evidence bundles from `traces_dir`, but `CommanderV3`'s live path never wrote any — `BundleWriter` existed but was only ever called from the old, unused `commander.py`. Second fix: `CommanderV3` now writes a bundle after every incident when given a `bundle_writer` (opt-in, off by default so unit tests don't pollute the real `traces/` dir — an early version defaulted it on and 23 test-run trace folders leaked into the real project directory before that was caught), and `scripts/tune_weights.py` runs the full analyze → tune → version → apply cycle against the real DB, discovering tenants from actual `IncidentHistory` rows. Verified live end to end: fired the same incident 3x (agent `retrain` wins each time), ran the tuning script, fired it again — winner flipped to `threshold`, reward improved 0.150 → 0.250.
- **Global weights vs. per-tenant config is a real tension, not fully resolved.** The fix above applies one set of tuned weights to every tenant. But `TenantTierConfig` is explicitly per-tenant (a free-tier customer's cost sensitivity shouldn't be overwritten by what worked for an enterprise customer). The evidence-bundle analyzer doesn't currently segment by tenant either — it pools all incidents together. Correct fix is per-tenant analysis and per-tenant tuning, not a global broadcast; flagged honestly below rather than papered over.
- **Test isolation bug, found while fixing the above.** `db/session.py`'s `get_engine()` is `@lru_cache`d and points at the real on-disk `sqlite:///./pipeline.db` by default. Tests that call `Base.metadata.create_all(get_engine())` share that same file and cached engine across the whole test session — so test order and prior local runs affect results. That's the actual reason `test_multi_tenant_isolation` in `test_integration_full_loop.py` is skipped. My new tests avoid this by creating their own in-memory engine per test (`create_engine("sqlite:///:memory:")`) instead of the cached global one.
- **Per-tenant concurrency without a central lock server.** Two incidents on the same tenant must not both get remediated at once (race on `TenantTierConfig`/model registry writes), but two different tenants should run in parallel, not queue behind each other. Solved with an `asyncio.Lock` per tenant_id in the gateway (`gateway.py`), not a single global lock — verified in `test_edge_cases.py::TestConcurrentIncidents`.

## 12. What you'd improve — shows maturity

- Condition `historical_success` on incident subtype/severity bucket, not just agent identity (fixes the confound above).
- Make meta-harness tuning per-tenant instead of global — segment the evidence-bundle analysis by tenant_id before computing significance, so a fintech tenant's history doesn't silently reweight a bank tenant's commander.
- Point `get_engine()` at a per-test-run DB in the test suite (or stop using the cached global engine in tests entirely) so `test_multi_tenant_isolation` can be un-skipped.
- Real drift detection (PSI, KS-test) instead of normalized mean-shift — current method is simple and works for this dataset but wouldn't generalize to high-dimensional or streaming features.
- Model delayed/partial label arrival — currently assumes labels available immediately, unrealistic for real credit-risk timelines (defaults surface months later).
- Thin React/TypeScript frontend over the existing typed API responses (`/incidents/recent`, `/agents/summary`) — backend already returns structured data, so this is additive, not a redesign.

## 13. Biggest lesson

Building the agents and the commander was the "fun" part. The part that actually made the project defensible was building **the thing that checks whether the fix worked** — most auto-remediation demos stop at "action executed," and that's not the same as "problem solved." Once verification and auto-rollback existed, it forced honesty everywhere else in the system: agents' self-reported confidence and expected effects couldn't be trusted blindly, so reward had to come from real measured deltas, not predictions. The lesson: an autonomous system is only as trustworthy as its ability to check its own work — the verification loop, not the decision logic, is what makes "self-healing" a real claim instead of a marketing name.

## 14. Signals

Not about the project — about what an interviewer should walk away inferring. Checklist to run through before any interview: did the conversation leave all six on the table?

| Signal | Evidence from the story |
|---|---|
| Systems thinking | End-to-end closed-loop architecture: detect → decide → execute → verify → adapt |
| Experimental rigor | Controlled comparison against 4 baselines over 60 deterministic trials |
| Engineering judgment | Deterministic multi-objective scoring, LLM used only as a tie-breaker |
| Reliability mindset | Verification loop with automatic rollback on guardrail failure |
| Intellectual honesty | Unprompted disclosure of limitations, and of the new tension the meta-harness fix introduced |
| Debugging ability | Meta-harness/commander disconnect — missed by per-phase unit tests, found by tracing data flow end-to-end |
