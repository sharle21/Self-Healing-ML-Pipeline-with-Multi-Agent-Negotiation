# Methodology

How claims in this project are validated: what's real vs. simulated, how
baselines were derived, how the baseline policies being compared against
are defined, how statistical significance is tested, and how correctness
is verified via the test suite.

---

## Scope and Simulation Boundaries

### Implemented as real system behavior

- LightGBM model training and inference
- FastAPI prediction endpoint with measured latency
- UCI dataset rows replayed as real prediction requests
- Prometheus metric instrumentation and live querying
- Calculated feature drift scores and data-quality metrics
- Windowed model-quality metrics (AUC, precision, recall)
- Tenant-specific configuration and SLA persistence in SQLite
- Threshold updates written to the configuration store
- Model retraining against held-out validation data
- Model version rollback switching active model in registry
- Multi-objective proposal ranking via UtilityScorer
- Post-action state snapshot and guardrail verification
- Auto-rollback when guardrails fail and regression detected
- Outcome reward and evidence bundle persistence

### Simulated

- Production traffic source (historical dataset replay, not live)
- Tenant business profiles (documented cost model, not real billing)
- Infrastructure cost (calculated per prediction, not cloud metered)
- Fault injection (controlled scripts, not spontaneous failures)
- Upstream warehouse repair (replay environment, not real pipeline)
- Production-scale deployment infrastructure

The environment is simulated, but telemetry is computed from actual model
requests and injected data conditions rather than generated as random mock
values. Rationale for where this line was drawn:
[design.md](design.md#whats-real-vs-simulated-and-why-that-boundary-was-drawn-there).

---

## Model and Dataset

| Property | Value |
|---|---|
| Dataset | UCI Default of Credit Card Clients |
| Rows | ~30,000 |
| Features | 23 |
| Model | LightGBM binary classifier |
| Task | Predict credit-card default probability |
| Metric | ROC-AUC |
| Baseline ROC-AUC | ~0.77 |

The dataset is split into training, baseline calibration, and replay
partitions. The replay partition is sent through the real prediction API
to generate runtime telemetry.

---

## Baseline and Threshold Derivation

Tenant configuration values are derived from model evaluation and runtime
benchmarking, not hardcoded guesses.

| Configuration | Source |
|---|---|
| `baseline_auc` | Validation set evaluation |
| `min_auc` | Baseline minus tolerated degradation |
| `decision_threshold` | Threshold search on validation data |
| `baseline_latency_p95_ms` | Measured inference benchmark |
| `latency_sla_ms` | Baseline plus tolerance or explicit SLA |
| `max_missing_rate` | Operator-defined data-quality limit |
| `daily_cost_budget_usd` | Tenant business profile |
| `false_positive_cost` / `false_negative_cost` | Tenant business profile |

Business limits remain operator-defined because they represent
requirements, not model properties.

### Multi-tenant configuration

| Tenant | Priority | Latency SLA | Primary Risk |
|---|---|---:|---|
| Enterprise | Model quality | 150 ms | False negatives ($5,000 each) |
| Standard | Balanced | 100 ms | Quality and latency |
| Free | Cost | 200 ms | Inference expense |

---

## Baseline Policies Compared Against

Each is a fixed, non-adaptive strategy used as a control in
[evaluation.md](evaluation.md):

- **Highest confidence** — picks the eligible agent with the highest
  self-reported confidence, ignoring utility scoring entirely.
- **Always retrain** — retrains whenever eligible, regardless of incident
  type; never considers rollback or fallback.
- **Fixed priority** — tries agents in a fixed order
  (threshold → retrain → rollback → fallback → data_repair), picks the
  first eligible one.
- **Cheapest eligible** — picks the lowest dollar-cost eligible agent
  (threshold is $0 whenever eligible, so it wins most ties).

The adaptive commander (utility scoring + guardrails + auto-rollback) is
compared against all four, across the same 12 deterministic scenarios, for
an apples-to-apples 60-trial comparison.

---

## Statistical Methodology

### Scenario design

12 scenarios were hand-designed to each stress-test a distinct failure
mode (severe vs. mild drift, deployment regression, latency breach
severity levels, data-quality variants, cost overrun, and combinations),
rather than randomly sampled from a distribution. This is deliberate:
random sampling from a small simulated environment would mostly generate
easy, non-discriminating cases; hand-designed scenarios guarantee coverage
of the cases where naive baselines are known to fail structurally.

**What this proves:** the adaptive commander beats naive baselines on the
scenario families it was designed to stress-test.
**What this doesn't prove:** population-level statistical significance —
12 scenarios is small-n and deterministic (fixed seeds), not a large
random sample. Worth stating proactively; it's a maturity signal about the
evaluation, not a weakness to hide.

### Meta-harness significance testing

`WeightTuner` uses a one-tailed binomial test (`scipy.stats.binomtest`) to
decide whether an agent's observed success rate is significantly different
from the population baseline before adjusting any weight — `p < 0.05`,
minimum sample size `n ≥ 5` per agent. Below that sample size, no
adjustment is made regardless of the observed rate, to avoid tuning on
noise from a handful of incidents.

---

## Test Methodology

```bash
uv run pytest tests/                    # 465 tests, 1 skipped
```

| Test module | What it covers |
|---|---|
| `test_policy_comparison.py` | 4-policy comparison across 12 scenarios |
| `test_scenarios.py` | 30 named end-to-end scenario assertions |
| `test_verification_guardrails.py` | GuardrailChecker multi-dimensional checks |
| `test_utility_scorer.py` | UtilityScorer normalisation and ranking |
| `test_outcome_reward.py` | OutcomeReward calculation from before/after states |
| `test_phase9_10.py` | Per-type severity formulas and threshold search |
| `test_layer3_integration.py` | Full Commander end-to-end |
| `test_meta_harness_apply.py` | Tuned `ScoringWeights` → `TenantTierConfig` → live `UtilityScorer` |
| `test_commander_v3_bundle_writer.py` | Evidence bundles match what `EvidenceBundleAnalyzer` expects |
| `test_tune_weights_script.py` | Full analyze → tune → version → apply cycle against a real DB session |
| `test_weight_tuner.py`, `test_weight_tuner_significance.py` | Significance-gated weight adjustment |
| ... | 20+ additional test modules |

### Edge cases covered (`test_edge_cases.py`)

- all remediation agents fail → escalation logged
- agent execution timeout → fallback to next-ranked agent
- tied proposal scores → reconciliation debate picks winner
- concurrent incidents on same tenant → serialized
- concurrent incidents on different tenants → parallel
- memory tracks execution success/failure across retries

### Integration test approach

- `test_layer3_integration.py` — full 3-layer pipeline (observe → decide →
  verify) against real agents, mock telemetry.
- `test_integration_full_loop.py` — multi-tenant end-to-end; one test is
  skipped due to a shared cached DB engine, not flakiness — see
  [bug_diary.md](bug_diary.md#4-test-isolation-cached-global-db-engine-leaks-across-tests).
- `test_commander_v3_bundle_writer.py` — specifically asserts
  `bundle_writer` is opt-in and off by default, a regression test for the
  bug documented in
  [bug_diary.md](bug_diary.md#2-self-introduced-bundle-writer-defaulted-to-always-on).
