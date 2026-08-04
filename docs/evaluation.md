# Evaluation

## Scenarios

The system is evaluated across 12 deterministic incident scenarios with
defined pre- and post-action states per agent type. Scenario families:

- Severe drift (model 30 days old)
- Mild drift (threshold sufficient, retrain wasteful)
- Recent-deployment regression (rollback optimal)
- Mild and severe latency breach
- High missing-rate data quality
- Schema-violation data quality
- Cost overrun
- Drift with concurrent data quality issues
- Post-upgrade latency spike
- Near-threshold drift (AUC already above minimum)
- Cost and quality conflict

Each scenario is deterministic (fixed seeds), designed to stress-test a
distinct failure mode rather than sample a large population — see
[Metrics](#metrics) below for what that does and doesn't prove.

## Metrics

| Metric | Definition |
|---|---|
| Resolution rate | % scenarios where all policy guardrails pass after action |
| Mean reward | Average outcome-based reward across scenarios |
| Guardrail violation rate | % scenarios with at least one guardrail violation |

---

## Baseline Comparison

12 scenarios × 5 policies = 60 deterministic trials
(`tests/test_policy_comparison.py`).

| Policy | Resolution | Mean Reward | Guardrail Violations |
|---|---:|---:|---:|
| **Adaptive commander** | **75%** | **0.346** | **50%** |
| Highest confidence | 58% | 0.285 | 50% |
| Always retrain | 50% | 0.186 | 75% |
| Fixed priority | 33% | 0.143 | 67% |
| Cheapest eligible | 33% | 0.143 | 67% |

"Cheapest eligible" always picks the lowest dollar-cost eligible agent
(threshold at $0 whenever eligible) — it ties fixed-priority here because
both gravitate to threshold first; it fails severe-drift and
deployment-regression scenarios where only retrain or rollback fully
resolve the incident.

### Key failure modes of baselines

| Scenario | Adaptive | Always Retrain | Fixed Priority |
|---|---|---|---|
| Deployment regression | rollback ✓ (+0.65) | retrain ✗ (−0.08) | threshold ✗ (+0.09) |
| Severe latency breach | fallback ✓ (+0.70) | threshold ✗ (+0.11) | threshold ✗ (+0.11) |
| Post-upgrade latency | fallback ✓ (+0.70) | threshold ✗ (+0.09) | threshold ✗ (+0.09) |
| High missing rate | data_repair ✓ (+0.61) | data_repair ✓ (+0.61) | fallback ✗ (−0.05) |

The pattern: single-strategy baselines each have a scenario family where
they're structurally wrong — retrain can't fix a bad deploy, fixed-priority
never reaches data_repair or fallback when threshold is eligible but
insufficient. The adaptive commander's advantage isn't raw score, it's not
being wrong in a *specific, predictable way* for an entire scenario family.

## Statistical caveat

12 scenarios is small-n and deterministic, not a large random sample. It
demonstrates the mechanism beats naive baselines on the scenarios it was
designed to stress-test; it is not a claim of population-level
significance. Worth saying proactively — it's a maturity signal, not a
weakness to hide.
