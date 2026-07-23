# Self-Healing ML Pipeline with Multi-Agent Negotiation

A production-grade ML reliability system where specialist agents autonomously detect, diagnose, and resolve model incidents through utility-scored negotiation, explicit guardrail verification, and offline weight optimization.

## What This Is

ML models fail in production for many reasons — data drift, quality degradation, latency spikes, cost overruns. This system resolves those incidents without human intervention:

1. **Detect** — 4 monitors watch for drift, data quality, latency, and cost incidents
2. **Observe** — `IncidentStateBuilder` builds a unified state snapshot from Prometheus metrics + SQLite policy DB, replacing hardcoded constants with real live values
3. **Propose** — 5 specialist agents each analyze the state and produce a `RemediationPlan` with an `expected_effect` vector (AUC delta, FNR delta, latency delta, cost delta)
4. **Score** — `UtilityScorer` converts each plan's effects into a utility ∈ [-1, +1] using per-incident-type weights; Commander selects the highest-utility agent
5. **Negotiate** — when top-2 utilities are within 10%, a LangGraph reconciliation debate decides the winner
6. **Execute** — winning plan runs; fallback chain tries next-best if it fails; escalation fires if all fail
7. **Verify** — `GuardrailChecker` checks explicit multi-dimensional thresholds after a stabilization window; auto-rollback triggers if guardrails fail AND regression detected
8. **Learn** — `EvidenceBundleAnalyzer` + `WeightTuner` run offline over historical traces to recommend weight adjustments; `CanaryWeightManager` rolls them out gradually with automatic rollback

## Architecture

```
Incident
  │
  ▼ Layer 1: OBSERVE
  IncidentStateBuilder ──► Prometheus + SQLite ──► IncidentState
  SeverityCalculator ──► per-type formula ──► (severity, SeverityBreakdown)
  │
  ▼ Layer 2: DECIDE
  5 Agents ──► RemediationPlan {action, expected_effect, confidence, risk}
  UtilityScorer ──► utility ∈ [-1,+1] per plan ──► rank
  [LangGraph reconciliation if top-2 within 10%]
  │
  ▼ Layer 2b: EXECUTE
  Winner.execute() ──► fallback chain ──► escalation
  │
  ▼ Layer 3: VERIFY
  stabilization window
  IncidentStateBuilder.build() ──► post-action IncidentState
  RewardCalculator ──► OutcomeReward (quality/cost/reliability/latency gains)
  GuardrailChecker ──► resolved? no_regression? should_rollback?
  [auto-rollback if should_rollback=True and rollback agent available]
  │
  ▼ LEARN (offline batch)
  EvidenceBundleAnalyzer ──► per-agent success + calibration
  WeightTuner ──► adjusted ScoringWeights (statistical significance gated)
  CanaryWeightManager ──► gradual rollout + auto-rollback
```

## Agents

| Agent | Incident Type | Key Signal | Expected Effect |
|-------|---------------|-----------|-----------------|
| `ThresholdAdjustmentAgent` | Drift, latency, cost | FPR/FNR cost model; 61-candidate search | `new_threshold`, `false_positive_rate_delta`, `false_negative_rate_delta` |
| `RetrainAgent` | Drift | AUC drop, data quality signal, model age | `auc_delta`, `cost_delta_usd` |
| `RollbackAgent` | Drift | `deployment_prob`, AUC regression vs previous version | `auc_delta`, `false_negative_rate_delta` |
| `FallbackAgent` | Latency, cost | Latency breach ratio, availability | `latency_p95_delta_ms`, `availability_delta` |
| `DataRepairAgent` | Data quality | Missing rate, duplicate rate, schema violations | `missing_rate_delta`, `false_negative_rate_delta` |

## Key Design Decisions

**Per-type severity formulas (Phase 9)**
Each incident type has named formula components with calibrated weights:
- Drift: `0.45·auc_drop + 0.35·drift + 0.20·volume`
- Data quality: `0.40·missing + 0.25·schema + 0.15·duplicates + 0.20·volume`
- Latency: `0.60·latency_ratio + 0.25·error_rate + 0.15·traffic`
- Cost: `0.70·budget_overrun + 0.30·cost_growth`

**Unified expected_effect vocabulary (Phase 10)**
All agents produce the same keys (`auc_delta`, `false_negative_rate_delta`, `latency_p95_delta_ms`, `cost_delta_usd`) so `UtilityScorer` can compare cross-agent without knowing which agent produced the plan.

**Utility scoring (Phase 11)**
Replaces raw confidence ranking. Each dimension of `expected_effect` is normalized to [-1, +1] and weighted by incident type:
- Drift: quality weight 0.40 → retrain wins on large AUC promises
- Latency: speed weight 0.40 → fallback wins on latency reduction
- Cost: cost weight 0.40 → cost-reducing agents win
- Data quality: reliability weight 0.35 → data repair on FNR improvement

**Explicit guardrail checking (Phase 13)**
Resolution is authoritative from `GuardrailChecker`, not from a reward score threshold:
- `resolved = AUC ≥ min_auc AND latency ≤ SLA AND missing ≤ max_missing`
- `no_regression = cost ≤ before×1.10 AND FNR ≤ before+0.05`
- `should_rollback = not resolved AND not no_regression`

## Quick Start

### Prerequisites
- Python 3.12+, `uv` package manager
- Docker + Docker Compose (Prometheus + Grafana)

### Setup
```bash
uv sync
uv run python scripts/train.py          # Train initial model on UCI credit dataset
```

### Run Demo
```bash
uv run python src/self_healing_pipeline/demo_week3.py
```

### API + Monitoring
```bash
docker-compose up -d                    # Prometheus + Grafana
uv run python -m uvicorn src.self_healing_pipeline.api.main:app --reload
```

- API: `http://localhost:8000/health`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (admin/admin)

### Replay Traffic
```bash
uv run python scripts/replay_traffic.py --duration 60
```

### Trigger Incident
```bash
uv run python scripts/trigger_incident.py
```

## Testing

```bash
uv run pytest tests/                    # 446 tests, 1 skipped
```

Test coverage:
- Unit tests: agents, severity, utility scoring, guardrails, reward
- Integration tests: 3-layer Commander end-to-end
- Scenario suite: 30 named deterministic scenarios (all 4 incident types)
- Meta-harness: analyzer, tuner, version control, canary

## Project Structure

```
src/self_healing_pipeline/
├── agents/              # 5 specialist agents (threshold, retrain, rollback, fallback, datarepair)
├── commander/           # CommanderV3, UtilityScorer, LangGraph reconciliation
├── gateway/             # Incident events, API gateway
├── meta_harness/        # Offline weight optimization (analyzer, tuner, canary, version_control)
├── monitors/            # Drift, quality, business monitors
├── observability/       # IncidentState, IncidentStateBuilder, SeverityCalculator, TelemetryCollector
├── verification/        # RewardCalculator (OutcomeReward), GuardrailChecker
└── api/                 # FastAPI endpoints
tests/
├── test_scenarios.py    # 30 named scenario evaluations
├── test_verification_guardrails.py
├── test_utility_scorer.py
├── test_phase9_10.py
├── test_outcome_reward.py
└── ...                  # 20+ additional test modules
```

## What the Meta-Harness Actually Does

The meta-harness is an **offline batch system**, not a real-time learning loop:

1. `EvidenceBundleAnalyzer` — reads JSON traces from resolved incidents; computes per-agent success rates and confidence calibration scores
2. `WeightTuner` — applies scipy t-tests; only adjusts weights when performance difference is statistically significant (p < 0.05)
3. `CanaryWeightManager` — rolls new weights to a configurable percentage of traffic; auto-rolls-back if success rate drops below threshold

> Note: The meta-harness tunes `ScoringWeights` (legacy scoring path). Phase 11's `UtilityScorer` uses `UtilityWeights` (per-dimension) from `TenantTierConfig`. These two weight systems are not yet wired together.

## Portfolio Signal

- **Multi-agent systems**: 5 specialist agents with distinct strategies, negotiation via utility scoring and LangGraph debate
- **Production observability**: Prometheus metrics, Grafana dashboards, per-incident evidence bundles
- **Explicit verification loop**: before/after state snapshots, multi-dimensional guardrails, auto-rollback
- **Cost optimization**: asymmetric FP/FN cost model, per-tenant thresholds, $ impact tracking
- **Statistical rigor**: significance-gated weight tuning, canary rollout, outcome-based reward (not estimated)
- **Test coverage**: 446 tests across unit, integration, scenario, and meta-harness layers

---

**Status:** All 17 phases complete. 446 tests passing.
