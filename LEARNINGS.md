# Engineering Learnings & War Stories

Difficulties encountered during build — useful for interviews, retros, and future reference.

---

## Multi-Process Prometheus Registry Isolation

**What happened:** API ran on port 8000, Prometheus scraped port 8001 (old API process). Replay pushed metrics to 8000. Prometheus always saw zero — completely empty dashboards.

**Root cause:** Each Python process has its own in-memory Prometheus registry. `Counter.inc()` in process A is invisible to process B's `/metrics` endpoint. Two API instances = two registries = writes and reads never touching the same data.

**Fix:** Restart the API on the exact port Prometheus scrapes. Kill the old process first. One source of truth per metric.

**Lesson:** When Prometheus shows nothing, ask: "Are the writer and the scrape target the same process?"

---

## Feature Importance Beats Domain Intuition for Drift Injection

**What happened:** Injected covariate drift via `LIMIT_BAL × 3` and `PAY_0 = 8` (max late payment). Expected model accuracy to drop and realized cost to spike. Instead, cost went *down*.

**Root cause:** The LightGBM model's top features were `BILL_AMT1`, `PAY_AMT2`, `AGE`, `LIMIT_BAL` — not `PAY_0`. Masking `PAY_0` had minimal effect. Additionally, enterprise has `fn_cost=$5000` vs `fp_cost=$200`. Making the model predict "default" for everyone (via `PAY_0=8`) eliminated the expensive FNs while only adding cheap FPs — net lower cost.

**Fix:** Read `model.feature_importances_` first, then target the features that actually matter. For label-aware drift, mask `BILL_AMT1=0`, `PAY_AMT*=100000`, `LIMIT_BAL=1M` for actual defaulters — these are the signals the model uses to identify risky customers.

**Lesson:** Never guess which features matter. Check feature importances before designing drift injection.

---

## Asymmetric Cost Models Have Non-Obvious Failure Modes

**What happened:** Enterprise tenant has `fn_cost=$5000`, `fp_cost=$200`. Tried to spike `cost_per_prediction` by injecting drift. Multiple injection strategies made cost *decrease* instead.

**Root cause:** With 15.4% positive rate and high `fn_cost`, the model's baseline cost is already dominated by FNs. Injections that made the model more "aggressive" (predict default for everyone) eliminated FNs at the cost of cheap FPs — reducing total cost. The model was accidentally optimizing business cost by being trigger-happy.

**Mathematical ceiling:** Max FN cost = `positive_rate × fn_cost = 0.154 × $5000 = $770/prediction`. Any threshold above $770 is unreachable. Set thresholds below the mathematical ceiling.

**Fix:** Threshold = $660 (above clean baseline $530-$630, below ceiling $770). Drift injection masks actual defaulters to look like ideal customers → FN rate spikes → cost approaches ceiling.

**Lesson:** Model the math before setting alert thresholds. Cost metrics have hard ceilings determined by dataset label distribution and cost asymmetry.

---

## Grafana 13 Ignores File Provisioning

**What happened:** Created JSON dashboard files in `docker/grafana/provisioning/dashboards/`. Grafana showed "Create your first dashboard" after login. All provisioning silently ignored.

**Root cause:** Grafana 13 migrated to "unified storage" for dashboards. File-based provisioning from the `provisioning/` directory is no longer applied on startup. No error messages — it just skips it.

**Fix:** Import dashboards via REST API after Grafana starts:
```bash
curl -X POST http://admin:admin@localhost:3000/api/dashboards/import \
  -H 'Content-Type: application/json' \
  -d "{\"dashboard\": $(cat dashboard.json), \"overwrite\": true}"
```

**Lesson:** Always verify Grafana version behavior. File provisioning documentation is written for Grafana 8-10. Grafana 13 broke the contract silently.

---

## Mock Telemetry Blocks All Real Agent Decisions

**What happened:** All 4 incident types wired up. Only ThresholdAdjustmentAgent ever won. Other agents never appeared on leaderboard.

**Root cause:** `CommanderV3` used `TelemetryCollector(use_mock=True)`. Mock returns fixed values: `missing_rate=0.08` (below DataRepair's 0.15 threshold), `auc=0.75` (above RetrainAgent's 0.70 threshold), etc. No matter what real Prometheus showed, agents' `can_handle()` check always saw mock values → only threshold agent was ever eligible.

**Fix:** Pass `incident.payload` into `_get_agent_state()` which overrides mock state with real metric values from the incident. Surgical fix — no need to rewire TelemetryCollector to Prometheus.

**Lesson:** Mocks baked into production code paths are invisible until you trace why the "wrong" branch always wins. Check `can_handle()` return values before debugging agent logic.

---

## Per-Tenant Drift Baselines Are Not Optional

**What happened:** Drift scores wildly different per tenant — enterprise always showed high drift, standard and free showed none, even with identical data.

**Root cause:** Drift was computed as deviation from *overall population mean*. Enterprise customers have much higher `LIMIT_BAL` (high-value B2B). Their values deviated strongly from the population mean even in baseline operation. Standard customers clustered near the mean.

**Fix:** Compute `(mean, std)` per tenant separately from their own test rows. Each tenant's drift baseline is 0 by definition. Drift > 1.5σ means deviation from *that tenant's own normal*, not from the global average.

**Lesson:** Any metric comparing against a global baseline in a multi-tenant system will be biased toward tenants whose distribution differs most from the average. Always compute baselines per-segment.

---

## Meta-Harness: What It Actually Does vs. What Was Claimed

**Original claim:** "Learns by analyzing evidence bundles and auto-tuning decision weights."

**Accurate description:**

The meta-harness runs as an **offline batch job** — not in real-time during incident handling. It has four moving parts:

1. `CommanderV3` — writes an evidence bundle (`traces/run_<incident_id>/evidence_bundle.json`) after every incident it resolves, when constructed with a `bundle_writer` (wired into `demo.py` and `scripts/trigger_incidents.py`; off by default so unit tests don't touch the real `traces/` dir).
2. `EvidenceBundleAnalyzer` — reads those bundles, aggregates per-agent success rates and confidence calibration accuracy.
3. `WeightTuner` — applies scipy t-tests (`p < 0.05`) to determine statistical significance before adjusting `ScoringWeights`. High-performing agents (statistically significant) get a higher confidence weight; low performers are reduced. No gradient descent, no neural networks.
4. `CanaryWeightManager` — gradually rolls out new `ScoringWeights` to a configurable percentage of incidents (`canary_percentage`). Watches live success rate; if it drops below `rollback_threshold` (default 0.95), rolls back to the previous version automatically. Implemented and tested, but not yet invoked by the production entrypoint — `scripts/tune_weights.py` applies directly, no canary gate.

**Architectural gap, closed:**

Phase 11 introduced `UtilityScorer` with `UtilityWeights` (per-dimension: quality, cost, reliability, speed, confidence, risk) — the weights the Commander actually reads at decision time, via `UtilityScorer.weights_from_tier_config()` on the tenant's `TenantTierConfig` row. The meta-harness tunes the legacy `ScoringWeights` dataclass (business_value, confidence, risk_inverse, cost_efficiency, time_inverse, historical_success), which pre-dates the utility scoring refactor.

For a while these were two disconnected systems that happened to share column names. A first pass (`meta_harness/apply.py::sync_tuned_weights()`) closed the mapping but was only exercised against a throwaway in-memory DB inside `demo_meta_harness.py` — never the real one. Two things were still missing for it to matter in practice: nothing ran the tune→apply step against the production DB, and `CommanderV3`'s live path never wrote evidence bundles in the first place (`BundleWriter` existed but was only wired into the old, unused `commander.py`). Both are now fixed: `CommanderV3` writes bundles when given a `bundle_writer`, and `scripts/tune_weights.py` runs analyze → tune → version → apply against the real DB, discovering tenants from actual `IncidentHistory` rows rather than a hardcoded list. Verified live: after tuning, `demo.py`'s winning agent flipped from `retrain` to `threshold` for the same incident.

Two round-trip gaps remain, tracked but not fixed: `historical_success_weight` has a DB column but no `UtilityWeights` field reads it back, and `UtilityWeights.reliability` has no DB column at all (always falls back to the per-incident-type default). Tuning still nudges those two dimensions on paper; they just don't reach live scoring.

**Lesson:** Batch-offline learning + canary rollout is architecturally cleaner than online gradient-based weight updates for an inference-time system — but "the wiring exists in code" and "the wiring runs against production data" are different claims. The gap here survived multiple phases because per-component unit tests (all using in-memory DBs and mocked evidence) couldn't see it; it only showed up when tracing the real data path end to end.

---

## Dead Code Audit Findings (2026-07)

Features built but never wired into the live pipeline:

| Component | Status | Impact |
|-----------|--------|--------|
| `false_positives_total` / `false_negatives_total` counters | Defined, never incremented | FP/FN invisible in Prometheus |
| `cost_per_prediction` Gauge | Defined, never `.set()` | COST_THRESHOLD incidents can't fire |
| `data_duplicate_rate` / `data_schema_violations` | Hardcoded `0.0`/`0` in replay | Always show clean data |
| `monitors/business.py` | Computes FP/FN/cost correctly, results go nowhere | Not wired into trigger or commander |
| `monitors/quality.py` / `monitors/drift.py` | Exported from `__init__`, never imported in live code | Dead code |
| `SystemMetrics.cpu_usage` / `memory_usage` | In telemetry dataclass, never pushed by replay | Always mock values |

Fix priority: `cost_per_prediction` first (blocks COST_THRESHOLD incidents) → FP/FN counters → wire monitors.
