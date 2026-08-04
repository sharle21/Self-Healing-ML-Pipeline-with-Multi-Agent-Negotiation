# Interview Notes — Self-Healing ML Pipeline

Personal recall doc. Not the README. Format: Q you might get asked → short answer
you should be able to say cold. Read this the morning of, not the night before.

> **If the interviewer remembers only one thing, it should be:**
> *"Execution isn't the finish line — verification is. I built the layer
> that checks whether a fix actually worked, not just whether it ran."*

---

## 1. Design decisions + alternatives considered

**Q: Why utility scoring instead of just picking the most confident agent?**
A: Tried highest-confidence first. Confidence measures "how sure is this agent
in itself," not "is this the right fix for this tenant." Replaced with a
multi-objective utility function: quality + reliability + cost + speed +
confidence − risk, weighted per incident type and per tenant. Proved it beats
confidence-only in the baseline comparison (75% vs 58% resolution rate).

**Q: Why 5 separate agents instead of one decision function?**
A: Each remediation type has genuinely different eligibility logic and inputs
(threshold needs precision/recall, retrain needs drift + label volume,
rollback needs deployment age, fallback needs error rate, data-repair needs
missing/schema rates). Cramming into one function means one giant if-tree.
Separate agents = each owns its own `can_handle()` gate + `analyze()` — easier
to test and reason about in isolation. Tradeoff: coordination complexity moved
to the commander instead.

**Q: Why not just always retrain when quality drops?**
A: Tested this as a baseline (`always_retrain` policy). Loses on deployment
regressions — retraining doesn't fix a bad deploy, it trains a new model on
the same environment and does nothing for the actual cause. 50% resolution
vs 75% for adaptive. This is the single best story for "why multi-agent
was necessary, not just cool."

**Q: Why SQLite instead of Postgres?**
A: Portability + zero-setup local reproducibility for a solo/demo project.
Docker-compose does include a Postgres service for a "prod-shaped" path, but
default dev/test path is SQLite. Interfaces (SQLAlchemy ORM) don't change if
swapped — that was a deliberate constraint, not an afterthought.

**Q: Why is LLM (Claude/LangGraph) usage minimal?**
A: Only invoked when top-2 utility scores are within 10% — a tie-break
reconciliation debate. Deliberately not used for core decision-making,
because non-deterministic LLM calls in the primary decision path would make
the system hard to test and hard to explain to an auditor. Heuristics decide;
LLM only breaks close ties. System runs fully without an API key.

**Q: Why per-incident-type severity formulas instead of one score?**
A: Drift, data-quality, latency, and cost incidents have different signals in
different units (AUC drop vs missing-rate vs latency-ratio). Forcing one
formula would mean either ignoring signal or badly normalizing everything
into one scale. Chose 4 separate weighted formulas, each with named,
stored components — so severity is auditable, not a black-box number.

---

## 2. Implementation details (the "how" — say these precisely, don't hand-wave)

- **Commander scoring formula:**
  `utility = quality_w·auc_delta_norm + reliability_w·(-fnr_delta_norm) + cost_w·(-cost_delta_norm) + speed_w·(-latency_delta_norm) + confidence_w·confidence − risk_w·risk`,
  clipped to [-1, +1]. Normalizers come from the live `IncidentState`, not
  hardcoded constants (e.g. cost_ref = 20% of current cost/1000 predictions).

- **Reconciliation trigger:** top-2 utility scores within 10% margin →
  LangGraph debate compares guardrails, historical success, predicted side
  effects before picking.

- **Verification loop (7 steps):** snapshot pre-action state → execute →
  wait for stabilization window → re-query Prometheus → rebuild post-action
  `IncidentState` → run `GuardrailChecker` (resolution guardrails: AUC/latency/
  missing-rate; regression guardrails: cost ≤ 1.10×before, FNR ≤ before+0.05)
  → accept or auto-rollback.

- **Reward is computed only after verification**, from real before/after
  deltas — not from agent's self-reported expected_effect. This was
  deliberate: letting agents grade their own homework would make reward
  meaningless.

- **Concurrency:** per-tenant `asyncio.Lock` in `gateway.py` — two incidents
  on the same tenant serialize (can't both mutate the same model/config),
  different tenants run in parallel. Not one global lock.

- **Meta-harness pipeline:** `EvidenceBundleAnalyzer` (reads JSON traces,
  computes per-agent success rate + confidence calibration) → `WeightTuner`
  (binomial significance test, p<0.05, before adjusting any weight) →
  `CanaryWeightManager` (rolls new weights to a % of traffic, hash-routed
  deterministically by incident_id, auto-reverts if canary underperforms
  stable by more than `rollback_threshold`).

- **Weight write-back (`meta_harness/apply.py`):** `ScoringWeights` fields map
  1:1 to `TenantTierConfig` DB columns (`business_value_weight`,
  `confidence_weight`, `risk_inverse_weight`, `cost_efficiency_weight`,
  `time_inverse_weight`, `historical_success_weight`). `sync_tuned_weights()`
  loads-or-creates the row and commits. This is what makes tuning actually
  reach `commander_v3.py`, which reads weights via
  `UtilityScorer.weights_from_tier_config()`.

---

## 3. Metrics + benchmarks (numbers you must not fumble)

- Dataset: UCI Default of Credit Card Clients, ~30,000 rows, 23 features.
- Model: LightGBM binary classifier. Baseline validation ROC-AUC ≈ **0.77**.
- Evaluation: 12 deterministic incident scenarios × 5 policies = **60 trials**
  (`tests/test_policy_comparison.py`).

| Policy | Resolution | Mean Reward | Guardrail Violations |
|---|---:|---:|---:|
| Adaptive commander | **75%** | **0.346** | 50% |
| Highest confidence | 58% | 0.285 | 50% |
| Always retrain | 50% | 0.186 | 75% |
| Fixed priority | 33% | 0.143 | 67% |
| Cheapest eligible | 33% | 0.143 | 67% |

- **465 tests passing, 1 skipped** (`uv run pytest tests/`).
- Reward components example: quality_gain 0.52, reliability_gain 0.18,
  cost_gain 0.04, resolution_score 0.80, exec_cost_penalty −0.04,
  time_penalty −0.01, regression_penalty 0.0 → total 0.65.

**If asked "is this statistically significant" —** be honest: 12 scenarios is
small-n, deterministic (fixed seeds), designed to cover distinct failure
modes rather than a large random sample. It demonstrates the mechanism beats
naive baselines on the scenarios it was designed to stress-test; it is not a
claim of population-level significance. Say this proactively if pushed —
it's a maturity signal, not a weakness to hide.

---

## 4. Limitations (say these before they ask — controls the framing)

- Traffic is historical replay, not live production traffic.
- UCI dataset is small and tabular — results may not generalize to
  high-dimensional or streaming data.
- Drift detection uses normalized mean shift, not PSI/KS-test — simpler,
  works here, wouldn't scale to more complex feature spaces.
- Cost values come from a documented simulation model, not real cloud billing.
- Labels assumed available immediately — real credit-risk labels (defaults)
  surface over months. Delayed-label handling isn't modeled.
- SQLite is fine for local demo, not high-concurrency production.
- Data-repair agent operates on the replay environment, not a real upstream
  warehouse.
- Meta-harness tuning is **global**, not per-tenant — see mistakes below.

---

## 5. Mistakes made (own these — this is what separates senior from junior in an interview)

1. **Built the meta-harness and the live commander as two disconnected
   systems — twice, because the first fix was also disconnected.**
   `WeightTuner` computed and version-controlled new `ScoringWeights`
   correctly. `commander_v3.py` scored live proposals using `UtilityWeights`
   from `TenantTierConfig`. The two shared a naming convention
   (`business_value_weight` etc.) and nothing else — nothing ever wrote
   tuned weights back into the DB row the commander actually reads. The
   offline learning loop ran end-to-end, produced sensible output, had a
   demo script and tests — and never once influenced a live decision.
   Found this by literally tracing "where does this value get read" backward
   from the commander, not from a bug report. **First fix:**
   `meta_harness/apply.py::sync_tuned_weights()` + `tests/test_meta_harness_apply.py`
   (verifies tune → apply → commander sees new value).
   That fix was correct in isolation but its only caller, `demo_meta_harness.py`,
   ran it against a throwaway in-memory SQLite engine — so the wiring existed
   in code, passed its own test, and *still* never touched the real database.
   Same mistake, one layer deeper. Digging further surfaced a second gap:
   the evidence bundles the analyzer reads were never written by the live
   commander path at all (`BundleWriter` existed, but only the old, unused
   `commander.py` called it) — so even a production entrypoint would have
   analyzed zero real incidents. **Second fix:** wired `bundle_writer` into
   `CommanderV3` (opt-in, off by default) and built `scripts/tune_weights.py`
   to run analyze → tune → version → apply against the real DB, discovering
   tenants from actual `IncidentHistory` rows. While fixing this I also
   caught myself defaulting `bundle_writer` to always-on, which silently
   wrote real trace files into the project's `traces/` directory on every
   unit test run — 23 stray folders before I noticed via `git status`.
   Verified the whole thing live: fired the same incident 3x (`retrain` wins
   each time), ran the tuning script, fired it again — winner flipped to
   `threshold`.
   *Why it happened:* built bottom-up in phases (agents → commander → verify
   → reward → meta-harness) and never wrote an integration test that
   spanned the full loop end-to-end until asked to explain it out loud. The
   first fix repeated the pattern at a smaller scale: fixed the mapping,
   tested the mapping, didn't ask "does anything actually call this against
   real data."
   *Lesson:* a system "working" phase-by-phase with passing unit tests is not
   the same as the phases being wired together — and "I added a test for the
   fix" is not the same as "I verified the fix reaches production," if that
   test also uses a fake DB. The only way I actually caught the second gap
   was by asking "has this ever run, even once, against real data" and then
   checking.

2. **Historical success is tracked per-agent, not per-agent-per-incident-type.**
   If retrain succeeds often on severe drift, that inflates its global score,
   which then also nudges it (weakly — eligibility gates still apply) toward
   winning mild-drift cases where threshold was the cheaper right call.
   Not caught until explicitly asked "what if the history is skewed toward
   one incident type." Still unresolved — see "what's next."

3. **Global weight tuning overwrites per-tenant customization.** After fixing
   mistake #1, the first version of the fix applied one tuned weight set to
   *every* tenant. That's wrong: `TenantTierConfig` exists specifically to let
   a free-tier tenant weight cost differently than an enterprise tenant.
   Applying one global tuning result erases that distinction. Flagged
   honestly rather than hidden — it's in Limitations and "what's next," not
   swept under "meta-harness works now."

4. **Test suite pollution from a cached global DB engine.** `db/session.py`'s
   `get_engine()` is `@lru_cache`d and points at the real on-disk
   `pipeline.db` by default. A test that calls
   `Base.metadata.create_all(get_engine())` shares that file/engine across
   the whole test run — so a multi-tenant isolation test became flaky
   depending on run order and got skipped rather than fixed at the time.
   Root cause only surfaced when tracing session handling for the weight
   write-back fix. New tests use their own in-memory engine to avoid this;
   the original skipped test is still skipped.

---

## 6. What I'd build next (priority order, and why each one)

1. **Per-tenant meta-harness tuning.** Segment evidence-bundle analysis by
   tenant_id before running significance tests, so tuning respects each
   tenant's own history instead of pooling everyone together. Directly fixes
   mistake #3 and is the most "production-realistic" gap left.
2. **Condition historical_success on incident subtype/severity**, not just
   agent identity — fixes mistake #2.
3. **Fix the DB engine test isolation issue** so `test_multi_tenant_isolation`
   can run instead of being permanently skipped.
4. **Real drift detection (PSI/KS-test)** to replace normalized mean-shift —
   needed before this could handle higher-dimensional or streaming features.
5. **Delayed/partial label modeling** — real credit-risk labels are not
   immediate; current system assumes they are.
6. **Thin React/TypeScript frontend** over the existing typed API responses
   (`/incidents/recent`, `/agents/summary`) — lowest priority, purely
   presentational, backend already returns structured data for it.

---

## 7. If I freeze, say this

"This project detects when an ML model is failing in production, has five
specialist strategies compete to fix it, picks the best one for the specific
tenant's priorities using a weighted scoring function instead of one fixed
rule, and — the important part — actually verifies the fix worked and rolls
back automatically if it didn't. Tested against four dumber strategies across
60 trials, mine resolved 75% of incidents versus 33-58% for the others."

Then let them steer with follow-ups. Don't try to say everything at once.
