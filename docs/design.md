# Design

The engineering decisions behind the system and why the alternatives were
rejected. For the concrete mechanics (formulas, schemas, code paths), see
[implementation.md](implementation.md).

---

## Competing policies, not a chat between agents

The agents in this system do not chat or delegate tasks to one another.
Each agent represents a distinct remediation strategy with its own
eligibility rules, state inputs, expected effects, confidence estimate,
execution cost, and operational risk. The commander converts each proposal
into a utility score under the tenant's objective weights, selects the
highest-scoring agent, and verifies the outcome after execution.

**Why not one decision function?** Each remediation type has genuinely
different eligibility logic and inputs — threshold needs precision/recall,
retrain needs drift + label volume, rollback needs deployment age, fallback
needs error rate, data-repair needs missing/schema rates. Cramming into one
function means one giant if-tree. Separate agents each own their own
`can_handle()` gate + `analyze()` — easier to test and reason about in
isolation. Tradeoff: coordination complexity moved to the commander instead
of being distributed across branches of a monolith.

**Why not always retrain when quality drops?** Tested this as a baseline
(`always_retrain` policy). It loses specifically on deployment
regressions — retraining doesn't fix a bad deploy, it trains a new model on
the same environment and does nothing for the actual cause. 50% resolution
vs. 75% for the adaptive commander. This is the clearest evidence that
multi-agent competition was necessary, not just an architectural
preference — see [evaluation.md](evaluation.md) for the numbers.

---

## Utility scoring instead of confidence-only ranking

Early version picked the agent most "sure of itself." Confidence measures
"how sure is this agent in itself," not "is this the right fix for this
tenant." Replaced with a multi-objective utility function: quality +
reliability + cost + speed + confidence − risk, weighted per incident type
and per tenant. This was the single highest-leverage design change in the
project, proven by the baseline comparison (75% vs. 58% resolution rate
against a highest-confidence baseline). Formula and weight tables:
[implementation.md](implementation.md#commander-scoring).

---

## LLM used only as a tie-breaker

Only invoked when top-2 utility scores are within 10% — a reconciliation
debate, not a decision-maker. Deliberately not used for core
decision-making: non-deterministic LLM calls in the primary decision path
would make the system hard to test and hard to explain to an auditor.
Heuristics decide; the LLM only breaks close ties. The system runs fully
without an API key.

---

## Per-incident-type severity formulas instead of one score

Drift, data-quality, latency, and cost incidents have different signals in
different units (AUC drop vs. missing-rate vs. latency-ratio). Forcing one
formula would mean either ignoring signal or badly normalizing everything
onto one scale. Chose 4 separate weighted formulas, each with named, stored
components — so severity is auditable, not a black-box number.

---

## Guardrails are authoritative, reward is descriptive

The resolution verdict comes from `GuardrailChecker`, not from crossing a
reward threshold. Reward is useful for ranking and for the meta-harness,
but it's a soft, weighted signal — using it as the actual accept/reject
gate would let a system with several small wins mask one guardrail
violation that actually matters (e.g., a real SLA breach). Guardrails are
explicit boolean checks against policy limits; there's no way for them to
average away a violation.

## Reward from measured deltas only, never from agent self-report

Agents predict their own expected effect during proposal; naively trusting
that would let an agent "grade its own homework." Reward is computed
strictly from measured before/after state — agent predictions are used
only for initial ranking, never for the outcome signal.

---

## Offline batch tuning instead of online gradient updates

The meta-harness re-tunes decision weights from accumulated outcomes as a
scheduled batch job, gated by significance testing, with canary rollout —
not a continuous online-learning loop. Batch-offline + canary is
architecturally cleaner than online gradient-based weight updates for an
inference-time decision system: it's auditable (you can point at exactly
which incidents produced a given weight change), reversible (canary can
roll back), and doesn't risk a single bad incident perturbing live scoring
mid-stream. The tradeoff is documented explicitly, not left implicit:
"auto-tuning" here means "batch job + canary," not a live feedback loop.
See [implementation.md](implementation.md#meta-harness) for the pipeline
and [bug_diary.md](bug_diary.md) for how long it took to actually reach
production.

---

## What's real vs. simulated, and why that boundary was drawn there

See [methodology.md](methodology.md#scope-and-simulation-boundaries) for
the full real-vs-simulated table and the reasoning per boundary — the
short version: anything that determines *whether the decision logic is
correct* (model training, inference, metric computation, state
transitions) is real; anything that would just require infrastructure
scale without changing the decision logic (traffic source, cloud billing,
production-scale deployment) is simulated.
