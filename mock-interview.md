# Mock Interview — Self-Healing ML Pipeline (1 hour, deep-dive)

Simulated FDE/ML-systems interview. Escalating depth: intro → architecture →
component deep-dives → gotchas/tradeoffs → behavioral → closing. Each
question has a model answer grounded in the actual code, not invented.

> **If the interviewer remembers only one thing, it should be:**
> *"Execution isn't the finish line — verification is. I built the layer
> that checks whether a fix actually worked, not just whether it ran."*

---

## Part 0 — Fundamentals / first principles (10 min)

Interviewers use these to check you understand *why* the tools you used are
the right tools, not just that you can operate them. Each answer ties back
to a real moment in this project — don't answer in pure textbook terms.

**Q0.1. What is ROC-AUC, and why is it the metric you picked over accuracy?**
A: "ROC-AUC measures how well the model ranks positives above negatives
across every possible decision threshold, not just one. Accuracy at a single
threshold is misleading here because the UCI credit-default dataset is
imbalanced — most customers don't default — so a model that always predicts
'no default' gets high accuracy and is useless. AUC is threshold-independent,
which matters directly for this project because the threshold agent's whole
job is to move the decision threshold — I need a metric that still means
something as that threshold changes."

**Q0.2. Explain precision and recall in this project's actual terms, not
abstractly.**
A: "Precision: of the customers I flag as likely to default, how many
actually do. Recall: of the customers who actually default, how many I
catch. In credit risk, missing a real default (low recall / false negative)
usually costs more than wrongly flagging a safe customer (false positive) —
that's why `false_negative_cost` is a distinct tenant-configured value in
this system, and why several agents' expected effects are tracked in terms
of `false_negative_rate_delta` specifically, not just accuracy."

**Q0.3. What's the difference between covariate drift and concept drift, and
which does this project detect?**
A: "Covariate drift is the input feature distribution shifting while the
true relationship between features and outcome stays the same — e.g. income
levels in the traffic shift, but a given income still predicts default the
same way. Concept drift is the relationship itself changing — the same
income now means something different for default risk (say, after a
recession). This project's fault injection and drift scoring is built around
covariate-style shifts (`max_feature_drift`, per-feature distribution
change) — it does not model true concept drift, and I'd say that plainly if
asked whether it's covered."

**Q0.4. Why train/validation/replay split instead of just train/test?**
A: "Training data fits the model. Validation/baseline-calibration data is
used once, to set the operating point — `baseline_auc`, the initial decision
threshold, latency benchmarks — values the whole system treats as ground
truth for 'normal.' The replay partition is what actually flows through the
live `/predict` API to generate runtime telemetry. Using the same data for
calibration and replay would let the system's baseline quietly overfit to
exactly the data used to judge it later — three-way split keeps that
honest."

**Q0.5. What is statistical significance, and where does it actually gate a
decision in this system?**
A: "It's the probability an observed effect could occur by chance if there
were really no effect. The `WeightTuner` uses a one-tailed binomial test
(`scipy.stats.binomtest`) to check whether an agent's success rate is
significantly above or below the population mean, at p < 0.05, before
adjusting any weight — and requires at least 5 samples before even running
the test. Without that gate, a single lucky or unlucky agent run could
swing the commander's future decisions on pure noise."

**Q0.6. Why normalize deltas before scoring them, instead of comparing raw
numbers?**
A: "Because raw units aren't comparable across dimensions — an AUC delta of
0.05 and a latency delta of 5ms are different scales entirely, and you can't
add them meaningfully. Every delta in the utility formula is divided by a
live reference (10% AUC gain = 1.0, cost_ref = 20% of current cost, etc.)
before being weighted and summed, so the six dimensions are on a comparable
[-1, 1] scale before they're combined."

**Q0.7. Why REST/FastAPI here instead of gRPC or a message queue?**
A: "The prediction path is a synchronous request/response — send features,
get a probability back — which is exactly what REST is for, and FastAPI
gives me typed request/response models via Pydantic almost for free, which
matters because those same models double as the OpenAPI schema a frontend
could generate types from. gRPC would make sense if this were
service-to-service at high throughput with strict latency budgets; a message
queue would make sense if predictions were async/batch. Neither describes
this system's actual traffic pattern."

**Q0.8. Why does Prometheus use a pull model, and why does that matter for
this system?**
A: "Prometheus scrapes metrics from the app on an interval, rather than the
app pushing metrics to it. That decouples metric collection from the
prediction request path — instrumenting a counter in `prediction.py` doesn't
add a network call to the hot path, it just increments an in-memory value
that Prometheus reads later. That's part of why 'measured latency' in this
project is real: the act of recording a metric doesn't itself distort the
latency being measured."

**Q0.9. What is a race condition, concretely, and where could one occur
here if you hadn't guarded against it?**
A: "Two incidents on the same tenant executing remediation concurrently —
say, threshold-agent and rollback-agent both winning separate incidents at
nearly the same moment — could interleave writes to the same
`TenantTierConfig` or model-registry row, leaving it in a state neither
agent intended. Guarded with a per-tenant `asyncio.Lock` in the gateway, so
same-tenant incidents serialize instead of racing."

**Q0.10. In one sentence, what makes a system 'self-healing' versus just
'automated'?**
A: "Automated means it takes an action without a human; self-healing means
it also verifies the action worked and corrects itself if it didn't — the
verification-and-rollback loop is the actual distinction, not the action
itself."

---

## Part 1 — Warm-up (5 min)

**Q1. Walk me through this project in 2 minutes.**
A: "This is a closed-loop ML operations system. It replays a real dataset
through a live FastAPI prediction endpoint to generate real telemetry —
latency, AUC, drift, data quality. When telemetry crosses a tenant's
threshold, an incident detector flags it. Five specialist agents — threshold
adjustment, retrain, rollback, fallback, data repair — each independently
propose a fix if they're eligible. A commander scores every proposal with a
multi-objective utility function weighted per tenant and per incident type,
picks the best one, executes it, waits, re-measures, and checks explicit
guardrails. If the fix didn't actually work, it auto-rolls-back. All of this
is measured against four naive baseline strategies across 60 deterministic
trials — mine resolves 75% of incidents, the best naive baseline gets 58%."

**Q2. Why did you build this?**
A: "Production ML models degrade silently — drift, broken pipelines, bad
deploys — and by the time a human notices via a downstream business metric,
damage is done. I wanted to build the decision-making layer that most
monitoring dashboards stop short of: not just detect, but decide and verify."

---

## Part 2 — Architecture (10 min)

**Q3. Draw the data flow for me, verbally.**
A: "Dataset replay hits `/predict` → Prometheus scrapes real metrics →
telemetry collector builds a snapshot → incident detector compares it to
tenant baselines/SLAs → if breached, `IncidentState` is built → the 5 agents
each check `can_handle()` and if eligible produce a `RemediationPlan` →
`UtilityScorer.rank()` scores every plan → commander executes the winner →
guardrail check → outcome and reward persisted → offline meta-harness later
re-tunes the commander's weights from accumulated outcomes."

**Q4. Why not just use one model to decide what to do — like a classifier
that maps incident features to an action?**
A: "Two reasons. One, I don't have enough real incident-outcome pairs to
train that reliably — this is a portfolio project with synthetic incidents,
not years of production data. Two, and more important: a black-box
classifier can't explain *why* it picked an action, and it can't be audited
per-dimension. The utility-scoring approach is explicit — I can point to
exactly which weight (quality/cost/reliability/speed/risk) drove the
decision, and that's what let me build a comparison against naive baselines
in the first place. If I ever did have enough outcome data, the natural next
step is exactly what the meta-harness does: use the data to *tune the
weights*, not replace the scoring function with an opaque model."

**Q5. Why separate agents instead of one function with a big if/else?**
A: "Each remediation type has genuinely different eligibility conditions and
inputs — rollback cares about deployment age, retrain cares about drift and
label volume, fallback cares about error rate. Separating them means each
agent's `can_handle()` and `analyze()` can be unit-tested in isolation, and
adding a 6th remediation type later doesn't touch the other 5. The tradeoff
is I pushed coordination complexity into the commander — that's a deliberate
trade, not a free lunch."

---

## Part 3 — Deep dive: scoring and decision-making (15 min)

**Q6. Walk me through the utility formula precisely. Don't summarize — the
actual formula.**
A: "`utility = quality_w·clip(auc_delta/0.10) + reliability_w·clip(-fnr_delta/fnr_ref)
+ cost_w·clip(-cost_delta/cost_ref) + speed_w·clip(-latency_delta/sla)
+ confidence_w·plan.confidence − risk_w·plan.risk`, then clipped to [-1, 1].
Every delta is normalized against something live — `cost_ref` is 20% of the
tenant's current cost per 1000 predictions, `fnr_ref` is the tenant's current
false-negative rate, not a hardcoded constant. That matters because a $5
cost delta means something different to a free-tier tenant than an
enterprise one."

**Q7. What happens when two agents score almost identically?**
A: "If the top-2 scores are within a 10% margin, it triggers a LangGraph
reconciliation debate — a second pass that compares guardrail history,
historical success, and predicted side effects before finalizing. This is
the only place an LLM touches the decision path; the rest is deterministic
heuristics."

**Q8. Why weight confidence at all if you already have a whole utility
function? Isn't that redundant?**
A: "Confidence is the agent's own self-assessment of how sure it is its
prediction is right — it's one input among six, not the decider. Early
version *was* confidence-only ranking, and it lost to the adaptive utility
version in testing — 58% vs 75% resolution. I kept confidence as a small
weighted term because it's still informative, just not sufficient alone."

**Q9. Your baseline comparison — walk me through the weakest baseline and
why it fails.**
A: "'Fixed priority' always tries agents in a hardcoded order — threshold
first, then retrain, then rollback, etc, regardless of incident type. It
gets 33% resolution. It fails because it applies the same static preference
to every incident — a severe latency spike needs fallback, not
threshold-tuning, but fixed-priority tries threshold every time since it's
always eligible. 'Cheapest eligible' ties it at 33% for a similar reason —
picking on cost alone with no context about whether cheap actually works."

**Q10. Is 12 scenarios / 60 trials statistically significant?**
A: "No, and I'd say that upfront rather than wait to be caught. It's
deterministic, fixed-seed, and designed to cover 12 distinct failure modes
rather than a large random sample — it proves the mechanism beats naive
alternatives *on the failure modes it's built to stress*, it's not a
population-level significance claim. If I wanted that, I'd need many more
scenarios with randomized parameters and confidence intervals on the
resolution-rate difference."

---

## Part 4 — Deep dive: verification and trust (10 min)

**Q11. How do you know the fix actually worked, not just that it ran without
crashing?**
A: "The verifier does 7 things: snapshot pre-action state → execute →
stabilization window → re-query Prometheus → rebuild post-action
`IncidentState` → run `GuardrailChecker` → accept or auto-rollback.
Resolution guardrails are AUC ≥ tenant minimum, latency ≤ SLA, missing-rate
≤ limit — all three must pass. Regression guardrails separately check the
fix didn't make something *else* worse: cost ≤ 1.10× before-value, FNR ≤
before + 0.05. If either set fails, `should_rollback = True` and rollback
executes automatically, unless the failed action was already a rollback."

**Q12. Why not trust the agent's own predicted `expected_effect`?**
A: "Because that would let an agent grade its own homework. Reward is
computed strictly from the measured before/after `IncidentState` deltas —
the agent's `expected_effect` is only used pre-execution, for ranking. If I
trusted predicted savings as the reward signal, an overconfident agent could
look successful in the metrics forever without ever being checked against
reality."

**Q13. What if verification itself times out or Prometheus is unreachable?**
A: "That's one of the edge cases explicitly tested — `test_edge_cases.py`
covers agent execution timeout with fallback to the next-ranked agent, and
if *every* agent fails, an escalation record is written rather than the
system silently doing nothing. I don't currently have a distinct
'Prometheus unreachable mid-verification' test — if pushed on it, I'd say
that's a real gap, the system assumes Prometheus is reachable during the
stabilization re-query, and a production version would need a
circuit-breaker / retry-with-backoff there before declaring rollback."

---

## Part 5 — Deep dive: the meta-harness bug (10 min, this is the strongest material — the interviewer WILL find this compelling if you tell it well)

**Q14. Tell me about a bug you found in your own system.**
A: "The meta-harness — the offline loop that's supposed to re-tune the
commander's decision weights from accumulated outcomes — was completely
disconnected from the live commander. I built it bottom-up in phases:
agents, then commander, then verification, then reward, then meta-harness.
Each phase had passing unit tests and a demo script that looked correct in
isolation. But the commander reads its weights from a `TenantTierConfig` DB
row via `UtilityScorer.weights_from_tier_config()`. The meta-harness computed
new `ScoringWeights` and version-controlled them to JSON — and stopped
there. The DB columns happened to share a naming convention with the
`ScoringWeights` fields, which made it *look* connected on a surface read,
but nothing ever wrote the tuned values back into that row. The self-tuning
loop ran end to end, produced sensible output, had tests — and never once
influenced a real decision."

**Q15. How did you find it, and how did you fix it?**
A: "I found it by tracing backward from the commander — 'where does this
weight value actually come from at decision time' — instead of trusting that
because each phase had tests, the phases were wired together. I fixed it by
adding `meta_harness/apply.py::sync_tuned_weights()`, which loads-or-creates
the tenant's `TenantTierConfig` row and writes the tuned fields in. I wrote a
new test file that proves the mapping: tune weights → apply → confirm
`UtilityScorer.weights_from_tier_config` returns the new values.

That would've been a fine place to stop, except I asked one more question
before calling it done: 'has this ever actually run against the real
database?' It hadn't — the only caller was a demo script using a throwaway
in-memory SQLite engine. So the fix was correct and tested, and still never
touched production, for the exact same reason as the original bug. Digging
further, I found the fix was masking a second problem: the live commander
never wrote evidence bundles in the first place, so even a real entrypoint
would've analyzed zero incidents. I wired bundle-writing into `CommanderV3`
and built `scripts/tune_weights.py` to run the whole cycle — analyze, tune,
version, apply — against the real DB, pulling tenants from actual incident
history instead of a hardcoded list. Then I verified it end to end by hand:
fired the same incident three times, `retrain` won every time, ran the
tuning script, fired it a fourth time — `threshold` won instead. Full suite
after both fixes: 465 passing."

**Q16. Does your fix have a new problem of its own?**
A: "Yes, and I'd rather surface it than let you find it: the fix applies one
globally-tuned weight set to *every* tenant. But `TenantTierConfig` exists
specifically so a free-tier tenant can weight cost differently than an
enterprise tenant — a global broadcast can silently overwrite that
per-tenant intent. The evidence-bundle analyzer also pools all incidents
together regardless of tenant, so the tuning signal itself isn't segmented
either. Correct fix is per-tenant analysis and per-tenant weight application
— that's the top item in my 'what's next' list, not something I'm treating
as done."

**Q17. Why should I trust that there aren't more bugs like this?**
A: "You shouldn't fully trust that on my word — that's exactly why I write
integration tests that span the whole loop instead of only per-phase unit
tests. The lesson I took from this specific bug is that phase-by-phase
development with passing tests is not evidence the phases are actually
wired together; I now treat 'does this data reach the component that
consumes it' as its own explicit test, not an assumption. I'd also point at
a smaller thing that happened while fixing this as evidence I actually check
my own work rather than just claiming to: my first version of the bundle-
writer wiring defaulted it to always-on, which meant unit tests started
silently writing real trace files into the project's actual `traces/`
directory — 23 stray folders by the time I ran `git status` and asked why
there were untracked files I didn't create on purpose. I didn't find that
from a review comment; I found it because I habitually check `git status`
before assuming a change is clean, and it looked wrong."

---

## Part 6 — Systems / scaling questions (5 min)

**Q18. This uses SQLite. What breaks at scale?**
A: "SQLite single-writer semantics would bottleneck under concurrent writes
from many tenants' incidents. Docker-compose already has a Postgres service
defined as the prod-shaped path — SQLAlchemy ORM means swapping `DB_URL`
doesn't touch agent or commander logic. I haven't load-tested that swap."

**Q19. Two incidents fire for the same tenant at once — what happens?**
A: "Gateway holds a per-tenant `asyncio.Lock`, so same-tenant incidents
serialize — you can't have two remediations racing on the same model
registry or config row. Different tenants don't block each other; each gets
its own lock. Verified in `test_edge_cases.py::TestConcurrentIncidents`,
both the same-tenant-serializes and different-tenants-parallel cases."

**Q20. What's the actual reason one of your tests is skipped, not just
'flaky'?**
A: "`get_engine()` is `@lru_cache`d and defaults to the real on-disk SQLite
file. A test calling `Base.metadata.create_all(get_engine())` shares that
cached engine and file across the whole test session, so results depend on
run order and prior local state. That's why `test_multi_tenant_isolation` is
skipped rather than flaky-and-ignored. My newer tests sidestep it by
creating a private in-memory engine instead of using the cached global one —
but the root fix (making the test suite not depend on the cached global
engine at all) isn't done."

---

## Part 7 — Behavioral (5 min)

**Q21. Tell me about a mistake and what you'd do differently.**
A: (Use the meta-harness disconnect from Part 5 — it's real, root-caused,
and shows the process fix, not just the bug fix.)

**Q22. What would you cut if you had half the time to build this?**
A: "The LangGraph/Claude reconciliation tie-breaker — it only fires when two
proposals are within 10% utility, which is a minority of cases, and the
system runs fine without an API key anyway. I'd keep all 5 agents and the
verification loop; those are the load-bearing parts of the actual claim
'self-healing,' the LLM tie-break is a refinement on top."

**Q23. What was the single highest-leverage design decision?**
A: "Replacing confidence-only ranking with the multi-objective utility
function. It's the one change I can point to a number for — 58% → 75%
resolution rate in the same test harness, same scenarios, only the scoring
function changed."

---

## Part 8 — Closing (5 min)

**Q24. What would you build next if this became a real product?**
A: "In priority order: per-tenant meta-harness tuning, since global tuning
can currently overwrite a tenant's intentional cost/quality tradeoff;
conditioning historical-success on incident subtype instead of just agent
identity; real drift detection (PSI/KS-test) instead of normalized
mean-shift; and delayed-label modeling, since real credit-risk defaults
surface over months, not immediately like this simulation assumes."

**Q25. Any questions for me?**
A: (Have 2 ready, tailored to FDE: "How does your team decide when a
customer-specific config diverges enough from the default that it needs its
own tuned policy rather than a shared one?" — ties directly to the
per-tenant tuning gap you just discussed, shows you're already thinking in
their problem space.)
