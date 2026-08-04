# Engineering Story Library — Self-Healing ML Pipeline

Pick-and-choose stories for interviews, one page each, same fixed
structure throughout. Categories 10 (project you're most proud of — already
built as 30s/2min/5min in `story-versions.md`), 13 (tell me about yourself),
and 14 (why this company) are personal/per-application, not per-project —
don't duplicate them here.

| # | Category | Signal |
|---|---|---|
| 1 | Biggest Technical Challenge | Deep technical reasoning |
| 2 | Failure | Humility, iteration, learning |
| 3 | Debugging | Systems thinking, persistence, root-cause analysis |
| 4 | Ambiguity | 0→1 thinking |
| 5 | Tradeoff | Engineering judgment |
| 6 | Leadership | Ownership |
| 7 | Customer Obsession | *skipped — no real customer, synthetic tenants only* |
| 8 | Fast Learning | Learning velocity |
| 9 | Disagreement | Independent thinking |
| 11 | Biggest Mistake | Ownership, honesty |
| 12 | Decision With Incomplete Information | Judgment under uncertainty |

Template per story: Situation (2 sentences) → Problem (1 sentence) →
Constraints (Time/Compute/Data/Team/Budget) → Options Considered (A/B/C) →
Decision → Why → Implementation → Outcome (numbers) → What I'd Do
Differently → Lesson → Signals Demonstrated.

---

## 1. Biggest Technical Challenge — Multi-Objective Utility Scoring

**Situation.** The commander had to pick one remediation out of up to five
competing proposals per incident, across three tenants with different
cost/quality/latency priorities on the same underlying model. **Problem.**
Ranking by an agent's own self-reported confidence doesn't capture whether
the action fits *this* tenant's tradeoffs for *this* incident.

**Constraints.** Time: solo, one phase of a 17-phase build. Compute: none —
a scoring formula, not a training problem. Data: no real production outcome
data yet to learn weights from — had to hand-set defaults per incident
type. Team: solo. Budget: n/a.

**Options Considered.**
- A — Confidence-only ranking (already implemented, simplest).
- B — Fixed priority order (deterministic, but static regardless of
  incident type or tenant).
- C — Multi-objective utility function: score expected effect
  (quality/cost/reliability/speed) against tenant-specific weights, plus
  confidence and risk terms.

**Decision.** Built C — `UtilityScorer`: six weighted, normalized
dimensions, different default weights per incident type, per-tenant
overrides from `TenantTierConfig`.

**Why.** Confidence measures how sure an agent is in itself, not whether
its fix suits this incident and tenant. A cost-sensitive and a
quality-sensitive tenant facing the identical drift incident should get
different remediations — no confidence ranking expresses that; a weighted
utility function does.

**Implementation.** Every expected-effect delta (AUC, cost, FNR, latency)
is normalized against the tenant's *live* operating point before weighting
and summing — e.g. `cost_ref` is 20% of current cost/1000 predictions, not
a hardcoded constant. Clipped to [-1, 1]. Top-2 scores within 10% trigger an
LLM tie-break.

**Outcome.** Resolution rate: 58% (confidence-only baseline, tested
explicitly) → 75% (utility-scored), same 60 deterministic trials.

**What I'd Do Differently.** Build the baseline-comparison harness *before*
finalizing the utility weights, not after — would have caught weaker
defaults sooner instead of relying on intuition for the first pass.

**Lesson.** A signal that measures the proposer's self-belief isn't the
same as one that measures fitness for the actual decision — they only
coincide by accident. Testing against a naive alternative is the only way
to know a more complex approach earns its complexity.

**Signals Demonstrated.** Deep technical reasoning, systems design,
willingness to replace a working-but-weaker approach with a
harder-to-build correct one.

---

## 2. Failure — Historical Success Wasn't the Clean Signal It Looked Like

**Situation.** The meta-harness re-tunes commander weights by tracking each
agent's historical success rate from evidence bundles. **Problem.** Success
rate is tracked globally per agent type — not per incident subtype or
severity — so it silently conflates "retrain succeeds on severe drift"
with "retrain should win mild drift too."

**Constraints.** Time: found this during a documentation/review pass, not
budgeted as its own phase. Compute/Budget: n/a. Data: the evidence-bundle
schema was already fixed by the time this was noticed. Team: solo.

**Options Considered.**
- A — Assume global per-agent success rate is sufficient (the original,
  unexamined assumption).
- B — Condition success rate on incident subtype/severity bucket.
- C — Leave it, rely on eligibility gates as the only real guard.

**Decision.** Documented it honestly as an open limitation (C for now,
B queued as future work) rather than quietly shipping a deeper fix under
time pressure or pretending it wasn't a real gap.

**Why.** Eligibility gates (`can_handle()`) prevent the confound from
causing outright wrong-type selections — an agent ineligible for an
incident never enters the candidate pool regardless of its global score.
But within an eligible pool, the blunt signal can still bias ranking in
ways that don't generalize across severity.

**Implementation.** No code change yet — the fix is scoped (segment
`AnalysisResult.agent_metrics` by incident subtype before computing
significance) but deliberately not rushed in without more outcome data to
validate against.

**Outcome.** A named, understood limitation instead of a silent one —
documented in the README, PROJECT_BIBLE, and interview notes rather than
discovered by an interviewer asking a sharp question I hadn't prepared for.

**What I'd Do Differently.** Would design the evidence-bundle schema with
incident subtype as a first-class dimension from the start, rather than
adding it as a follow-on fix.

**Lesson.** An evaluation signal that's *directionally* useful (agents that
work well do get rewarded) isn't automatically *precise* enough to drive
fine-grained decisions — the granularity of what you measure has to match
the granularity of the decision you're using it to make.

**Signals Demonstrated.** Humility, willingness to surface a real gap
unprompted, iterative thinking.

---

## 3. Debugging — The Meta-Harness Was Never Actually Reaching the Commander

**Situation.** The offline meta-harness computes tuned `ScoringWeights` from
incident outcomes; the live commander scores proposals using
`UtilityWeights` loaded from a `TenantTierConfig` DB row. **Problem.** The
two systems shared a naming convention on their fields and nothing else —
nothing ever wrote tuned weights back into the row the commander reads.

**Constraints.** Time: found this while writing documentation, not as a
scheduled debugging task. Team: solo — no second engineer to catch it in
review. Data: each phase (agents, commander, verify, reward, meta-harness)
had its own passing unit tests, which is exactly what made this invisible.

**Options Considered.**
- A — Trust that passing per-phase tests meant the phases were wired
  together (the default assumption up to this point).
- B — Trace backward from the commander: "where does this weight value
  actually come from at decision time," verify every hop.

**Decision.** B. Traced `commander_v3._load_tenant_tier_config()` →
`UtilityScorer.weights_from_tier_config()` back to the DB schema, then
checked what, if anything, in `meta_harness/` ever wrote to that table.
Nothing did.

**Why.** A demo script and passing tests are evidence a component works in
isolation, not evidence it's connected to the thing that's supposed to
consume its output — those are different claims and only the second one
matters for whether the system does what it claims to do end to end.

**Implementation.** Added `meta_harness/apply.py::sync_tuned_weights()` —
loads-or-creates the tenant's `TenantTierConfig` row, writes the tuned
fields, commits. Wrote `tests/test_meta_harness_apply.py` proving the
mapping: tune → apply → `UtilityScorer` reads the new value.

That fix's only caller turned out to be a demo script running against a
throwaway in-memory SQLite engine — so it was tested, correct, and still
never touched the real database, for the same underlying reason as the
original bug. Asking "has this ever run against real data" surfaced a
second gap under the first one: the live commander never wrote evidence
bundles at all, so a real entrypoint would've had nothing to analyze. Fixed
both — wired opt-in bundle writing into `CommanderV3`, built
`scripts/tune_weights.py` to run analyze → tune → version → apply against
the real DB, tenants discovered from actual `IncidentHistory` rows instead
of a hardcoded list.

**Outcome.** Full test suite: 454 → 465 passing. Verified live, not just via
tests: fired the same incident three times (`retrain` won each time), ran
the tuning script, fired it a fourth time — `threshold` won instead, reward
0.150 → 0.250.

**What I'd Do Differently.** Write one integration test spanning the full
loop *when the meta-harness phase started*, even as a stub, instead of
waiting until asked to explain the system out loud to actually trace it —
and the first time I "fixed" this, ask whether the fix's own test used real
data before calling it closed.

**Lesson.** Phase-by-phase development with passing unit tests is not
evidence the phases are wired together — "does this data reach the
component that consumes it" needs to be its own explicit test, not an
assumption inherited from unrelated tests passing. And a fix can repeat the
exact mistake it's fixing, one layer down, if its own test also uses a fake
data source.

**Signals Demonstrated.** Systems thinking, root-cause tracing rather than
symptom-patching, persistence.

---

## 4. Ambiguity — Building the Whole Project With No Spec

**Situation.** No customer, no ticket, no existing system to extend —
"build something that demonstrates closed-loop ML operations" was the
entire brief, self-assigned as a portfolio project. **Problem.** Every
major decision — dataset, architecture, what "self-healing" even means
operationally, how to prove it works — had to be made from nothing.

**Constraints.** Time: self-paced, no external deadline pressure, no team
to divide ambiguity across. Data: had to pick a dataset (UCI credit
default) that could plausibly support a multi-tenant simulation. Budget:
zero — no cloud infra spend, no labeled incident data to start from.

**Options Considered.**
- A — Build a single end-to-end demo script and stop once something "runs."
- B — Define what would make the claim "self-healing" *falsifiable* first
  — i.e., what would prove the system doesn't just execute actions but
  verifies them — then build backward from that bar.

**Decision.** B — treated "does it actually verify outcomes, not just
execute actions" as the north star before writing the agents, and treated
"does it beat naive baselines" as the proof bar before calling any phase
done.

**Why.** Without a customer or spec setting the bar, it's easy to build
something that *looks* sophisticated (multiple agents, an LLM, a scoring
formula) without it being *provably* better than a simple alternative.
Deciding the falsifiability bar up front prevented that.

**Implementation.** 17 build phases, each ending in passing tests; the last
three phases specifically built the baseline-comparison harness (12
scenarios × 5 policies) because "it works" isn't a claim without a
comparison to something naive.

**Outcome.** A working, tested, honestly-documented system — 465 tests,
75% resolution rate vs. 33-58% for naive baselines, with limitations
written down rather than hidden.

**What I'd Do Differently.** Would define the evaluation/comparison
methodology in phase 1 instead of phase 15 — having it early would have
shaped every architectural decision in between, not just validated them
after the fact.

**Lesson.** In the absence of external requirements, the discipline has to
come from self-imposed falsifiability — decide up front what would prove
the thing wrong, or the project drifts toward "looks impressive" instead of
"is actually better."

**Signals Demonstrated.** 0→1 thinking, self-direction, resistance to
building unfalsifiable "impressive-looking" systems.

---

## 5. Tradeoff — Deterministic Scoring vs. LLM Autonomy

**Situation.** Five agents propose remediations; something has to pick the
winner when proposals are close in quality. **Problem.** An LLM could
plausibly make every ranking decision, but that trades away explainability
and determinism for flexibility the system may not need.

**Constraints.** Time/Budget: an LLM call on every incident costs money and
adds latency to every decision, not just close calls. Data: no ground truth
to validate "the LLM chose better" against, at least not yet.

**Options Considered.**
- A — LLM decides every remediation directly (full agent autonomy).
- B — Fully deterministic scoring, no LLM anywhere in the decision path.
- C — Deterministic scoring for the primary decision; LLM only as a
  tie-break when top-2 utility scores land within 10%.

**Decision.** C.

**Why.** Most incidents aren't close calls — the utility function already
separates them clearly. Spending an LLM call (cost, latency, and
non-determinism) on cases that don't need it is waste; reserving it for the
genuinely ambiguous 10%-margin cases is where an LLM's flexibility is
actually worth its cost, and the rest of the system stays fully explainable
and auditable.

**Implementation.** LangGraph reconciliation step, triggered only inside
`UtilityScorer.rank()`'s top-2-within-10%-margin branch, comparing
guardrail history, historical success, and predicted side effects before
finalizing.

**Outcome.** System runs fully and passes its full test suite with zero
API key configured — the LLM path is additive, not load-bearing, which is
itself evidence the deterministic core carries the actual claim.

**What I'd Do Differently.** Would log how often the 10%-margin branch
actually fires in the 60-trial evaluation, to know empirically whether 10%
is the right margin or just a reasonable-sounding number.

**Lesson.** The right question for "should an LLM decide this" isn't
"could it help" — almost anything could — it's "is this specific case
genuinely ambiguous enough that flexibility beats determinism," and most
decisions in most systems aren't.

**Signals Demonstrated.** Engineering judgment, cost/latency awareness,
resistance to LLM-everywhere design by default.

---

## 6. Leadership — Standardizing the Baseline-Comparison Evaluation Methodology

**Situation.** By phase 15, the system had agents, a commander, verification,
and reward — but no proof any of it was better than something simpler.
**Problem.** "It works" isn't a claim without something to compare against,
and nobody was going to build that comparison but me.

**Constraints.** Team: solo — no one to delegate methodology design to or
push back on it. Time: three phases dedicated specifically to building this
(scenario suite, comparison harness, documentation) that could otherwise
have gone to more features.

**Options Considered.**
- A — Ship the adaptive commander and describe it qualitatively ("smarter
  than fixed rules").
- B — Design and build a controlled comparison: fixed scenarios, multiple
  naive policies, identical reward calculation, only the decision logic
  varying.

**Decision.** B — designed 12 deterministic incident scenarios and 5
selection policies (adaptive, always-retrain, fixed-priority,
highest-confidence, cheapest-eligible), holding everything else constant.

**Why.** A qualitative claim ("smarter") isn't falsifiable and doesn't
survive a skeptical technical interviewer's second question. A controlled
comparison with the same scenarios and reward math, varying only the
decision policy, isolates exactly the variable being claimed to matter.

**Implementation.** `tests/test_policy_comparison.py` — later personally
extended with a 5th baseline (`cheapest_eligible`) after realizing the
methodology itself had a gap, not just the system under test.

**Outcome.** 60 trials, resolution rate 75% (adaptive) vs. 33-58% (naive
baselines) — a defensible, specific number instead of a vibe.

**What I'd Do Differently.** Would have made this methodology the first
deliverable of the whole project, not the fifteenth — see the Ambiguity
story above.

**Lesson.** Leadership on a solo project means holding yourself to the
standard an external reviewer would — nobody was going to ask "compared to
what?" so I had to ask myself before someone else did.

**Signals Demonstrated.** Ownership, driving methodology nobody assigned,
self-imposed rigor.

---

## 8. Fast Learning — Multi-Agent Orchestration and Observability From Scratch

**Situation.** Never previously built a LangGraph-based reconciliation flow
or a full OpenTelemetry/Prometheus/Grafana observability stack.
**Problem.** Needed both working correctly for the project's core claims
(explainable tie-breaking; "telemetry is measured, not mocked") to be true,
not just aspirational.

**Constraints.** Time: solo, no mentor or team member who'd built either
before. Compute/Budget: local Docker only, no managed observability
service.

**Options Considered.**
- A — Skip real Prometheus, mock the metrics for the demo (faster to build,
  weaker claim).
- B — Build a real FastAPI prediction service instrumented with real
  Prometheus counters/histograms, scraped genuinely, dashboarded in
  Grafana.

**Decision.** B, for both LangGraph reconciliation and the observability
stack.

**Why.** The entire "environment is simulated but telemetry is real"
positioning (a specific, deliberately chosen claim throughout the project's
docs) is false if the metrics are mocked — the whole point was proving
metrics come from actual inference and injected data conditions, not
generated values.

**Implementation.** Prometheus scrape config, Grafana auto-provisioned
dashboard and datasource, OTel spans around STT/extraction-equivalent
service calls — learned by building the smallest working version first
(one counter, one panel), then expanding.

**Outcome.** Working docker-compose stack with real scrape-based metrics
backing every claim in the "what's real vs. simulated" section of the
project's documentation.

**What I'd Do Differently.** Would read the Prometheus/Grafana
provisioning docs fully before the first attempt — lost time initially
guessing at the datasource auto-provisioning config instead of reading it
straight through.

**Lesson.** The fastest way to learn unfamiliar infrastructure is to build
the smallest possible real version of it first (one metric, one dashboard
panel) rather than trying to design the full system before anything is
running.

**Signals Demonstrated.** Learning velocity, comfort being a beginner in a
new tool while still shipping.

---

## 9. Disagreement — Rejecting My Own "Done" Fix

**Situation.** After finding the meta-harness/commander disconnect (see
Debugging story), I wrote a fix — `sync_tuned_weights()` — that closed the
loop and passed its tests. **Problem.** The instinct after a passing test
suite is to call it done; instead, re-examining the fix surfaced that it
applies one globally-tuned weight set to *every* tenant.

**Constraints.** Time: this reflection happened in the same session as the
fix, no external deadline forcing a stop.

**Options Considered.**
- A — Ship the fix as complete since tests pass and the original bug (loop
  disconnected) is genuinely resolved.
- B — Flag that the fix, while correct for the original bug, introduces a
  new problem: overwriting per-tenant customization that `TenantTierConfig`
  exists specifically to preserve.

**Decision.** B — documented the new limitation explicitly in the README,
PROJECT_BIBLE, and interview notes, rather than presenting the fix as
unqualified progress.

**Why.** "The tests I wrote for this fix pass" only proves the fix does
what I designed it to do — it says nothing about whether what I designed
it to do is actually correct in every dimension that matters. A
free-tier tenant's deliberate cost-weighting shouldn't silently get
overwritten by what worked for an enterprise tenant's incident history.

**Implementation.** No code change beyond documenting the gap and adding it
as the top item in "what's next" — the honest move here was disclosure, not
a rushed second fix.

**Outcome.** A more credible-sounding fix story precisely because it isn't
presented as flawless — "I fixed X, and while fixing it I found and flagged
Y" is a stronger signal than "I fixed X" alone.

**What I'd Do Differently.** Would build the per-tenant segmentation into
the fix from the start rather than as a documented follow-up, if time
allowed — but not at the cost of rushing an unvalidated per-tenant tuning
design into the same change.

**Lesson.** The habit worth building isn't "find bugs in the system" — it's
"find bugs in your own fix for the bug," which requires actively arguing
against your own most recent work instead of relaxing once it's green.

**Signals Demonstrated.** Independent thinking, intellectual honesty,
resistance to declaring victory at the first passing test run.

---

## 11. Biggest Mistake — Building the Meta-Harness Without an End-to-End Test

**Situation.** Same underlying incident as the Debugging story — the
meta-harness never reached the live commander — but the angle here is
ownership of *why* it happened, not the mechanics of finding it.
**Problem.** I built the system bottom-up in phases (agents → commander →
verify → reward → meta-harness), and never wrote a test that spanned the
full loop until asked to explain the system out loud.

**Constraints.** Team: solo, no code reviewer to ask "does this actually
connect to the thing that reads it." Time: each phase was time-boxed
individually, which is exactly what encouraged treating "phase tests pass"
as "phase done" without a cross-phase check.

**Options Considered (in hindsight — what should have been considered at
the time).**
- A — What I actually did: trust per-phase test coverage as sufficient
  evidence of integration.
- B — Write a stub integration test spanning the full loop the moment a
  new phase's output is meant to feed an earlier phase's input.

**Decision (retrospective).** Should have been B from the meta-harness
phase onward — this is a decision about process, not about the bug itself.

**Why this was a mistake, not just a failure.** A failure is something
external revealing a flaw in a reasonable plan. This was self-caused: the
plan itself — validate each phase in isolation and assume composition —
was the gap, not any single line of code. Nobody made me skip the
integration test; time-boxing each phase made it easy to move on once
green.

**Implementation.** Fixed via `meta_harness/apply.py` (see Debugging story
for mechanics) — the important part here is the process change: now treat
"does this data reach its consumer" as its own explicit test whenever a
component's whole purpose is to feed another component, not an assumption
carried over from unrelated passing tests.

**Outcome.** 465 tests passing post-fix, but the more durable outcome is
the process lesson applied going forward, not just this one bug closed —
proven when the same "does this reach production" question caught a second,
deeper instance of the identical gap in my own first fix (see Debugging
story).

**What I'd Do Differently.** Write the cross-phase integration test as a
failing stub *before* building the meta-harness's internals, so the gap
would have been visible from day one of that phase instead of discovered
retroactively.

**Lesson.** "I caused this" is a more useful sentence than "this happened"
— it forces asking what about *my process*, not just what about *this
code*, needs to change so the same shape of gap doesn't recur elsewhere.

**Signals Demonstrated.** Ownership, honest root-cause attribution, process
maturity over one-off bug-fixing.

---

## 12. Decision With Incomplete Information — Setting Utility Weights With No Outcome Data

**Situation.** `UtilityScorer` needs per-incident-type default weights
(quality/cost/reliability/speed/confidence/risk) before the commander can
score anything. **Problem.** At design time, there was zero real production
outcome data to learn these weights from — the meta-harness that's
supposed to tune them from data didn't exist yet, and couldn't, without
weights to start from.

**Constraints.** Data: none — this is the literal chicken-and-egg problem
of bootstrapping a learning system. Time: needed working defaults to
unblock every downstream phase (execution, verification, reward). Team:
solo, no domain expert to consult on "correct" weight values.

**Options Considered.**
- A — Uniform weights across all incident types (simplest, ignores that
  different incidents obviously call for different priorities).
- B — Hand-set defaults per incident type based on domain reasoning (e.g.
  latency incidents should weight speed highest), explicitly flagged as a
  starting hypothesis, not a validated value.
- C — Block the whole project until real outcome data existed to fit
  weights properly.

**Decision.** B.

**Why.** C is a nonstarter — the data can only exist after a working system
generates outcomes to learn from. A is defensible but throws away known
domain structure for no reason (a latency incident obviously should weight
speed more than a drift incident does). B makes the hypothesis explicit and
gives the meta-harness something concrete to correct later, rather than
pretending the defaults are more validated than they are.

**Implementation.** Per-incident-type weight tables in
`commander/utility.py` (e.g. drift: quality 0.40, cost 0.05; latency:
speed 0.40, quality 0.05) — reasoned from domain structure, not fit to
data, and documented as such.

**Outcome.** Good enough to unblock every downstream phase and to produce a
75% resolution rate against naive baselines in the 60-trial evaluation —
but explicitly still a hypothesis the meta-harness exists to eventually
correct with real outcome data, not a claimed-final answer.

**What I'd Do Differently.** Would log the reasoning behind each specific
weight value at the time (not just the final numbers), so later tuning
could tell whether a weight changed because the domain reasoning was wrong
or because real data genuinely diverged from a reasonable starting guess.

**Lesson.** When you can't wait for data before making a decision, the
honest move is making the reasoning explicit and falsifiable — not
pretending a hand-set value is more validated than it is, and building the
mechanism (the meta-harness) that will eventually replace judgment with
evidence.

**Signals Demonstrated.** Judgment under uncertainty, comfort making a
defensible call without perfect information, planning for the decision to
be revisited rather than treating it as permanent.
