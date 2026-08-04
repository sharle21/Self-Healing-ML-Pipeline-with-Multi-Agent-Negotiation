# Story Versions — Self-Healing ML Pipeline

Resume bullets get you the interview in 10 seconds. These get you through
the next 10 minutes. Interviews are conversations, not presentations — the
5-minute version below is modular, not a script to recite start to finish.

> **If the interviewer remembers only one thing, it should be:**
> *"Execution isn't the finish line — verification is. I built the layer
> that checks whether a fix actually worked, not just whether it ran."*
>
> Come back to this line no matter where you get interrupted — 30 seconds
> in or 5 minutes in, this is what should stick.

**Be ready to show, not just tell:** every number below is backed by a real
file — `tests/test_policy_comparison.py` (60-trial results),
`tests/test_meta_harness_apply.py` (the weight write-back mapping),
`tests/test_tune_weights_script.py` + `tests/test_commander_v3_bundle_writer.py`
(the production entrypoint that actually runs it against real data),
`uv run pytest tests/` (465 passing). If asked "can I see that," you can.

---

## Why I built this (say before architecture, not after)

"I kept coming back to a simple question: if a system can automatically
detect that a model is drifting, why do we still wake a human up to decide
the fix? Most monitoring tools stop at detection — they page someone and
stop. I wanted to see if the decision itself could be automated safely,
while staying explainable enough that I'd trust it."

---

## 15 seconds — elevator / "what are you working on?"

"I built a system that watches an AI model in production, notices when it's
quietly going bad, and picks the right fix automatically instead of someone
scrambling at 3am."

*(Say this, then stop. Let them ask the next question.)*

---

## 30 seconds — recruiter screen (hook first)

"Most ML monitoring systems stop at detection — they tell you something's
wrong and page an engineer. I wanted to build the next step: a system that
decides the right remediation automatically, and proves it actually solved
the problem instead of just executing something.

Five specialist strategies compete for each incident — adjust a threshold,
retrain, roll back, fall back to a simpler rule, repair bad data. A
commander picks the best one for the situation using a weighted scoring
function, not one fixed rule, then verifies the fix worked and rolls back
automatically if it didn't. Tested against simpler strategies, mine resolved
75% of incidents versus 33–58% for the naive approaches."

---

## 2 minutes — most technical interviews (this is the one to polish most — you'll use it 80% of the time)

Start with the hook + why-I-built-this, then:

"I ended up splitting the problem into three parts.

**First, detection.** A real dataset gets replayed through a live prediction
API, so metrics — latency, AUC, drift, data quality — come from real
inference, not mocked data. Prometheus scrapes that continuously, and an
incident gets raised when a metric crosses a tenant's threshold.

**Second, decision.** Five agents each check if they're eligible to help and
propose a fix. Early version picked whichever agent was most confident in
itself — that's not the same as picking the *right* fix. I replaced it with
a weighted utility function across quality, cost, reliability, speed, and
risk, weighted differently per tenant and per incident type. A cost-sensitive
tenant might prefer a cheap threshold tweak for the same incident that
triggers a full retrain for a quality-sensitive one.

**Third, verification.** This is the part I think actually matters most —
it doesn't just trust that the action worked. It re-measures state
afterward against explicit guardrails and auto-rolls-back if the fix failed
or made something else worse.

I proved this whole approach beats the obvious alternatives, not just sounds
nicer — compared against four naive strategies across 60 deterministic
trials, and mine won on both resolution rate and reward."

*(Natural stopping point. If they want more, they'll ask "how does scoring
work" or "how do you verify it" — that's your cue into the deep dive.)*

---

## 5 minutes — modular deep dive

Don't recite this linearly. Give the overview, pause, and let the
interviewer pick which module to go deeper on.

```
Overview → Architecture → Scoring → Verification → Bug → Evaluation → Limitations → Lessons
```

**Overview:** the 2-minute version above.

**Architecture module** (if asked "walk me through the system"): dataset
replay → FastAPI prediction → Prometheus → incident detector → 5 agents →
commander → executor → guardrail check → outcome store → offline
meta-harness that periodically re-tunes the commander's weights.

**Scoring module** (if asked "how does it decide"): "Utility is a weighted
sum of six normalized dimensions — AUC delta, cost delta, false-negative-rate
delta, latency delta, the agent's own confidence, minus risk. Every delta is
normalized against the tenant's *live* operating point, not a hardcoded
constant, so a $5 cost change means something different to a free-tier
tenant than an enterprise one. When two proposals land within 10% of each
other, a secondary reconciliation step runs — the only place an LLM touches
the decision path, since I wanted the core logic to stay fully explainable."

**Verification module** (if asked "how do you know it worked"): "Seven
steps: snapshot before, execute, wait for stabilization, re-query real
metrics, rebuild state, check two guardrail sets — resolution guardrails
like AUC and latency, and regression guardrails checking the fix didn't
quietly break something else — then accept or auto-rollback. Reward comes
strictly from those measured deltas, never from what the agent predicted,
otherwise an agent could look good by just being confident, not by being
right."

**Bug module** (if asked "tell me about a challenge" or a mistake): "I built
this bottom-up in phases, each with passing tests. Tracing backward from
'where does the commander's weight actually come from at decision time,' I
found the offline meta-harness computed new weights and saved them to
JSON — and never wrote them back into the database row the commander reads.
Two systems sharing a naming convention and nothing else. I fixed it and
wrote a test proving the mapping worked — then asked one more question
before calling it done: had that fix ever run against the real database?
It hadn't; its only caller used a throwaway in-memory one. Same mistake,
one layer deeper, plus a second gap under it — the live commander never
wrote the evidence the tuner needs in the first place. Fixed both, built a
real entrypoint script that runs the whole cycle against production data,
and verified it by hand: same incident three times picks one agent, run the
tuner, run it a fourth time, a different agent wins. And I'd rather surface
this myself than have you find it: it still applies one tuned weight set to
*every* tenant, which can overwrite a tenant's deliberate cost/quality
tradeoff. Per-tenant tuning is next, not something I'm calling done."

**Evaluation module** (if asked "how did you measure success"): 12
deterministic scenarios × 5 policies, 60 trials, adaptive commander at 75%
resolution / 0.346 mean reward vs. 33–58% / 0.143–0.285 for naive baselines.

**Limitations module** (say some of this unprompted — it reads as maturity):
replayed historical traffic, not live production; 12 scenarios prove the
mechanism beats naive baselines on the failure modes it's built to stress,
not a population-level statistical claim; SQLite is fine for the demo, not
concurrent production load, though Postgres is already the swap-in path.

**Lessons module:** "The biggest lesson was that execution isn't the finish
line — verification is. Once I built the verification loop, it forced every
upstream component to become more rigorous, because success was measured by
observed outcomes, not predicted ones."

---

## One sentence I want them to remember

"I wasn't trying to automate retraining — I was trying to automate
engineering judgment, safely."

---

## Whiteboard version (practice drawing this with just a marker)

```
Data → Prediction API → Telemetry → Incident → Five Agents → Commander
→ Execute → Verify → Rollback?
```

---

## Questions I hope they ask (strongest territory — steer here if you can)

- Why five agents instead of one policy?
- Why not reinforcement learning?
- Why weighted utility instead of confidence-only ranking?
- Why deterministic scoring, with the LLM only as a tie-breaker?
- Why replay instead of live traffic?
- Why verification *after* remediation, not just execution?
- Why canary rollout for weight changes?
- How did you actually evaluate success?

## Weak opening questions (answer briefly, then steer back up a level)

- "How did you use FastAPI?" / "Which database?" / "Which Docker image?" /
  "Which Python version?"
- These are implementation trivia, not judgment. Answer in one sentence,
  then redirect: *"FastAPI fit the synchronous request/response shape of
  prediction — the more interesting decision was how remediation policies
  get selected once an incident fires..."*

---

## The takeaway, not a feature list

"Most auto-remediation demos stop at 'the action executed' — that's not the
same as 'the problem is solved.' Execution isn't the finish line;
verification is. That's the thing I'd want you to remember about this
project tomorrow morning, more than any single number in it."
