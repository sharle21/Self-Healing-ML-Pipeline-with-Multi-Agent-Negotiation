# Resume Bullets — Self-Healing ML Pipeline

> **If the interviewer remembers only one thing, it should be:**
> *"Execution isn't the finish line — verification is. I built the layer
> that checks whether a fix actually worked, not just whether it ran."*

**This is a library, not a resume.** A resume gets 2 bullets from this file,
matched to the job. The rest become interview stories, application-question
answers, LinkedIn posts, or README material.

**Rules applied to every entry below:**
- One bullet sells **one** engineering strength, self-contained — a reader
  skimming for 8 seconds should know *what the project is* from the bullet
  alone, not just the isolated technique. Every bullet names the system
  ("ML incident-response system") even when the sentence is really about one
  component of it.
- The number has to *prove the idea*, not just sound big. "75% resolution
  rate" proves the decision system works. Vanity counts (test counts, metric
  counts) are cut from bullets and kept as follow-up material instead.
- Lead with the capability unlocked; technology is the answer to "how," not
  the subject of the sentence.
- Varied verbs (built / designed / replaced / shipped / found), not "designed"
  five times in a row — a resume has rhythm.
- A good bullet makes the reader ask a follow-up question.

Each row: bullet, the one signal it sells, a 1-5 star resume-strength rating,
and the follow-up question it should provoke.

---

## The library

| Bullet | Signal | Strength | Likely follow-up |
|---|---|:---:|---|
| Built a closed-loop ML incident-response system that selects among five competing remediation strategies using adaptive utility scoring, resolving 75% of simulated production incidents vs. 33–58% for rule-based baselines across 60 controlled trials. | ML systems design | ★★★★★ | "How did you compute the score?" |
| Designed a multi-agent ML incident-response system where an LLM is invoked only when deterministic scoring produces statistically indistinguishable remediation options, keeping routine decisions explainable while reserving model reasoning for genuinely ambiguous cases. | Engineering judgment | ★★★★★ | "Why only for ties? Why not full agent autonomy?" |
| Found that an entire self-tuning subsystem in an ML remediation pipeline was silently disconnected from production decision-making for multiple build phases — invisible to per-component unit tests, and still true after a first fix whose own test used a fake database — then built the real production entrypoint and verified live that tuned weights change which agent wins. | Debugging ability | ★★★★★ | "How did you find it if the tests were passing?" |
| Built a policy-adaptation loop for an ML remediation system that re-tunes decision weights from accumulated outcomes, but only after a significance test confirms the performance difference isn't noise. | Experimental rigor | ★★★★☆ | "How did you determine statistical significance?" |
| Architected a multi-tenant control plane that safely coordinates concurrent ML incident remediation across independent customer environments without cross-tenant interference. | Distributed systems | ★★★★☆ | "How did you prevent race conditions between tenants?" |
| Shipped remediation-policy changes through canary rollout with automatic rollback when a new policy underperformed the current one, applying safe-deployment discipline to an AI decision system, not just code. | Production engineering | ★★★★☆ | "What triggers the rollback, exactly?" |
| Introduced automatic post-action verification for an ML remediation system, re-measuring state after every action and reverting changes that fail explicit guardrails, so unsafe remediation never silently reaches production. | Reliability | ★★★★☆ | "What counts as a guardrail failure?" |
| Replaced simulated monitoring with live operational telemetry for an automated ML remediation system, so policy decisions are driven entirely by runtime model health instead of synthetic inputs. | Observability | ★★★☆☆ | "How do you know the telemetry is real and not mocked?" |

---

## If I only had room for two

Closed-loop adaptive incident response + LLM-only-as-tie-breaker. First
communicates the overall technical ambition with strong quantitative
evidence; second demonstrates judgment about *when not* to reach for AI,
which is rarer and more memorable than "I used an LLM."

## Pick per company, not per favorite

- **Anthropic-style (judgment/safety-flavored)** → closed-loop remediation + LLM tie-breaker
- **Databricks-style (platform/infra)** → multi-tenant control plane + live telemetry
- **Scale AI-style (data/eval-flavored)** → verification/guardrails + policy adaptation
- **Big-tech generalist** → adaptive scoring + verification

Same library, different 2 bullets selected per target — don't build one
resume and reuse it everywhere.

---

## Kept as interview material, not resume bullets

True and worth having ready, but proof-details that belong in the follow-up
answer, not the headline:
- 465 passing tests, 1 documented skip
- 15+ Prometheus metrics across 4 categories
- SQLite/Postgres interchangeable via SQLAlchemy
- Per-tenant `asyncio.Lock`
- p<0.05 binomial significance gate, n≥5 minimum sample
