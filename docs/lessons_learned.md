# Lessons Learned

Distilled takeaways, not the blow-by-blow — see [bug_diary.md](bug_diary.md)
for the specific incidents these were pulled from.

---

## Verification is the load-bearing part, not the decision logic

Building the agents and the commander was the "fun" part. The part that
actually made the project defensible was building **the thing that checks
whether the fix worked** — most auto-remediation demos stop at "action
executed," and that's not the same as "problem solved." Once verification
and auto-rollback existed, it forced honesty everywhere else in the
system: agents' self-reported confidence and expected effects couldn't be
trusted blindly, so reward had to come from real measured deltas, not
predictions. An autonomous system is only as trustworthy as its ability to
check its own work.

## "Tested" and "wired together" are different claims

Phase-by-phase development with passing unit tests is not evidence the
phases are actually connected. This project was built bottom-up (agents →
commander → verify → reward → meta-harness), each phase with its own
passing tests — and the meta-harness sat disconnected from the live
commander for multiple phases, invisible to every one of those tests,
because none of them asked "does this data reach the component that's
supposed to consume it." That question has to be its own explicit test,
not an assumption inherited from unrelated tests passing.

## A fix can repeat the exact mistake it's fixing, one layer down

The first fix for the meta-harness disconnect (`sync_tuned_weights`) was
correct, and its own test proved the mapping worked — but the test used an
in-memory database, and the only real caller used a throwaway one too. The
fix was tested and still never touched production, for the same root
reason as the bug it fixed. The generalizable version: if a fix's own test
uses a fake version of the exact resource the original bug was about
(a real database, a real file, a real network call), the test can pass
while the underlying claim — "this now works in production" — stays
unverified. Ask "has this ever run against real data, even once" before
calling a fix done, not just "does the test pass."

## Batch-offline + canary is a defensible tradeoff, but say so explicitly

Batch-offline learning with significance-gated updates and canary rollout
is architecturally cleaner than online gradient-based weight updates for
an inference-time decision system — auditable, reversible, no risk of one
bad incident perturbing live scoring mid-stream. But "auto-tuning" as a
term implies continuous learning to most readers. The documentation needs
to say explicitly that this means "batch job + canary," not a live
feedback loop — an accurate claim stated vaguely reads as a bigger claim
than it is.

## A correct fix can introduce a new, narrower problem — say that too

Fixing the meta-harness→commander wiring made tuned weights reach
production, but the fix applies one globally-tuned weight set to *every*
tenant. `TenantTierConfig` exists specifically so a free-tier tenant can
weight cost differently than an enterprise tenant — a global broadcast can
silently overwrite that per-tenant intent. The instinct after a passing
test suite is to call it done; the more useful habit is arguing against
your own most recent work before moving on, not just once it's green.

## Global, cached test infrastructure will eventually leak into results

`get_engine()` being `@lru_cache`d against the real on-disk SQLite file
meant tests sharing that cache depended on run order and prior local
state — invisible until a specific multi-tenant isolation test needed
guaranteed clean state and had to be skipped instead. The lesson isn't
"don't cache" — it's that anything cached and shared across test runs
needs an explicit test-mode override from day one, not a workaround added
after the first flaky failure.

## Habitually checking `git status`/diffs catches self-introduced bugs fast

The bundle-writer default-on bug (see bug_diary.md #2) wasn't found by a
review comment — it was found by noticing untracked files that shouldn't
have existed, right after making a change. Checking `git status` after
every meaningful change, and reading *why* a file is dirty before moving
on, is cheap and catches exactly this class of bug: side effects that are
correct in isolation but wrong by default.

## Local commits and pushed commits are different states — verify both

A fix can be fully committed locally and still be effectively incomplete
in the shared repository, if a file it depends on was never tracked in
the first place (`meta_harness/apply.py`, see bug_diary.md #3). "I
committed it" and "it's in the repo everyone else clones" are different
claims when a dependency slips through `git add`.
