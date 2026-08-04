# Bug Diary

Chronological record of real bugs found in this system, how they were
found, and how they were fixed — kept separate from
[lessons_learned.md](lessons_learned.md), which is the distilled takeaways
rather than the blow-by-blow.

---

## 1. Meta-harness never reached the live commander (found twice)

**Found:** Tracing backward from the commander — "where does this weight
value actually come from at decision time" — instead of trusting that
per-phase passing tests meant the phases were wired together.

`WeightTuner` computed and version-controlled new `ScoringWeights`
correctly. `commander_v3.py` scored live proposals using `UtilityWeights`
from `TenantTierConfig`. The two shared a naming convention
(`business_value_weight` etc.) and nothing else — nothing ever wrote tuned
weights back into the DB row the commander actually reads. The offline
loop ran end-to-end, produced sensible output, had a demo script and
tests — and never once influenced a live decision.

**First fix:** `meta_harness/apply.py::sync_tuned_weights()` — loads or
creates the tenant's `TenantTierConfig` row, writes the tuned fields,
commits. New test file (`tests/test_meta_harness_apply.py`) proved the
mapping: tune → apply → `UtilityScorer` reads the new value.

**Found again, one layer deeper:** That fix's only caller,
`demo_meta_harness.py`, ran it against a throwaway in-memory SQLite
engine — so the fix was correct, tested, and still never touched
production, for the same underlying reason as the original bug. Asking
"has this ever run against real data" surfaced it. Digging further found a
second gap under the first: the live `CommanderV3` never wrote evidence
bundles at all — `BundleWriter` existed but only the old, unused
`commander.py` ever called it — so even a production entrypoint would
have analyzed zero real incidents.

**Second fix:** wired opt-in `bundle_writer` into `CommanderV3`; built
`scripts/tune_weights.py` to run analyze → tune → version → apply against
the real database, discovering tenants from actual `IncidentHistory` rows
instead of a hardcoded list.

**Verified live, not just via tests:** fired the same incident three times
against a fresh DB — `retrain` won every time (utility 0.291). Ran
`scripts/tune_weights.py` (15 incidents analyzed). Fired the incident a
fourth time — `threshold` won instead (utility 0.255, reward 0.150 →
0.250). Full suite: 454 → 465 passing across both fixes.

Mechanics: [implementation.md#meta-harness](implementation.md#meta-harness).
Design rationale: [design.md](design.md#offline-batch-tuning-instead-of-online-gradient-updates).

---

## 2. Self-introduced: bundle writer defaulted to always-on

**Found:** while verifying fix #1 above, `git status` showed 23 untracked
folders under `traces/` that hadn't been there before — created by running
the test suite, not by the manual verification steps.

**Root cause:** the first version of the `bundle_writer` wiring
constructed a default `BundleWriter(settings.traces_dir)` whenever no
writer was explicitly passed to `CommanderV3`. Most existing unit tests
construct `CommanderV3(agents)` without a `bundle_writer` — so every one of
those tests started silently writing real evidence-bundle files into the
project's actual `traces/` directory on every run.

**Fix:** changed the default to `None` (no writes) instead of an
auto-constructed writer — writing is now opt-in, matching how `db_session`
already works on the same class. Production entrypoints
(`trigger_incidents.py`, `demo.py`) pass `bundle_writer` explicitly; tests
that need it (e.g. `test_commander_v3_bundle_writer.py`) pass a
`tmp_path`-scoped one. Added a regression test
(`test_handle_incident_skips_bundle_write_by_default`) asserting no writes
happen without an explicit writer.

**Cleanup:** removed the 23 stray trace folders and the DB rows created
during verification, restoring `pipeline.db`/`traces/` to their
pre-session state before committing anything.

---

## 3. `meta_harness/apply.py` was never actually committed to git

**Found:** user asked "why do I not see commit messages on my repo" —
traced to commits being local-only (never pushed), which surfaced the
question "did everything I changed actually make it into a commit." Running
`git ls-files | grep apply.py` showed it wasn't tracked at all, despite
`scripts/tune_weights.py` (which *was* committed and pushed) importing
`sync_tuned_weights` from it.

**Impact:** a fresh clone of the repo at that point would `ImportError` the
first time `scripts/tune_weights.py` ran, and `tests/test_meta_harness_apply.py`
wouldn't exist either — the fix from bug #1 above looked complete locally
but was silently incomplete in the actual pushed repository.

**Fix:** `git add` + commit `meta_harness/apply.py` and
`tests/test_meta_harness_apply.py`, verified the tests still pass, pushed.

**Lesson applied:** the same "does this actually reach the place that
depends on it" question that caught bug #1 caught this one too, just
applied to git state instead of runtime state.

---

## 4. Test isolation: cached global DB engine leaks across tests

**Found:** while investigating bug #1, noticed `db/session.py`'s
`get_engine()` is `@lru_cache`d and defaults to the real on-disk
`sqlite:///./pipeline.db`. Any test calling
`Base.metadata.create_all(get_engine())` shares that same cached engine
and file across the whole test session — so results depend on run order
and prior local state, not just the test's own setup.

**Impact:** `test_multi_tenant_isolation` in `test_integration_full_loop.py`
is skipped rather than flaky-and-ignored, with the reason documented
in-line.

**Workaround, not a fix:** newer tests (including the ones added for bugs
#1–3) sidestep this by creating a private in-memory engine per test
(`create_engine("sqlite:///:memory:")`) instead of using the cached global
one. The root fix — making the test suite not depend on the cached global
engine at all — is still open, tracked in
[lessons_learned.md](lessons_learned.md).

---

## 5. Historical-success confound (open, not fixed)

Agent historical-success is tracked globally per agent type, not
conditioned on incident subtype/severity — so a boost from "retrain wins
on severe drift" leaks into "retrain vs. mild drift" ranking. Eligibility
gates prevent this from causing outright wrong-type selection, and canary
rollout would catch regressions before full rollout if it were wired into
the production path, but the root cause isn't fixed. Listed honestly as
still open rather than papered over.
