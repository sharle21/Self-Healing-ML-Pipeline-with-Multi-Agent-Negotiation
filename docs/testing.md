# Testing

```bash
uv run pytest tests/                    # 465 tests, 1 skipped
```

| Test module | What it covers |
|---|---|
| `test_policy_comparison.py` | 4-policy comparison across 12 scenarios |
| `test_scenarios.py` | 30 named end-to-end scenario assertions |
| `test_verification_guardrails.py` | GuardrailChecker multi-dimensional checks |
| `test_utility_scorer.py` | UtilityScorer normalisation and ranking |
| `test_outcome_reward.py` | OutcomeReward calculation from before/after states |
| `test_phase9_10.py` | Per-type severity formulas and threshold search |
| `test_layer3_integration.py` | Full Commander end-to-end |
| `test_meta_harness_apply.py` | Tuned `ScoringWeights` -> `TenantTierConfig` -> live `UtilityScorer` |
| `test_commander_v3_bundle_writer.py` | Evidence bundles written match what `EvidenceBundleAnalyzer` expects |
| `test_tune_weights_script.py` | Full analyze -> tune -> version -> apply cycle against a real DB session |
| `test_weight_tuner.py`, `test_weight_tuner_significance.py` | Significance-gated weight adjustment |
| `test_weight_version_control.py` | Weight version save/load/list |
| ... | 20+ additional test modules |

## Edge cases covered (`test_edge_cases.py`)

- all remediation agents fail → escalation logged
- agent execution timeout → fallback to next-ranked agent
- tied proposal scores → reconciliation debate picks winner
- concurrent incidents on same tenant → serialized
- concurrent incidents on different tenants → parallel
- memory tracks execution success/failure across retries

## Integration tests

- `test_layer3_integration.py` — full 3-layer pipeline (observe → decide →
  verify) against real agents, mock telemetry.
- `test_integration_full_loop.py` — multi-tenant end-to-end; one test
  (`test_multi_tenant_isolation`) is skipped. Reason: `db/session.py`'s
  `get_engine()` is `@lru_cache`d and defaults to the real on-disk
  `sqlite:///./pipeline.db`. Tests that call
  `Base.metadata.create_all(get_engine())` share that cached engine and
  file across the whole test session, so results depend on run order and
  prior local state. Newer tests avoid this by creating a private
  in-memory engine per test instead of using the cached global one — the
  root fix (making the suite not depend on the cached global engine at
  all) isn't done yet.
- `test_commander_v3_bundle_writer.py` — confirms `bundle_writer` is
  opt-in and off by default specifically so ordinary unit tests don't
  write real files into the project's `traces/` directory. This guard
  exists because an earlier version of the wiring defaulted it to
  always-on and silently wrote 23 stray trace folders into the real repo
  before that was caught via `git status`.
