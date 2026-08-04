# Storage

SQLite is used as the persistent control-plane store.

| Table | Purpose |
|---|---|
| `tenant_policy` | SLAs, cost limits, agent eligibility |
| `tenant_tier_config` | Commander utility weights per tenant |
| `model_validation_report` | Registered model versions and AUC |
| `runtime_deployment_profile` | Latency benchmarks per version |
| `incident_history` | Detected incidents with severity |
| `remediation_action` | Selected actions, outcomes, rewards |

SQLite was chosen for portability and local reproducibility. The interfaces
could be backed by PostgreSQL without changing agent or commander logic —
`docker-compose.yml` already has a Postgres service defined as the
prod-shaped path; swapping `DB_URL` doesn't touch agent or commander code.

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | `""` | Claude API key (optional — heuristic-only mode without it) |
| `DB_URL` | `sqlite:///./pipeline.db` | control-plane database connection |
| `PROMETHEUS_URL` | `http://localhost:9090` | Prometheus HTTP API |
| `API_HOST` | `127.0.0.1` | API bind host |
| `API_PORT` | `8000` | API bind port |
| `LOG_LEVEL` | `INFO` | logging verbosity |
| `USE_REPLAY_FIXTURES` | `false` | use recorded fixtures instead of live replay |
| `HAIKU_MODEL` | `claude-haiku-4-5-20251001` | model used for lightweight LLM calls |
| `SONNET_MODEL` | `claude-sonnet-4-6` | model used for reconciliation debates |

Set via `.env` file or environment. See
`src/self_healing_pipeline/config/settings.py`.

---

## Repository Structure

```
src/self_healing_pipeline/
├── agents/              # 5 specialist agents
├── commander/           # CommanderV3, UtilityScorer, LangGraph reconciliation
├── gateway/             # Incident events, API gateway
├── meta_harness/        # Offline weight optimisation (analyzer, tuner, canary, apply)
├── monitors/            # Drift, quality, business monitors
├── observability/       # IncidentState, SeverityCalculator, TelemetryCollector
├── verification/        # RewardCalculator (OutcomeReward), GuardrailChecker
└── api/                 # FastAPI endpoints

scripts/
├── train.py               # train baseline model on UCI dataset
├── replay.py               # replay UCI test set through live prediction API
├── trigger_incidents.py    # watch Prometheus, fire real incidents into CommanderV3
└── tune_weights.py         # analyze evidence -> tune -> version -> apply to live DB
```

See [testing.md](testing.md) for the `tests/` layout.

---

## Future Work

- PostgreSQL-backed control-plane storage (interfaces already support the
  swap; not yet load-tested)
- Production deployment (Docker, Kubernetes)
