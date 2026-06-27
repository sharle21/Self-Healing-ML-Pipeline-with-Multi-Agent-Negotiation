# Self-Healing ML Pipeline with Multi-Agent Negotiation

A 28-day project demonstrating a production-grade self-healing ML pipeline where specialist agents autonomously resolve incidents through negotiation and business-cost optimization.

## Overview

This pipeline addresses a common problem: ML models fail in production for many reasons (data drift, quality degradation, latency breaches, cost overruns), and engineers need to detect, diagnose, and fix them fast. Rather than a single static fix, this system:

1. **Detects** incidents via 4 specialized monitors (drift, data quality, cost, latency)
2. **Proposes** solutions through 5 specialist agents (threshold, retrain, rollback, fallback, data repair)
3. **Negotiates** via a Commander agent using weighted scoring + optional reconciliation debates
4. **Learns** by analyzing evidence bundles and auto-tuning decision weights
5. **Observes** every step with audit trails and Grafana dashboards

## Quick Start

### Prerequisites
- Python 3.12+
- `uv` package manager
- Docker + Docker Compose (for Prometheus + Grafana)

### Setup
```bash
uv sync
uv run python scripts/train.py          # Train initial model
```

### Run End-to-End Demo
```bash
uv run python src/self_healing_pipeline/demo_week3.py
```

### Start API + Monitoring
```bash
docker-compose up -d                    # Start Prometheus + Grafana
uv run python -m uvicorn src.self_healing_pipeline.api.main:app --reload
```

Then visit:
- **API:** http://localhost:8000/health
- **Metrics:** http://localhost:9090
- **Dashboards:** http://localhost:3000 (admin/admin)

## Architecture

5 agents propose → Commander scores → Reconciliation debate (optional) → Execute with fallback chain → Memory learns → Meta-harness tunes weights

## Testing

```bash
uv run pytest tests/                    # All 183 tests
```

## Key Features

- **Multi-agent negotiation** (scoring formula, optional debates)
- **Business-cost optimization** (per-tenant thresholds, $ impact tracking)
- **3-tier memory system** (cache → DB → evidence bundles)
- **Self-optimization** (meta-harness: analyze outcomes → auto-tune weights)
- **Production observability** (Prometheus metrics, Grafana dashboards)
- **Graceful degradation** (timeout fallback chains, escalation)
- **Full audit trails** (evidence bundles per incident)

## Portfolio Value

Demonstrates multi-agent systems, production observability, self-optimization, concurrent task handling, and test-driven development.

For full architecture details, API reference, and design decisions, see the GitHub repository.

---

**Status:** Week 3 complete (183 tests passing). Days 1-25 implemented. Days 26-28 documentation.
