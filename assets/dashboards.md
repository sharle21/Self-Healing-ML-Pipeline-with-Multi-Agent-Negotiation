# Grafana Dashboards

No screenshot here — these require `docker-compose up -d` and a live
Prometheus feed to render real data, which isn't something this repo can
ship as a static image without it going stale immediately. Panel list below
is generated directly from the provisioning JSON
(`docker/grafana/provisioning/dashboards/*.json`), so it stays accurate
without needing a manual screenshot each time a panel changes.

To see it live: `docker-compose up -d`, then
`http://localhost:3000` (admin/admin).

## Self-Healing ML Pipeline (main dashboard)
- Model AUC by Tenant (rolling 200-row window)
- Feature Drift Score (σ from tenant baseline — >2 = alert)
- LIMIT_BAL Drift Score by Tenant (current)
- Prediction Latency p95 / p99 by Tenant
- Missing Feature Rate by Tenant (>15% = incident)
- Prediction Rate by Tenant
- Incident Rate
- Agent Wins (5m window)

## Agent Leaderboard
- Agent Win Count (24h)
- Agent Proposal Count (24h)
- Agent Leaderboard (table)
- Agent Win Rate Over Time (5m)

## Incident Feed
- Incident Distribution (24h)
- Incident Rate Timeline (5m)
- Incident Counts by Type (Last Hour)

## Memory State Dashboard
- Agent Proposal Rate (5m window)
- Decision Distribution by Agent (24h)
- Memory-Informed Decision Quality (gauge)
- Agent Success Rate Trends

If a real screenshot is wanted for the portfolio README, take it after
running a replay + a few incidents so the panels aren't empty, and drop it
in as `assets/dashboards.png` — this file's panel list will still be
accurate as the caption/reference.
