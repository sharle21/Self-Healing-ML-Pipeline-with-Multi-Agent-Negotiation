# Incident Detection

How the system decides an incident exists, what it looks like on the wire,
and the dataset/tenant baselines the severity math is computed against.

---

## Model and Dataset

| Property | Value |
|---|---|
| Dataset | UCI Default of Credit Card Clients |
| Rows | ~30,000 |
| Features | 23 |
| Model | LightGBM binary classifier |
| Task | Predict credit-card default probability |
| Metric | ROC-AUC |
| Baseline ROC-AUC | ~0.77 |

The dataset is split into training, baseline calibration, and replay
partitions. The replay partition is sent through the real prediction API to
generate runtime telemetry — see [Traffic Replay](#traffic-replay) below.

---

## Baseline and Threshold Initialization

Tenant configuration values are derived from model evaluation and runtime
benchmarking, not hardcoded guesses.

| Configuration | Source |
|---|---|
| `baseline_auc` | Validation set evaluation |
| `min_auc` | Baseline minus tolerated degradation |
| `decision_threshold` | Threshold search on validation data |
| `baseline_latency_p95_ms` | Measured inference benchmark |
| `latency_sla_ms` | Baseline plus tolerance or explicit SLA |
| `max_missing_rate` | Operator-defined data-quality limit |
| `daily_cost_budget_usd` | Tenant business profile |
| `false_positive_cost` / `false_negative_cost` | Tenant business profile |

Business limits remain operator-defined because they represent requirements,
not model properties.

### Multi-Tenant Configuration

One shared model; per-tenant policy, thresholds, SLAs, and cost weights.

| Tenant | Priority | Latency SLA | Primary Risk |
|---|---|---:|---|
| Enterprise | Model quality | 150 ms | False negatives ($5,000 each) |
| Standard | Balanced | 100 ms | Quality and latency |
| Free | Cost | 200 ms | Inference expense |

Commander utility weights are also per-tenant: a cost-sensitive tenant may
prefer threshold adjustment for the same drift incident that triggers
retraining for an accuracy-sensitive tenant.

---

## Observability

### Model-quality metrics
- `ml_model_auc`, `ml_model_precision`, `ml_model_recall`
- `ml_false_positive_rate`, `ml_false_negative_rate`

### Data-quality metrics
- `ml_missing_rate`, `ml_duplicate_rate`, `ml_schema_violations_total`
- `ml_batch_volume`, `ml_feature_drift_score`, `ml_max_feature_drift_score`

### Prediction metrics
- `ml_predictions_total`, `ml_prediction_latency_seconds`
- `ml_prediction_errors_total`, `ml_positive_predictions_total`

### Control-plane metrics
- `ml_remediation_proposals_total`, `ml_agent_wins_total`
- `ml_incidents_total`, `ml_remediation_duration_seconds`
- `ml_remediation_reward`, `ml_current_decision_threshold`

Metrics are labeled by tenant and model version. High-cardinality values
(request IDs) are intentionally excluded.

---

## Traffic Replay

The replay engine sends held-out dataset rows through the real `/predict`
endpoint. For each request the system records tenant, model version,
prediction probability, class decision, request latency, and error status.

```bash
uv run python scripts/replay.py
```

---

## Fault Injection Scenarios

| Scenario | Mechanism |
|---|---|
| Covariate drift | Multiply high-importance feature values |
| Missing-data incident | Set selected fields to null |
| Duplicate-data incident | Repeat rows within the replay window |
| Schema incident | Remove required fields |
| Latency incident | Add controlled delay inside inference path |
| Cost incident | Increase traffic volume beyond budget |

```bash
uv run python scripts/trigger_incidents.py --once
```

---

## Evidence Schema

The incident detector compares Prometheus snapshots against policy limits
from SQLite. Detected incidents carry severity, affected metrics, baseline
values, current values, and supporting evidence.

```json
{
  "tenant_id": "enterprise",
  "incident_type": "drift",
  "severity": 0.81,
  "evidence": {
    "baseline_auc": 0.773,
    "current_auc": 0.701,
    "auc_drop": 0.072,
    "max_feature_drift": 1.84,
    "drifted_features": ["LIMIT_BAL", "BILL_AMT1"]
  }
}
```

---

## Severity Formulas

Severity is computed by a per-type formula with named, stored components —
so severity is auditable, not a black-box number.

**Drift severity:**
```
0.45 × normalised AUC drop
+ 0.35 × normalised feature drift
+ 0.20 × normalised affected volume
```

**Data-quality severity:**
```
0.40 × missing + 0.25 × schema + 0.15 × duplicates + 0.20 × volume
```

**Latency severity:**
```
0.60 × latency_ratio + 0.25 × error_rate + 0.15 × traffic_volume
```

**Cost severity:**
```
0.70 × budget_overrun + 0.30 × cost_growth
```

All component values are stored with the incident.

---

## Future Work

- PSI and KS-test drift metrics (current method: normalised mean shift —
  simple, works for this dataset, wouldn't generalise to high-dimensional
  or streaming features)
- Delayed and partially observed labels (currently assumes labels available
  immediately; real credit-risk defaults surface over months)
