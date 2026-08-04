# Example Workflow Diagram

The end-to-end scenario from README, as a sequence diagram: drift incident
on `enterprise` → retrain selected → verified → resolved.

```mermaid
sequenceDiagram
    participant R as Replay Engine
    participant P as Prometheus
    participant D as Incident Detector
    participant Ag as Agents (Threshold/Retrain/Rollback)
    participant C as Commander
    participant G as GuardrailChecker

    R->>P: replay batch (drift injected in payment-history features)
    P-->>D: max_feature_drift = 1.84, AUC 0.773 -> 0.701
    D->>C: Incident(type=drift, severity=0.81, tenant=enterprise)
    C->>Ag: analyze(state)
    Ag-->>C: 3 proposals (retrain 0.62, rollback 0.38, threshold 0.21)
    C->>C: select retrain (highest utility, enterprise quality_weight=0.40)
    C->>Ag: execute(retrain plan)
    Ag-->>C: LightGBM v4 trained, validated, registered
    C->>C: wait stabilization window
    C->>P: re-query metrics
    C->>G: check(before, after)
    G-->>C: AUC 0.758 >= 0.75 ✓, latency 88ms ✓, missing 2% ✓ -> resolved
    C->>C: reward = 0.64, evidence bundle stored
```

Also see [../docs/policy-agents.md](../docs/policy-agents.md) for the
per-agent decision-flow diagrams and [../docs/verification.md](../docs/verification.md)
for the full verification sequence (this diagram is the happy path only).
