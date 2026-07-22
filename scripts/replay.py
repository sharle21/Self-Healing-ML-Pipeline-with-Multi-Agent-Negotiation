"""Replay UCI test set through live Prediction API to generate real Prometheus metrics.

Instead of mocking telemetry, this script replays the same test rows that were
held out during training — one request at a time — so latency, drift, and error
metrics are real numbers produced by actual inference.

Usage:
    uv run python scripts/replay.py                        # normal replay
    uv run python scripts/replay.py --rate 50              # 50 req/s
    uv run python scripts/replay.py --incident drift       # inject drift mid-replay
    uv run python scripts/replay.py --incident missing     # inject missing features
    uv run python scripts/replay.py --limit 500            # first 500 rows only
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import deque
from typing import Any

import httpx
import numpy as np

logger = logging.getLogger("replay")


def _build_test_set() -> "pd.DataFrame":
    """Re-derive test set using same seed as train.py → deterministic replay."""
    import pandas as pd
    from sklearn.model_selection import train_test_split

    from self_healing_pipeline.pipeline import (
        LABEL_COL,
        fetch_uci_credit_default,
        split_by_tenant,
    )

    df = fetch_uci_credit_default()
    df = split_by_tenant(df, cold_start_frac=0.05, rng=np.random.default_rng(0))

    features = df.drop(columns=[LABEL_COL])
    labels = df[LABEL_COL].astype(int)

    _, x_test, _, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=0, stratify=labels
    )
    x_test = x_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    x_test["_label"] = y_test
    return x_test


def _inject_drift(row: dict[str, Any], label: int = 0) -> dict[str, Any]:
    """Simulate upstream data corruption that causes expensive false negatives.

    For actual defaulters (label=1): override the model's top features to look
    like a low-risk customer → model predicts no default → FN → fn_cost spikes.
    Feature importance order: BILL_AMT1, PAY_AMT*, LIMIT_BAL, AGE, BILL_AMT*.
    For non-defaulters: covariate shift on LIMIT_BAL only (harmless noise).
    """
    out = dict(row)
    if label == 1:
        # Mask actual defaulters as "ideal" customers using top-importance features
        out["BILL_AMT1"] = 0        # no outstanding bill
        out["BILL_AMT2"] = 0
        out["LIMIT_BAL"] = 1_000_000.0  # ultra-high credit limit → looks wealthy
        for col in ("PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"):
            if col in out:
                out[col] = 100_000  # paying off massive amounts → looks responsible
    else:
        if "LIMIT_BAL" in out:
            out["LIMIT_BAL"] = float(out["LIMIT_BAL"]) * 3.0
    return out


def _inject_missing(row: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    """Drop 3 random payment-history features — simulates upstream data loss."""
    pay_cols = [k for k in row if k.startswith("PAY_")]
    if pay_cols:
        drop = rng.choice(pay_cols, size=min(3, len(pay_cols)), replace=False)
        return {k: v for k, v in row.items() if k not in drop}
    return row


def _compute_drift_scores(
    window_features: list[dict[str, Any]],
    training_stats: dict[str, tuple[float, float]],
) -> dict[str, float]:
    """KS-like drift: |mean_window - mean_train| / std_train per numeric feature.

    training_stats must be from the same tenant's baseline, not the overall population.
    """
    if not window_features or not training_stats:
        return {}
    scores: dict[str, float] = {}
    for feat, (train_mean, train_std) in training_stats.items():
        vals = [row[feat] for row in window_features if feat in row and isinstance(row[feat], (int, float))]
        if vals and train_std > 0:
            window_mean = np.mean(vals)
            scores[feat] = float(abs(window_mean - train_mean) / train_std)
    return scores


# Realized cost per wrong prediction by tenant (from tenants.yaml)
_TENANT_COSTS: dict[str, tuple[float, float]] = {
    "standard":   (5.0,   50.0),
    "enterprise": (200.0, 5000.0),
    "free":       (10.0,  100.0),
}


def _push_metrics(
    client: httpx.Client,
    api_url: str,
    tenant_id: str,
    window_preds: deque,
    window_latencies: deque,
    missing_count: int,
    total_count: int,
    training_stats: dict[str, tuple[float, float]],
    window_features: list[dict[str, Any]],
) -> None:
    """Compute rolling stats from window and push to API's /internal/metrics/update."""
    if not window_preds:
        return

    probas = np.array([p for p, _ in window_preds])
    labels = np.array([y for _, y in window_preds])

    try:
        from sklearn.metrics import precision_score, recall_score, roc_auc_score

        auc = float(roc_auc_score(labels, probas)) if len(np.unique(labels)) > 1 else 0.5
        preds_binary = (probas >= 0.5).astype(int)
        precision = float(precision_score(labels, preds_binary, zero_division=0))
        recall = float(recall_score(labels, preds_binary, zero_division=0))
        error_rate = float(np.mean(preds_binary != labels))
        # Expected calibration error approximation
        calibration_error = float(np.mean(np.abs(probas - labels)))
    except ImportError:
        auc, precision, recall, error_rate, calibration_error = 0.5, 0.0, 0.0, 0.0, 0.0

    latencies = np.array(list(window_latencies)) * 1000  # → ms
    p95 = float(np.percentile(latencies, 95)) if len(latencies) > 0 else 0.0
    p99 = float(np.percentile(latencies, 99)) if len(latencies) > 0 else 0.0
    missing_rate = missing_count / max(total_count, 1)

    drift = _compute_drift_scores(window_features, training_stats)

    # --- DataQualityMonitor: real duplicate_rate + schema_violations ---
    import pandas as pd
    from self_healing_pipeline.monitors.quality import DataQualityMonitor

    window_df = pd.DataFrame(window_features) if window_features else pd.DataFrame()
    dq_result = DataQualityMonitor().detect(window_df) if not window_df.empty else None
    duplicate_rate = float(dq_result.duplicate_rate) if dq_result else 0.0
    schema_violations = int(dq_result.schema_violations) if dq_result else 0

    # --- DriftMonitor: fraction of features drifted beyond 1σ ---
    from self_healing_pipeline.monitors.drift import DriftMonitor

    drift_pct = 0.0
    if training_stats and window_features:
        drifted = sum(
            1 for feat, (t_mean, t_std) in training_stats.items()
            if t_std > 0 and feat in drift and drift[feat] > 1.0
        )
        drift_pct = drifted / max(len(training_stats), 1)

    # --- BusinessCostMonitor: cross-validate realized cost ---
    from self_healing_pipeline.monitors.business import BusinessCostMonitor

    fp_cost, fn_cost = _TENANT_COSTS.get(tenant_id, (10.0, 100.0))
    preds_binary = (probas >= 0.5).astype(int)
    n = len(labels)
    fp_rate = float(((preds_binary == 1) & (labels == 0)).sum() / n)
    fn_rate = float(((preds_binary == 0) & (labels == 1)).sum() / n)

    biz_monitor = BusinessCostMonitor(
        false_positive_cost=fp_cost,
        false_negative_cost=fn_cost,
        window_size=n,
    )
    for y_true, y_pred in zip(labels.tolist(), preds_binary.tolist()):
        biz_monitor.record_prediction(int(y_true), int(y_pred))
    biz_result = biz_monitor.detect()
    avg_cost = biz_result.cost_per_prediction  # authoritative from monitor

    try:
        client.post(
            f"{api_url}/internal/metrics/update",
            json={
                "tenant_id": tenant_id,
                "auc": auc,
                "precision": precision,
                "recall": recall,
                "error_rate": error_rate,
                "calibration_error": calibration_error,
                "missing_rate": missing_rate,
                "duplicate_rate": duplicate_rate,
                "schema_violations": schema_violations,
                "latency_p95_ms": p95,
                "latency_p99_ms": p99,
                "cost_per_prediction": avg_cost,
                "false_positive_rate": fp_rate,
                "false_negative_rate": fn_rate,
                "drift_percentage": drift_pct,
                "feature_drift": drift,
            },
            timeout=5.0,
        )
        logger.debug(
            "pushed metrics: tenant=%s auc=%.3f missing_rate=%.3f p95=%.1fms drift=%s",
            tenant_id, auc, missing_rate, p95,
            {k: f"{v:.2f}" for k, v in list(drift.items())[:3]},
        )
    except httpx.RequestError as exc:
        logger.debug("metrics push failed: %s", exc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--rate", type=float, default=10.0, help="requests per second (0=max)")
    parser.add_argument("--limit", type=int, default=0, help="max rows to replay (0=all)")
    parser.add_argument(
        "--incident",
        choices=["drift", "missing", "none"],
        default="none",
        help="inject incident starting at 50%% of rows",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    from self_healing_pipeline.pipeline.loader import TENANT_COL

    logger.info("building test set from UCI dataset (seed=%d)", args.seed)
    test = _build_test_set()
    if args.limit > 0:
        test = test.iloc[: args.limit]

    total = len(test)
    inject_from = total // 2
    rng = np.random.default_rng(args.seed)
    interval = 1.0 / args.rate if args.rate > 0 else 0.0

    # Compute per-tenant baseline stats for drift detection.
    # Must use each tenant's own distribution — not the overall population.
    # Mixing tenants would make enterprise (high limits) always look "drifted"
    # vs the average, and miss drift in standard (low limits) entirely.
    numeric_cols = ["LIMIT_BAL", "AGE", "BILL_AMT1", "BILL_AMT2", "PAY_AMT1", "PAY_AMT2"]
    tenant_training_stats: dict[str, dict[str, tuple[float, float]]] = {}
    for tenant in test[TENANT_COL].unique():
        tenant_rows = test[test[TENANT_COL] == tenant]
        tenant_training_stats[str(tenant)] = {
            col: (float(tenant_rows[col].mean()), float(tenant_rows[col].std() + 1e-9))
            for col in numeric_cols
            if col in tenant_rows.columns
        }

    # Rolling windows per tenant (last 200 rows)
    WINDOW = 200
    windows: dict[str, deque] = {}        # tenant → deque of (proba, label)
    lat_windows: dict[str, deque] = {}    # tenant → deque of latency_s
    feat_windows: dict[str, list] = {}    # tenant → list of feature dicts
    missing_counts: dict[str, int] = {}
    total_counts: dict[str, int] = {}
    push_every = 100  # push metrics every N rows per tenant

    ok = err = 0
    t0 = time.perf_counter()
    logger.info("replaying %d rows → %s  rate=%.1f/s  incident=%s", total, args.api, args.rate, args.incident)

    with httpx.Client(timeout=10.0) as client:
        for i, row in test.iterrows():
            tenant_id = str(row[TENANT_COL])
            label = int(row["_label"])

            features: dict[str, Any] = {
                k: (int(v) if hasattr(v, "item") else v)
                for k, v in row.items()
                if k not in (TENANT_COL, "_label")
            }

            injected_missing = False
            if args.incident != "none" and i >= inject_from:
                if args.incident == "drift":
                    features = _inject_drift(features, label)
                elif args.incident == "missing":
                    features = _inject_missing(features, rng)
                    injected_missing = True

            # Init per-tenant state
            if tenant_id not in windows:
                windows[tenant_id] = deque(maxlen=WINDOW)
                lat_windows[tenant_id] = deque(maxlen=WINDOW)
                feat_windows[tenant_id] = []
                missing_counts[tenant_id] = 0
                total_counts[tenant_id] = 0

            total_counts[tenant_id] += 1
            t_req = time.perf_counter()

            try:
                resp = client.post(
                    f"{args.api}/predict/{tenant_id}",
                    json={"features": features},
                )
                lat_s = time.perf_counter() - t_req

                if resp.status_code == 200:
                    ok += 1
                    proba = resp.json()["probability"]
                    windows[tenant_id].append((proba, label))
                    lat_windows[tenant_id].append(lat_s)
                    feat_windows[tenant_id] = (feat_windows[tenant_id] + [features])[-WINDOW:]
                else:
                    err += 1
                    if injected_missing or resp.status_code == 422:
                        missing_counts[tenant_id] += 1
                    logger.debug("row %d → %d %s", i, resp.status_code, resp.text[:80])
            except httpx.RequestError as exc:
                err += 1
                logger.warning("row %d request failed: %s", i, exc)

            # Push rolling metrics every push_every rows per tenant
            if total_counts[tenant_id] % push_every == 0:
                _push_metrics(
                    client, args.api, tenant_id,
                    windows[tenant_id], lat_windows[tenant_id],
                    missing_counts[tenant_id], total_counts[tenant_id],
                    tenant_training_stats[tenant_id], feat_windows[tenant_id],
                )

            if interval > 0:
                time.sleep(interval)

            if (i + 1) % 200 == 0:
                elapsed = time.perf_counter() - t0
                logger.info(
                    "progress: %d/%d  ok=%d  err=%d  rps=%.1f",
                    i + 1, total, ok, err, (ok + err) / elapsed,
                )

    # Final push for all tenants
    with httpx.Client(timeout=10.0) as client:
        for tenant_id in windows:
            _push_metrics(
                client, args.api, tenant_id,
                windows[tenant_id], lat_windows[tenant_id],
                missing_counts[tenant_id], total_counts[tenant_id],
                tenant_training_stats[tenant_id], feat_windows[tenant_id],
            )

    elapsed = time.perf_counter() - t0
    logger.info(
        "done: %d rows in %.1fs  ok=%d  err=%d  avg_rps=%.1f",
        total, elapsed, ok, err, total / elapsed,
    )
    return 0 if err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
