"""Train LightGBM on UCI Default of Credit Card Clients, split by MS-tier tenants."""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np

from self_healing_pipeline.config import get_settings
from self_healing_pipeline.pipeline import (
    fetch_uci_credit_default,
    persist_model,
    split_by_tenant,
    train_model,
)

logger = logging.getLogger("train")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cold-start-frac", type=float, default=0.05)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = get_settings()

    logger.info("fetching UCI Default of Credit Card Clients")
    df = fetch_uci_credit_default()
    logger.info("fetched %d rows, %d cols", len(df), df.shape[1])

    logger.info("tertile-splitting into MS-tier tenants")
    df = split_by_tenant(
        df, cold_start_frac=args.cold_start_frac, rng=np.random.default_rng(args.seed)
    )
    logger.info("tenant counts: %s", df["tenant_id"].value_counts().to_dict())

    logger.info("training LightGBM")
    result = train_model(df, test_size=args.test_size, random_state=args.seed)
    logger.info("overall ROC-AUC: %.4f", result.overall_auc)
    for tenant, auc in result.per_tenant_auc.items():
        logger.info("  tenant=%s ROC-AUC=%.4f", tenant, auc)

    path = persist_model(result, settings.model_path)
    logger.info("persisted model to %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
