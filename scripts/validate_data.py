"""
FraudShield — Data Validation
==============================
Validates processed data quality before training.
Run: python scripts/validate_data.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from loguru import logger

from data.pipeline.features import FEATURE_COLUMNS


def validate(df: pd.DataFrame, split_name: str) -> bool:
    passed = True
    errors = []

    # Check shape
    if len(df) == 0:
        errors.append(f"{split_name}: Empty DataFrame")
        passed = False

    # Check required columns
    required = FEATURE_COLUMNS + ["is_fraud", "timestamp", "user_id"]
    for col in required:
        if col not in df.columns:
            errors.append(f"{split_name}: Missing column '{col}'")
            passed = False

    # No amount > 0 check here: by this point `amount` has already been
    # StandardScaler-transformed (it's in FEATURE_COLUMNS), so ~half of any
    # real dataset is legitimately <= 0. Raw-dollar positivity is guaranteed
    # at generation time in data/generator/synthesizer.py instead.

    # Check for NaN in features
    nan_count = df[FEATURE_COLUMNS].isna().sum().sum()
    if nan_count > 0:
        errors.append(f"{split_name}: {nan_count} NaN values in features")
        passed = False

    # Check for Inf
    inf_count = np.isinf(df[FEATURE_COLUMNS].values).sum()
    if inf_count > 0:
        errors.append(f"{split_name}: {inf_count} Inf values in features")
        passed = False

    # Check label distribution
    fraud_rate = df["is_fraud"].mean()
    if fraud_rate == 0:
        errors.append(f"{split_name}: No fraud cases in split")
        passed = False
    if fraud_rate > 0.3:
        errors.append(f"{split_name}: Suspiciously high fraud rate: {fraud_rate:.2%}")
        passed = False

    if errors:
        for e in errors:
            logger.error(f"  ❌ {e}")
    else:
        logger.success(f"  ✅ {split_name}: All checks passed ({len(df):,} rows, {fraud_rate:.2%} fraud)")

    return passed


def main():
    logger.info("Running data validation...")
    all_passed = True

    for split in ["train", "val", "test"]:
        path = f"data/processed/{split}.parquet"
        if not os.path.exists(path):
            logger.warning(f"  ⚠️ {split}.parquet not found — skipping")
            continue
        df = pd.read_parquet(path)
        result = validate(df, split)
        all_passed = all_passed and result

    if all_passed:
        logger.success("\n✅ All validation checks passed. Safe to train.")
        sys.exit(0)
    else:
        logger.error("\n❌ Validation failed. Fix data issues before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
