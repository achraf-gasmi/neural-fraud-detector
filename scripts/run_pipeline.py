"""
FraudShield — Full Pipeline Runner
===================================
Run: python scripts/run_pipeline.py
Executes: data generation → feature engineering → graph construction
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from loguru import logger
from data.generator.synthesizer import generate_dataset
from data.pipeline.features import run_pipeline

def main():
    logger.info("=" * 60)
    logger.info("  FraudShield — Data Pipeline")
    logger.info("=" * 60)

    # Step 1: Generate synthetic data
    logger.info("\n[1/2] Generating synthetic transactions...")
    generate_dataset(
        n_users=5000,
        n_merchants=500,
        n_transactions=500_000,
        fraud_rate=0.015,
        output_path="data/raw/transactions.parquet",
    )

    # Step 2: Feature engineering
    logger.info("\n[2/2] Running feature engineering pipeline...")
    train, val, test = run_pipeline(
        raw_path="data/raw/transactions.parquet",
        output_dir="data/processed",
        artifacts_dir="data/processed/artifacts",
    )

    logger.success("\n✅ Pipeline complete!")
    logger.info(f"  Train: {len(train):,} rows | {train['is_fraud'].sum()} fraud")
    logger.info(f"  Val:   {len(val):,} rows | {val['is_fraud'].sum()} fraud")
    logger.info(f"  Test:  {len(test):,} rows | {test['is_fraud'].sum()} fraud")
    logger.info("\nNext steps:")
    logger.info("  python -m training.train")

if __name__ == "__main__":
    main()
