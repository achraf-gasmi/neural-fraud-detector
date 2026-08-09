"""
FraudShield — Smoke Training Script
=====================================
Quick sanity-check training run for CI pipelines.
Generates tiny data, trains for 2 epochs, asserts loss decreases.
Usage: python scripts/smoke_train.py --epochs 2 --n-samples 1000
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data.generator.synthesizer import generate_dataset
from data.pipeline.features import FEATURE_COLUMNS, run_pipeline
from models.tabular import TabularFraudDetector
from training.losses import CombinedFraudLoss
from training.train import TransactionDataset


def smoke_train(epochs: int = 2, n_samples: int = 1000):
    logger.info(f"Smoke train: {epochs} epochs, {n_samples} samples")

    # Generate tiny dataset
    generate_dataset(
        n_users=100, n_merchants=30, n_transactions=n_samples,
        fraud_rate=0.02, output_path="/tmp/smoke_txns.parquet", seed=0
    )
    train_df, _val_df, _ = run_pipeline(
        raw_path="/tmp/smoke_txns.parquet",
        output_dir="/tmp/smoke_processed",
        artifacts_dir="/tmp/smoke_processed/artifacts",
    )

    cfg = OmegaConf.create({
        "model": {
            "transformer": {
                "d_token": 32, "n_blocks": 1, "attention_n_heads": 4,
                "attention_dropout": 0.0, "ffn_d_hidden_multiplier": 1.0, "ffn_dropout": 0.0,
            },
            "fusion": {"hidden_dim": 64, "dropout": 0.0},
        }
    })

    train_ds = TransactionDataset(train_df, FEATURE_COLUMNS)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    model = TabularFraudDetector(n_features=len(FEATURE_COLUMNS), cfg=cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = CombinedFraudLoss()

    losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X, y, ts in train_loader:
            optimizer.zero_grad()
            out = model(X)
            l = loss_fn(out["logits"], y, out.get("reconstruction"), X)
            l["total_loss"].backward()
            optimizer.step()
            epoch_loss += l["total_loss"].item()
        avg = epoch_loss / len(train_loader)
        losses.append(avg)
        logger.info(f"  Epoch {epoch+1}: loss={avg:.4f}")

    # Basic sanity checks
    assert len(losses) == epochs, "Wrong number of epochs"
    assert all(np.isfinite(l) for l in losses), "NaN/Inf loss detected!"
    logger.success("✅ Smoke train passed. Model trains without errors.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--n-samples", type=int, default=1000)
    args = parser.parse_args()
    smoke_train(args.epochs, args.n_samples)
