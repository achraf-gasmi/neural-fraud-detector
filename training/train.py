"""
FraudShield — Training Loop
============================
Full training pipeline with MLflow tracking, early stopping,
learning rate scheduling, and checkpoint management.
"""

import random
from pathlib import Path

import hydra
import mlflow
import numpy as np
import pandas as pd
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from data.pipeline.features import FEATURE_COLUMNS
from models.tabular import TabularFraudDetector
from training.losses import CombinedFraudLoss
from training.metrics import MetricTracker, compute_all_metrics, print_metrics

# ─────────────────────────────────────────────
# Seed
# ─────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class TransactionDataset(torch.utils.data.Dataset):
    """Simple tabular dataset for the FT-Transformer path (no graph)."""

    def __init__(self, df: pd.DataFrame, feature_cols: list[str]):
        self.X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.y = torch.tensor(df["is_fraud"].values, dtype=torch.long)
        # Normalized timestamps for temporal encoding
        ts = df["timestamp"].astype(np.int64) if hasattr(df["timestamp"], "astype") else df["timestamp"]
        ts_norm = (ts - ts.min()) / (ts.max() - ts.min() + 1e-8)
        self.timestamps = torch.tensor(ts_norm.values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.timestamps[idx]


def get_weighted_sampler(labels: torch.Tensor) -> WeightedRandomSampler:
    """Over-sample minority class via weighted random sampling."""
    class_counts = torch.bincount(labels)
    weights = 1.0 / class_counts.float()
    sample_weights = weights[labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


# ─────────────────────────────────────────────
# Training Functions
# ─────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: CombinedFraudLoss,
    device: torch.device,
    cfg: DictConfig,
) -> dict:
    model.train()
    total_losses = {"total_loss": 0, "focal_loss": 0, "anomaly_loss": 0}
    all_probs, all_labels = [], []

    for X, y, ts in tqdm(loader, desc="Training", leave=False):
        X, y, ts = X.to(device), y.to(device), ts.to(device)

        optimizer.zero_grad()
        out = model(X)

        losses = loss_fn(
            logits=out["logits"],
            targets=y,
            reconstruction=out.get("reconstruction"),
            original_features=X,
        )

        losses["total_loss"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k in total_losses:
            total_losses[k] += losses[k].item()

        all_probs.extend(out["probs"].detach().cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    n_batches = len(loader)
    avg_losses = {k: v / n_batches for k, v in total_losses.items()}
    metrics = compute_all_metrics(np.array(all_labels), np.array(all_probs))
    return {**avg_losses, **metrics}


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: CombinedFraudLoss,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss = 0
    all_probs, all_labels = [], []

    for X, y, ts in tqdm(loader, desc="Evaluating", leave=False):
        X, y, ts = X.to(device), y.to(device), ts.to(device)
        out = model(X)
        losses = loss_fn(
            logits=out["logits"],
            targets=y,
            reconstruction=out.get("reconstruction"),
            original_features=X,
        )
        total_loss += losses["total_loss"].item()
        all_probs.extend(out["probs"].cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    metrics = compute_all_metrics(np.array(all_labels), np.array(all_probs))
    metrics["total_loss"] = total_loss / len(loader)
    return metrics


# ─────────────────────────────────────────────
# Main Training Entry Point
# ─────────────────────────────────────────────

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def train(cfg: DictConfig):
    set_seed(cfg.project.seed)
    device = torch.device(cfg.project.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(OmegaConf.to_yaml(cfg))

    # ── Load Data ──
    train_df = pd.read_parquet(f"{cfg.data.output_dir}/train.parquet")
    val_df = pd.read_parquet(f"{cfg.data.output_dir}/val.parquet")

    train_ds = TransactionDataset(train_df, FEATURE_COLUMNS)
    val_ds = TransactionDataset(val_df, FEATURE_COLUMNS)

    train_sampler = get_weighted_sampler(train_ds.y)
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size,
                              sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size * 2,
                            shuffle=False, num_workers=4, pin_memory=True)

    n_features = len(FEATURE_COLUMNS)
    logger.info(f"Features: {n_features} | Train: {len(train_df):,} | Val: {len(val_df):,}")

    # ── Model ──
    model = TabularFraudDetector(n_features=n_features, cfg=cfg).to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ── Loss & Optimizer ──
    loss_fn = CombinedFraudLoss(
        focal_gamma=cfg.training.focal_loss_gamma,
        focal_alpha=cfg.training.focal_loss_alpha,
        anomaly_weight=cfg.training.anomaly_loss_weight,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.training.lr,
        epochs=cfg.training.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=cfg.training.warmup_epochs / cfg.training.epochs,
    )

    tracker = MetricTracker(primary_metric="auprc", patience=10)
    checkpoint_dir = Path(cfg.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── MLflow ──
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name=f"fraudshield-tabular-{cfg.project.seed}"):
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))

        for epoch in range(1, cfg.training.epochs + 1):
            logger.info(f"\n── Epoch {epoch}/{cfg.training.epochs} ──")

            train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, device, cfg)
            val_metrics = eval_epoch(model, val_loader, loss_fn, device)
            scheduler.step()

            # Log to MLflow
            mlflow.log_metrics({
                "train/loss": train_metrics["total_loss"],
                "train/auprc": train_metrics["auprc"],
                "train/f1": train_metrics["f1"],
                "val/loss": val_metrics["total_loss"],
                "val/auprc": val_metrics["auprc"],
                "val/f1": val_metrics["f1"],
                "val/precision": val_metrics["precision"],
                "val/recall": val_metrics["recall"],
                "lr": optimizer.param_groups[0]["lr"],
            }, step=epoch)

            print_metrics(val_metrics, prefix=f"Val Epoch {epoch}")

            # Checkpoint
            is_best = tracker.update(val_metrics, epoch)
            if is_best:
                ckpt_path = checkpoint_dir / "best_model.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": val_metrics,
                    "config": OmegaConf.to_container(cfg, resolve=True),
                }, ckpt_path)
                logger.success(f"New best AUPRC: {val_metrics['auprc']:.4f} — saved to {ckpt_path}")
                mlflow.log_artifact(str(ckpt_path))

            if tracker.should_stop():
                logger.info(f"Early stopping at epoch {epoch} (best: {tracker.best_epoch})")
                break

        # Log best metrics
        best_metrics = tracker.history[tracker.best_epoch - 1]
        mlflow.log_metrics({f"best/{k}": v for k, v in best_metrics.items() if isinstance(v, float)})
        logger.success(f"Training complete. Best AUPRC: {tracker.best_value:.4f} at epoch {tracker.best_epoch}")

        # Save metric history
        tracker.get_history_df().to_csv(checkpoint_dir / "metric_history.csv", index=False)


if __name__ == "__main__":
    train()
