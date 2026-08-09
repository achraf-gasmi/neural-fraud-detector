"""
FraudShield — Hybrid Model Training (FT-Transformer + Temporal GNN)
=====================================================================
Trains the full models.hybrid.FraudDetector using mini-batch neighborhood
sampling over the transaction graph built by data/pipeline/graph_builder.py.

This is the "deep" training path: it captures shared-device/shared-IP/
shared-merchant fraud rings that the tabular-only path (training/train.py)
cannot see, at the cost of a heavier, slower training loop. Run
training/train.py first for a fast baseline; run this once graphs exist
(python scripts/run_pipeline.py builds them) to train the full hybrid model.

Usage: python -m training.train_graph
"""

import pickle
from pathlib import Path

import hydra
import mlflow
import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

from data.pipeline.graph_builder import load_graph
from models.hybrid import FraudDetector
from training.losses import CombinedFraudLoss
from training.metrics import MetricTracker, compute_all_metrics, print_metrics
from training.train import set_seed

# ─────────────────────────────────────────────
# Graph Loading
# ─────────────────────────────────────────────

def load_split_graphs(graph_dir: str):
    train_graph = load_graph(f"{graph_dir}/train_graph.pt")
    val_graph = load_graph(f"{graph_dir}/val_graph.pt")

    with open(f"{graph_dir}/id_maps.pkl", "rb") as f:
        id_maps = pickle.load(f)

    entity_counts = {k: len(v) for k, v in id_maps.items()}
    return train_graph, val_graph, entity_counts


def make_loader(graph, num_neighbors: list, batch_size: int, shuffle: bool) -> NeighborLoader:
    return NeighborLoader(
        graph,
        num_neighbors={edge_type: num_neighbors for edge_type in graph.edge_types},
        input_nodes=("transaction", None),  # every transaction is a seed
        batch_size=batch_size,
        shuffle=shuffle,
    )


# ─────────────────────────────────────────────
# Train / Eval Epoch
# ─────────────────────────────────────────────

def run_epoch(
    model: torch.nn.Module,
    loader: NeighborLoader,
    loss_fn: CombinedFraudLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
) -> dict:
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    total_losses = {"total_loss": 0.0, "focal_loss": 0.0, "anomaly_loss": 0.0}
    all_probs, all_labels = [], []
    n_batches = 0

    for batch in tqdm(loader, desc="Training" if train_mode else "Evaluating", leave=False):
        batch = batch.to(device)
        txn_store = batch["transaction"]
        batch_size = txn_store.batch_size
        y = txn_store.y[:batch_size]
        x_seed = txn_store.x[:batch_size]

        if train_mode:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train_mode):
            out = model(batch)
            losses = loss_fn(
                logits=out["logits"],
                targets=y,
                reconstruction=out.get("reconstruction"),
                original_features=x_seed,
            )

        if train_mode:
            losses["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        for k in total_losses:
            total_losses[k] += losses[k].item()
        n_batches += 1

        all_probs.extend(out["probs"].detach().cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    avg_losses = {k: v / n_batches for k, v in total_losses.items()}
    metrics = compute_all_metrics(np.array(all_labels), np.array(all_probs))
    return {**avg_losses, **metrics}


# ─────────────────────────────────────────────
# Main Training Entry Point
# ─────────────────────────────────────────────

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def train(cfg: DictConfig):
    set_seed(cfg.project.seed)
    device = torch.device(cfg.project.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ── Load Graphs ──
    train_graph, val_graph, entity_counts = load_split_graphs(cfg.data.graph_dir)
    n_features = train_graph["transaction"].x.shape[1]
    logger.info(f"Features: {n_features} | Entities: {entity_counts}")

    num_neighbors = list(cfg.training.graph_neighbors_per_hop)
    train_loader = make_loader(train_graph, num_neighbors, cfg.training.batch_size, shuffle=True)
    val_loader = make_loader(val_graph, num_neighbors, cfg.training.batch_size * 2, shuffle=False)

    # ── Model ──
    gc = cfg.model.gnn
    tc = cfg.model.transformer
    fc = cfg.model.fusion
    model = FraudDetector(
        n_features=n_features,
        transformer_d_token=tc.d_token,
        transformer_n_blocks=tc.n_blocks,
        transformer_n_heads=tc.attention_n_heads,
        transformer_attn_dropout=tc.attention_dropout,
        transformer_ffn_d_hidden_mult=tc.ffn_d_hidden_multiplier,
        transformer_ffn_dropout=tc.ffn_dropout,
        gnn_hidden_channels=gc.hidden_channels,
        gnn_num_layers=gc.num_layers,
        gnn_heads=gc.heads,
        gnn_dropout=gc.dropout,
        gnn_time_encoding_dim=gc.time_encoding_dim,
        n_users=entity_counts["user"],
        n_merchants=entity_counts["merchant"],
        n_devices=entity_counts["device"],
        n_ips=entity_counts["ip"],
        fusion_hidden_dim=fc.hidden_dim,
        fusion_dropout=fc.dropout,
        anomaly_hidden_dim=cfg.model.anomaly_head.reconstruction_dim,
        use_anomaly_head=cfg.model.anomaly_head.enabled,
    ).to(device)

    # GATConv layers use lazy (-1, -1) in/out channels, so their weights
    # don't exist until the first forward pass — materialize them here
    # before touching model.parameters() (e.g. for the optimizer or logging).
    with torch.no_grad():
        warmup_batch = next(iter(train_loader)).to(device)
        model(warmup_batch)
    logger.info(f"Model parameters: {model.count_parameters():,}")

    # ── Loss & Optimizer ──
    loss_fn = CombinedFraudLoss(
        focal_gamma=cfg.training.focal_loss_gamma,
        focal_alpha=cfg.training.focal_loss_alpha,
        anomaly_weight=cfg.training.anomaly_loss_weight,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay,
    )

    tracker = MetricTracker(primary_metric="auprc", patience=10)
    checkpoint_dir = Path(cfg.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ── MLflow ──
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(f"{cfg.mlflow.experiment_name}-hybrid")

    with mlflow.start_run(run_name=f"fraudshield-hybrid-{cfg.project.seed}"):
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))

        for epoch in range(1, cfg.training.epochs + 1):
            logger.info(f"\n── Epoch {epoch}/{cfg.training.epochs} ──")

            train_metrics = run_epoch(model, train_loader, loss_fn, device, optimizer)
            val_metrics = run_epoch(model, val_loader, loss_fn, device, optimizer=None)

            mlflow.log_metrics({
                "train/loss": train_metrics["total_loss"],
                "train/auprc": train_metrics["auprc"],
                "train/f1": train_metrics["f1"],
                "val/loss": val_metrics["total_loss"],
                "val/auprc": val_metrics["auprc"],
                "val/f1": val_metrics["f1"],
                "val/precision": val_metrics["precision"],
                "val/recall": val_metrics["recall"],
            }, step=epoch)

            print_metrics(val_metrics, prefix=f"Val Epoch {epoch}")

            is_best = tracker.update(val_metrics, epoch)
            if is_best:
                ckpt_path = checkpoint_dir / "best_hybrid_model.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": val_metrics,
                    "entity_counts": entity_counts,
                    "config": OmegaConf.to_container(cfg, resolve=True),
                }, ckpt_path)
                logger.success(f"New best AUPRC: {val_metrics['auprc']:.4f} — saved to {ckpt_path}")
                mlflow.log_artifact(str(ckpt_path))

            if tracker.should_stop():
                logger.info(f"Early stopping at epoch {epoch} (best: {tracker.best_epoch})")
                break

        logger.success(f"Training complete. Best AUPRC: {tracker.best_value:.4f} at epoch {tracker.best_epoch}")
        tracker.get_history_df().to_csv(checkpoint_dir / "hybrid_metric_history.csv", index=False)


if __name__ == "__main__":
    train()
